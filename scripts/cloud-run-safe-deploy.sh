#!/usr/bin/env bash
# Deploy a Cloud Run revision without taking the live service down.
# Traffic stays on the previous revision until the new one answers its health
# check. If the new revision never becomes healthy, traffic is put back.
set -euo pipefail

: "${SERVICE:?SERVICE is required}"
: "${REGION:?REGION is required}"
: "${IMAGE:?IMAGE is required}"
: "${HEALTH_PATH:?HEALTH_PATH is required}"

PROBE_FAILURE_THRESHOLD="${PROBE_FAILURE_THRESHOLD:-24}"
PROBE_PERIOD_SECONDS="${PROBE_PERIOD_SECONDS:-10}"
PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-5}"
ATTEMPTS="${ATTEMPTS:-12}"

# Cloud Run requires timeoutSeconds < periodSeconds. A startup probe's
# failureThreshold * periodSeconds also cannot exceed 240 seconds.
if [ "$PROBE_PERIOD_SECONDS" -gt 240 ]; then
  PROBE_PERIOD_SECONDS=240
fi
if [ "$PROBE_PERIOD_SECONDS" -lt 2 ]; then
  PROBE_PERIOD_SECONDS=2
fi
if [ "$PROBE_TIMEOUT_SECONDS" -ge "$PROBE_PERIOD_SECONDS" ]; then
  PROBE_TIMEOUT_SECONDS=$((PROBE_PERIOD_SECONDS - 1))
fi
if [ "$PROBE_TIMEOUT_SECONDS" -lt 1 ]; then
  PROBE_TIMEOUT_SECONDS=1
fi
max_failures=$((240 / PROBE_PERIOD_SECONDS))
if [ "$PROBE_FAILURE_THRESHOLD" -gt "$max_failures" ]; then
  echo "Startup probe window cannot exceed 240s. Lowering failure threshold from ${PROBE_FAILURE_THRESHOLD} to ${max_failures}."
  PROBE_FAILURE_THRESHOLD="$max_failures"
fi
echo "Startup probe: timeout=${PROBE_TIMEOUT_SECONDS}s period=${PROBE_PERIOD_SECONDS}s failures=${PROBE_FAILURE_THRESHOLD}"

echo "Image: ${IMAGE}"
echo "Service: ${SERVICE} (${REGION})"
echo "Deploying with no traffic so the current revision keeps serving."

cat > /tmp/cr_status.py << 'PY'
import json, subprocess, sys

mode, service, region = sys.argv[1], sys.argv[2], sys.argv[3]
raw = subprocess.check_output(
    ["gcloud", "run", "services", "describe", service, "--region", region, "--format", "json"],
    text=True,
)
data = json.loads(raw)
status = data.get("status") or {}
traffic = status.get("traffic") or []

if mode == "previous":
    parts = []
    for item in traffic:
        percent = item.get("percent") or 0
        name = item.get("revisionName") or ""
        if name and percent > 0:
            parts.append(f"{name}={percent}")
    sys.stdout.write(",".join(parts))
elif mode == "urls":
    base = status.get("url") or ""
    candidate = ""
    for item in traffic:
        if item.get("tag") == "candidate" and item.get("url"):
            candidate = item["url"]
            break
    if not candidate and base.startswith("https://"):
        candidate = "https://candidate---" + base[len("https://"):]
    sys.stdout.write(base.rstrip("/") + "\n" + candidate.rstrip("/") + "\n")
elif mode == "latest":
    sys.stdout.write(status.get("latestCreatedRevisionName") or "")
else:
    sys.stderr.write("unknown mode %s\n" % mode)
    sys.exit(2)
PY

SERVICE_EXISTS=0
PREVIOUS_SPEC=""
if gcloud run services describe "$SERVICE" --region "$REGION" --format="value(metadata.name)" >/dev/null 2>&1; then
  SERVICE_EXISTS=1
  PREVIOUS_SPEC="$(python3 /tmp/cr_status.py previous "$SERVICE" "$REGION")"
  echo "Traffic that stays live until the new revision is healthy: ${PREVIOUS_SPEC:-none}"
else
  echo "No existing service. This is the first deploy."
fi

restore_previous() {
  if [ -n "$PREVIOUS_SPEC" ]; then
    echo "Restoring traffic to the previous revision: ${PREVIOUS_SPEC}"
    gcloud run services update-traffic "$SERVICE" \
      --region "$REGION" \
      --to-revisions "$PREVIOUS_SPEC" \
      --quiet
  fi
  gcloud run services update-traffic "$SERVICE" \
    --region "$REGION" \
    --remove-tags candidate \
    --quiet >/dev/null 2>&1 || true
}

deploy_args=(
  run deploy "$SERVICE"
  --image "$IMAGE"
  --region "$REGION"
  --platform managed
)

# A brand-new service cannot be created with --no-traffic.
if [ "$SERVICE_EXISTS" -eq 0 ]; then
  echo "Creating the first revision with traffic."
else
  deploy_args+=(--no-traffic --tag candidate)
fi

deploy_args+=(
  --startup-probe "httpGet.path=${HEALTH_PATH},initialDelaySeconds=0,timeoutSeconds=${PROBE_TIMEOUT_SECONDS},periodSeconds=${PROBE_PERIOD_SECONDS},failureThreshold=${PROBE_FAILURE_THRESHOLD}"
  --quiet
)

# Keep memory and CPU already set on a live service. Apply defaults only when
# the service is being created.
if [ "$SERVICE_EXISTS" -eq 0 ]; then
  if [ -n "${MEMORY:-}" ]; then
    deploy_args+=(--memory "$MEMORY")
  fi
  if [ -n "${CPU:-}" ]; then
    deploy_args+=(--cpu "$CPU")
  fi
fi

set +e
gcloud "${deploy_args[@]}" 2>&1 | tee /tmp/deploy-err
deploy_status="${PIPESTATUS[0]}"
set -e

# Older Cloud Run rejects --no-traffic on the very first revision. Retry once
# so a brand-new service can still be created. There is no previous version yet.
if [ "$deploy_status" -ne 0 ] && [ "$SERVICE_EXISTS" -eq 0 ] && grep -qi "no-traffic" /tmp/deploy-err; then
  echo "Initial deploy cannot use --no-traffic. Creating the first revision with traffic."
  fallback_args=()
  skip_next=0
  for arg in "${deploy_args[@]}"; do
    if [ "$skip_next" -eq 1 ]; then
      skip_next=0
      continue
    fi
    if [ "$arg" = "--no-traffic" ]; then
      continue
    fi
    if [ "$arg" = "--tag" ]; then
      skip_next=1
      continue
    fi
    fallback_args+=("$arg")
  done
  set +e
  gcloud "${fallback_args[@]}"
  deploy_status="$?"
  set -e
fi

if [ "$deploy_status" -ne 0 ]; then
  echo "gcloud run deploy failed before traffic moved."
  if [ -n "$PREVIOUS_SPEC" ]; then
    echo "The live service is still the previous revision."
    restore_previous
  else
    echo "There is no previous revision to serve."
  fi
  exit 1
fi

NEW_REV="$(python3 /tmp/cr_status.py latest "$SERVICE" "$REGION")"
if [ "$SERVICE_EXISTS" -eq 0 ]; then
  echo "Created the first revision ${NEW_REV} with traffic."
else
  echo "Created revision ${NEW_REV} with 0% traffic."
fi

URLS="$(python3 /tmp/cr_status.py urls "$SERVICE" "$REGION")"
SERVICE_URL="$(printf '%s\n' "$URLS" | sed -n '1p')"
CANDIDATE_URL="$(printf '%s\n' "$URLS" | sed -n '2p')"
if [ "$SERVICE_EXISTS" -eq 0 ]; then
  CANDIDATE_URL="$SERVICE_URL"
fi

if [ -z "$CANDIDATE_URL" ]; then
  echo "Could not resolve a candidate URL for ${NEW_REV}."
  restore_previous
  exit 1
fi

CALLER="$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" | head -n 1 || true)"
if [ -n "$CALLER" ]; then
  if [[ "$CALLER" == *.gserviceaccount.com ]]; then
    MEMBER="serviceAccount:${CALLER}"
  else
    MEMBER="user:${CALLER}"
  fi
  echo "Allowing ${MEMBER} to call the candidate URL for the health check."
  gcloud run services add-iam-policy-binding "$SERVICE" \
    --region "$REGION" \
    --member "$MEMBER" \
    --role "roles/run.invoker" \
    --quiet >/dev/null || echo "Invoker binding was not added. A private service may fail the health check and keep the previous revision."
fi

TOKEN=""
fetch_token() {
  TOKEN="$(gcloud auth print-identity-token --audiences="$SERVICE_URL" 2>/dev/null || true)"
}

probe() {
  local url="$1"
  local code
  if [ -n "$TOKEN" ]; then
    code="$(curl -sS -o /tmp/cloud-run-probe-body -w '%{http_code}' --max-time 90 \
      -H "Authorization: Bearer ${TOKEN}" \
      "$url" || true)"
  else
    code="$(curl -sS -o /tmp/cloud-run-probe-body -w '%{http_code}' --max-time 90 \
      "$url" || true)"
  fi
  if [ -z "$code" ]; then
    code="000"
  fi
  printf '%s' "$code"
}

wait_healthy() {
  local url="$1"
  local tries="$2"
  local i code
  for i in $(seq 1 "$tries"); do
    code="$(probe "$url")"
    echo "Health attempt ${i}/${tries} for ${url}: HTTP ${code}"
    case "$code" in
      2*|3*) return 0 ;;
    esac
    sleep 10
  done
  return 1
}

fetch_token
PROBE_URL="${CANDIDATE_URL}${HEALTH_PATH}"
if ! wait_healthy "$PROBE_URL" "$ATTEMPTS"; then
  echo "New revision ${NEW_REV} failed its health check."
  if [ -f /tmp/cloud-run-probe-body ]; then
    echo "Last response body:"
    head -c 300 /tmp/cloud-run-probe-body || true
    echo
  fi
  restore_previous
  if [ -n "$PREVIOUS_SPEC" ]; then
    echo "Traffic was not left on the failed revision. Users keep the previous version."
  else
    echo "There is no previous revision to serve."
  fi
  exit 1
fi

echo "Revision ${NEW_REV} is healthy. Moving 100% of traffic to it."
if ! gcloud run services update-traffic "$SERVICE" \
  --region "$REGION" \
  --to-revisions "${NEW_REV}=100" \
  --quiet; then
  echo "Traffic shift failed. Keeping the previous version."
  restore_previous
  exit 1
fi

fetch_token
if ! wait_healthy "${SERVICE_URL}${HEALTH_PATH}" 6; then
  echo "The service URL failed after promotion. Rolling back to the previous revision."
  restore_previous
  exit 1
fi

gcloud run services update-traffic "$SERVICE" \
  --region "$REGION" \
  --remove-tags candidate \
  --quiet >/dev/null 2>&1 || true

echo "Deployment finished. Traffic is on ${NEW_REV}."
