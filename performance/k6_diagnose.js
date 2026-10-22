// ============================================================
// ACOT diagnostics: per-endpoint latency attribution + DELETE
// failure classification at high concurrency.
// ============================================================

import http from "k6/http";
import { check, sleep } from "k6";
import exec from "k6/execution";
import { Counter } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const SUMMARY_USER_ID =
  __ENV.SUMMARY_USER_ID || "21005862-7337-4deb-b7e7-1ad3f7b0d11b";

const STATUS_ROOT = new Counter("diag_delete_status_root");
const DELETE_NON_200 = new Counter("diag_delete_non_200");

export const options = {
  scenarios: {
    diag: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "10s", target: 45 },
        { duration: "30s", target: 90 },
        { duration: "10s", target: 0 },
      ],
      gracefulRampDown: "0s",
      exec: "runDiag",
    },
  },
  // Per-name thresholds force k6 to emit per-endpoint sub-metrics.
  thresholds: {
    "http_req_duration{name:GET /}": ["p(95)<2000"],
    "http_req_duration{name:GET /health}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/session/stats}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/account/summary}": ["p(95)<2000"],
    "http_req_duration{name:DELETE /api/session/{id}}": ["p(95)<2000"],
  },
  summaryTrendStats: ["avg", "p(90)", "p(95)", "p(99)", "max"],
};

let loggedFailures = 0;

export function runDiag() {
  const vu = __VU;
  const iteration = exec.scenario.iterationInTest;

  http.get(`${BASE_URL}/`, { tags: { name: "GET /" } });

  http.get(`${BASE_URL}/health`, { tags: { name: "GET /health" } });

  http.get(`${BASE_URL}/api/session/stats`, {
    tags: { name: "GET /api/session/stats" },
  });

  const summary = http.get(
    `${BASE_URL}/api/account/summary/${SUMMARY_USER_ID}`,
    { tags: { name: "GET /api/account/summary" } }
  );
  check(summary, {
    "summary 200": (r) => r.status === 200,
  });

  const sessionId = `k6-diag-${vu}-${Math.floor(iteration % 5)}`;
  const del = http.del(
    `${BASE_URL}/api/session/${sessionId}`,
    null,
    { tags: { name: "DELETE /api/session/{id}" } }
  );

  if (del.status !== 200) {
    DELETE_NON_200.add(1);
    STATUS_ROOT.add(1);
    if (loggedFailures < 20) {
      loggedFailures += 1;
      console.warn(
        `DELETE non-200: vu=${vu} status=${del.status}` +
          ` err=${del.error_code || del.error || "none"}` +
          ` body=${(del.body || "").slice(0, 200)}`
      );
    }
  } else {
    check(del, { "DELETE 200": (r) => r.status === 200 });
  }

  sleep(1);
}
