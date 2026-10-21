// ============================================================
// ACOT light-mix load test (90 VUs) - cleaner final dataset
// ============================================================
// Same ramp profile as the 100-VU full test minus the ask probe
// (ask data is reused from the previous valid run). Adds:
//  - graceful ramp-down so no artificial connection resets
//  - per-endpoint latency sub-metrics
//  - DELETE failure classification (status + error code)
// ============================================================

import http from "k6/http";
import { check, sleep } from "k6";
import exec from "k6/execution";
import { Counter, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const SUMMARY_USER_ID =
  __ENV.SUMMARY_USER_ID || "21005862-7337-4deb-b7e7-1ad3f7b0d11b";

const DELETE_NON_200 = new Counter("light_delete_non_200");
const SUMMARY_NONEMPTY = new Rate("light_summary_nonempty_rate");

export const options = {
  scenarios: {
    light_mix: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 20 },
        { duration: "30s", target: 45 },
        { duration: "30s", target: 90 },
        { duration: "90s", target: 90 },
        { duration: "30s", target: 0 },
      ],
      gracefulRampDown: "10s",
      exec: "runLightMix",
    },
  },
  thresholds: {
    "http_req_duration{name:GET /}": ["p(95)<2000"],
    "http_req_duration{name:GET /health}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/session/stats}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/account/summary}": ["p(95)<2000"],
    "http_req_duration{name:DELETE /api/session/{id}}": ["p(95)<2000"],
    "http_req_failed{scenario:light_mix}": ["rate<0.01"],
    "checks{scenario:light_mix}": ["rate>0.99"],
  },
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

let loggedDeleteFailures = 0;
let dumpedSummaryShape = false;

export function runLightMix() {
  const vu = __VU;
  const iteration = exec.scenario.iterationInTest;

  http.get(`${BASE_URL}/`, { tags: { name: "GET /" } });

  const health = http.get(`${BASE_URL}/health`, {
    tags: { name: "GET /health" },
  });
  check(health, {
    "GET /health 200": (r) => r.status === 200,
  });

  const stats = http.get(`${BASE_URL}/api/session/stats`, {
    tags: { name: "GET /api/session/stats" },
  });
  check(stats, {
    "GET /api/session/stats 200": (r) => r.status === 200,
  });

  const summary = http.get(
    `${BASE_URL}/api/account/summary/${SUMMARY_USER_ID}`,
    { tags: { name: "GET /api/account/summary" } }
  );
  check(summary, {
    "GET /api/account/summary 200": (r) => r.status === 200,
  });

  if (!dumpedSummaryShape && summary.status === 200 && summary.body) {
    dumpedSummaryShape = true;
    console.info(
      `SUMMARY_SHAPE len=${summary.body.length} head=${summary.body.slice(0, 300)}`
    );
  }
  SUMMARY_NONEMPTY.add(
    summary.status === 200 && summary.body && summary.body.length >= 120
  );

  const sessionId = `k6-light-final-${vu}-${Math.floor(iteration % 20)}`;
  const del = http.del(
    `${BASE_URL}/api/session/${sessionId}`,
    null,
    { tags: { name: "DELETE /api/session/{id}" } }
  );
  check(del, {
    "DELETE /api/session 200": (r) => r.status === 200,
  });

  if (del.status !== 200) {
    DELETE_NON_200.add(1);
    if (loggedDeleteFailures < 30) {
      loggedDeleteFailures += 1;
      console.warn(
        `DELETE_FAIL vu=${vu} status=${del.status}` +
          ` err_code=${del.error_code || "none"}` +
          ` err=${(del.error || "").slice(0, 120)}`
      );
    }
  }

  sleep(1);
}
