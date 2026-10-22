// ============================================================
// ACOT Backend - k6 FINAL load test (user range: 100)
// ============================================================
// Peak: 90 VUs light mix + 10 VUs ask probe = 100 concurrent VUs.
// Fixes vs first run: graceful ramp-down (no artificial resets),
// DELETE failure classification, per-endpoint sub-metrics,
// token accounting, account-summary shape check.
// ============================================================

import http from "k6/http";
import { check, sleep } from "k6";
import exec from "k6/execution";
import { Trend, Counter, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const SUMMARY_USER_ID =
  __ENV.SUMMARY_USER_ID || "21005862-7337-4deb-b7e7-1ad3f7b0d11b";

const askRewriteMs = new Trend("acot_pipeline_rewrite_ms", true);
const askAnswerMs = new Trend("acot_pipeline_answer_ms", true);
const askSummaryMs = new Trend("acot_pipeline_ai_summary_ms", true);
const askTotalMs = new Trend("acot_pipeline_total_ms", true);
const askTokens = new Counter("acot_tokens_used");
const askQueue503 = new Counter("acot_ask_503");
const askNoAnswer = new Counter("acot_ask_empty_answer");
const DELETE_NON_200 = new Counter("light_delete_non_200");
const SUMMARY_NONEMPTY = new Rate("light_summary_nonempty_rate");

export const options = {
  scenarios: {
    ramp_light: {
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
      tags: { test_type: "light_mix" },
      exec: "runLightMix",
    },
    ask_probe: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 0 },
        { duration: "30s", target: 4 },
        { duration: "60s", target: 10 },
        { duration: "60s", target: 10 },
        { duration: "30s", target: 0 },
      ],
      gracefulRampDown: "10s",
      tags: { test_type: "ask_probe" },
      exec: "runAskProbe",
    },
  },
  thresholds: {
    "http_req_failed{scenario:ramp_light}": ["rate<0.01"],
    "http_req_duration{scenario:ramp_light}": ["p(95)<2000"],
    "http_req_failed{scenario:ask_probe}": ["rate<0.50"],
    "http_req_duration{name:POST /api/ask}": ["max<90000"],
    "http_req_duration{name:GET /}": ["p(95)<2000"],
    "http_req_duration{name:GET /health}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/session/stats}": ["p(95)<2000"],
    "http_req_duration{name:GET /api/account/summary}": ["p(95)<5000"],
    "http_req_duration{name:DELETE /api/session/{id}}": ["p(95)<2000"],
    "checks{scenario:ramp_light}": ["rate>0.99"],
  },
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

const PROBE_QUESTIONS = [
  "What are the starting prices of projects in Dubai Marina?",
  "Which of these have 2-bedroom options?",
  "Compare the first and second projects based on price, bedroom range, and handover date.",
  "What are the best properties in Jumeirah Village Circle?",
  "What amenities do those projects offer according to their documents?",
  "hi",
];

let loggedDeleteFailures = 0;
let dumpedSummaryShape = false;

export function runLightMix() {
  const vu = __VU;
  const iteration = exec.scenario.iterationInTest;

  http.get(`${BASE_URL}/`, { tags: { name: "GET /" } });

  const health = http.get(`${BASE_URL}/health`, {
    tags: { name: "GET /health" },
  });
  check(health, { "GET /health 200": (r) => r.status === 200 });

  const stats = http.get(`${BASE_URL}/api/session/stats`, {
    tags: { name: "GET /api/session/stats" },
  });
  check(stats, { "GET /api/session/stats 200": (r) => r.status === 200 });

  const summary = http.get(
    `${BASE_URL}/api/account/summary/${SUMMARY_USER_ID}`,
    { tags: { name: "GET /api/account/summary" } }
  );
  check(summary, { "GET /api/account/summary 200": (r) => r.status === 200 });

  if (!dumpedSummaryShape && summary.status === 200 && summary.body) {
    dumpedSummaryShape = true;
    console.info(
      `SUMMARY_SHAPE len=${summary.body.length} head=${summary.body.slice(0, 300)}`
    );
  }
  SUMMARY_NONEMPTY.add(
    summary.status === 200 && summary.body && summary.body.length >= 120
  );

  const sessionId = `k6-final-${vu}-${Math.floor(iteration % 20)}`;
  const del = http.del(
    `${BASE_URL}/api/session/${sessionId}`,
    null,
    { tags: { name: "DELETE /api/session/{id}" } }
  );
  check(del, { "DELETE /api/session 200": (r) => r.status === 200 });

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

export function runAskProbe() {
  const vu = __VU;
  const iteration = exec.scenario.iterationInTest;
  const sessionId = `k6-probe-${vu}`;
  const question =
    PROBE_QUESTIONS[iteration % PROBE_QUESTIONS.length];

  const payload = JSON.stringify({
    question: question,
    session_id: sessionId,
  });

  const res = http.post(
    `${BASE_URL}/api/ask`,
    payload,
    {
      headers: {
        "Content-Type": "application/json",
      },
      tags: { name: "POST /api/ask" },
      timeout: "90s",
    }
  );

  if (res.status === 503) {
    askQueue503.add(1);
  }

  let body = null;
  try {
    if (res.body) {
      body = res.json();
    }
  } catch (_err) {
    body = null;
  }

  check(res, {
    "POST /api/ask 200": (r) => r.status === 200,
    "POST /api/ask has answer": () =>
      body !== null &&
      typeof body.answer === "string" &&
      body.answer.length > 0,
  });

  if (body) {
    if (typeof body.answer !== "string" || body.answer.length === 0) {
      askNoAnswer.add(1);
    }

    const timing =
      body.meta && body.meta.timing_ms ? body.meta.timing_ms : {};

    if (typeof timing.rewrite === "number") askRewriteMs.add(timing.rewrite);
    if (typeof timing.answer === "number") askAnswerMs.add(timing.answer);
    if (typeof timing.ai_summary === "number") {
      askSummaryMs.add(timing.ai_summary);
    }
    if (typeof timing.total === "number") askTotalMs.add(timing.total);

    if (typeof body.tokens_used === "number") askTokens.add(body.tokens_used);
  }

  sleep(3);
}
