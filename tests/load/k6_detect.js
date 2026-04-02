import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend, Counter } from "k6/metrics";
import { SharedArray } from "k6/data";

const BASE_URL = __ENV.BASE_URL || "http://host.docker.internal:8000";
const DETECT_PATH = __ENV.DETECT_PATH || "/detect";
const THINK_TIME_MS = Number(__ENV.THINK_TIME_MS || "0");
const SCENARIO = (__ENV.SCENARIO || "steady").toLowerCase();
const AUTH_HEADER = __ENV.AUTH_TOKEN
  ? { Authorization: `Bearer ${__ENV.AUTH_TOKEN}` }
  : {};

const DEFAULT_HEADERS = {
  "Content-Type": "application/json",
  ...AUTH_HEADER,
};

const errorRate = new Rate("detect_errors");
const reviewRate = new Rate("detect_review_ratio");
const blockRate = new Rate("detect_block_ratio");
const passRate = new Rate("detect_pass_ratio");
const detectDuration = new Trend("detect_duration_ms");
const labelCounter = new Counter("detect_label_count");

const normalPayloads = new SharedArray("normal_payloads", function () {
  return JSON.parse(open("./payloads/normal.json"));
});

const noisyPayloads = new SharedArray("noisy_payloads", function () {
  return JSON.parse(open("./payloads/noisy.json"));
});

const boundaryPayloads = new SharedArray("boundary_payloads", function () {
  return JSON.parse(open("./payloads/boundary.json"));
});

const scenarioOptions = {
  smoke: {
    scenarios: {
      smoke: {
        executor: "constant-vus",
        vus: Number(__ENV.SMOKE_VUS || "1"),
        duration: __ENV.SMOKE_DURATION || "15s",
        gracefulStop: "0s",
      },
    },
  },
  steady: {
    scenarios: {
      steady: {
        executor: "constant-arrival-rate",
        rate: Number(__ENV.RATE || "20"),
        timeUnit: "1s",
        duration: __ENV.DURATION || "60s",
        preAllocatedVUs: Number(__ENV.PRE_ALLOCATED_VUS || "20"),
        maxVUs: Number(__ENV.MAX_VUS || "80"),
        gracefulStop: "5s",
      },
    },
  },
  ramping: {
    scenarios: {
      ramping: {
        executor: "ramping-vus",
        startVUs: Number(__ENV.START_VUS || "1"),
        stages: [
          { duration: __ENV.STAGE_1_DURATION || "30s", target: Number(__ENV.STAGE_1_TARGET || "10") },
          { duration: __ENV.STAGE_2_DURATION || "60s", target: Number(__ENV.STAGE_2_TARGET || "30") },
          { duration: __ENV.STAGE_3_DURATION || "30s", target: Number(__ENV.STAGE_3_TARGET || "0") },
        ],
        gracefulRampDown: "5s",
      },
    },
  },
};

export const options = {
  ...(scenarioOptions[SCENARIO] || scenarioOptions.steady),
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(50)<150", "p(95)<400", "p(99)<800"],
    detect_errors: ["rate<0.01"],
    detect_duration_ms: ["p(95)<400", "p(99)<800"],
  },
  summaryTrendStats: ["avg", "min", "med", "max", "p(90)", "p(95)", "p(99)"],
};

export function handleSummary(data) {
  return {
    stdout: buildConsoleSummary(data),
    "tests/load/summary.json": JSON.stringify(data, null, 2),
  };
}

export default function () {
  const payload = pickPayload();
  const response = http.post(
    `${BASE_URL}${DETECT_PATH}`,
    JSON.stringify({ text: payload.text }),
    { headers: DEFAULT_HEADERS, tags: { scenario: SCENARIO, payload_group: payload.group } }
  );

  detectDuration.add(response.timings.duration, { payload_group: payload.group });

  const ok = check(response, {
    "status is 200": (res) => res.status === 200,
    "response is json": (res) => (res.headers["Content-Type"] || "").includes("application/json"),
    "label exists": (res) => {
      const body = safeJson(res);
      return body !== null && typeof body.label === "string";
    },
    "score is number": (res) => {
      const body = safeJson(res);
      return body !== null && typeof body.score === "number";
    },
  });

  errorRate.add(!ok);

  const body = safeJson(response);
  if (body && typeof body.label === "string") {
    labelCounter.add(1, { label: body.label, payload_group: payload.group });
    blockRate.add(body.label === "BLOCK");
    reviewRate.add(body.label === "REVIEW");
    passRate.add(body.label === "PASS");
  } else {
    errorRate.add(true);
  }

  if (THINK_TIME_MS > 0) {
    sleep(THINK_TIME_MS / 1000);
  }
}

function pickPayload() {
  const ratio = Math.random();
  if (ratio < 0.50) {
    return withGroup(randomItem(normalPayloads), "normal");
  }
  if (ratio < 0.85) {
    return withGroup(randomItem(noisyPayloads), "noisy");
  }
  return withGroup(randomItem(boundaryPayloads), "boundary");
}

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function withGroup(item, group) {
  return {
    ...item,
    group,
  };
}

function safeJson(response) {
  try {
    return response.json();
  } catch (_) {
    return null;
  }
}

function buildConsoleSummary(data) {
  const failed = metricValue(data, "http_req_failed", "rate");
  const p95 = metricValue(data, "http_req_duration", "p(95)");
  const p99 = metricValue(data, "http_req_duration", "p(99)");
  const rps = metricValue(data, "http_reqs", "rate");
  return [
    "",
    "K6 detect summary",
    `scenario=${SCENARIO}`,
    `base_url=${BASE_URL}`,
    `rps=${formatNumber(rps)}`,
    `http_req_failed=${formatNumber(failed)}`,
    `p95_ms=${formatNumber(p95)}`,
    `p99_ms=${formatNumber(p99)}`,
    "",
  ].join("\n");
}

function metricValue(data, metricName, field) {
  const metric = data.metrics[metricName];
  if (!metric || !metric.values || metric.values[field] === undefined) {
    return 0;
  }
  return metric.values[field];
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}
