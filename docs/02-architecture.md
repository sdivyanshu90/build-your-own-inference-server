# Section 2 — Project Architecture

This section is the navigation map for the repository. Every file is enumerated, the per-request data flow is traced with line-level pointers, and every HTTP endpoint is documented with a full request/response example.

---

## 2.1 Directory structure

```text
inference-server/
├── Cargo.toml              # crate manifest + dependency rationale
├── Cargo.lock              # exact resolved versions (committed)
├── config/
│   └── default.toml        # default server/queue/model/pipeline/logging config
├── models/
│   └── README.md           # exact curl commands to fetch the SST-2 ONNX model
├── src/
│   ├── main.rs             # binary entry point — wires config, model, server, shutdown
│   ├── lib.rs              # public re-exports + build_app_runtime() factory
│   ├── config.rs           # AppConfig + sub-configs, serde + env layering, validate()
│   ├── error.rs            # AppError enum, ApiErrorBody, IntoResponse mapping
│   ├── metrics.rs          # global Prometheus recorder + emission helpers
│   ├── model/
│   │   ├── mod.rs          # InferenceModel trait, EncodedInput, ModelMetadata
│   │   ├── onnx.rs         # OnnxModel: ort 2.x Session inside Arc<Mutex<…>>
│   │   └── mock.rs         # deterministic MockModel for tests
│   ├── pipeline/
│   │   ├── mod.rs          # InferencePipeline orchestrator
│   │   ├── tokenizer.rs    # HuggingFace tokenizer + deterministic whitespace mock
│   │   └── postprocess.rs  # softmax, argmax, decode_label
│   ├── queue.rs            # InferenceQueue, QueueRuntime, dispatcher_loop
│   └── server/
│       ├── mod.rs          # build_router() — routes, body limit, middleware
│       ├── handlers.rs     # infer/health/ready/metrics/info handler functions
│       ├── middleware.rs   # x-request-id, per-request timeout, structured log
│       └── state.rs        # AppState — shared, Arc'd, atomically-mutable
└── tests/
    ├── integration_test.rs # spins a real server on :0, exercises every endpoint
    └── load_test.rs        # concurrency + percentile + overload assertions
```

### What every file does (and where the key code is)

| Path | Lines | Purpose | Key entry points |
|------|-------|---------|------------------|
| `Cargo.toml` | 96 | Crate manifest. Each dependency has a comment explaining *why* it was chosen. | `[dependencies]` block at line 18 |
| `config/default.toml` | 47 | Baseline config for the binary. Env vars (`INFERENCE_SERVER__…`) override on top. | — |
| `models/README.md` | 28 | Exact curl commands to download `tokenizer.json` and `model.onnx`. | — |
| `src/main.rs` | 120 | `#[tokio::main]`, CLI parsing, tracing init, model selection, graceful shutdown. | `parse_config_path()` line 23, `init_tracing()` line 34, `main()` line 47 |
| `src/lib.rs` | 107 | Public surface used by both the binary and the tests. Defines `AppRuntime` and `build_app_runtime()`. | `build_app_runtime()` line 62 |
| `src/config.rs` | 268 | `AppConfig` and five sub-configs (`Server`, `Queue`, `Model`, `Pipeline`, `Logging`); `load()` layers file + env; `validate()` rejects unsafe values. | `AppConfig::load()` line 43, `validate()` line 65 |
| `src/error.rs` | 206 | `AppError` enum (9 variants), `ApiErrorBody`, `status_code()` + `error_code()` mapping, `IntoResponse` impl. | `AppError` line 31, `IntoResponse` line 163 |
| `src/metrics.rs` | 84 | Global Prometheus recorder behind `OnceLock`; helpers for HTTP, inference, queue depth. | `install_metrics_recorder()` line 25 |
| `src/model/mod.rs` | 60 | `InferenceModel` trait, `EncodedInput`, `ModelOutput`, `ModelMetadata`, `ModelFuture<'a>` type alias. | trait at line 56 |
| `src/model/onnx.rs` | 121 | ONNX backend; runs `Session::run` inside `spawn_blocking`. | `OnnxModel::load()` line 28, `predict()` line 49 |
| `src/model/mock.rs` | 99 | Deterministic mock for tests; optional `delay` for backpressure/shutdown tests. | `predict()` line 34 |
| `src/pipeline/mod.rs` | 88 | `InferencePipeline` glues tokenizer → model → postprocess. Early validation rejects empty/oversized text. | `infer()` line 52 |
| `src/pipeline/tokenizer.rs` | 110 | `TokenizerWrapper` with two variants: `HuggingFace(Tokenizer)` and `Whitespace` (mock). | `encode()` line 50 |
| `src/pipeline/postprocess.rs` | 62 | Numerically stable softmax, argmax, label decoding. | `softmax()` line 4, `argmax()` line 26 |
| `src/queue.rs` | 331 | `InferenceQueue` (semaphore + mpsc); `QueueRuntime` for shutdown; `dispatcher_loop` fans jobs to bounded workers via a `JoinSet`. | `InferenceQueue::spawn()` line 52, `submit()` line 92, `dispatcher_loop()` line 146 |
| `src/server/mod.rs` | 30 | `build_router()` — five routes + body limit + request-context middleware. | `build_router()` line 21 |
| `src/server/handlers.rs` | 135 | Handler functions: `infer`, `health`, `ready`, `metrics`, `info`. | `infer()` line 60 |
| `src/server/middleware.rs` | 80 | Reads/generates `x-request-id`, applies per-request timeout, emits structured log + metrics. | `request_context()` line 28 |
| `src/server/state.rs` | 53 | `AppState` (config, queue, metadata, metrics handle, readiness atomic, start instant). | `AppState::new()` line 22 |
| `tests/integration_test.rs` | 370 | Spawns a real server on a random port and asserts behaviour for every endpoint, plus concurrency, overload, shutdown, and request-id propagation. | 12 `#[tokio::test]` functions |
| `tests/load_test.rs` | 193 | Concurrency storm; computes p50/p95/p99; asserts overload returns 503 instead of hanging. | `load_test_mock_model_latency_and_backpressure()` line 111 |

---

## 2.2 Data flow — a single request, step by step

```text
Step 1.  Client sends:  POST /v1/infer  Content-Type: application/json
                         { "text": "I love Rust" }

Step 2.  Axum routes the request to `infer` handler.
         The `Router` is built in src/server/mod.rs:21-30.

Step 3.  `request_context` middleware runs (src/server/middleware.rs:28-81):
           • reads x-request-id header or generates a UUIDv4
           • starts an Instant timer
           • inserts RequestContext into request extensions
           • wraps the inner call in tokio::time::timeout()

Step 4.  Body deserialization:  Json(payload): Json<InferRequest>
         (src/server/handlers.rs:60-64).  Bodies above
         server.max_payload_bytes (16 KB by default) fail here with 413.

Step 5.  Handler calls state.queue.submit(text, Some(enqueue_timeout))
         (src/server/handlers.rs:65-72).  submit() first checks
         self.accepting then acquires an OwnedSemaphorePermit from
         the queue's slot semaphore (src/queue.rs:96-106).

Step 6.  Handler builds (oneshot::Sender, oneshot::Receiver) and
         sends the InferenceJob through the mpsc channel
         (src/queue.rs:108-126).  The atomic depth gauge is bumped
         and exported to Prometheus.

Step 7.  Dispatcher task receives the job at src/queue.rs:155.
         It acquires a permit from the worker_limiter semaphore
         (bounded by config.queue.worker_count).

Step 8.  Worker future is spawned into a JoinSet
         (src/queue.rs:163-176).  It calls pipeline.infer(&job.text).

Step 9.  Pipeline (src/pipeline/mod.rs:52-88):
           • validates non-empty + max_characters
           • tokenizer.encode(text, max_tokens) → EncodedInput tensors
           • model.predict(&encoded).await
           • softmax(logits) → argmax → decode_label

Step 10. For the ONNX backend, predict() schedules Session::run on
         tokio::task::spawn_blocking (src/model/onnx.rs:60-110) so
         the model's blocking C++ call does not stall the Tokio
         reactor.

Step 11. Worker reports inference latency to metrics
         (src/queue.rs:168-170) and sends the PipelineOutput back
         through the oneshot channel.

Step 12. Handler resumes, builds the InferResponse, returns
         (StatusCode::OK, Json(body)).

Step 13. Middleware records:
           • http_requests_total{endpoint, status}
           • http_request_latency_ms{endpoint}
           • structured info!() log with request_id and latency_ms
         (src/server/middleware.rs:69-78).

Step 14. Client receives 200 OK with the JSON body and the
         x-request-id response header echoed.
```

Any error along the way is converted to a JSON `ApiErrorBody` by `AppError::into_response` (`src/error.rs:163-173`) with the appropriate HTTP status code (`src/error.rs:118-130`).

---

## 2.3 API contract

All endpoints are mounted in `src/server/mod.rs:21-30`. The body limit applies to every route.

### `POST /v1/infer`

Runs a single inference request through the queue.

**Request**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | `string` | yes | Non-empty after trimming; at most `pipeline.max_characters` characters (`4096` by default). |

Headers (optional):

| Header | Purpose |
|--------|---------|
| `x-request-id` | If set, it is propagated into logs and echoed in the response. Otherwise a UUIDv4 is generated. |

**Successful response (200)**

```json
{
  "request_id": "9c1bc35e-1c8a-4c5d-8c7a-9c4f4b1d3a7c",
  "label": "POSITIVE",
  "confidence": 0.9712,
  "token_count": 5,
  "logits": [-1.23, 2.46],
  "model_name": "distilbert-sst2",
  "model_version": "1"
}
```

Schema defined at `src/server/handlers.rs:27-36`.

**Error responses**

| Status | When | Example body |
|--------|------|--------------|
| `413` | Payload above `server.max_payload_bytes` (extractor failure) | `{ "error": "payload_too_large", "message": "...", "request_id": "..." }` |
| `422` | Empty/whitespace input or invalid JSON | `{ "error": "validation_error", "message": "validation error: input text must not be empty", "request_id": "..." }` |
| `503` | Queue full, queue closed (during shutdown), model unavailable, or model error | `{ "error": "queue_full", "message": "queue is full", "request_id": "..." }` |
| `504` | Per-request timeout (`server.request_timeout_ms`) exceeded | `{ "error": "timeout", "message": "request timed out", "request_id": "..." }` |
| `500` | Unexpected internal error | `{ "error": "internal_error", "message": "...", "request_id": "..." }` |

The full status-code mapping is in `src/error.rs:118-130`. The HTTP-413 case for raw payload-size overflow is handled by Axum's `DefaultBodyLimit` layer (`src/server/mod.rs:28`).

### `GET /health`

Liveness probe. Always returns 200 while the process is alive — orchestrators use it to decide whether to restart the container.

```json
{
  "status": "ok",
  "uptime_seconds": 142
}
```

Schema: `src/server/handlers.rs:38-42`; handler: `src/server/handlers.rs:88-96`.

### `GET /ready`

Readiness probe. Returns 200 once `AppState::mark_ready()` has been called *and* the queue is still accepting; otherwise it returns 503 with the `model_unavailable` body. Load balancers should hold traffic until this endpoint goes green.

Success:

```json
{ "status": "ready" }
```

Failure (during cold start or graceful shutdown):

```json
{
  "error": "model_unavailable",
  "message": "model unavailable: model is not ready yet",
  "request_id": null
}
```

Handler: `src/server/handlers.rs:98-112`. State check: `src/server/state.rs:46-48`.

### `GET /metrics`

Returns Prometheus text format (`text/plain; version=0.0.4`). The recorder is installed exactly once via `OnceLock` in `src/metrics.rs:25-53`. Counters/histograms registered:

- `http_requests_total{endpoint,status}` — counter
- `http_request_latency_ms{endpoint}` — histogram
- `inference_requests_total{model_name}` — counter
- `inference_latency_ms{model_name}` — histogram
- `queue_depth` — gauge

Excerpt of the response body:

```text
# HELP http_requests_total Total number of HTTP requests handled.
# TYPE http_requests_total counter
http_requests_total{endpoint="/v1/infer",status="200"} 42

# HELP inference_latency_ms Model pipeline latency in milliseconds.
# TYPE inference_latency_ms summary
inference_latency_ms{model_name="distilbert-sst2",quantile="0.5"} 7.21
inference_latency_ms{model_name="distilbert-sst2",quantile="0.9"} 12.04

# HELP queue_depth Current number of in-flight plus queued inference jobs.
# TYPE queue_depth gauge
queue_depth 3
```

Handler: `src/server/handlers.rs:114-120`.

### `GET /v1/info`

Returns the configured model metadata. Useful for human verification and for sidecars that need to know which model version is serving.

```json
{
  "name": "distilbert-sst2",
  "version": "1",
  "backend": "onnx",
  "input_names": ["input_ids", "attention_mask"],
  "labels": ["NEGATIVE", "POSITIVE"],
  "max_tokens": 512
}
```

Schema: `src/server/handlers.rs:50-57`; handler: `src/server/handlers.rs:122-136`. The metadata is populated at startup in `src/main.rs:54-61`.

---

## 2.4 Shared state

`AppState` (`src/server/state.rs:11-18`) is the single object handlers extract via Axum's `State<Arc<AppState>>`. It contains:

| Field | Type | Why |
|-------|------|-----|
| `config` | `Arc<AppConfig>` | Pipeline limits, timeouts, queue sizing. |
| `queue` | `Arc<InferenceQueue>` | Submission handle for inference jobs. |
| `model_metadata` | `ModelMetadata` | Snapshot returned by `/v1/info`. |
| `metrics_handle` | `Arc<PrometheusHandle>` | Renders the `/metrics` response body. |
| `ready` | `Arc<AtomicBool>` | Toggled by `mark_ready()`; read by `/ready`. |
| `started_at` | `Instant` | Source of `/health`'s `uptime_seconds`. |

`mark_ready()` (`src/server/state.rs:40-43`) flips both the readiness flag and the queue's `accepting` flag. `is_ready()` (`src/server/state.rs:46-48`) requires both to be true so a draining queue does not advertise readiness.
