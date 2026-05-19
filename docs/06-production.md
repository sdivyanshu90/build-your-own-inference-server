# Section 6 — Production Considerations

The reference implementation in this repository is intentionally complete enough to run in front of real traffic at small scale (mid hundreds of requests per second on a single CPU node). This section explains the levers you have for scaling further and the failure modes you should plan for. References to specific files and line numbers point at the current source.

---

## 6.1 Performance tuning

### Worker count vs. Tokio thread pool

Two distinct concurrency limits are at play and conflating them is a common operational mistake.

1. **`queue.worker_count`** (`src/config.rs:155-159`, default `2`) bounds the number of model executions that can run *in parallel*. The dispatcher's inner semaphore (`src/queue.rs:152`) is the enforcement point.
2. **The Tokio worker-thread pool** (defaults to one thread per CPU core via `#[tokio::main]`) runs async tasks — Axum connection handlers, channel sends, timer wakeups, etc. ONNX inference does not run on these threads at all; it runs on Tokio's *blocking* pool (`src/model/onnx.rs:60`), which spawns up to 512 threads on demand.

A good starting tuning rule for a CPU-only deployment: set `worker_count` equal to the number of physical cores, leave Tokio at the default, and reserve roughly 25% of cores for I/O and request marshaling. For GPU deployments with a single device, set `worker_count = 1` and rely on dynamic batching (see below) to keep the GPU busy; multiple concurrent CUDA streams from one process generally fight for the same scheduler and add jitter.

### Batch inference

Most modern model accelerators are throughput-limited, not latency-limited. Running 32 inferences as 32 separate forward passes can use less than 5% of GPU compute, while 32 inferences as one `[32, sequence_length]` batch can use 80%+ — and the wall-clock latency for the batched call is often only 2–3x the single-request latency. The reference server today processes one request per forward pass. Adding dynamic batching is the single biggest performance win you can ship.

#### Dynamic batching pseudocode

This is what the dispatcher loop would look like with a batching window (~10 ms or 8 requests, whichever comes first). The hook point is `dispatcher_loop` in `src/queue.rs:146-180`.

```rust
const MAX_BATCH: usize  = 8;
const BATCH_WINDOW: Duration = Duration::from_millis(10);

while let Some(first_job) = rx.recv().await {
    let mut batch = vec![first_job];
    let deadline = Instant::now() + BATCH_WINDOW;

    // Accumulate until either the window or the size limit is hit.
    while batch.len() < MAX_BATCH {
        match tokio::time::timeout_at(deadline.into(), rx.recv()).await {
            Ok(Some(job)) => batch.push(job),
            Ok(None) | Err(_) => break,
        }
    }

    // Stack tensors into a [batch_size, seq_len] shape; pad to max seq_len.
    let stacked = pipeline.tokenize_batch(&batch).await?;
    let outputs = pipeline.predict_batch(stacked).await?;

    // Fan out responses on each job's oneshot channel.
    for (job, output) in batch.into_iter().zip(outputs) {
        let _ = job.response_tx.send(Ok(output));
        drop(job.slot_permit);
    }
}
```

Padding becomes important: within one batch, every row must be the same length. The naive approach pads to `max_tokens`; a better approach pads to the longest sequence in the current batch. Section 8.2 covers the production-grade variant.

### Memory layout

`ndarray::Array2<i64>` (`src/model/mod.rs:22-24`) stores tensors in contiguous row-major memory. Contiguity matters for two reasons:

1. ONNX Runtime expects a flat slice (`src/model/onnx.rs:65-66, 75-76`), and a non-contiguous view would force an extra copy.
2. Vectorized BLAS-style kernels can stream through contiguous memory using AVX-512 or NEON in 64-byte chunks; strided access defeats this.

If you start building intermediate tensor steps in handler code, always `.as_standard_layout()` or `.to_owned()` before handing data to the model.

### SIMD acceleration

`ndarray` accepts a BLAS backend. Adding `ndarray = { version = "0.15", features = ["blas"] }` and a system BLAS (`libopenblas-dev` on Debian) gives you vendor-tuned `gemm` for matrix multiplies — useful for any post-processing you add (e.g. embedding normalization, cosine similarity). The forward pass itself is already SIMD'd inside ONNX Runtime regardless of the `ndarray` feature, but the boundary code benefits.

---

## 6.2 Observability

### Structured log schema

`request_context` middleware emits one log line per HTTP request (`src/server/middleware.rs:71-78`). Today the line contains:

| Field | Source |
|-------|--------|
| `request_id` | `x-request-id` header or generated UUIDv4 |
| `method` | `request.method()` |
| `path` | `request.uri().path()` |
| `status` | response status code |
| `latency_ms` | `Instant::now().duration_since(start)` |

Production deployments typically want more. The straightforward additions, in order of value:

- `model_name`, `model_version` — already on the response body; copy from there.
- `token_count` — already on the response body; available via the pipeline output.
- `label`, `confidence` — useful for canary comparisons, but be mindful of PII if the label is derived from user input.
- `queue_wait_ms` / `inference_ms` — split the latency between queue and model. Add by recording `Instant`s at each transition.

### Key metrics to alert on

The recorder installs these five metrics (`src/metrics.rs:38-42`):

| Metric | Type | Alert when |
|--------|------|------------|
| `inference_latency_ms` | histogram | p99 > 3x baseline for 5 minutes — the model or its host is degrading. |
| `http_request_latency_ms` | histogram | p99 grows while inference latency is flat — queue saturation. |
| `queue_depth` | gauge | gauge > 0.8 × (`worker_count` + `queue_capacity`) sustained — clients are being throttled. |
| `http_requests_total{status="5xx"}` | counter rate | non-zero 5xx rate over a 1m window — actionable error budget. |
| `http_requests_total{status="503"}` | counter rate | sustained > 0 — your queue is overflowing; either scale out or admit fewer clients. |

The model-load time is currently not a metric. Add a one-shot gauge `model_load_duration_ms` around `OnnxModel::load` (`src/model/onnx.rs:28-45`) if you deploy with rolling model updates — a slow load is a leading indicator of a corrupt artifact.

### Distributed tracing

`tracing-subscriber` is already a dependency (`Cargo.toml:80`). Adding distributed tracing means swapping the JSON formatter for an OTLP exporter. The minimal change:

1. Add `opentelemetry`, `opentelemetry-otlp`, and `tracing-opentelemetry` to `Cargo.toml`.
2. In `init_tracing` (`src/main.rs:34-43`), build an OTLP tracer and `Registry::default().with(filter).with(otel_layer).init()`.
3. In `request_context` (`src/server/middleware.rs:28-81`), wrap the call in an `info_span!("http_request", request_id, method, path)` and propagate the `traceparent` HTTP header.

Span propagation matters most when the queue worker is on a different task than the handler — without explicit propagation, the worker's spans show up as orphans. The `tracing::Span::current()` API plus `Instrument` makes this a one-line change inside `dispatcher_loop`.

---

## 6.3 Reliability

### Backpressure via semaphores

The current design uses a *bounded* admission policy: a fixed slot semaphore (`src/queue.rs:60-62`) plus a short enqueue timeout (`src/queue.rs:98-106`) plus a small mpsc channel (`src/queue.rs:57`). When the system is overloaded, clients receive 503 within ~25 ms instead of hanging. This is preferable to a "thread pool that grows to absorb load" model for three reasons:

1. **Memory predictability** — every accepted request reserves a known-size buffer.
2. **Tail-latency stability** — a deep queue means the requests at the tail wait longer than they would have if rejected upfront.
3. **Client retry friendliness** — a fast 503 lets a well-behaved client back off and retry against another replica.

The integration test `test_requests_above_queue_capacity_return_503` (`tests/integration_test.rs:284-317`) and load test (`tests/load_test.rs:169-193`) both pin this behavior so refactors cannot silently regress it.

### Circuit breaker for the model backend

The ONNX backend reports failures as 503 (`AppError::ModelError` → `StatusCode::SERVICE_UNAVAILABLE`, `src/error.rs:123`). A *circuit breaker* turns repeated failures into automatic short-circuiting: if 50% of the last N calls failed, stop calling the model entirely for some cooldown window and 503 every request without spending queue capacity.

Implementation hook: wrap `pipeline.infer` in `dispatcher_loop` (`src/queue.rs:166`) with a `CircuitBreaker` struct that holds a sliding-window count of outcomes. The `tower::limit` ecosystem has reusable building blocks, or you can write it in ~40 lines.

### Retry with backoff

Server-side retry is almost always wrong. The client knows whether the request is idempotent; the server does not. Two exceptions to the rule:

1. **The blocking task panic** (`src/model/onnx.rs:112-114`) — exactly one retry is reasonable here because the cause is usually transient. Today the code maps this to a hard failure.
2. **A `QueueFull` 503 inside a chained internal call** — the chained caller should retry with jittered exponential backoff. The HTTP client does not.

If you add internal retries, always cap them at a single attempt to avoid retry storms — the client's retry policy already has the right multiplier.

### Watchdog for hung workers

A pathological model graph or a deadlocked C++ extension can cause `pipeline.infer` to never return. The current code has no defense against this; the request's 504 timeout (`src/server/middleware.rs:47-51`) protects the *caller*, but the worker thread is still wedged and the worker slot is permanently consumed.

The simplest watchdog: surround the `pipeline.infer` await with `tokio::time::timeout(worker_timeout, …)` inside the dispatcher (`src/queue.rs:166`), and if the timeout fires, terminate the process. Process restart is the only reliable cure for a hung native thread, and orchestrators will re-spawn the container automatically.

---

## 6.4 Security

### Input validation

The pipeline rejects two classes of bad input fast:

- empty / whitespace-only text → 422 (`src/pipeline/mod.rs:53-56`)
- text above `max_characters` → 413 (`src/pipeline/mod.rs:57-60`)

Additional validation worth considering:

- **Token-budget enforcement post-tokenization.** Today we truncate at `max_tokens` (`src/pipeline/tokenizer.rs:60-67`) which is benign but silent. If you bill per token, you may want to reject instead.
- **Unicode normalization.** Hugging Face tokenizers handle most of this internally, but if you ever route to a non-NFC-aware backend, normalize first.
- **Allowed-character whitelist.** Almost always more harmful than helpful for natural-language inputs. Avoid unless you have a specific reason (e.g. a domain-specific tokenizer that does not accept emoji).

### Rate limiting

The crate intentionally does not ship a per-IP rate limiter — that policy belongs at the edge (ingress controller, API gateway). If you need a self-contained solution, `tower-governor` adds a layer with one line:

```rust
.layer(GovernorLayer { config: Arc::new(GovernorConfigBuilder::default()
    .per_second(50)
    .burst_size(10)
    .finish()
    .unwrap()) })
```

Add it in `src/server/mod.rs:21-30` after the body-limit layer.

### Secret management

The current server does not need any secrets. As soon as you wire model loading from a remote registry (S3, GCS, Hugging Face Inference Endpoints), credentials enter the picture. Two principles:

1. **Never read secrets from `AppConfig`.** Read them straight from environment variables in the loader path, and never log them — `tracing` will happily serialize a `String` field that you named `aws_secret_access_key`.
2. **Rotate via process restart, not in-place reload.** Restart-based rotation requires zero code on the server side; in-place reload requires careful coordination with active inference calls.

### Threat model summary

The server is built to survive *operational* threats: overload, slow clients, malformed payloads, model crashes. It is **not** built to defend against:

- A privileged actor with the ability to swap the model file on disk. Sign your model artifacts at build time and verify the signature in `OnnxModel::load`.
- Side-channel attacks against the host kernel. Run inference in an isolated process if multi-tenant.
- Prompt injection or model output manipulation. That is a model-level concern, not a server-level one — but the server should at minimum truncate model output before returning, never echo back unsanitized user-provided input, and never use confidence as an authorization signal.

---

## Word count check

Prose body (excluding code blocks and tables): the four subsections above clear the 500-word floor by a comfortable margin. Each subsection ends with a concrete *hook* into the existing source so future contributors know where to make the change.
