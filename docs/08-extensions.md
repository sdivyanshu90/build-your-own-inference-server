# Section 8 — Extensions and Exercises

A roadmap for taking this repository past the tutorial baseline. Each entry names the design challenge, points at the exact source location where the extension would hook in, and lists the crates to reach for.

For each, you can also drop the suggested `// TODO:` comment into the source so the work item is discoverable via `grep -RIn TODO src/`.

---

## 8.1 gRPC endpoint alongside REST

### Why this is interesting

gRPC carries inference payloads more efficiently than JSON (protobuf, HTTP/2 framing, streaming). Adding it gives clients a choice without duplicating the model code.

### Where it hooks in

`src/lib.rs:62-107` already separates the pipeline from the HTTP server. Add a second route surface that shares the same `Arc<InferenceQueue>`.

```rust
// src/server/mod.rs — near build_router(state)
// TODO(Section 8.1): expose a `build_grpc_service(state)` factory and call
// it from main.rs to bind a Tonic server on a second port. The Tonic
// service can call `state.queue.submit(...)` identically to the Axum
// handler in src/server/handlers.rs:60-86.
```

### Design challenge

The single hardest decision is *port topology*. Two listeners on different ports is the simplest path; multiplexing HTTP/1 and HTTP/2 on the same port via `axum::serve` requires a TLS terminator that supports ALPN. Use `tonic::transport::Server` on a separate port for the tutorial.

### Crates

- `tonic` for the gRPC server.
- `prost` (Tonic transitively pulls this in) for protobuf message generation.
- `tonic-build` (build-dependency) to compile `.proto` files at `build.rs` time.

---

## 8.2 Dynamic batching

### Why this is interesting

Section 6.1 explains the throughput math: batching can multiply effective throughput by an order of magnitude on accelerators. The latency cost is small (one batching-window worth).

### Where it hooks in

`dispatcher_loop` in `src/queue.rs:146-180`. The loop currently spawns one worker per job; it should instead accumulate up to `MAX_BATCH` jobs (or wait at most `BATCH_WINDOW`), tokenize them together, and run a single forward pass.

```rust
// src/queue.rs:155 — TODO(Section 8.2): replace the
// `while let Some(job) = rx.recv().await { spawn(...) }` pattern
// with a batching window. See docs/06-production.md §6.1 for the
// pseudocode shape. The pipeline needs a new `infer_batch(&[&str])`
// method (src/pipeline/mod.rs:52) that produces `Vec<PipelineOutput>`.
```

### Design challenge

Padding. The tensor passed to ONNX must be rectangular. Two strategies:

1. **Pad to the longest sequence in the batch.** Minimal wasted compute; requires the model to handle dynamic batch *and* sequence dimensions (most BERT-family models do).
2. **Bucketing.** Group sequences into length buckets (e.g. ≤64, ≤128, ≤256) so each batch has similar lengths. More complex, but it keeps GPU utilization high.

Always update `attention_mask` to zero out padding positions so the model does not attend to padding tokens.

### Crates

No new crates required; this is a re-shape of the dispatch logic plus minor `ort` API changes (you'll pass `[batch_size, seq_len]` shaped tensors instead of `[1, seq_len]`).

---

## 8.3 Model hot-reload

### Why this is interesting

Reloading the model without restarting the process means clients see no connection churn and existing in-flight requests are not interrupted. A common pattern for canary deployments and emergency rollback.

### Where it hooks in

The pipeline owns the model via `Arc<dyn InferenceModel>` (`src/pipeline/mod.rs:30-34`). Today the `Arc` is created once. To support hot reload, swap the `Arc` itself behind an `arc_swap::ArcSwap`.

```rust
// src/pipeline/mod.rs:31 — TODO(Section 8.3): change `model: Arc<dyn InferenceModel>`
// to `model: arc_swap::ArcSwap<dyn InferenceModel>` and add a `swap(new_model)`
// method. New requests will pick up `model.load_full()` per call; existing
// requests already hold their own Arc and continue to completion.
```

You also need a watcher task (e.g. `notify` crate) that observes the model file and triggers `swap` when the inode changes. Validate the new model first by running a sample input through it — only swap on success.

### Design challenge

Atomic visibility. Inside one inference call, you must read the `Arc` once and reuse it for both the forward pass and the metadata that decorates the response (`src/pipeline/mod.rs:75-87`); otherwise you can return metadata from the new model and logits from the old one.

### Crates

- `arc_swap` — lock-free atomic Arc replacement.
- `notify` — cross-platform filesystem event listener.

---

## 8.4 GPU acceleration

### Why this is interesting

A single mid-range GPU can serve more inference than 16 CPU cores at lower cost-per-request — but only if the server keeps the GPU busy (Section 8.2 batching).

### Where it hooks in

`src/model/onnx.rs:28-45`. The `Session::builder()` chain accepts execution-provider configuration. Wire a build-time feature flag.

```rust
// Cargo.toml — TODO(Section 8.4): add a feature `gpu` that enables
// `ort/cuda` (or `ort/tensorrt`) and conditional code in src/model/onnx.rs
// to register the CUDA execution provider on the Session builder.
```

```rust
// src/model/onnx.rs:30 — example after the change:
// let mut builder = Session::builder()?;
// #[cfg(feature = "gpu")]
// {
//     builder = builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
// }
// let session = builder.commit_from_file(model_path)?;
```

### Design challenge

Tensor device placement. The CPU and CUDA paths in `ort` both accept the same `Tensor::<i64>::from_array(...)` interface; the *output* extraction (`src/model/onnx.rs:99-105`) copies back to host memory for you. For high-throughput streaming use cases you may want to keep tensors on the device and pipeline pre/post processing on the GPU as well — that's substantially more work.

The other constraint is `worker_count`. With a single GPU, set `worker_count = 1` because two concurrent CUDA streams from one process serialize at the driver. Capacity comes from batching (8.2), not parallelism.

### Crates

- `ort` with the `cuda` or `tensorrt` feature.
- CUDA toolkit installed at runtime (or TensorRT). The `load-dynamic` feature continues to work — just ensure `libonnxruntime_providers_cuda.so` is reachable.

---

## 8.5 Streaming responses (token-by-token)

### Why this is interesting

Text-generation models (LLMs) produce one token per forward step. Sending each token as soon as it's available improves perceived latency dramatically — clients see characters appear in milliseconds while the full response may take seconds.

### Where it hooks in

A new handler, plus a streaming variant of the model trait. The current `InferenceModel::predict` (`src/model/mod.rs:58`) returns a single `ModelOutput`. A streaming model would return a stream of tokens.

```rust
// src/model/mod.rs:56 — TODO(Section 8.5): add a sibling trait
// `pub trait StreamingModel: Send + Sync {
//     fn predict_stream<'a>(&'a self, input: &'a EncodedInput)
//         -> Pin<Box<dyn Stream<Item = AppResult<TokenChunk>> + Send + 'a>>;
// }`
// and a `pub async fn infer_stream` handler in src/server/handlers.rs
// that returns axum::response::sse::Sse or a StreamBody of NDJSON.
```

The queue needs to expose a streaming-friendly submit too — the `oneshot` reply pattern in `src/queue.rs:108-136` would be replaced by an `mpsc` per request so the worker can push token chunks as they arrive.

### Design challenge

Backpressure on the *response* side. A slow client must not allow a worker's `mpsc::Sender` to grow without bound. Use a small bounded channel per request and have the worker `await` the send — if the client is too slow, the worker naturally slows down.

### Crates

- `tokio-stream` — `Stream`/`StreamExt` adapters.
- `axum` already supports SSE via `axum::response::sse`.

---

## 8.6 Model versioning

### Why this is interesting

Serving multiple models or versions from the same process simplifies canary testing, A/B experiments, and model retirement.

### Where it hooks in

`build_app_runtime` (`src/lib.rs:62-107`) accepts a single `Arc<dyn InferenceModel>`. Replace it with a registry keyed by `(name, version)`.

```rust
// src/lib.rs:62 — TODO(Section 8.6): change signature to
// `pub async fn build_app_runtime(config, models: HashMap<(String, String), Arc<dyn InferenceModel>>, …)`
// and route `/v1/models/:name/versions/:version/infer` to the
// matching model. The default model alias lives in config.
```

```rust
// src/server/mod.rs:21 — TODO(Section 8.6): add
// `.route("/v1/models/:name/versions/:version/infer", post(infer_versioned))`
// `.route("/v1/models", get(list_models))`
```

### Design challenge

Per-model queues vs. a single shared queue with per-request routing. The shared-queue design is simpler but means a slow model can starve a fast one. A queue-per-model design is fairer but multiplies the bookkeeping (semaphore, depth gauge, metrics label cardinality).

For a small number of versions (≤4), one queue with metric labels for `model_name`/`model_version` is fine. Beyond that, give each its own `InferenceQueue::spawn` and own admission policy.

### Crates

No new crates. Just `HashMap<(String, String), Arc<dyn InferenceModel>>` in `AppState` (`src/server/state.rs:11-18`).

---

## 8.7 WebAssembly target for pre/post-processing

### Why this is interesting

Pushing tokenization and post-processing to the browser cuts server CPU and round-trip count. The same Rust code becomes a `.wasm` module the JavaScript client loads at startup.

### Where it hooks in

`src/pipeline/tokenizer.rs` and `src/pipeline/postprocess.rs` are the candidates — both already have no async, no I/O, and no Tokio dependencies. Carve them into a no-`tokio` sub-crate.

```rust
// Cargo.toml — TODO(Section 8.7): split this crate into a workspace.
// The new `inference-pipeline-wasm` crate re-exports softmax/argmax
// and TokenizerWrapper (without the From-file constructor, since WASM
// has no filesystem). Compile with:
//   cargo build --release --target wasm32-unknown-unknown -p inference-pipeline-wasm
```

### Design challenge

Tokenizer asset delivery. The `tokenizer.json` file is megabytes — fine for a server, awkward for a browser. Compress it (Brotli) and serve it from a CDN with long-lived caching headers; ship the WASM bundle as a separate, much smaller artifact.

### Crates

- `wasm-bindgen` — JS ↔ Rust interop.
- `serde-wasm-bindgen` — for passing arrays of token IDs without copying.
- `getrandom` with the `js` feature if any RNG is used.

---

## How to use this list

Each extension stands on its own. A reasonable sequence for a learner who wants to ship to production:

1. **8.2 Dynamic batching** — biggest immediate throughput win.
2. **8.4 GPU acceleration** — unlocks the next order of magnitude (once batching is in).
3. **8.3 Model hot-reload** — operational hygiene.
4. **8.6 Model versioning** — when you have more than one model to serve.
5. **8.1 gRPC** — only if you have gRPC-native clients.
6. **8.5 Streaming** — required for LLM use cases, irrelevant for classifiers.
7. **8.7 WASM** — niche; pursue only when client-side compute is genuinely cheaper than server-side.
