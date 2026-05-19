# Section 4 — Line-by-Line Walkthrough

This section steps through the four most subtle code paths in the project. Each line is annotated in a two-column format so beginners can follow exactly what the compiler sees and why the line is needed.

The walkthroughs reference the current source verbatim. Pin the relevant file open in a second pane:

1. `src/queue.rs` (queue + worker loop)
2. `src/server/handlers.rs` (the `infer` handler)
3. `src/main.rs` (graceful shutdown)
4. `src/model/onnx.rs` (ONNX forward pass)

---

## 4.1 The inference queue and worker loop — `src/queue.rs`

This is the heart of the project. It owns three concurrency primitives at once (`Semaphore`, `mpsc`, `oneshot`) and bridges synchronous job submission with an async worker pool. Read it after Section 1.3.

### 4.1.1 The `InferenceJob` struct (`src/queue.rs:18-23`)

| Code | Explanation |
|------|-------------|
| `pub struct InferenceJob {` | A plain data carrier moved from the producer (handler) to the consumer (worker). |
| `    pub text: String,` | Owned input text. Moved into the worker so its lifetime is independent of the HTTP task. |
| `    pub response_tx: oneshot::Sender<AppResult<PipelineOutput>>,` | The "return address" for this specific job. `oneshot` is a one-time channel: send once, receive once. |
| `    pub slot_permit: OwnedSemaphorePermit,` | Holding this permit keeps the queue's overall slot count accurate. When the worker drops it, the slot is released and a new request can enter. |
| `}` | The struct has no methods — its only job is to carry these three values across the channel. |

### 4.1.2 The `InferenceQueue::spawn` factory (`src/queue.rs:52-79`)

| Code | Explanation |
|------|-------------|
| `pub fn spawn(config: QueueConfig, pipeline: Arc<InferencePipeline>, initially_ready: bool) -> (Arc<Self>, QueueRuntime) {` | Returns a *pair*: an `Arc<Self>` that producers clone freely, and a `QueueRuntime` consumed once by the binary to shut down. |
| `    let (tx, rx) = mpsc::channel(config.queue_capacity);` | Bounded mpsc channel. Capacity is the backpressure limit at the channel level — `send().await` will suspend (not panic) when full. |
| `    let total_slots = config.worker_count + config.queue_capacity;` | The semaphore must allow at least one permit per worker plus one per waiting slot, otherwise the system can deadlock: the channel could be empty while the semaphore says "full." |
| `    let queue = Arc::new(Self {` | Wrap in `Arc` so handlers can `clone()` cheaply. |
| `        sender: Mutex::new(Some(tx)),` | The sender lives behind a mutex with an `Option` so `close()` can take it and drop it (`src/queue.rs:140-143`), causing the dispatcher's `rx.recv().await` to return `None`. |
| `        slots: Arc::new(Semaphore::new(total_slots)),` | The outer semaphore. Acquired by `submit` *before* the job is enqueued, so the queue never silently overflows. |
| `        depth: Arc::new(AtomicUsize::new(0)),` | A gauge bookkeeping value exported to Prometheus. |
| `        accepting: Arc::new(AtomicBool::new(initially_ready)),` | A "soft" admission flag flipped by `mark_ready()` and `close()`. Lets the binary serve before the model is warm without crashing on traffic that arrives early. |
| `    });` | End of struct construction. |
| `    let queue_for_runtime = Arc::clone(&queue);` | One reference for the public `QueueRuntime`. |
| `    let depth_for_dispatcher = Arc::clone(&queue.depth);` | Dispatcher only needs the depth counter, not the whole queue. |
| `    let dispatcher = tokio::spawn(async move { Self::dispatcher_loop(rx, pipeline, config.worker_count, depth_for_dispatcher).await; });` | `tokio::spawn` returns a `JoinHandle`. The future is `'static` because it owns `rx`, `pipeline`, and the depth `Arc`. The `move` keyword forces ownership transfer into the future. |
| `    (queue, QueueRuntime { queue: queue_for_runtime, dispatcher })` | Return both handles. The dispatcher is exposed only to the runtime so user code can't accidentally await it. |

### 4.1.3 `submit` — the producer path (`src/queue.rs:92-137`)

This function blends three things that beginners often confuse: a semaphore acquisition, an mpsc send, and a oneshot await.

| Code | Explanation |
|------|-------------|
| `pub async fn submit(&self, text: String, enqueue_timeout: Option<Duration>) -> AppResult<PipelineOutput> {` | Async because we may await the slot permit, the channel send, and the worker reply. |
| `    if !self.is_accepting() { return Err(AppError::QueueClosed { request_id: None }); }` | Fast-fail when the queue is draining. The atomic read is wait-free. |
| `    let slot_future = Arc::clone(&self.slots).acquire_owned();` | `acquire_owned()` returns an `OwnedSemaphorePermit` that does not borrow from the semaphore — important because the permit travels across a channel into a different task. |
| `    let slot_permit = match enqueue_timeout {` | Two branches: with or without a deadline. |
| `        Some(timeout) => tokio::time::timeout(timeout, slot_future).await.map_err(|_| AppError::QueueFull { request_id: None })?.map_err(|_| AppError::QueueClosed { request_id: None })?,` | Outer `.map_err` converts a timeout into 503 `QueueFull`; inner `.map_err` converts a closed semaphore into 503 `QueueClosed`. The double `?` is necessary because `timeout()` returns `Result<Result<…,…>,Elapsed>`. |
| `        None => slot_future.await.map_err(|_| AppError::QueueClosed { request_id: None })?,` | Unbounded wait — only used from tests; production handlers always pass `Some(enqueue_timeout)` (see `src/server/handlers.rs:65-72`). |
| `    };` | At this point we *own* a permit. It will be dropped automatically when `InferenceJob` is dropped. |
| `    let (response_tx, response_rx) = oneshot::channel();` | Create the per-request reply channel. |
| `    let sender = self.sender.lock().await.as_ref().cloned().ok_or(AppError::QueueClosed { request_id: None })?;` | Take a fresh clone of the `mpsc::Sender`. `mpsc::Sender` is cheap to clone — it's an `Arc` internally. `.as_ref().cloned()` peels off the outer `Option<Sender>` without taking ownership of the option itself. |
| `    self.depth.fetch_add(1, Ordering::AcqRel);` | Atomically bump the queue depth gauge. `AcqRel` makes both the read and write visible across threads. |
| `    metrics::set_queue_depth(self.depth.load(Ordering::Acquire));` | Publish the new depth through the metrics facade. |
| `    let send_result = sender.send(InferenceJob { text, response_tx, slot_permit }).await;` | This await *could* suspend if the channel is full — but the slot semaphore already gates us, so in practice this completes immediately. |
| `    if send_result.is_err() { self.depth.fetch_sub(1, Ordering::AcqRel); … return Err(AppError::QueueClosed …); }` | If the dispatcher has shut down, undo the depth bump and report the error. |
| `    response_rx.await.map_err(|_| AppError::internal("worker dropped response channel before replying"))?` | Wait for the worker to send its `PipelineOutput`. A drop here is an *internal* bug, not a user error. |

### 4.1.4 `dispatcher_loop` — the consumer/worker fan-out (`src/queue.rs:146-180`)

| Code | Explanation |
|------|-------------|
| `async fn dispatcher_loop(mut rx: mpsc::Receiver<InferenceJob>, pipeline: Arc<InferencePipeline>, worker_count: usize, depth: Arc<AtomicUsize>) {` | Single instance per queue. Owns the receiver — no other task can read from the channel. |
| `    let worker_limiter = Arc::new(Semaphore::new(worker_count));` | Independent inner semaphore that bounds concurrent model executions. This is the *true* parallelism limit, separate from queue depth. |
| `    let mut in_flight = JoinSet::new();` | `JoinSet` is a Tokio collection of in-flight spawned tasks. We use it (not bare `spawn`) so we can `await` until *every* worker finishes during shutdown. |
| `    while let Some(job) = rx.recv().await {` | `recv()` resolves to `None` only when *all* senders are dropped. Closing the sender in `close()` ends this loop cleanly. |
| `        let worker_permit = Arc::clone(&worker_limiter).acquire_owned().await.expect("worker semaphore should stay open while dispatcher is alive");` | Acquire a worker slot *before* spawning — back-pressure inside the dispatcher. The `expect()` is safe because we own the only reference path that could close this semaphore. |
| `        let pipeline = Arc::clone(&pipeline);` | Each spawned worker captures its own Arc clone — the original stays in the dispatcher. |
| `        let depth = Arc::clone(&depth);` | Same for the depth counter. |
| `        in_flight.spawn(async move {` | Spawn a detached worker task tracked in the `JoinSet`. |
| `            let _worker_permit = worker_permit;` | Bind the permit to a local; it is dropped when the future ends, releasing the worker slot. |
| `            let start = std::time::Instant::now();` | Monotonic timer for the inference-latency histogram. |
| `            let result = pipeline.infer(&job.text).await;` | The actual model call. The pipeline borrows `&job.text` — the future is local so this borrow is fine. |
| `            if let Ok(output) = &result { metrics::record_inference(&output.model_name, start.elapsed().as_secs_f64() * 1_000.0); }` | Only record on success — failed inferences are accounted via the HTTP histogram instead, to avoid skewing inference latency with error fast paths. |
| `            let _ = job.response_tx.send(result);` | Send the result back. If the receiver has been dropped (caller gave up), the send returns `Err` — we ignore it. |
| `            drop(job.slot_permit);` | Explicitly drop the queue's slot permit so the next request can enter. The order matters: we drop *after* sending the reply so a new request cannot displace the in-flight one in latency-sensitive paths. |
| `            depth.fetch_sub(1, Ordering::AcqRel);` | Adjust the gauge. |
| `            metrics::set_queue_depth(depth.load(Ordering::Acquire));` | Publish. |
| `        });` | End of worker future. |
| `    }` | When the loop exits, `rx` has returned `None`, meaning `close()` ran. |
| `    while in_flight.join_next().await.is_some() {}` | Drain remaining workers before returning. This is the line that makes `runtime.shutdown().await` actually wait for in-flight work. |

---

## 4.2 The `/v1/infer` HTTP handler — `src/server/handlers.rs:60-86`

| Code | Explanation |
|------|-------------|
| `pub async fn infer(` | Async because it awaits the queue. |
| `    State(state): State<Arc<AppState>>,` | Axum extracts the shared state. `Arc<AppState>` clones in O(1). |
| `    Extension(context): Extension<RequestContext>,` | Inserted by the `request_context` middleware. Carries the request id (`src/server/middleware.rs:43-45`). |
| `    Json(payload): Json<InferRequest>,` | Body extractor. Failures here (bad JSON, missing fields, body too large) short-circuit *before* the handler body runs. |
| `) -> AppResult<impl IntoResponse> {` | Return type is a custom `Result`. The `Err` branch is converted by `AppError::into_response` (`src/error.rs:163-173`). |
| `    let result = state.queue.submit(payload.text, Some(Duration::from_millis(state.config.queue.enqueue_timeout_ms))).await.map_err(|error| error.with_request_id(context.request_id.clone()))?;` | Submit the inference. The `enqueue_timeout_ms` (default 25ms) is the *only* deadline that protects against unbounded queue waits. `with_request_id` attaches the request id to whatever error variant comes back so the JSON body has a usable trace key. The trailing `?` propagates. |
| `    Ok((` | Tuple response: status + body. |
| `        StatusCode::OK,` | Explicit 200 even though Axum would default to it — explicit beats implicit. |
| `        Json(InferResponse { request_id: context.request_id, label: result.label, … }),` | Build the response struct and serialize via `serde`. `Json(_)` sets `Content-Type: application/json` automatically. |
| `    ))` | End of `Ok`. |

The handler intentionally does *no* business logic — all validation and dispatch live in the pipeline and queue. Tests can exercise the same logic without the HTTP layer by calling `pipeline.infer(text).await` directly.

---

## 4.3 Graceful shutdown — `src/main.rs:46-119`

The shutdown path is short but easy to get wrong. The full sequence:

| Code | Explanation |
|------|-------------|
| `#[tokio::main]` | Macro expands to a `main()` that creates a Tokio runtime and runs the user's `async fn main` on it. |
| `async fn main() -> Result<()> {` | `Result<()>` is `anyhow::Result<()>` — convenient for top-level error chaining. |
| `    let config_path = parse_config_path();` | Plain CLI parsing; no external dependency. |
| `    let config = AppConfig::load(config_path.as_deref()).context("failed to load server configuration")?;` | Layer file + env into one validated struct (`src/config.rs:43-62`). `.context()` wraps the inner error with a human-readable message. |
| `    init_tracing(&config);` | Install the logger before any `info!` call. Logs are routed to stdout. |
| `    let metadata = ModelMetadata { … };` | Build the metadata snapshot. It is cloned into `AppState`, returned by `/v1/info`, and embedded in metric labels. |
| `    let model: Arc<dyn InferenceModel> = match config.model.backend.as_str() { "onnx" => Arc::new(OnnxModel::load(…)?), _ => Arc::new(MockModel::new(metadata.clone(), Duration::ZERO)), };` | Backend selection. Note the trait object: from this point on, neither the queue nor the pipeline nor any handler cares whether the backend is real or mock. |
| `    let tokenizer = match … { "onnx" => TokenizerWrapper::from_file(…)?, _ => TokenizerWrapper::mock(), };` | Same pattern for the tokenizer. The mock tokenizer is deterministic — see `src/pipeline/tokenizer.rs:72-92`. |
| `    let runtime = build_app_runtime(config.clone(), model, tokenizer, true).await.context("failed to build application runtime")?;` | Factory in `src/lib.rs:62-107` that installs the metrics recorder, builds the pipeline, spawns the queue dispatcher, and assembles the router. |
| `    runtime.state.mark_ready();` | Flip readiness *after* the queue is alive but *before* binding the listener — otherwise a `/ready` probe could 503 in the gap. |
| `    let listener = TcpListener::bind(format!("{}:{}", config.server.host, config.server.port)).await.context("failed to bind TCP listener")?;` | Bind explicitly so we can read back the local address if `port = 0` was passed (used in tests). |
| `    info!(…);` | Single structured log line indicating successful startup. |
| `    let router = runtime.router.clone();` | Cheap — `Router` is `Arc`-based internally. |
| `    let graceful = axum::serve(listener, router).with_graceful_shutdown(async {` | `with_graceful_shutdown` accepts any future. When that future resolves, Axum stops accepting new connections but lets in-flight responses finish. |
| `        let ctrl_c = async { let _ = signal::ctrl_c().await; };` | First signal source: Ctrl-C in any environment. |
| `        #[cfg(unix)] let terminate = async { let mut signal = signal::unix::signal(signal::unix::SignalKind::terminate()).expect("…"); let _ = signal.recv().await; };` | Second signal source on Unix: `SIGTERM`. This is the signal Kubernetes and systemd send. |
| `        #[cfg(not(unix))] let terminate = std::future::pending::<()>();` | On Windows, `SIGTERM` does not exist. `pending()` is a future that *never* completes — meaning the `select!` below will fall through to Ctrl-C exclusively. |
| `        tokio::select! { _ = ctrl_c => {}, _ = terminate => {}, }` | `select!` polls both futures in parallel and resolves with whichever completes first. The empty arm bodies mean "we don't care which signal fired, just return." |
| `    });` | End of the shutdown future. |
| `    graceful.await.context("HTTP server exited with an error")?;` | Now we block here until either: (a) a signal arrives → graceful shutdown begins → in-flight responses complete → `graceful` resolves, or (b) an irrecoverable error occurs. |
| `    runtime.shutdown().await.context("failed to drain queue runtime")?;` | After Axum has stopped accepting connections, drain the worker pool. `QueueRuntime::shutdown` closes the queue's mpsc sender, then awaits the dispatcher's `JoinHandle` (`src/queue.rs:40-47`). The dispatcher in turn waits for every spawned worker via the `JoinSet` (`src/queue.rs:179`). |
| `    Ok(())` | Process exits cleanly. |

The order matters:

1. Stop accepting new TCP connections (Axum).
2. Allow currently-accepted requests to complete (Axum's in-flight handling).
3. Drain the queue so any handler awaiting `response_rx` gets its reply.
4. Exit.

If we reversed steps 3 and 4, in-flight requests inside the queue would be cancelled and clients would see dropped TCP connections.

---

## 4.4 The ONNX forward pass — `src/model/onnx.rs:48-117`

This is the only place we touch a non-Rust runtime. The trick is **never block the Tokio reactor with C++ FFI calls.**

| Code | Explanation |
|------|-------------|
| `impl InferenceModel for OnnxModel {` | Implement the same trait `MockModel` does. Callers cannot distinguish them. |
| `    fn predict<'a>(&'a self, input: &'a EncodedInput) -> ModelFuture<'a> {` | Returns the manual async-trait future type (`src/model/mod.rs:52`). The `'a` lifetime ties the future to both `&self` and `&input`. |
| `        let session = Arc::clone(&self.session);` | The session lives behind `Arc<Mutex<Session>>` (`src/model/onnx.rs:21`). Cloning the `Arc` does not clone the model — it only bumps the refcount. |
| `        let input_ids = input.input_ids.iter().copied().collect::<Vec<_>>();` | Copy the contiguous `i64` token-id buffer. We *must* own the buffer inside the spawned blocking task because `&EncodedInput` borrows for `'a` and `spawn_blocking` requires `'static`. |
| `        let attention_mask = input.attention_mask.iter().copied().collect::<Vec<_>>();` | Same for the attention mask. |
| `        let sequence_length = input.input_ids.shape()[1];` | Record the dynamic dimension; ONNX tensors are `[batch=1, seq=sequence_length]`. |
| `        Box::pin(async move {` | Heap-allocate and pin the async block so it can be returned through a trait object. |
| `            tokio::task::spawn_blocking(move || -> AppResult<ModelOutput> {` | Hand the heavy synchronous work to Tokio's blocking-pool thread. The Tokio reactor stays free to drive other futures. The `move` closure owns `session`, `input_ids`, `attention_mask`, and `sequence_length`. |
| `                let mut session = session.lock().map_err(|_| AppError::internal("failed to lock ONNX session"))?;` | `std::sync::Mutex::lock()` returns a `PoisonError` if a previous holder panicked. We surface that as an internal error rather than panicking again. |
| `                let input_ids_tensor = Tensor::<i64>::from_array(([1_usize, sequence_length], input_ids.into_boxed_slice()))` | Build an `ort::value::Tensor` from a `(shape, boxed_slice)` tuple. `into_boxed_slice()` hands ownership to `ort` so it can manage the lifetime. |
| `                    .map_err(|error| AppError::ModelError { message: format!("failed to build input_ids tensor: {error}"), request_id: None, })?;` | Map any `ort` error into our 503 `ModelError`. |
| `                let attention_mask_tensor = Tensor::<i64>::from_array((…)).map_err(…)?;` | Same for the mask. |
| `                let inputs = ort::inputs! { "input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor, }.map_err(…)?;` | Build the named-input map the SST-2 model expects. The `inputs!` macro produces a `Result` of `SessionInputValue` pairs. |
| `                let outputs = session.run(inputs).map_err(|error| AppError::ModelError { message: format!("failed to execute ONNX session: {error}"), request_id: None, })?;` | The actual forward pass — synchronous C++ inside a managed thread. |
| `                let output = &outputs[0];` | Single-output model; index 0 is the logits tensor. |
| `                let (_shape, values) = output.try_extract_raw_tensor::<f32>().map_err(…)?;` | Pull out a `f32` slice. We discard the shape because softmax/argmax only need the flat values. |
| `                Ok(ModelOutput { logits: values.to_vec(), })` | Copy into an owned `Vec<f32>` so the result outlives the locked session. |
| `            })` | End of the blocking closure. |
| `            .await` | Await the spawn_blocking handle. Cancellation is cooperative — if the awaiting task is dropped, the blocking thread still finishes the current Session::run. |
| `            .map_err(|error| AppError::ModelError { message: format!("blocking ONNX task failed to join: {error}"), request_id: None, })?` | If the worker thread panicked, surface that as a model error. |
| `        })` | End of the outer `async move` / `Box::pin`. |
| `    }` | End of `predict`. |
| `    fn metadata(&self) -> ModelMetadata { self.metadata.as_ref().clone() }` | Cheap snapshot — used by `/v1/info` and metric labels. |
| `}` | End of trait impl. |

The structural shape worth memorizing:

```text
async fn returning ModelFuture<'a>
   └─ Box::pin(async move {
          └─ spawn_blocking(move || {
                └─ session.lock() → build tensors → session.run → extract
             })
          .await
      })
```

This is the canonical Rust pattern for wrapping any blocking C++ inference engine (Torch, TensorRT, etc.) into an async server.
