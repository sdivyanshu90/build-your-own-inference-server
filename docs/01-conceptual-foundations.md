# Section 1 — Conceptual Foundations

This document is the conceptual on-ramp for the rest of the repository. It assumes nothing more than basic familiarity with HTTP and Python-style programming. Every Rust feature used in the project is introduced here so the source files can be read without surprise.

---

## 1.1 What Is an Inference Server?

### Inference vs. training

Machine-learning workflows have two distinct phases:

| Phase | Goal | Cost profile | Frequency |
|-------|------|--------------|-----------|
| **Training** | Compute model weights that minimize a loss function over a dataset. | Hours to weeks; usually on a GPU cluster. | Performed rarely. |
| **Inference** | Apply already-trained weights to a single new input and produce a prediction. | Milliseconds to seconds; CPU or single GPU. | Performed continuously, often millions of times per day. |

An **inference server** is the long-running process that exposes the second phase as a network service. The weights are loaded once at startup; client requests flow through a fixed pipeline (tokenize → forward pass → decode) and return a structured response.

The model file used by this project is `models/distilbert-sst2/model.onnx` (a quantized DistilBERT sentiment classifier — see `models/README.md`). The pipeline that wraps it is `src/pipeline/mod.rs:36-88`.

### Request lifecycle

A single inference call traverses these stages:

```text
┌──────────────────────────────────────────────────────────────┐
│ POST /v1/infer  { "text": "I love Rust" }                    │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
   ┌─────────────────┐  inject x-request-id, start latency timer
   │  Middleware     │  enforce per-request timeout
   └─────────────────┘  src/server/middleware.rs
            │
            ▼
   ┌─────────────────┐  deserialize JSON → InferRequest
   │  Handler        │  acquire semaphore permit (backpressure)
   └─────────────────┘  send job over mpsc channel + oneshot reply
            │           src/server/handlers.rs:60-86
            ▼
   ┌─────────────────┐  dispatcher receives job; spawns a worker
   │  Queue          │  worker calls pipeline.infer(text)
   └─────────────────┘  src/queue.rs:145-180
            │
            ▼
   ┌─────────────────┐  text → token IDs + attention mask tensors
   │  Tokenizer      │  src/pipeline/tokenizer.rs
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐  ONNX Session::run() or MockModel logits
   │  Model forward  │  src/model/onnx.rs:48-117  /  mock.rs:33-55
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐  softmax → argmax → decode_label
   │  Postprocess    │  src/pipeline/postprocess.rs
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐  worker sends PipelineOutput back over oneshot
   │  Handler resume │  serialize InferResponse to JSON
   └─────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ 200 OK  { "label": "POSITIVE", "confidence": 0.97, ... }     │
└──────────────────────────────────────────────────────────────┘
```

### Architecture at a glance (basic)

```text
[Client] → [HTTP Layer] → [Request Queue] → [Model Runner] → [Response]
                               ↓
                        [Backpressure / Rate Limiting]
```

### Architecture in detail

```text
                            ┌───────────────────────────────────┐
                            │           Tokio Runtime           │
                            │  ┌─────────────────────────────┐  │
        HTTP request ────►  │  │  Axum task per connection   │  │
                            │  └────────────┬────────────────┘  │
                            │               │ Json body          │
                            │               ▼                    │
                            │  ┌─────────────────────────────┐  │
                            │  │  infer() handler            │  │
                            │  │  - acquire semaphore slot   │  │
                            │  │  - mpsc::Sender::send(job)  │  │
                            │  │  - await oneshot::Receiver  │  │
                            │  └────────────┬────────────────┘  │
                            │               │                    │
                            │               ▼                    │
                            │  ┌─────────────────────────────┐  │
                            │  │  Dispatcher task            │  │
                            │  │  - rx.recv() from mpsc      │  │
                            │  │  - worker_limiter permit    │  │
                            │  │  - spawn worker fn          │  │
                            │  └────────────┬────────────────┘  │
                            │               │                    │
                            │  ┌────────────▼────────────┐       │
                            │  │  Worker tasks (JoinSet) │       │
                            │  │  - pipeline.infer()     │       │
                            │  │  - send via oneshot     │◄────┐ │
                            │  │  - drop slot permit     │     │ │
                            │  └────────────┬────────────┘     │ │
                            └───────────────┼──────────────────┘ │
                                            ▼                    │
                                   spawn_blocking pool           │
                                  (ONNX Session::run)            │
                                            │                    │
                                            ▼                    │
                            ┌───────────────────────────────────┐│
                            │ Arc<Mutex<Session>>  (weights in  ││
                            │ RAM, shared across worker tasks)  ││
                            └───────────────────────────────────┘│
                                                                 │
                       ┌──── metrics side-channel ───────────────┘
                       ▼
                ┌────────────────────────────────────┐
                │  metrics crate facade              │
                │  → metrics-exporter-prometheus     │
                │  → /metrics endpoint               │
                └────────────────────────────────────┘
```

Three key flows are visible:

1. **Request flow** — Axum task → handler → mpsc → dispatcher → worker → pipeline.
2. **Reply flow** — worker → oneshot → handler → JSON body.
3. **Observability side-channel** — every layer emits counters/histograms through the global recorder installed in `src/metrics.rs:25-53`.

---

## 1.2 Why Rust for Inference Servers?

### Zero-cost abstractions

"Zero-cost" means a high-level construct compiles down to the same machine code you would have written by hand at a lower level. In this project, `Arc<dyn InferenceModel>` (a heap-allocated reference-counted trait object) emits exactly two instructions on the hot path: an atomic refcount increment and an indirect call through a vtable. There is no garbage-collected wrapper, no reflection, no hidden allocator detour. The trait is defined in `src/model/mod.rs:54-61`.

### No garbage-collector pauses

A latency-sensitive inference call typically takes 20–80 ms. A stop-the-world GC pause of even 50 ms doubles the tail latency. Rust has no GC; memory is freed deterministically when ownership ends. Worker tasks that allocate token buffers (see `src/pipeline/tokenizer.rs:71-92`) deallocate them on the same task, so heap pressure is predictable.

### Ownership as a correctness guarantee

The compiler refuses to let two threads mutate the same model state without explicit synchronization. The ONNX `Session` is wrapped in `Arc<Mutex<Session>>` (`src/model/onnx.rs:19-24`) because `Session::run` needs `&mut self`. The compiler enforces — at compile time — that the lock is held before the mutation runs.

### `unsafe`-free path to SIMD

`ndarray` is the tensor container used by `EncodedInput` (`src/model/mod.rs:20-27`). When compiled with the `blas` feature, it dispatches matrix operations to vendor-tuned SIMD kernels without exposing any `unsafe` blocks to the user.

### Rust vs. Python vs. Go (illustrative)

| Concern | Rust (this server) | Python (FastAPI + ONNX Runtime) | Go (net/http + onnxruntime-go) |
|---------|--------------------|---------------------------------|--------------------------------|
| Cold start | tens of ms | seconds (interpreter + imports) | tens of ms |
| Tail latency (p99) under load | predictable, no GC pauses | spiky due to GIL contention and GC | predictable, but GC pauses are occasionally visible |
| Concurrency model | async/await on Tokio, M:N | thread pool gated by GIL | goroutines, M:N |
| Memory footprint | small; deterministic | large; reference cycles + interpreter | medium |
| Backend FFI | thin (`ort` crate) | thin (Python C-API) | thin (cgo) |
| Refactor safety | compiler enforces invariants | tests only | partial via type system |

The numbers vary by workload, but the *shape* of the trade-off is consistent: Rust gives you predictable tail latency and small binaries; Python gives you the fastest research-to-prototype iteration; Go sits in the middle with simpler concurrency syntax but a GC.

---

## 1.3 Rust Concepts Used in This Project (Beginner Primer)

Every concept below is paired with a tiny standalone example plus a pointer to where it shows up in the codebase.

### Ownership & borrowing

Every value in Rust has exactly one owner. Passing it to a function moves it; passing `&value` borrows it temporarily.

```rust
fn consume(x: String) { /* x is dropped at the end of this function */ }
fn inspect(x: &String) { /* x is borrowed, caller still owns the value */ }

let name = String::from("rust");
inspect(&name);   // OK — name is still usable afterwards
consume(name);    // moves; using `name` after this is a compile error
```

In the server, the user's text is *moved* into the inference job (`src/queue.rs:121-126`), because the worker needs to own it for the lifetime of the inference call.

### `Arc<T>` — Atomically Reference-Counted

Imagine a shared library book with a counter on the cover; each new reader bumps the counter, returning the book decrements it. When the counter hits zero, the book is shelved (dropped). `Arc<T>` does this with atomic CPU instructions so it works across threads.

```rust
use std::sync::Arc;
let shared = Arc::new(vec![1, 2, 3]);
let copy_for_other_thread = Arc::clone(&shared);
std::thread::spawn(move || println!("{:?}", copy_for_other_thread));
```

This server uses `Arc<dyn InferenceModel>` so every worker task holds a cheap reference to the same model weights (`src/lib.rs:62-87`).

### `async` / `await` and Tokio

`async fn` returns a *future* — a state machine that produces a value when polled. `.await` suspends the current task until the future is ready, freeing the OS thread to run other tasks. Tokio is the runtime that polls these futures across a small thread pool.

```rust
async fn fetch() -> u32 { 42 }

#[tokio::main]
async fn main() {
    let value = fetch().await;  // suspends until fetch() finishes
    println!("{value}");
}
```

The HTTP entry point uses `#[tokio::main]` at `src/main.rs:46-47`; every handler in `src/server/handlers.rs` is `async`.

### `Result<T, E>` and `?`

Rust has no exceptions. Functions that can fail return `Result<T, E>`. The `?` operator unwraps `Ok` or returns the `Err` to the caller — the equivalent of a checked exception, but visible in the type signature.

```rust
fn parse_port(s: &str) -> Result<u16, std::num::ParseIntError> {
    let n: u16 = s.parse()?;  // ? propagates the error
    Ok(n)
}
```

`AppResult<T>` is the project-wide alias defined at `src/error.rs:19`, used everywhere from config loading to handler responses.

### Traits and trait objects

A trait is an interface. `dyn TraitName` is a type-erased pointer-to-an-implementor — like a Java interface reference.

```rust
trait Shape { fn area(&self) -> f32; }
struct Square { side: f32 }
impl Shape for Square { fn area(&self) -> f32 { self.side * self.side } }

let s: Box<dyn Shape> = Box::new(Square { side: 2.0 });
```

`InferenceModel` is defined at `src/model/mod.rs:56-61` and implemented by both `OnnxModel` and `MockModel`. The server only ever sees `Arc<dyn InferenceModel>`, so swapping backends is a one-line change at startup (`src/main.rs:63-69`).

### Enums and pattern matching

Rust enums are *tagged unions* — each variant can carry its own data. `match` exhaustively destructures them.

```rust
enum Status { Idle, Running { since_ms: u64 }, Failed(String) }

let s = Status::Running { since_ms: 42 };
match s {
    Status::Idle => println!("idle"),
    Status::Running { since_ms } => println!("running for {since_ms}ms"),
    Status::Failed(reason) => println!("failed: {reason}"),
}
```

`AppError` (`src/error.rs:31-74`) uses this pattern to keep one structured type that maps to many HTTP status codes (`src/error.rs:118-130`).

### `impl Trait` and generics

`impl Trait` in a return position means "I'm returning *some* concrete type that implements this trait — don't ask which one." It is a compile-time abstraction with zero runtime cost.

```rust
fn make_doubler() -> impl Fn(i32) -> i32 { |x| x * 2 }
```

The `infer` handler uses `impl IntoResponse` (`src/server/handlers.rs:60-64`) so it can return any concrete response type Axum can serialize.

### Channels: `mpsc` and `oneshot`

A **`mpsc::channel`** is a multi-producer, single-consumer queue. **`oneshot::channel`** sends exactly one value. In this project:

- `mpsc<InferenceJob>` is the request queue from handlers to the dispatcher (`src/queue.rs:57`).
- `oneshot::Sender<AppResult<PipelineOutput>>` is the reply path back to the handler that submitted the job (`src/queue.rs:21-23`).

```rust
let (tx, rx) = tokio::sync::oneshot::channel::<u32>();
tokio::spawn(async move { tx.send(7).unwrap(); });
let received = rx.await.unwrap();
```

### `Mutex` vs. `RwLock`

A `Mutex` allows one writer at a time. A `RwLock` allows many readers OR one writer. We use `tokio::sync::Mutex` to guard the queue's sender handle (`src/queue.rs:27`), because the sender is *replaced* (not just read) at shutdown — there is no read-heavy access pattern that would justify an `RwLock`. For the ONNX session, we use `std::sync::Mutex` (`src/model/onnx.rs:21`) because the model call already happens inside `spawn_blocking`, so locking is synchronous.

### Derive macros

`#[derive(Debug, Serialize, Deserialize)]` auto-generates implementations at compile time. Examples are everywhere in `src/config.rs` and `src/server/handlers.rs:22-57`.

### Lifetimes — focus on `'static`

A lifetime annotation tells the compiler how long a reference is valid. The most common one in async code is `'static`: a value that does not borrow from anything with a shorter lifetime. Tokio tasks must be `'static` because the runtime may keep them alive across thread moves.

```rust
fn spawn(future: impl Future<Output = ()> + Send + 'static) { /* ... */ }
```

`Pipeline` lifetimes appear in `src/model/mod.rs:52` — the future borrows the model and input for `'a`. This is *not* `'static`: the future is constructed inside `Box::pin` and awaited immediately within the same scope (`src/queue.rs:166`), so the borrows are valid.

### `Pin<Box<dyn Future + Send>>`

This is the "manual async trait" pattern used at `src/model/mod.rs:52`:

```rust
pub type ModelFuture<'a> = Pin<Box<dyn Future<Output = AppResult<ModelOutput>> + Send + 'a>>;
```

Decomposed:

- `dyn Future<…>` — an unknown concrete future type behind a trait object.
- `Box<…>` — heap-allocated so the size is known at compile time.
- `Pin<…>` — promises the future will not be moved in memory after polling begins, which is required because self-referential async state machines rely on stable addresses.
- `+ Send` — Tokio may move the future between worker threads.
- `+ 'a` — the future may borrow data with lifetime `'a`.

Without this pattern, a trait method cannot return an async value on stable Rust without the `async-trait` macro. The crate uses the manual approach to keep the dependency tree small.

---

## What's next

- **Section 2** maps the directory layout and full request data flow to specific file/line locations.
- **Section 4** walks through the four most subtle code paths (queue worker loop, infer handler, graceful shutdown, ONNX forward pass) line by line.
