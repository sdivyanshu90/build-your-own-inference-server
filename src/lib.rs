// --- Public module declarations ---
// Expose the configuration layer so both the binary and tests can build the same application.
pub mod config;
// Expose the application error type so handlers, middleware, and tests share one error contract.
pub mod error;
// Expose metrics helpers so the server and tests can inspect `/metrics` consistently.
pub mod metrics;
// Expose model backends so the binary can choose ONNX or mock mode at startup.
pub mod model;
// Expose the inference pipeline so queue workers can tokenize, run the model, and decode outputs.
pub mod pipeline;
// Expose the queue so HTTP handlers can submit work safely.
pub mod queue;
// Expose the HTTP server assembly so the binary and tests can build routers the same way.
pub mod server;

// --- Standard library imports ---
// `Arc` is the thread-safe shared pointer we use for state shared across async tasks.
use std::sync::Arc;

// --- Local imports ---
// `AppConfig` holds all validated runtime settings.
use crate::config::AppConfig;
// `AppResult` is our project-wide result alias.
use crate::error::AppResult;
// `install_metrics_recorder` creates or reuses the global Prometheus recorder.
use crate::metrics::install_metrics_recorder;
// `InferenceModel` is the trait object that hides whether the backend is mock or ONNX.
use crate::model::InferenceModel;
// `InferencePipeline` is the tokenizer -> model -> postprocess assembly.
use crate::pipeline::InferencePipeline;
// `TokenizerWrapper` loads a real tokenizer or the deterministic mock tokenizer.
use crate::pipeline::tokenizer::TokenizerWrapper;
// `InferenceQueue` accepts user work and runs it on background workers.
use crate::queue::{InferenceQueue, QueueRuntime};
// `build_router` assembles the Axum routes and middleware.
use crate::server::build_router;
// `AppState` is the shared state extracted by handlers.
use crate::server::state::AppState;

// --- Application runtime ---
// This struct bundles the router and the queue runtime so callers can serve HTTP and later drain workers.
pub struct AppRuntime {
    // The router is the HTTP surface that Axum serves.
    pub router: axum::Router,
    // Shared state is exposed so tests can inspect readiness or metrics when needed.
    pub state: Arc<AppState>,
    // The queue runtime owns the worker dispatcher join handle.
    pub queue_runtime: QueueRuntime,
}

impl AppRuntime {
    // Shut down background queue workers after the HTTP server has stopped accepting new requests.
    pub async fn shutdown(self) -> AppResult<()> {
        // Delegate the actual drain behavior to the queue runtime.
        self.queue_runtime.shutdown().await
    }
}

// --- Application construction ---
// Build the full application around any model backend that implements `InferenceModel`.
pub async fn build_app_runtime(
    // The validated config controls ports, queue sizing, and model metadata.
    config: AppConfig,
    // The trait object lets tests swap in a mock model without changing server code.
    model: Arc<dyn InferenceModel>,
    // The tokenizer wrapper can be a real Hugging Face tokenizer or a mock whitespace tokenizer.
    tokenizer: TokenizerWrapper,
    // This flag controls `/ready`, which is useful in tests that simulate model warmup.
    initially_ready: bool,
) -> AppResult<AppRuntime> {
    // Wrap config in `Arc` so state, middleware, and handlers can all share it cheaply.
    let shared_config = Arc::new(config);
    // Install or reuse the global Prometheus recorder before any metrics are emitted.
    let metrics_handle = install_metrics_recorder()?;
    // Build the inference pipeline that every queue worker will call into.
    let pipeline = Arc::new(InferencePipeline::new(
        Arc::clone(&model),
        tokenizer,
        shared_config.pipeline.clone(),
    ));
    // Create the queue plus its background dispatcher task.
    let (queue, queue_runtime) = InferenceQueue::spawn(
        shared_config.queue.clone(),
        Arc::clone(&pipeline),
        initially_ready,
    );
    // Build the immutable metadata snapshot returned by `/v1/info`.
    let metadata = model.metadata();
    // Assemble all server-visible shared state.
    let state = Arc::new(AppState::new(
        Arc::clone(&shared_config),
        Arc::clone(&queue),
        metadata,
        metrics_handle,
        initially_ready,
    ));
    // Build the Axum router with routes, middleware, and body limits.
    let router = build_router(Arc::clone(&state));

    // Return the assembled runtime bundle to the caller.
    Ok(AppRuntime {
        router,
        state,
        queue_runtime,
    })
}