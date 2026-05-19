// --- Standard library imports ---
use std::{sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Instant};

// --- Prometheus imports ---
use metrics_exporter_prometheus::PrometheusHandle;

// --- Local imports ---
use crate::{config::AppConfig, model::ModelMetadata, queue::InferenceQueue};

// --- Shared application state ---
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub queue: Arc<InferenceQueue>,
    pub model_metadata: ModelMetadata,
    pub metrics_handle: Arc<PrometheusHandle>,
    ready: Arc<AtomicBool>,
    started_at: Instant,
}

impl AppState {
    // Construct the one shared state object every handler sees.
    pub fn new(
        config: Arc<AppConfig>,
        queue: Arc<InferenceQueue>,
        model_metadata: ModelMetadata,
        metrics_handle: Arc<PrometheusHandle>,
        ready: bool,
    ) -> Self {
        Self {
            config,
            queue,
            model_metadata,
            metrics_handle,
            ready: Arc::new(AtomicBool::new(ready)),
            started_at: Instant::now(),
        }
    }

    // Mark the model as ready for traffic.
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
        self.queue.mark_ready();
    }

    // Query readiness for `/ready`.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) && self.queue.is_accepting()
    }

    // Return the process uptime in seconds for health responses.
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}