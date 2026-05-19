// --- Standard library imports ---
use std::sync::{Arc, Mutex, OnceLock};

// --- metrics imports ---
use metrics::{
    counter,
    describe_counter,
    describe_gauge,
    describe_histogram,
    gauge,
    histogram,
};

// --- Prometheus exporter imports ---
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

// --- Local imports ---
use crate::error::{AppError, AppResult};

// --- Global recorder installation state ---
static METRICS_INSTALL_LOCK: Mutex<()> = Mutex::new(());
static METRICS_HANDLE: OnceLock<Arc<PrometheusHandle>> = OnceLock::new();

// --- Recorder installation ---
pub fn install_metrics_recorder() -> AppResult<Arc<PrometheusHandle>> {
    if let Some(existing) = METRICS_HANDLE.get() {
        return Ok(Arc::clone(existing));
    }

    let _guard = METRICS_INSTALL_LOCK
        .lock()
        .expect("metrics installation mutex should not be poisoned");

    if let Some(existing) = METRICS_HANDLE.get() {
        return Ok(Arc::clone(existing));
    }

    describe_counter!("http_requests_total", "Total number of HTTP requests handled.");
    describe_counter!("inference_requests_total", "Total number of inference requests submitted.");
    describe_gauge!("queue_depth", "Current number of in-flight plus queued inference jobs.");
    describe_histogram!("http_request_latency_ms", "End-to-end HTTP request latency in milliseconds.");
    describe_histogram!("inference_latency_ms", "Model pipeline latency in milliseconds.");

    let handle = Arc::new(
        PrometheusBuilder::new()
            .install_recorder()
            .map_err(|error| AppError::internal(format!("failed to install Prometheus recorder: {error}")))?,
    );

    let _ = METRICS_HANDLE.set(Arc::clone(&handle));

    Ok(handle)
}

// --- Metric emission helpers ---
pub fn record_http_request(endpoint: &str, status: u16, latency_ms: f64) {
    counter!(
        "http_requests_total",
        "endpoint" => endpoint.to_string(),
        "status" => status.to_string(),
    )
    .increment(1);
    histogram!(
        "http_request_latency_ms",
        "endpoint" => endpoint.to_string(),
    )
    .record(latency_ms);
}

pub fn record_inference(model_name: &str, latency_ms: f64) {
    counter!(
        "inference_requests_total",
        "model_name" => model_name.to_string(),
    )
    .increment(1);
    histogram!(
        "inference_latency_ms",
        "model_name" => model_name.to_string(),
    )
    .record(latency_ms);
}

pub fn set_queue_depth(depth: usize) {
    gauge!("queue_depth").set(depth as f64);
}