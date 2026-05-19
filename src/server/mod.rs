// --- Child modules ---
pub mod handlers;
pub mod middleware;
pub mod state;

// --- Axum imports ---
use axum::{
    extract::DefaultBodyLimit,
    middleware::from_fn_with_state,
    routing::{get, post},
    Router,
};

// --- Standard library imports ---
use std::sync::Arc;

// --- Local imports ---
use crate::server::{handlers::{health, info, infer, metrics, ready}, state::AppState};

// --- Router assembly ---
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/v1/info", get(info))
        .route("/v1/infer", post(infer))
        .layer(DefaultBodyLimit::max(state.config.server.max_payload_bytes))
        .layer(from_fn_with_state(Arc::clone(&state), middleware::request_context))
        .with_state(state)
}