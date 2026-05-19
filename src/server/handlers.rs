// --- Standard library imports ---
use std::{sync::Arc, time::Duration};

// --- Axum imports ---
use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};

// --- Serde imports ---
use serde::{Deserialize, Serialize};

// --- Local imports ---
use crate::{
    error::{AppError, AppResult},
    server::{middleware::RequestContext, state::AppState},
};

// --- Inference API types ---
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub request_id: String,
    pub label: String,
    pub confidence: f32,
    pub token_count: usize,
    pub logits: Vec<f32>,
    pub model_name: String,
    pub model_version: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct InfoResponse {
    pub name: String,
    pub version: String,
    pub backend: String,
    pub input_names: Vec<String>,
    pub labels: Vec<String>,
    pub max_tokens: usize,
}

// --- Handler implementations ---
pub async fn infer(
    State(state): State<Arc<AppState>>,
    Extension(context): Extension<RequestContext>,
    Json(payload): Json<InferRequest>,
) -> AppResult<impl IntoResponse> {
    let result = state
        .queue
        .submit(
            payload.text,
            Some(Duration::from_millis(state.config.queue.enqueue_timeout_ms)),
        )
        .await
        .map_err(|error| error.with_request_id(context.request_id.clone()))?;

    Ok((
        StatusCode::OK,
        Json(InferResponse {
            request_id: context.request_id,
            label: result.label,
            confidence: result.confidence,
            token_count: result.token_count,
            logits: result.logits,
            model_name: result.model_name,
            model_version: result.model_version,
        }),
    ))
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            uptime_seconds: state.uptime_seconds(),
        }),
    )
}

pub async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.is_ready() {
        (
            StatusCode::OK,
            Json(ReadyResponse { status: "ready" }),
        )
            .into_response()
    } else {
        AppError::ModelUnavailable {
            message: "model is not ready yet".to_string(),
            request_id: None,
        }
        .into_response()
    }
}

pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics_handle.render(),
    )
}

pub async fn info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metadata = &state.model_metadata;

    (
        StatusCode::OK,
        Json(InfoResponse {
            name: metadata.name.clone(),
            version: metadata.version.clone(),
            backend: metadata.backend.clone(),
            input_names: metadata.input_names.clone(),
            labels: metadata.labels.clone(),
            max_tokens: metadata.max_tokens,
        }),
    )
}