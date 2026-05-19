// --- Standard library imports ---
use std::{sync::Arc, time::Duration};

// --- Axum imports ---
use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue},
    middleware::Next,
    response::{IntoResponse, Response},
};

// --- Tracing imports ---
use tracing::info;

// --- UUID imports ---
use uuid::Uuid;

// --- Local imports ---
use crate::{error::AppError, metrics, server::state::AppState};

// --- Request context ---
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
}

// --- Request middleware ---
pub async fn request_context(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    request.extensions_mut().insert(RequestContext {
        request_id: request_id.clone(),
    });

    let response_or_timeout = tokio::time::timeout(
        Duration::from_millis(state.config.server.request_timeout_ms),
        next.run(request),
    )
    .await;

    let mut response = match response_or_timeout {
        Ok(response) => response,
        Err(_) => AppError::Timeout {
            request_id: Some(request_id.clone()),
        }
        .into_response(),
    };

    // DESIGN NOTE: UUIDs contain only ASCII-safe characters, so converting them into an HTTP header is infallible in practice.
    let header_value = HeaderValue::from_str(&request_id)
        .expect("UUID request IDs should always produce a valid header value");
    response.headers_mut().insert(
        HeaderName::from_static("x-request-id"),
        header_value,
    );

    let latency_ms = start.elapsed().as_secs_f64() * 1_000.0;
    metrics::record_http_request(&path, response.status().as_u16(), latency_ms);
    info!(
        request_id = %request_id,
        method = %method,
        path = %path,
        status = response.status().as_u16(),
        latency_ms = latency_ms,
        "request completed"
    );

    response
}