// --- Axum imports ---
// `IntoResponse` turns Rust errors into HTTP responses automatically.
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

// --- Serde imports ---
// We serialize the API error body as JSON.
use serde::Serialize;

// --- Thiserror imports ---
// `Error` derives a human-readable `Display` implementation for our enum.
use thiserror::Error;

// --- Result alias ---
// A project-wide alias keeps signatures readable without hiding the concrete error type.
pub type AppResult<T> = Result<T, AppError>;

// --- API error body ---
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

// --- Application error enum ---
#[derive(Debug, Error, Clone)]
pub enum AppError {
    #[error("configuration error: {message}")]
    Config {
        message: String,
        request_id: Option<String>,
    },
    #[error("validation error: {message}")]
    Validation {
        message: String,
        request_id: Option<String>,
    },
    #[error("model unavailable: {message}")]
    ModelUnavailable {
        message: String,
        request_id: Option<String>,
    },
    #[error("inference failed: {message}")]
    ModelError {
        message: String,
        request_id: Option<String>,
    },
    #[error("queue is full")]
    QueueFull {
        request_id: Option<String>,
    },
    #[error("queue is closed")]
    QueueClosed {
        request_id: Option<String>,
    },
    #[error("payload too large")]
    PayloadTooLarge {
        request_id: Option<String>,
    },
    #[error("request timed out")]
    Timeout {
        request_id: Option<String>,
    },
    #[error("internal server error: {message}")]
    Internal {
        message: String,
        request_id: Option<String>,
    },
}

impl AppError {
    // Build a config error without repeating the enum name at call sites.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            request_id: None,
        }
    }

    // Build a validation error for bad user input or invalid config.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            request_id: None,
        }
    }

    // Build an internal error for unexpected conditions.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            request_id: None,
        }
    }

    // Attach a request ID after the error is created, which is useful in handlers.
    pub fn with_request_id(self, request_id: impl Into<String>) -> Self {
        let request_id = Some(request_id.into());
        match self {
            Self::Config { message, .. } => Self::Config { message, request_id },
            Self::Validation { message, .. } => Self::Validation { message, request_id },
            Self::ModelUnavailable { message, .. } => Self::ModelUnavailable { message, request_id },
            Self::ModelError { message, .. } => Self::ModelError { message, request_id },
            Self::QueueFull { .. } => Self::QueueFull { request_id },
            Self::QueueClosed { .. } => Self::QueueClosed { request_id },
            Self::PayloadTooLarge { .. } => Self::PayloadTooLarge { request_id },
            Self::Timeout { .. } => Self::Timeout { request_id },
            Self::Internal { message, .. } => Self::Internal { message, request_id },
        }
    }

    // Map each error variant to the HTTP status code clients should see.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Validation { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::ModelUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::ModelError { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::QueueFull { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::QueueClosed { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::PayloadTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            Self::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    // Give each error a stable machine-readable code.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config_error",
            Self::Validation { .. } => "validation_error",
            Self::ModelUnavailable { .. } => "model_unavailable",
            Self::ModelError { .. } => "model_error",
            Self::QueueFull { .. } => "queue_full",
            Self::QueueClosed { .. } => "queue_closed",
            Self::PayloadTooLarge { .. } => "payload_too_large",
            Self::Timeout { .. } => "timeout",
            Self::Internal { .. } => "internal_error",
        }
    }

    // Return the optional request ID stored on the error.
    pub fn request_id(&self) -> Option<String> {
        match self {
            Self::Config { request_id, .. }
            | Self::Validation { request_id, .. }
            | Self::ModelUnavailable { request_id, .. }
            | Self::ModelError { request_id, .. }
            | Self::QueueFull { request_id }
            | Self::QueueClosed { request_id }
            | Self::PayloadTooLarge { request_id }
            | Self::Timeout { request_id }
            | Self::Internal { request_id, .. } => request_id.clone(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ApiErrorBody {
            error: self.error_code().to_string(),
            message: self.to_string(),
            request_id: self.request_id(),
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    // --- Axum imports ---
    use axum::response::IntoResponse;

    // --- HTTP imports ---
    use axum::http::StatusCode;

    // --- Local imports ---
    use super::AppError;

    #[test]
    fn test_model_error_maps_to_503() {
        // WHAT: Model execution failures become HTTP 503.
        // WHY: A temporary model backend issue is a service availability problem, not a client bug.
        let response = AppError::ModelError {
            message: "backend offline".to_string(),
            request_id: None,
        }
        .into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_validation_error_maps_to_422() {
        // WHAT: Validation failures become HTTP 422.
        // WHY: Clients need a precise signal that their payload shape or content is invalid.
        let response = AppError::validation("bad input").into_response();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}