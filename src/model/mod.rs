// --- Standard library imports ---
// `Future` and `Pin` are the building blocks of async trait methods without relying on an extra macro crate.
use std::{future::Future, pin::Pin};

// --- Third-party imports ---
// Serde lets us serialize model metadata on the `/v1/info` endpoint.
use serde::{Deserialize, Serialize};

// --- Local imports ---
// `AppResult` keeps trait signatures short and consistent.
use crate::error::AppResult;

// --- Child modules ---
pub mod mock;
pub mod onnx;

// --- Shared model input ---
// This struct represents the tokenized tensors that both the mock backend and ONNX backend consume.
#[derive(Debug, Clone)]
pub struct EncodedInput {
    // `input_ids` are the integer token IDs the language model understands.
    pub input_ids: ndarray::Array2<i64>,
    // `attention_mask` tells the model which token positions are real input versus padding.
    pub attention_mask: ndarray::Array2<i64>,
    // `token_count` is convenient metadata for logs, metrics, and API responses.
    pub token_count: usize,
}

// --- Shared model output ---
#[derive(Debug, Clone)]
pub struct ModelOutput {
    // Raw, unnormalized class scores from the model.
    pub logits: Vec<f32>,
}

// --- Model metadata ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub backend: String,
    pub input_names: Vec<String>,
    pub labels: Vec<String>,
    pub max_tokens: usize,
}

// --- Async model future alias ---
// `Pin<Box<dyn Future + Send>>` is the classic way to express an async trait method on stable Rust.
// `Pin` promises the future will not be moved in memory after polling starts,
// `Box` gives it a known size, `dyn Future` allows different concrete futures behind one trait,
// and `Send` means Tokio may move the future between worker threads safely.
pub type ModelFuture<'a> = Pin<Box<dyn Future<Output = AppResult<ModelOutput>> + Send + 'a>>;

// --- Model trait ---
// This trait hides whether inference happens through ONNX Runtime or the deterministic mock backend.
pub trait InferenceModel: Send + Sync {
    // Run the forward pass for one already-tokenized input.
    fn predict<'a>(&'a self, input: &'a EncodedInput) -> ModelFuture<'a>;
    // Return immutable descriptive metadata used by API responses and metrics labels.
    fn metadata(&self) -> ModelMetadata;
}