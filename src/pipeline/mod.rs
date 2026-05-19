// --- Child modules ---
pub mod postprocess;
pub mod tokenizer;

// --- Standard library imports ---
use std::sync::Arc;

// --- Local imports ---
use crate::{
    config::PipelineConfig,
    error::{AppError, AppResult},
    model::{InferenceModel, ModelMetadata},
};

// --- Child module imports ---
use self::{postprocess::{argmax, decode_label, softmax}, tokenizer::TokenizerWrapper};

// --- Pipeline output ---
#[derive(Debug, Clone, serde::Serialize)]
pub struct PipelineOutput {
    pub label: String,
    pub confidence: f32,
    pub token_count: usize,
    pub logits: Vec<f32>,
    pub model_name: String,
    pub model_version: String,
}

// --- Inference pipeline ---
pub struct InferencePipeline {
    model: Arc<dyn InferenceModel>,
    tokenizer: TokenizerWrapper,
    config: PipelineConfig,
}

impl InferencePipeline {
    // Assemble the tokenizer and model into one reusable pipeline object.
    pub fn new(model: Arc<dyn InferenceModel>, tokenizer: TokenizerWrapper, config: PipelineConfig) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }

    // Return model metadata for `/v1/info` without making a model call.
    pub fn metadata(&self) -> ModelMetadata {
        self.model.metadata()
    }

    // Run the full text -> tensor -> logits -> label path for one request.
    pub async fn infer(&self, text: &str) -> AppResult<PipelineOutput> {
        // Reject empty or all-whitespace input early so we do not waste queue capacity.
        if text.trim().is_empty() {
            return Err(AppError::validation("input text must not be empty"));
        }
        // Cheap character-limit validation happens before tokenization.
        if text.chars().count() > self.config.max_characters {
            return Err(AppError::PayloadTooLarge { request_id: None });
        }

        // Tokenize the user text into model-ready integer tensors.
        let encoded = self.tokenizer.encode(text, self.config.max_tokens)?;
        // Run the model backend.
        let model_output = self.model.predict(&encoded).await?;
        // Convert raw logits into a probability distribution.
        let probabilities = softmax(&model_output.logits);
        // Pick the most likely class.
        let (best_index, confidence) = argmax(&probabilities).ok_or_else(|| {
            AppError::ModelError {
                message: "model returned no logits".to_string(),
                request_id: None,
            }
        })?;
        // Fetch descriptive metadata after inference so the response can name the model.
        let metadata = self.model.metadata();
        // Decode the class index into a human-readable label.
        let label = decode_label(&metadata.labels, best_index);

        Ok(PipelineOutput {
            label,
            confidence,
            token_count: encoded.token_count,
            logits: model_output.logits,
            model_name: metadata.name,
            model_version: metadata.version,
        })
    }
}