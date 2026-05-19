// --- Standard library imports ---
use std::path::Path;

// --- ndarray imports ---
use ndarray::Array2;

// --- tokenizers imports ---
use tokenizers::Tokenizer;

// --- Local imports ---
use crate::{
    error::{AppError, AppResult},
    model::EncodedInput,
};

// --- Internal tokenizer variants ---
enum InnerTokenizer {
    // A real Hugging Face tokenizer loaded from `tokenizer.json`.
    HuggingFace(Tokenizer),
    // A deterministic whitespace tokenizer used by tests and the default mock mode.
    Whitespace,
}

// --- Public wrapper ---
pub struct TokenizerWrapper {
    inner: InnerTokenizer,
}

impl TokenizerWrapper {
    // Load a production tokenizer from disk.
    pub fn from_file(path: &Path) -> AppResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| AppError::ModelUnavailable {
            message: format!("failed to load tokenizer from {}: {error}", path.display()),
            request_id: None,
        })?;

        Ok(Self {
            inner: InnerTokenizer::HuggingFace(tokenizer),
        })
    }

    // Construct the deterministic tokenizer used in tests and the mock backend.
    pub fn mock() -> Self {
        Self {
            inner: InnerTokenizer::Whitespace,
        }
    }

    // Convert raw user text into the `EncodedInput` tensors the model consumes.
    pub fn encode(&self, text: &str, max_tokens: usize) -> AppResult<EncodedInput> {
        match &self.inner {
            InnerTokenizer::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(text, true)
                    .map_err(|error| AppError::validation(format!("failed to tokenize input text: {error}")))?;

                let token_ids = encoding
                    .get_ids()
                    .iter()
                    .take(max_tokens)
                    .map(|value| i64::from(*value))
                    .collect::<Vec<_>>();
                let attention_mask = encoding
                    .get_attention_mask()
                    .iter()
                    .take(max_tokens)
                    .map(|value| i64::from(*value))
                    .collect::<Vec<_>>();

                Self::build_encoded_input(token_ids, attention_mask)
            }
            InnerTokenizer::Whitespace => {
                let mut token_ids = Vec::with_capacity(max_tokens.min(text.len() + 2));
                let mut attention_mask = Vec::with_capacity(max_tokens.min(text.len() + 2));

                token_ids.push(101);
                attention_mask.push(1);

                for token in text.split_whitespace().take(max_tokens.saturating_sub(2)) {
                    let hashed = token
                        .bytes()
                        .fold(0_u64, |accumulator, byte| accumulator.wrapping_mul(31).wrapping_add(u64::from(byte)));
                    token_ids.push((hashed % 30_000) as i64 + 100);
                    attention_mask.push(1);
                }

                token_ids.push(102);
                attention_mask.push(1);

                Self::build_encoded_input(token_ids, attention_mask)
            }
        }
    }

    // Build the 2D tensors expected by our model abstraction.
    fn build_encoded_input(token_ids: Vec<i64>, attention_mask: Vec<i64>) -> AppResult<EncodedInput> {
        let token_count = token_ids.len();
        let input_ids = Array2::from_shape_vec((1, token_count), token_ids).map_err(|error| {
            AppError::internal(format!("failed to shape input_ids tensor: {error}"))
        })?;
        let attention_mask = Array2::from_shape_vec((1, token_count), attention_mask).map_err(|error| {
            AppError::internal(format!("failed to shape attention_mask tensor: {error}"))
        })?;

        Ok(EncodedInput {
            input_ids,
            attention_mask,
            token_count,
        })
    }
}