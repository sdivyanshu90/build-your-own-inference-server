// --- Standard library imports ---
use std::{sync::Arc, time::Duration};

// --- Tokio imports ---
use tokio::time::sleep;

// --- Local imports ---
use crate::{
    model::{EncodedInput, InferenceModel, ModelFuture, ModelMetadata, ModelOutput},
};

// --- Mock model ---
// This backend is deterministic and lightweight, which makes it ideal for tests and tutorials.
#[derive(Debug, Clone)]
pub struct MockModel {
    // Metadata is returned verbatim from the trait.
    metadata: Arc<ModelMetadata>,
    // An optional delay helps us test backpressure and graceful shutdown behavior.
    delay: Duration,
}

impl MockModel {
    // Construct a mock model using the same metadata shape as the ONNX backend.
    pub fn new(metadata: ModelMetadata, delay: Duration) -> Self {
        Self {
            metadata: Arc::new(metadata),
            delay,
        }
    }
}

impl InferenceModel for MockModel {
    fn predict<'a>(&'a self, input: &'a EncodedInput) -> ModelFuture<'a> {
        // Clone the delay because the async block captures by value.
        let delay = self.delay;
        // Copy the token IDs so the future owns the data it needs.
        let token_values: Vec<i64> = input.input_ids.iter().copied().collect();

        Box::pin(async move {
            // Sleep only when a delay is configured; this keeps the common fast-path cheap.
            if !delay.is_zero() {
                sleep(delay).await;
            }

            // Sum token IDs into a deterministic pseudo-score.
            let sum = token_values.iter().copied().sum::<i64>();
            // Fold the sum into a bounded floating-point range.
            let positive_score = ((sum.rem_euclid(1_000)) as f32 / 1_000.0) + 0.25;
            // Build two logits so postprocessing can produce a binary classification.
            let logits = vec![1.0 - positive_score, positive_score];

            Ok(ModelOutput { logits })
        })
    }

    fn metadata(&self) -> ModelMetadata {
        self.metadata.as_ref().clone()
    }
}

#[cfg(test)]
mod tests {
    // --- Standard library imports ---
    use std::time::Duration;

    // --- Local imports ---
    use crate::model::{EncodedInput, InferenceModel, ModelMetadata};

    // --- ndarray imports ---
    use ndarray::Array2;

    #[tokio::test]
    async fn test_mock_model_is_deterministic() {
        // WHAT: The mock model returns the same logits for the same input every time.
        // WHY: Deterministic tests are essential when the mock backend stands in for a real model.
        let model = super::MockModel::new(
            ModelMetadata {
                name: "mock".to_string(),
                version: "1".to_string(),
                backend: "mock".to_string(),
                input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
                labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
                max_tokens: 8,
            },
            Duration::ZERO,
        );

        let input = EncodedInput {
            input_ids: Array2::from_shape_vec((1, 3), vec![101, 200, 102]).expect("shape should be valid"),
            attention_mask: Array2::from_shape_vec((1, 3), vec![1, 1, 1]).expect("shape should be valid"),
            token_count: 3,
        };

        let first = model.predict(&input).await.expect("first inference should work");
        let second = model.predict(&input).await.expect("second inference should work");

        assert_eq!(first.logits, second.logits);
    }
}