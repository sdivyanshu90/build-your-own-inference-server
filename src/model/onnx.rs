// --- Standard library imports ---
use std::{path::Path, sync::{Arc, Mutex}};

// --- ONNX Runtime imports ---
// `Session` loads the model and runs inference, while `Tensor` wraps raw input arrays.
use ort::{
    session::Session,
    value::Tensor,
};

// --- Local imports ---
use crate::{
    error::{AppError, AppResult},
    model::{EncodedInput, InferenceModel, ModelFuture, ModelMetadata, ModelOutput},
};

// --- ONNX model backend ---
// `Session::run` needs `&mut self`, so we protect it with a `Mutex` because multiple async tasks share one model instance.
pub struct OnnxModel {
    // The session holds the loaded model weights and execution context.
    session: Arc<Mutex<Session>>,
    // Metadata is cloned into API responses and metrics labels.
    metadata: Arc<ModelMetadata>,
}

impl OnnxModel {
    // Load an ONNX model from disk.
    pub fn load(model_path: &Path, metadata: ModelMetadata) -> AppResult<Self> {
        // Let `ort` create its default environment lazily and commit a session from the model file.
        let session = Session::builder()
            .map_err(|error| AppError::ModelUnavailable {
                message: format!("failed to create ONNX session builder: {error}"),
                request_id: None,
            })?
            .commit_from_file(model_path)
            .map_err(|error| AppError::ModelUnavailable {
                message: format!("failed to load ONNX model from {}: {error}", model_path.display()),
                request_id: None,
            })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            metadata: Arc::new(metadata),
        })
    }
}

impl InferenceModel for OnnxModel {
    fn predict<'a>(&'a self, input: &'a EncodedInput) -> ModelFuture<'a> {
        // Clone the session handle because the blocking task owns its captures.
        let session = Arc::clone(&self.session);
        // Copy input tensors into contiguous vectors because `ort` accepts raw shape + data tuples.
        let input_ids = input.input_ids.iter().copied().collect::<Vec<_>>();
        let attention_mask = input.attention_mask.iter().copied().collect::<Vec<_>>();
        // Record the dynamic sequence length for the ONNX input tensor shape.
        let sequence_length = input.input_ids.shape()[1];

        Box::pin(async move {
            // ONNX Runtime inference is CPU-bound and uses a blocking mutex, so we move it onto Tokio's blocking pool.
            tokio::task::spawn_blocking(move || -> AppResult<ModelOutput> {
                // Lock the session so only one thread mutates the `Session` at a time.
                let session = session.lock().map_err(|_| AppError::internal("failed to lock ONNX session"))?;
                // Build the `input_ids` tensor in the `[batch, sequence]` shape expected by BERT-like models.
                let input_ids_tensor = Tensor::<i64>::from_array((
                    [1_usize, sequence_length],
                    input_ids.into_boxed_slice(),
                ))
                .map_err(|error| AppError::ModelError {
                    message: format!("failed to build input_ids tensor: {error}"),
                    request_id: None,
                })?;
                // Build the matching attention mask tensor.
                let attention_mask_tensor = Tensor::<i64>::from_array((
                    [1_usize, sequence_length],
                    attention_mask.into_boxed_slice(),
                ))
                .map_err(|error| AppError::ModelError {
                    message: format!("failed to build attention_mask tensor: {error}"),
                    request_id: None,
                })?;
                // Build named ONNX inputs separately so any macro error maps into our application error type.
                let inputs = ort::inputs! {
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                }
                .map_err(|error| AppError::ModelError {
                    message: format!("failed to build ONNX inputs: {error}"),
                    request_id: None,
                })?;
                // Run the model by name because the SST-2 DistilBERT graph exposes named inputs.
                let outputs = session
                    .run(inputs)
                    .map_err(|error| AppError::ModelError {
                        message: format!("failed to execute ONNX session: {error}"),
                        request_id: None,
                    })?;
                // The first output contains the classification logits for binary sentiment.
                let output = &outputs[0];
                // Extract the tensor into raw dimensions and a float slice without depending on `ort`'s optional ndarray feature.
                let (_shape, values) = output
                    .try_extract_raw_tensor::<f32>()
                    .map_err(|error| AppError::ModelError {
                        message: format!("failed to extract ONNX output tensor: {error}"),
                        request_id: None,
                    })?;

                Ok(ModelOutput {
                    logits: values.to_vec(),
                })
            })
            .await
            .map_err(|error| AppError::ModelError {
                message: format!("blocking ONNX task failed to join: {error}"),
                request_id: None,
            })?
        })
    }

    fn metadata(&self) -> ModelMetadata {
        self.metadata.as_ref().clone()
    }
}