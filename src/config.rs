// --- Standard library imports ---
// `Path` and `PathBuf` let us refer to config files without hard-coding platform-specific separators.
use std::path::{Path, PathBuf};

// --- Third-party imports ---
// Serde derives let us deserialize TOML and environment variables into typed Rust structs.
use serde::{Deserialize, Serialize};

// --- Local imports ---
// `AppError` gives us one consistent validation and startup error type.
use crate::error::{AppError, AppResult};

// --- Top-level application config ---
// `Clone` is useful because tests often tweak a base config, and `Serialize` helps documentation examples and debugging.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    // Server network and timeout settings.
    pub server: ServerConfig,
    // Queue sizing and backpressure behavior.
    pub queue: QueueConfig,
    // Model backend selection and metadata.
    pub model: ModelConfig,
    // Tokenization and input validation limits.
    pub pipeline: PipelineConfig,
    // Logging preferences for the tracing subscriber.
    pub logging: LoggingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            queue: QueueConfig::default(),
            model: ModelConfig::default(),
            pipeline: PipelineConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl AppConfig {
    // Load config from an optional file path and then layer environment variables on top.
    pub fn load(path: Option<&Path>) -> AppResult<Self> {
        // Use the provided path when present, or fall back to the conventional default config file.
        let config_path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("config/default.toml"));
        // Build the layered configuration source list.
        let settings = config::Config::builder()
            .add_source(config::File::from(config_path).required(false))
            .add_source(config::Environment::with_prefix("INFERENCE_SERVER").separator("__"))
            .build()
            .map_err(|error| AppError::config(format!("failed to read configuration sources: {error}")))?;
        // Deserialize the layered data into our strongly typed struct.
        let loaded = settings
            .try_deserialize::<AppConfig>()
            .map_err(|error| AppError::config(format!("failed to deserialize configuration: {error}")))?;
        // Validate user input before the server starts.
        loaded.validate()?;
        // Return the safe, typed, validated config.
        Ok(loaded)
    }

    // Validate config values that cannot be expressed directly in the TOML type system.
    pub fn validate(&self) -> AppResult<()> {
        // Zero worker threads would mean the queue never executes work.
        if self.queue.worker_count == 0 {
            return Err(AppError::validation(
                "queue.worker_count must be greater than zero",
            ));
        }
        // Zero queue capacity means no waiting room for bursts of traffic.
        if self.queue.queue_capacity == 0 {
            return Err(AppError::validation(
                "queue.queue_capacity must be greater than zero",
            ));
        }
        // The enqueue timeout must be positive so callers cannot wait forever accidentally.
        if self.queue.enqueue_timeout_ms == 0 {
            return Err(AppError::validation(
                "queue.enqueue_timeout_ms must be greater than zero",
            ));
        }
        // Requests must have a timeout so slow dependencies do not pin resources indefinitely.
        if self.server.request_timeout_ms == 0 {
            return Err(AppError::validation(
                "server.request_timeout_ms must be greater than zero",
            ));
        }
        // Grace periods must also be positive or graceful shutdown becomes impossible.
        if self.server.shutdown_grace_period_ms == 0 {
            return Err(AppError::validation(
                "server.shutdown_grace_period_ms must be greater than zero",
            ));
        }
        // Input validation limits must be present to defend memory usage.
        if self.pipeline.max_tokens == 0 {
            return Err(AppError::validation(
                "pipeline.max_tokens must be greater than zero",
            ));
        }
        // Character limits should never be zero because that would reject all input.
        if self.pipeline.max_characters == 0 {
            return Err(AppError::validation(
                "pipeline.max_characters must be greater than zero",
            ));
        }
        // The model metadata is surfaced to clients, so it should not be empty.
        if self.model.labels.is_empty() {
            return Err(AppError::validation(
                "model.labels must contain at least one label",
            ));
        }
        // Restrict the backend string to the two supported implementations.
        if self.model.backend != "mock" && self.model.backend != "onnx" {
            return Err(AppError::validation(
                "model.backend must be either 'mock' or 'onnx'",
            ));
        }
        // Logging level is parsed at runtime by tracing, but we still reject the obviously empty case.
        if self.logging.level.trim().is_empty() {
            return Err(AppError::validation(
                "logging.level must not be empty",
            ));
        }
        // Everything looked sane.
        Ok(())
    }
}

// --- Server config ---
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_payload_bytes: usize,
    pub request_timeout_ms: u64,
    pub shutdown_grace_period_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            max_payload_bytes: 16 * 1024,
            request_timeout_ms: 5_000,
            shutdown_grace_period_ms: 10_000,
        }
    }
}

// --- Queue config ---
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueueConfig {
    pub worker_count: usize,
    pub queue_capacity: usize,
    pub enqueue_timeout_ms: u64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            worker_count: 2,
            queue_capacity: 16,
            enqueue_timeout_ms: 25,
        }
    }
}

// --- Model config ---
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub backend: String,
    pub model_path: String,
    pub tokenizer_path: String,
    pub name: String,
    pub version: String,
    pub labels: Vec<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backend: "mock".to_string(),
            model_path: "models/distilbert-sst2/model.onnx".to_string(),
            tokenizer_path: "models/distilbert-sst2/tokenizer.json".to_string(),
            name: "distilbert-sst2".to_string(),
            version: "1".to_string(),
            labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
        }
    }
}

// --- Pipeline config ---
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineConfig {
    pub max_tokens: usize,
    pub max_characters: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            max_characters: 4_096,
        }
    }
}

// --- Logging config ---
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub json: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json: true,
        }
    }
}

#[cfg(test)]
mod tests {
    // --- Standard library imports ---
    use std::fs;

    // --- Tokio imports ---
    use tokio::runtime::Runtime;

    // --- Local imports ---
    use super::AppConfig;

    #[test]
    fn test_config_validation_rejects_zero_workers() {
        // WHAT: Reject a queue config with zero workers.
        // WHY: A queue with no workers would accept traffic and never complete requests.
        let mut config = AppConfig::default();
        config.queue.worker_count = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_env_override() {
        // WHAT: Environment variables override file config values.
        // WHY: Containerized deployments rely on env vars to override baked-in defaults safely.
        let temp_path = std::env::temp_dir().join("inference-server-config-test.toml");
        fs::write(
            &temp_path,
            "[queue]\nworker_count = 1\nqueue_capacity = 2\nenqueue_timeout_ms = 10\n\n[server]\nhost = \"127.0.0.1\"\nport = 3000\nmax_payload_bytes = 1024\nrequest_timeout_ms = 1000\nshutdown_grace_period_ms = 1000\n\n[model]\nbackend = \"mock\"\nmodel_path = \"unused\"\ntokenizer_path = \"unused\"\nname = \"mock\"\nversion = \"1\"\nlabels = [\"NEGATIVE\", \"POSITIVE\"]\n\n[pipeline]\nmax_tokens = 8\nmax_characters = 128\n\n[logging]\nlevel = \"info\"\njson = true\n",
        )
        .expect("temp config file should be writable");

        std::env::set_var("INFERENCE_SERVER__QUEUE__WORKER_COUNT", "7");
        let loaded = AppConfig::load(Some(&temp_path)).expect("config should load with env overrides");
        std::env::remove_var("INFERENCE_SERVER__QUEUE__WORKER_COUNT");
        let _ = fs::remove_file(&temp_path);

        assert_eq!(loaded.queue.worker_count, 7);

        // Keep Tokio linked into the test target so async-only dependencies are exercised by `cargo test`.
        let runtime = Runtime::new().expect("Tokio runtime should construct in unit tests");
        runtime.block_on(async {});
    }
}