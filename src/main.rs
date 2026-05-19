// --- Standard library imports ---
use std::{path::PathBuf, sync::Arc, time::Duration};

// --- Anyhow imports ---
use anyhow::{Context, Result};

// --- Tokio imports ---
use tokio::{net::TcpListener, signal};

// --- Tracing imports ---
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

// --- Local imports ---
use inference_server::{
    build_app_runtime,
    config::AppConfig,
    model::{mock::MockModel, onnx::OnnxModel, InferenceModel, ModelMetadata},
    pipeline::tokenizer::TokenizerWrapper,
};

// --- CLI parsing ---
fn parse_config_path() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--config" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

// --- Tracing setup ---
fn init_tracing(config: &AppConfig) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.logging.level.clone()));

    if config.logging.json {
        fmt().json().with_env_filter(filter).init();
    } else {
        fmt().with_env_filter(filter).init();
    }
}

// --- Main entry point ---
#[tokio::main]
async fn main() -> Result<()> {
    let config_path = parse_config_path();
    let config = AppConfig::load(config_path.as_deref())
        .context("failed to load server configuration")?;

    init_tracing(&config);

    let metadata = ModelMetadata {
        name: config.model.name.clone(),
        version: config.model.version.clone(),
        backend: config.model.backend.clone(),
        input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
        labels: config.model.labels.clone(),
        max_tokens: config.pipeline.max_tokens,
    };

    let model: Arc<dyn InferenceModel> = match config.model.backend.as_str() {
        "onnx" => Arc::new(
            OnnxModel::load(PathBuf::from(&config.model.model_path).as_path(), metadata.clone())
                .context("failed to load ONNX model")?,
        ),
        _ => Arc::new(MockModel::new(metadata.clone(), Duration::ZERO)),
    };

    let tokenizer = match config.model.backend.as_str() {
        "onnx" => TokenizerWrapper::from_file(PathBuf::from(&config.model.tokenizer_path).as_path())
            .context("failed to load tokenizer")?,
        _ => TokenizerWrapper::mock(),
    };

    let runtime = build_app_runtime(config.clone(), model, tokenizer, true)
        .await
        .context("failed to build application runtime")?;
    runtime.state.mark_ready();

    let listener = TcpListener::bind(format!("{}:{}", config.server.host, config.server.port))
        .await
        .context("failed to bind TCP listener")?;

    info!(
        host = %config.server.host,
        port = config.server.port,
        backend = %config.model.backend,
        "server listening"
    );

    let router = runtime.router.clone();

    let graceful = axum::serve(listener, router).with_graceful_shutdown(async {
        let ctrl_c = async {
            let _ = signal::ctrl_c().await;
        };

        #[cfg(unix)]
        let terminate = async {
            let mut signal = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("SIGTERM signal handler should install on Unix");
            let _ = signal.recv().await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    });

    graceful.await.context("HTTP server exited with an error")?;
    runtime.shutdown().await.context("failed to drain queue runtime")?;

    Ok(())
}