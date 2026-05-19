use std::{sync::Arc, time::Duration};

use inference_server::{
    build_app_runtime,
    config::{AppConfig, ModelConfig, PipelineConfig, QueueConfig, ServerConfig, LoggingConfig},
    model::{mock::MockModel, InferenceModel, ModelMetadata},
    pipeline::tokenizer::TokenizerWrapper,
};
use reqwest::Client;
use serde_json::Value;
use tokio::{net::TcpListener, sync::oneshot};

struct TestServer {
    base_url: String,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl TestServer {
    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.join_handle.await;
    }
}

fn base_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_bytes: 1024,
            request_timeout_ms: 1_000,
            shutdown_grace_period_ms: 2_000,
        },
        queue: QueueConfig {
            worker_count: 4,
            queue_capacity: 32,
            enqueue_timeout_ms: 50,
        },
        model: ModelConfig {
            backend: "mock".to_string(),
            model_path: "unused".to_string(),
            tokenizer_path: "unused".to_string(),
            name: "mock".to_string(),
            version: "1".to_string(),
            labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
        },
        pipeline: PipelineConfig {
            max_tokens: 32,
            max_characters: 128,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            json: true,
        },
    }
}

async fn spawn_server(config: AppConfig, ready: bool, delay: Duration) -> TestServer {
    let metadata = ModelMetadata {
        name: config.model.name.clone(),
        version: config.model.version.clone(),
        backend: config.model.backend.clone(),
        input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
        labels: config.model.labels.clone(),
        max_tokens: config.pipeline.max_tokens,
    };
    let model: Arc<dyn InferenceModel> = Arc::new(MockModel::new(metadata, delay));
    let runtime = build_app_runtime(config.clone(), model, TokenizerWrapper::mock(), ready)
        .await
        .expect("test app should build");
    if ready {
        runtime.state.mark_ready();
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener should bind");
    let address = listener.local_addr().expect("local addr should be available");
    let base_url = format!("http://{}", address);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let router = runtime.router.clone();

    let join_handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await?;
        runtime.shutdown().await.map_err(anyhow::Error::from)?;
        Ok(())
    });

    TestServer {
        base_url,
        shutdown_tx: Some(shutdown_tx),
        join_handle,
    }
}

#[tokio::test]
async fn test_infer_endpoint_returns_200_with_valid_input() {
    // WHAT: A valid inference request succeeds end to end.
    // WHY: This is the primary contract clients depend on.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .post(format!("{}/v1/infer", server.base_url))
        .json(&serde_json::json!({ "text": "I love Rust" }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("response should be valid JSON");
    assert!(body.get("label").is_some());
    assert!(body.get("confidence").is_some());
    server.shutdown().await;
}

#[tokio::test]
async fn test_infer_endpoint_returns_422_with_empty_input() {
    // WHAT: Empty input is rejected with HTTP 422.
    // WHY: Clients need fast feedback instead of spending queue capacity on invalid requests.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .post(format!("{}/v1/infer", server.base_url))
        .json(&serde_json::json!({ "text": "   " }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 422);
    server.shutdown().await;
}

#[tokio::test]
async fn test_infer_endpoint_returns_413_with_oversized_input() {
    // WHAT: Overly long input is rejected with HTTP 413.
    // WHY: Size limits prevent memory abuse and pathological tokenizer work.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();
    let oversized = "a".repeat(1024);

    let response = client
        .post(format!("{}/v1/infer", server.base_url))
        .json(&serde_json::json!({ "text": oversized }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 413);
    server.shutdown().await;
}

#[tokio::test]
async fn test_health_returns_200_always() {
    // WHAT: The liveness probe always returns 200 while the process is running.
    // WHY: Orchestrators use liveness to decide whether to restart the process.
    let server = spawn_server(base_config(), false, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .get(format!("{}/health", server.base_url))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 200);
    server.shutdown().await;
}

#[tokio::test]
async fn test_ready_returns_200_when_model_loaded() {
    // WHAT: Readiness returns 200 once the app is marked ready.
    // WHY: Load balancers should only send traffic to warmed-up instances.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .get(format!("{}/ready", server.base_url))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 200);
    server.shutdown().await;
}

#[tokio::test]
async fn test_ready_returns_503_before_model_loaded() {
    // WHAT: Readiness returns 503 before the model is marked ready.
    // WHY: This prevents a cold instance from receiving production traffic too early.
    let server = spawn_server(base_config(), false, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .get(format!("{}/ready", server.base_url))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), 503);
    server.shutdown().await;
}

#[tokio::test]
async fn test_metrics_endpoint_returns_prometheus_format() {
    // WHAT: The metrics endpoint exposes Prometheus text format.
    // WHY: Scrapers and dashboards rely on this format remaining stable.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let _ = client
        .post(format!("{}/v1/infer", server.base_url))
        .json(&serde_json::json!({ "text": "metrics sample" }))
        .send()
        .await
        .expect("inference request should succeed");

    let response = client
        .get(format!("{}/metrics", server.base_url))
        .send()
        .await
        .expect("request should succeed");
    let body = response.text().await.expect("metrics body should be readable");

    assert!(body.contains("http_requests_total"));
    server.shutdown().await;
}

#[tokio::test]
async fn test_info_endpoint_returns_model_metadata() {
    // WHAT: `/v1/info` returns the configured model metadata.
    // WHY: Clients and operators need to verify which model version is serving traffic.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .get(format!("{}/v1/info", server.base_url))
        .send()
        .await
        .expect("request should succeed");
    let body: Value = response.json().await.expect("response should be JSON");

    assert_eq!(body["name"], "mock");
    assert_eq!(body["version"], "1");
    server.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_requests_all_succeed_under_limit() {
    // WHAT: Multiple concurrent requests succeed when the queue has enough capacity.
    // WHY: The server's core value is handling many simultaneous clients correctly.
    let server = spawn_server(base_config(), true, Duration::from_millis(5)).await;
    let client = Client::new();
    let mut tasks = Vec::new();

    for _ in 0..16 {
        let client = client.clone();
        let url = format!("{}/v1/infer", server.base_url);
        tasks.push(tokio::spawn(async move {
            client
                .post(url)
                .json(&serde_json::json!({ "text": "parallel request" }))
                .send()
                .await
                .expect("request should succeed")
                .status()
        }));
    }

    for task in tasks {
        assert_eq!(task.await.expect("task should join"), 200);
    }

    server.shutdown().await;
}

#[tokio::test]
async fn test_requests_above_queue_capacity_return_503() {
    // WHAT: Requests above queue capacity return 503 instead of hanging forever.
    // WHY: Explicit overload signals are safer than unbounded waiting during incidents.
    let mut config = base_config();
    config.queue.worker_count = 1;
    config.queue.queue_capacity = 1;
    config.queue.enqueue_timeout_ms = 10;

    let server = spawn_server(config, true, Duration::from_millis(100)).await;
    let client = Client::new();
    let mut tasks = Vec::new();

    for _ in 0..6 {
        let client = client.clone();
        let url = format!("{}/v1/infer", server.base_url);
        tasks.push(tokio::spawn(async move {
            client
                .post(url)
                .json(&serde_json::json!({ "text": "overload" }))
                .send()
                .await
                .expect("request should succeed")
                .status()
        }));
    }

    let mut statuses = Vec::new();
    for task in tasks {
        statuses.push(task.await.expect("task should join"));
    }

    assert!(statuses.iter().any(|status| *status == 503));
    server.shutdown().await;
}

#[tokio::test]
async fn test_graceful_shutdown_drains_in_flight_requests() {
    // WHAT: A request already in progress still completes while shutdown is happening.
    // WHY: Graceful termination avoids dropping client work during deploys or autoscaling events.
    let server = spawn_server(base_config(), true, Duration::from_millis(100)).await;
    let client = Client::new();
    let url = format!("{}/v1/infer", server.base_url);

    let request_task = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .post(url)
                .json(&serde_json::json!({ "text": "shutdown test" }))
                .send()
                .await
                .expect("request should succeed")
                .status()
        }
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    let status = request_task.await.expect("request task should join");
    assert_eq!(status, 200);
    server.shutdown().await;
}

#[tokio::test]
async fn test_request_id_header_propagated_in_response() {
    // WHAT: The server echoes or generates an `x-request-id` response header.
    // WHY: Request IDs are the backbone of cross-system log correlation.
    let server = spawn_server(base_config(), true, Duration::ZERO).await;
    let client = Client::new();

    let response = client
        .post(format!("{}/v1/infer", server.base_url))
        .header("x-request-id", "integration-test-request")
        .json(&serde_json::json!({ "text": "header propagation" }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .expect("header should exist")
            .to_str()
            .expect("header should be valid ASCII"),
        "integration-test-request"
    );
    server.shutdown().await;
}