use std::{sync::Arc, time::{Duration, Instant}};

use inference_server::{
    build_app_runtime,
    config::{AppConfig, LoggingConfig, ModelConfig, PipelineConfig, QueueConfig, ServerConfig},
    model::{mock::MockModel, InferenceModel, ModelMetadata},
    pipeline::tokenizer::TokenizerWrapper,
};
use reqwest::Client;
use tokio::{net::TcpListener, sync::oneshot};

struct LoadServer {
    base_url: String,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl LoadServer {
    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.join_handle.await;
    }
}

async fn spawn_load_server(queue_capacity: usize, worker_count: usize, delay: Duration) -> LoadServer {
    let config = AppConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_bytes: 1024,
            request_timeout_ms: 5_000,
            shutdown_grace_period_ms: 5_000,
        },
        queue: QueueConfig {
            worker_count,
            queue_capacity,
            enqueue_timeout_ms: 20,
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
            max_characters: 256,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            json: true,
        },
    };

    let metadata = ModelMetadata {
        name: config.model.name.clone(),
        version: config.model.version.clone(),
        backend: config.model.backend.clone(),
        input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
        labels: config.model.labels.clone(),
        max_tokens: config.pipeline.max_tokens,
    };
    let model: Arc<dyn InferenceModel> = Arc::new(MockModel::new(metadata, delay));
    let runtime = build_app_runtime(config, model, TokenizerWrapper::mock(), true)
        .await
        .expect("load test app should build");
    runtime.state.mark_ready();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener should bind");
    let address = listener.local_addr().expect("listener should expose local addr");
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

    LoadServer {
        base_url,
        shutdown_tx: Some(shutdown_tx),
        join_handle,
    }
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("latencies should be comparable"));
    let index = ((values.len() as f64 - 1.0) * percentile).round() as usize;
    values[index]
}

fn print_histogram(values: &[f64]) {
    let buckets = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0];
    println!("Latency histogram (ms):");
    for bucket in buckets {
        let count = values.iter().filter(|value| **value <= bucket).count();
        println!("<= {:>6.1} ms : {}", bucket, count);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn load_test_mock_model_latency_and_backpressure() {
    // WHAT: The mock-backed server maintains low latency under healthy concurrency and rejects overload with 503.
    // WHY: Load tests verify that throughput, latency, and overload behavior match production expectations.
    let healthy_worker_count = 32;
    let healthy_client_count = healthy_worker_count;
    let requests_per_client = 64;
    let healthy_server = spawn_load_server(256, healthy_worker_count, Duration::ZERO).await;
    let client = Client::new();
    let mut tasks = Vec::new();

    // Keep healthy concurrency aligned with available workers so this phase measures steady-state latency.
    // The overload phase below is responsible for validating queue saturation and 503 behavior.
    for worker in 0..healthy_client_count {
        let client = client.clone();
        let url = format!("{}/v1/infer", healthy_server.base_url);
        tasks.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            let mut statuses = Vec::new();

            for request_index in 0..requests_per_client {
                let started = Instant::now();
                let response = client
                    .post(&url)
                    .json(&serde_json::json!({ "text": format!("load-{worker}-{request_index}") }))
                    .send()
                    .await
                    .expect("load request should succeed");
                latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
                statuses.push(response.status().as_u16());
            }

            (latencies, statuses)
        }));
    }

    let mut all_latencies = Vec::new();
    let mut all_statuses = Vec::new();
    for task in tasks {
        let (latencies, statuses) = task.await.expect("load worker should join");
        all_latencies.extend(latencies);
        all_statuses.extend(statuses);
    }

    let p50 = percentile(all_latencies.clone(), 0.50);
    let p95 = percentile(all_latencies.clone(), 0.95);
    let p99 = percentile(all_latencies.clone(), 0.99);

    print_histogram(&all_latencies);
    println!("p50 = {:.2} ms, p95 = {:.2} ms, p99 = {:.2} ms", p50, p95, p99);

    assert!(p50 < 50.0);
    assert!(p95 < 200.0);
    assert!(all_statuses.iter().all(|status| *status < 500 || *status == 503));
    assert_eq!(all_statuses.iter().filter(|status| **status >= 500).count(), 0);

    healthy_server.shutdown().await;

    let overload_server = spawn_load_server(1, 1, Duration::from_millis(100)).await;
    let mut overload_tasks = Vec::new();

    for _ in 0..32 {
        let client = client.clone();
        let url = format!("{}/v1/infer", overload_server.base_url);
        overload_tasks.push(tokio::spawn(async move {
            client
                .post(url)
                .json(&serde_json::json!({ "text": "overload" }))
                .send()
                .await
                .expect("overload request should complete")
                .status()
                .as_u16()
        }));
    }

    let mut overload_statuses = Vec::new();
    for task in overload_tasks {
        overload_statuses.push(task.await.expect("overload task should join"));
    }

    assert!(overload_statuses.iter().any(|status| *status == 503));
    overload_server.shutdown().await;
}