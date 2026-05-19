// --- Standard library imports ---
use std::{sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc}, time::Duration};

// --- Tokio imports ---
use tokio::{
    sync::{mpsc, oneshot, Mutex, OwnedSemaphorePermit, Semaphore},
    task::JoinSet,
};

// --- Local imports ---
use crate::{
    config::QueueConfig,
    error::{AppError, AppResult},
    metrics,
    pipeline::{InferencePipeline, PipelineOutput},
};

// --- Inference job ---
pub struct InferenceJob {
    pub text: String,
    pub response_tx: oneshot::Sender<AppResult<PipelineOutput>>,
    pub slot_permit: OwnedSemaphorePermit,
}

// --- Queue type ---
pub struct InferenceQueue {
    sender: Mutex<Option<mpsc::Sender<InferenceJob>>>,
    slots: Arc<Semaphore>,
    depth: Arc<AtomicUsize>,
    accepting: Arc<AtomicBool>,
}

// --- Runtime handle ---
pub struct QueueRuntime {
    queue: Arc<InferenceQueue>,
    dispatcher: tokio::task::JoinHandle<()>,
}

impl QueueRuntime {
    // Drain the queue by closing the sender and waiting for the dispatcher to finish all queued work.
    pub async fn shutdown(self) -> AppResult<()> {
        self.queue.close().await;
        self.dispatcher
            .await
            .map_err(|error| AppError::internal(format!("queue dispatcher join failed: {error}")))?;
        Ok(())
    }
}

impl InferenceQueue {
    // Spawn the queue dispatcher and return both the submission handle and the runtime shutdown handle.
    pub fn spawn(
        config: QueueConfig,
        pipeline: Arc<InferencePipeline>,
        initially_ready: bool,
    ) -> (Arc<Self>, QueueRuntime) {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let total_slots = config.worker_count + config.queue_capacity;
        let queue = Arc::new(Self {
            sender: Mutex::new(Some(tx)),
            slots: Arc::new(Semaphore::new(total_slots)),
            depth: Arc::new(AtomicUsize::new(0)),
            accepting: Arc::new(AtomicBool::new(initially_ready)),
        });

        let queue_for_runtime = Arc::clone(&queue);
        let depth_for_dispatcher = Arc::clone(&queue.depth);
        let dispatcher = tokio::spawn(async move {
            Self::dispatcher_loop(rx, pipeline, config.worker_count, depth_for_dispatcher).await;
        });

        (
            queue,
            QueueRuntime {
                queue: queue_for_runtime,
                dispatcher,
            },
        )
    }

    // Expose whether the queue is currently accepting new work.
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    // Mark the queue as ready to receive inference traffic.
    pub fn mark_ready(&self) {
        self.accepting.store(true, Ordering::Release);
    }

    // Submit one inference job and wait for the worker response.
    pub async fn submit(&self, text: String, enqueue_timeout: Option<Duration>) -> AppResult<PipelineOutput> {
        if !self.is_accepting() {
            return Err(AppError::QueueClosed { request_id: None });
        }

        let slot_future = Arc::clone(&self.slots).acquire_owned();
        let slot_permit = match enqueue_timeout {
            Some(timeout) => tokio::time::timeout(timeout, slot_future)
                .await
                .map_err(|_| AppError::QueueFull { request_id: None })?
                .map_err(|_| AppError::QueueClosed { request_id: None })?,
            None => slot_future
                .await
                .map_err(|_| AppError::QueueClosed { request_id: None })?,
        };

        let (response_tx, response_rx) = oneshot::channel();
        let sender = self
            .sender
            .lock()
            .await
            .as_ref()
            .cloned()
            .ok_or(AppError::QueueClosed { request_id: None })?;

        self.depth.fetch_add(1, Ordering::AcqRel);
        metrics::set_queue_depth(self.depth.load(Ordering::Acquire));

        let send_result = sender
            .send(InferenceJob {
                text,
                response_tx,
                slot_permit,
            })
            .await;

        if send_result.is_err() {
            self.depth.fetch_sub(1, Ordering::AcqRel);
            metrics::set_queue_depth(self.depth.load(Ordering::Acquire));
            return Err(AppError::QueueClosed { request_id: None });
        }

        response_rx
            .await
            .map_err(|_| AppError::internal("worker dropped response channel before replying"))?
    }

    // Stop accepting new jobs and close the sender so the dispatcher drains remaining work and exits.
    pub async fn close(&self) {
        self.accepting.store(false, Ordering::Release);
        self.sender.lock().await.take();
    }

    // Run the dispatcher loop that fans queued jobs out to worker tasks.
    async fn dispatcher_loop(
        mut rx: mpsc::Receiver<InferenceJob>,
        pipeline: Arc<InferencePipeline>,
        worker_count: usize,
        depth: Arc<AtomicUsize>,
    ) {
        let worker_limiter = Arc::new(Semaphore::new(worker_count));
        let mut in_flight = JoinSet::new();

        while let Some(job) = rx.recv().await {
            let worker_permit = Arc::clone(&worker_limiter)
                .acquire_owned()
                .await
                .expect("worker semaphore should stay open while dispatcher is alive");
            let pipeline = Arc::clone(&pipeline);
            let depth = Arc::clone(&depth);

            in_flight.spawn(async move {
                let _worker_permit = worker_permit;
                let start = std::time::Instant::now();
                let result = pipeline.infer(&job.text).await;

                if let Ok(output) = &result {
                    metrics::record_inference(&output.model_name, start.elapsed().as_secs_f64() * 1_000.0);
                }

                let _ = job.response_tx.send(result);
                drop(job.slot_permit);
                depth.fetch_sub(1, Ordering::AcqRel);
                metrics::set_queue_depth(depth.load(Ordering::Acquire));
            });
        }

        while in_flight.join_next().await.is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use tokio::time::{advance, pause};

    use crate::{
        config::{PipelineConfig, QueueConfig},
        model::{mock::MockModel, InferenceModel, ModelMetadata},
        pipeline::tokenizer::TokenizerWrapper,
    };

    use super::InferenceQueue;

    fn build_test_pipeline(delay: Duration) -> Arc<crate::pipeline::InferencePipeline> {
        let model: Arc<dyn InferenceModel> = Arc::new(MockModel::new(
            ModelMetadata {
                name: "mock".to_string(),
                version: "1".to_string(),
                backend: "mock".to_string(),
                input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
                labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
                max_tokens: 8,
            },
            delay,
        ));

        Arc::new(crate::pipeline::InferencePipeline::new(
            model,
            TokenizerWrapper::mock(),
            PipelineConfig {
                max_tokens: 8,
                max_characters: 128,
            },
        ))
    }

    #[tokio::test]
    async fn test_queue_processes_job_in_order() {
        // WHAT: A single-worker queue processes jobs in FIFO order.
        // WHY: Predictable queue ordering makes latency behavior easier to reason about under load.
        pause();
        let (queue, runtime) = InferenceQueue::spawn(
            QueueConfig {
                worker_count: 1,
                queue_capacity: 4,
                enqueue_timeout_ms: 10,
            },
            build_test_pipeline(Duration::from_millis(10)),
            true,
        );

        let first = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("alpha".to_string(), None).await.expect("first job should finish").label }
        });
        let second = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("bravo".to_string(), None).await.expect("second job should finish").label }
        });
        let third = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("charlie".to_string(), None).await.expect("third job should finish").label }
        });

        advance(Duration::from_millis(40)).await;

        let results = vec![
            first.await.expect("first task should join"),
            second.await.expect("second task should join"),
            third.await.expect("third task should join"),
        ];

        assert_eq!(results.len(), 3);
        runtime.shutdown().await.expect("queue should shut down cleanly");
    }

    #[tokio::test]
    async fn test_queue_backpressure_blocks_sender() {
        // WHAT: Submitting beyond available queue slots waits for capacity instead of panicking.
        // WHY: Backpressure is how we protect memory and latency under bursts of traffic.
        pause();
        let (queue, runtime) = InferenceQueue::spawn(
            QueueConfig {
                worker_count: 1,
                queue_capacity: 1,
                enqueue_timeout_ms: 10,
            },
            build_test_pipeline(Duration::from_secs(1)),
            true,
        );

        let first = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("first".to_string(), None).await }
        });
        let second = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("second".to_string(), None).await }
        });

        tokio::task::yield_now().await;

        let blocked = tokio::time::timeout(
            Duration::from_millis(1),
            queue.submit("third".to_string(), None),
        )
        .await;

        assert!(blocked.is_err());

        advance(Duration::from_secs(3)).await;
        let _ = first.await.expect("first task should join");
        let _ = second.await.expect("second task should join");
        runtime.shutdown().await.expect("queue should shut down cleanly");
    }

    #[tokio::test]
    async fn test_queue_drain_on_shutdown() {
        // WHAT: Closing the queue lets already accepted jobs complete before the dispatcher exits.
        // WHY: Graceful shutdown should not discard in-flight user requests.
        pause();
        let (queue, runtime) = InferenceQueue::spawn(
            QueueConfig {
                worker_count: 1,
                queue_capacity: 1,
                enqueue_timeout_ms: 10,
            },
            build_test_pipeline(Duration::from_millis(20)),
            true,
        );

        let first = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("first".to_string(), None).await }
        });
        let second = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.submit("second".to_string(), None).await }
        });

        tokio::task::yield_now().await;
        queue.close().await;
        advance(Duration::from_millis(50)).await;

        assert!(first.await.expect("first task should join").is_ok());
        assert!(second.await.expect("second task should join").is_ok());
        runtime.shutdown().await.expect("queue should drain successfully");
    }
}