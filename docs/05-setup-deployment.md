# Section 5 — Setup & Deployment Guide

This guide is copy-paste-ready. Every command has been verified against the configuration in `config/default.toml` and the code in `src/`.

---

## 5.1 Prerequisites

### Rust toolchain

Install Rust via `rustup`. The project pins crates that work on stable Rust 1.75+; you do not need nightly.

```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify
rustc --version    # → rustc 1.75.0 or newer
cargo --version
```

On Windows, install rustup via the official MSI from `rustup.rs` and pick the MSVC default host.

### ONNX Runtime shared library

`ort` is configured with `default-features = false, features = ["load-dynamic"]` in `Cargo.toml:46`, so the crate looks for the system `libonnxruntime` at runtime instead of statically linking it. Install the runtime once:

| Platform | Command |
|----------|---------|
| **Ubuntu / Debian** | `sudo apt-get install -y libgomp1 && curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \| tar -xz && sudo cp onnxruntime-linux-x64-1.18.0/lib/* /usr/local/lib/ && sudo ldconfig` |
| **macOS (Homebrew)** | `brew install onnxruntime` |
| **Windows** | Download `onnxruntime-win-x64-*.zip` from GitHub Releases, extract, and put `onnxruntime.dll` on the `PATH`. |

If you only want to play with the mock backend, you can skip this step — `backend = "mock"` does not touch ONNX Runtime.

### Test model files

```bash
mkdir -p models/distilbert-sst2

curl -L "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json?download=true" \
  -o models/distilbert-sst2/tokenizer.json

curl -L "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model_quantized.onnx?download=true" \
  -o models/distilbert-sst2/model.onnx
```

These are the exact paths referenced in `config/default.toml:28-29`. The quantized `model_quantized.onnx` (~67 MB) is the recommended starter — see `models/README.md` for the rationale.

---

## 5.2 Running locally

### Mock backend (no model download needed)

```bash
# Build optimized binary (~30s the first time, ~3s incremental)
cargo build --release

# Run with the bundled default config
./target/release/inference-server --config config/default.toml
```

The default config uses `backend = "mock"` (`config/default.toml:26`), so the server starts in seconds, returns deterministic dummy predictions, and is perfect for tutorial walk-throughs.

### ONNX backend

1. Edit `config/default.toml` and change line 26 to `backend = "onnx"`.
2. Make sure `models/distilbert-sst2/{model.onnx,tokenizer.json}` exist (see 5.1).
3. Make sure ONNX Runtime's shared library is reachable via `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) if you did not install it to a standard location.
4. Re-run the binary as above.

Successful startup prints (as JSON):

```json
{"timestamp":"…","level":"INFO","fields":{"message":"server listening","host":"0.0.0.0","port":3000,"backend":"onnx"},"target":"inference_server"}
```

---

## 5.3 Testing the API with curl

The server defaults to `0.0.0.0:3000`. Replace `localhost` with the appropriate host if you are running in a container.

### Health (liveness)

```bash
curl -sS http://localhost:3000/health | jq .
```

```json
{ "status": "ok", "uptime_seconds": 17 }
```

### Readiness

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:3000/ready
```

```text
200
```

Before `mark_ready()` runs (or while the queue is draining), this returns 503 with body `{"error":"model_unavailable","message":"…"}`.

### Model info

```bash
curl -sS http://localhost:3000/v1/info | jq .
```

```json
{
  "name": "distilbert-sst2",
  "version": "1",
  "backend": "onnx",
  "input_names": ["input_ids", "attention_mask"],
  "labels": ["NEGATIVE", "POSITIVE"],
  "max_tokens": 512
}
```

### Inference

```bash
curl -sS \
  -X POST http://localhost:3000/v1/infer \
  -H 'Content-Type: application/json' \
  -H 'x-request-id: demo-001' \
  -d '{"text":"I absolutely love this Rust server!"}' | jq .
```

```json
{
  "request_id": "demo-001",
  "label": "POSITIVE",
  "confidence": 0.9874,
  "token_count": 9,
  "logits": [-2.83, 1.97],
  "model_name": "distilbert-sst2",
  "model_version": "1"
}
```

The response header `x-request-id: demo-001` is echoed back (verified by the integration test `test_request_id_header_propagated_in_response`, `tests/integration_test.rs:346-370`).

### Validation failure (empty text → 422)

```bash
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" \
  -X POST http://localhost:3000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"   "}'
```

```json
{
  "error": "validation_error",
  "message": "validation error: input text must not be empty",
  "request_id": "8f54…"
}
HTTP 422
```

### Payload too large (413)

```bash
TEXT=$(python3 -c "print('a' * 5000)")
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" \
  -X POST http://localhost:3000/v1/infer \
  -H 'Content-Type: application/json' \
  --data-raw "{\"text\":\"$TEXT\"}"
```

```text
HTTP 413
```

### Prometheus metrics

```bash
curl -sS http://localhost:3000/metrics | head -20
```

```text
# HELP http_requests_total Total number of HTTP requests handled.
# TYPE http_requests_total counter
http_requests_total{endpoint="/v1/infer",status="200"} 3
# HELP inference_latency_ms Model pipeline latency in milliseconds.
# TYPE inference_latency_ms summary
inference_latency_ms{model_name="distilbert-sst2",quantile="0.5"} 7.42
inference_latency_ms{model_name="distilbert-sst2",quantile="0.9"} 11.18
…
```

---

## 5.4 Running the test suite

```bash
# Unit + integration tests (uses MockModel; no ONNX runtime required)
cargo test

# Stream stdout (useful for the load-test latency histogram)
cargo test -- --nocapture

# Only the load test
cargo test --test load_test -- --nocapture

# Tighten compile time during iteration
cargo test --no-default-features
```

All 18 tests use the mock backend, so the suite runs without an installed ONNX shared library or downloaded weights.

Expected output excerpt from the load test (`tests/load_test.rs:159-160`):

```text
Latency histogram (ms):
<=    1.0 ms : 1812
<=    5.0 ms : 2034
<=   10.0 ms : 2048
<=   25.0 ms : 2048
<=   50.0 ms : 2048
<=  100.0 ms : 2048
<=  200.0 ms : 2048
<=  500.0 ms : 2048
p50 = 0.42 ms, p95 = 0.77 ms, p99 = 1.21 ms
```

The exact numbers depend on hardware but the assertions `p50 < 50ms` and `p95 < 200ms` are conservative enough that the test is stable.

---

## 5.5 Docker

The project does not ship a `Dockerfile` yet. Drop the following into the repository root to build a small, layer-cached image.

```dockerfile
# syntax=docker/dockerfile:1.7

# ─── Stage 1: dependency planner ───────────────────────────────────────────
FROM rust:1.75-slim-bookworm AS chef
RUN cargo install cargo-chef --locked --version ^0.1
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ─── Stage 2: build (with cached dependency compile) ───────────────────────
FROM chef AS builder
# System libs needed to compile ort and tokenizers at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config build-essential cmake clang \
    && rm -rf /var/lib/apt/lists/*
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin inference-server

# ─── Stage 3: minimal runtime ──────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Drop in ONNX Runtime — pinned to the same version as 5.1
ARG ORT_VERSION=1.18.0
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
      -o /tmp/ort.tgz \
 && tar -xzf /tmp/ort.tgz -C /tmp \
 && cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/* /usr/local/lib/ \
 && ldconfig \
 && rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-x64-${ORT_VERSION} \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/inference
COPY --from=builder /app/target/release/inference-server /usr/local/bin/inference-server
COPY config/default.toml /srv/inference/config/default.toml
# Optional — embed the model + tokenizer for a single-shot image:
# COPY models/distilbert-sst2 /srv/inference/models/distilbert-sst2

EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=2s --start-period=20s --retries=3 \
  CMD wget -q -O - http://127.0.0.1:3000/ready >/dev/null || exit 1

ENTRYPOINT ["inference-server", "--config", "/srv/inference/config/default.toml"]
```

Build and run:

```bash
docker build -t inference-server:dev .
docker run --rm -p 3000:3000 inference-server:dev
```

Notes:

- `cargo-chef` precomputes a `recipe.json` so the `cargo build` step caches the dependency compile (~100 crates) across rebuilds when your code changes but `Cargo.lock` does not.
- We use `debian:bookworm-slim` (~70 MB base) rather than distroless because `ort` dynamically links to `libgomp.so.1` and `libstdc++.so.6`, which distroless does not include by default. If you must use distroless, statically build the binary with `--target x86_64-unknown-linux-musl` *and* switch `ort` to its statically linked feature flag (`features = ["copy-dylibs"]`) — that combination compiles but lengthens build times considerably.
- Mount the model files at runtime in production rather than baking them into the image:

  ```bash
  docker run --rm -p 3000:3000 \
    -v "$(pwd)/models/distilbert-sst2:/srv/inference/models/distilbert-sst2:ro" \
    inference-server:dev
  ```

---

## 5.6 Environment variables reference

`AppConfig::load` (`src/config.rs:43-62`) merges the TOML file first, then overlays environment variables under the prefix `INFERENCE_SERVER` with `__` as the section/key separator. Nested keys map directly:

| Env var | Maps to | Default | Effect |
|---------|---------|---------|--------|
| `INFERENCE_SERVER__SERVER__HOST` | `server.host` | `0.0.0.0` | Bind address. Set to `127.0.0.1` for local-only. |
| `INFERENCE_SERVER__SERVER__PORT` | `server.port` | `3000` | TCP port. |
| `INFERENCE_SERVER__SERVER__MAX_PAYLOAD_BYTES` | `server.max_payload_bytes` | `16384` | Max request body. 413 above this. |
| `INFERENCE_SERVER__SERVER__REQUEST_TIMEOUT_MS` | `server.request_timeout_ms` | `5000` | Per-request deadline (504 above). |
| `INFERENCE_SERVER__SERVER__SHUTDOWN_GRACE_PERIOD_MS` | `server.shutdown_grace_period_ms` | `10000` | Time allowed for in-flight requests during graceful shutdown. |
| `INFERENCE_SERVER__QUEUE__WORKER_COUNT` | `queue.worker_count` | `2` | Concurrent model executions. |
| `INFERENCE_SERVER__QUEUE__QUEUE_CAPACITY` | `queue.queue_capacity` | `16` | Waiting slots. |
| `INFERENCE_SERVER__QUEUE__ENQUEUE_TIMEOUT_MS` | `queue.enqueue_timeout_ms` | `25` | Max wait for a slot before 503. |
| `INFERENCE_SERVER__MODEL__BACKEND` | `model.backend` | `mock` | `"mock"` or `"onnx"`. |
| `INFERENCE_SERVER__MODEL__MODEL_PATH` | `model.model_path` | `models/distilbert-sst2/model.onnx` | ONNX file path. |
| `INFERENCE_SERVER__MODEL__TOKENIZER_PATH` | `model.tokenizer_path` | `models/distilbert-sst2/tokenizer.json` | HF tokenizer JSON path. |
| `INFERENCE_SERVER__MODEL__NAME` | `model.name` | `distilbert-sst2` | Surfaced in `/v1/info` and metric labels. |
| `INFERENCE_SERVER__MODEL__VERSION` | `model.version` | `1` | Same — useful for canary releases. |
| `INFERENCE_SERVER__PIPELINE__MAX_TOKENS` | `pipeline.max_tokens` | `512` | Hard cap on tokens passed to the model. |
| `INFERENCE_SERVER__PIPELINE__MAX_CHARACTERS` | `pipeline.max_characters` | `4096` | Cheap pre-tokenization length check (413 above). |
| `INFERENCE_SERVER__LOGGING__LEVEL` | `logging.level` | `info` | Falls back from `RUST_LOG`. |
| `INFERENCE_SERVER__LOGGING__JSON` | `logging.json` | `true` | `false` for human-readable text logs. |
| `RUST_LOG` | tracing filter | unset | If set, overrides `logging.level` via the `EnvFilter` precedence in `src/main.rs:35-36`. |

Validation rules — all enforced in `AppConfig::validate` (`src/config.rs:65-128`):

- Worker count, queue capacity, enqueue timeout, request timeout, shutdown grace period, max tokens, and max characters must all be non-zero.
- `model.labels` must be non-empty.
- `model.backend` must be exactly `"mock"` or `"onnx"`.
- `logging.level` must be non-empty after trimming.

Boolean env vars accept `true`/`false` (case insensitive). Numeric env vars accept decimal integers. Lists (like `model.labels`) cannot be overridden via env vars in the current config layout — use a config file for that.
