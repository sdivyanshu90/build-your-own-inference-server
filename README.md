# build-your-own-inference-server

A production-grade ML inference server in Rust. ONNX Runtime backend, Axum HTTP layer, semaphore-based backpressure, graceful shutdown, Prometheus metrics, and a deterministic mock backend so the tests run with no external dependencies.

## Quick start

```bash
cargo build --release
./target/release/inference-server --config config/default.toml   # uses the mock backend by default

curl -sS -X POST http://localhost:3000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"I love Rust"}' | jq .
```

## Documentation

Full reference docs live in [`docs/`](docs/README.md):

- [Section 1 — Conceptual Foundations](docs/01-conceptual-foundations.md)
- [Section 2 — Project Architecture](docs/02-architecture.md)
- [Section 4 — Line-by-Line Walkthrough](docs/04-walkthrough.md)
- [Section 5 — Setup & Deployment](docs/05-setup-deployment.md)
- [Section 6 — Production Considerations](docs/06-production.md)
- [Section 8 — Extensions](docs/08-extensions.md)

Sections 3 (source) and 7 (tests) live in [`src/`](src/) and [`tests/`](tests/). Every file is heavily commented for beginners.