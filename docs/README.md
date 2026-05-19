# Documentation

Reference documentation for the inference server. The numbering matches the original master prompt; Sections 3 and 7 (full source code and tests) are not duplicated here — they live in `src/` and `tests/` respectively, with every-line comments already in place.

| Section | Topic | File |
|---------|-------|------|
| **1** | Conceptual foundations — what inference servers are, why Rust, beginner Rust primer | [`01-conceptual-foundations.md`](01-conceptual-foundations.md) |
| **2** | Project architecture — directory map, request data flow, full API contract | [`02-architecture.md`](02-architecture.md) |
| **3** | Complete source code | `src/` (see `src/lib.rs` for the module map) |
| **4** | Line-by-line walkthrough of the four most subtle code paths | [`04-walkthrough.md`](04-walkthrough.md) |
| **5** | Setup, curl examples for every endpoint, Docker, env-var reference | [`05-setup-deployment.md`](05-setup-deployment.md) |
| **6** | Production considerations — performance, observability, reliability, security | [`06-production.md`](06-production.md) |
| **7** | Test suite | `tests/integration_test.rs`, `tests/load_test.rs`, plus `#[cfg(test)]` blocks in `src/` |
| **8** | Extensions and exercises — gRPC, batching, hot reload, GPU, streaming, versioning, WASM | [`08-extensions.md`](08-extensions.md) |

## Reading order

For new contributors:

1. Skim `Cargo.toml` to see the dependency surface.
2. Read [Section 1](01-conceptual-foundations.md) end-to-end.
3. Read [Section 2](02-architecture.md), keeping `src/` open in another window.
4. Run the server locally using [Section 5](05-setup-deployment.md).
5. Dive into [Section 4](04-walkthrough.md) once the source is no longer foreign.
6. Use [Section 6](06-production.md) and [Section 8](08-extensions.md) as reference material.
