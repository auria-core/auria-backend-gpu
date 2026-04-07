# auria-backend-gpu

GPU execution backend for AURIA Runtime Core.

## Features

Supports multiple GPU backends:
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal (Apple)

## Features

```toml
[dependencies]
auria-backend-gpu = { features = ["cuda"] }
auria-backend-gpu = { features = ["rocm"] }
auria-backend-gpu = { features = ["metal"] }
```

## Usage

```rust
use auria_backend_gpu::{GpuBackend, GpuBackendType, GpuExecutionEngine};

let engine = GpuExecutionEngine::new(backend);
let output = engine.execute(input, experts, state).await?;
```
