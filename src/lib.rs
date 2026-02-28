// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     GPU execution backend for AURIA Runtime Core.
//     Implements tensor operations and expert execution on GPU hardware
//     supporting CUDA, ROCm, and Metal backends for Pro tier.
//
use auria_core::{ExecutionOutput, ExecutionState, AuriaResult, Tensor};
use async_trait::async_trait;

#[async_trait]
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> GpuBackendType;
    async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput>;
}

#[derive(Debug, Clone, Copy)]
pub enum GpuBackendType {
    Cuda,
    Rocm,
    Metal,
    None,
}

pub struct GpuExecutionEngine<B: GpuBackend> {
    backend: B,
}

impl<B: GpuBackend> GpuExecutionEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput> {
        self.backend.execute(input, experts, state).await
    }
}
