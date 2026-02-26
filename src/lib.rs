use auria_core::{ExecutionOutput, ExecutionState, Result, Tensor};
use async_trait::async_trait;

#[async_trait]
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> GpuBackendType;
    async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> Result<ExecutionOutput>;
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

    pub async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> Result<ExecutionOutput> {
        self.backend.execute(input, experts, state).await
    }
}
