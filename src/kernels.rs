// File: kernels.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     GPU kernel implementations for AURIA Runtime Core.
//     Provides optimized kernels for common ML operations.
//

use auria_core::AuriaResult;
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_relu(data: *mut f32, size: i32, stream: *mut std::ffi::c_void);
    fn cuda_gelu(data: *mut f32, size: i32, stream: *mut std::ffi::c_void);
    fn cuda_silu(data: *mut f32, size: i32, stream: *mut std::ffi::c_void);
    fn cuda_matmul(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    );
    fn cuda_softmax(data: *mut f32, size: i32, stream: *mut std::ffi::c_void);
    fn cuda_layer_norm(
        data: *mut f32,
        mean: *mut f32,
        var: *mut f32,
        batch_size: i32,
        size: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    );
    fn cuda_rms_norm(
        data: *mut f32,
        batch_size: i32,
        size: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    );
    fn cuda_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        seq_len: i32,
        head_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn cuda_float_to_half(
        input: *const f32,
        output: *mut i16,
        size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn cuda_half_to_float(
        input: *const i16,
        output: *mut f32,
        size: i32,
        stream: *mut std::ffi::c_void,
    );
}

#[derive(Clone, Debug)]
pub struct KernelConfig {
    pub block_size: u32,
    pub grid_size: u32,
    pub shared_memory_bytes: u32,
    pub num_stages: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 0,
            shared_memory_bytes: 48 * 1024,
            num_stages: 2,
        }
    }
}

impl KernelConfig {
    pub fn for_matrix_size(rows: usize, cols: usize) -> Self {
        let block_size: u32 = 256;
        let grid_size: u32 = ((rows * cols + 255) / 256) as u32;

        Self {
            block_size,
            grid_size,
            shared_memory_bytes: 48 * 1024,
            num_stages: 2,
        }
    }
}

pub struct GpuKernel {
    name: String,
    config: KernelConfig,
}

impl GpuKernel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            config: KernelConfig::default(),
        }
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &KernelConfig {
        &self.config
    }
}

pub struct MatmulKernel {
    kernel: GpuKernel,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
}

impl MatmulKernel {
    pub fn new() -> Self {
        Self {
            kernel: GpuKernel::new("matmul"),
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    pub fn transpose_a(mut self) -> Self {
        self.transpose_a = true;
        self
    }

    pub fn transpose_b(mut self) -> Self {
        self.transpose_b = true;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn launch(&self, m: usize, n: usize, k: usize) -> KernelLaunchConfig {
        let block_size: usize = 256;
        let grid_x: usize = (n + block_size - 1) / block_size;
        let grid_y: usize = (m + block_size - 1) / block_size;

        KernelLaunchConfig {
            grid_x: grid_x as u32,
            grid_y: grid_y as u32,
            grid_z: 1,
            block_x: block_size as u32,
            block_y: 1,
            block_z: 1,
            shared_memory: 0,
        }
    }
}

impl Default for MatmulKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelLaunchConfig {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub shared_memory: u32,
}

impl KernelLaunchConfig {
    pub fn total_threads(&self) -> u32 {
        self.grid_x * self.grid_y * self.grid_z * self.block_x * self.block_y * self.block_z
    }
}

pub struct ActivationKernel {
    activation_type: ActivationType,
}

#[derive(Clone, Debug)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Sigmoid,
    Tanh,
}

impl ActivationKernel {
    pub fn new(activation: ActivationType) -> Self {
        Self {
            activation_type: activation,
        }
    }

    pub fn forward(&self, data: &mut [f32]) {
        match self.activation_type {
            ActivationType::ReLU => {
                for val in data.iter_mut() {
                    *val = val.max(0.0);
                }
            }
            ActivationType::GELU => {
                let sqrt_2_over_pi = 0.7978845608028654;
                for val in data.iter_mut() {
                    let x = *val;
                    let cdf = 0.5 * (1.0 + (sqrt_2_over_pi * x * (1.0 + 0.044715 * x * x)).tanh());
                    *val = x * cdf;
                }
            }
            ActivationType::SiLU => {
                for val in data.iter_mut() {
                    let x = *val;
                    *val = x / (1.0 + (-x).exp());
                }
            }
            ActivationType::Sigmoid => {
                for val in data.iter_mut() {
                    *val = 1.0 / (1.0 + (-*val).exp());
                }
            }
            ActivationType::Tanh => {
                for val in data.iter_mut() {
                    *val = val.tanh();
                }
            }
        }
    }

    pub fn launch_config(&self, element_count: usize) -> KernelLaunchConfig {
        let block_size: usize = 256;
        let grid: usize = (element_count + block_size - 1) / block_size;

        KernelLaunchConfig {
            grid_x: grid as u32,
            grid_y: 1,
            grid_z: 1,
            block_x: block_size as u32,
            block_y: 1,
            block_z: 1,
            shared_memory: 0,
        }
    }
}

pub struct NormalizationKernel {
    norm_type: NormalizationType,
    epsilon: f32,
    num_groups: usize,
}

#[derive(Clone, Debug)]
pub enum NormalizationType {
    LayerNorm,
    RMSNorm,
    GroupNorm,
    InstanceNorm,
}

impl NormalizationKernel {
    pub fn layer_norm(epsilon: f32) -> Self {
        Self {
            norm_type: NormalizationType::LayerNorm,
            epsilon,
            num_groups: 1,
        }
    }

    pub fn rms_norm(epsilon: f32) -> Self {
        Self {
            norm_type: NormalizationType::RMSNorm,
            epsilon,
            num_groups: 1,
        }
    }

    pub fn group_norm(num_groups: usize, epsilon: f32) -> Self {
        Self {
            norm_type: NormalizationType::GroupNorm,
            epsilon,
            num_groups,
        }
    }

    pub fn instance_norm(epsilon: f32) -> Self {
        Self {
            norm_type: NormalizationType::InstanceNorm,
            epsilon,
            num_groups: 1,
        }
    }

    pub fn forward(&self, data: &[f32], num_channels: usize) -> AuriaResult<Vec<f32>> {
        match self.norm_type {
            NormalizationType::LayerNorm => {
                let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                let variance: f32 =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
                let std = (variance + self.epsilon).sqrt();

                Ok(data.iter().map(|x| (x - mean) / std).collect())
            }
            NormalizationType::RMSNorm => {
                let ms = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
                let rms = (ms + self.epsilon).sqrt();

                Ok(data.iter().map(|x| x / rms).collect())
            }
            NormalizationType::GroupNorm => {
                let group_size = num_channels / self.num_groups;
                let mut result = Vec::with_capacity(data.len());

                for g in 0..self.num_groups {
                    let offset = g * group_size;
                    let slice = &data[offset..offset + group_size];

                    let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
                    let variance: f32 =
                        slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32;
                    let std = (variance + self.epsilon).sqrt();

                    for val in slice {
                        result.push((val - mean) / std);
                    }
                }

                Ok(result)
            }
            NormalizationType::InstanceNorm => self.forward(data, data.len()),
        }
    }
}

pub struct AttentionKernel {
    num_heads: usize,
    scale: f32,
    use_causal_mask: bool,
}

impl AttentionKernel {
    pub fn new(num_heads: usize) -> Self {
        Self {
            num_heads,
            scale: 1.0 / (64.0_f32.sqrt()),
            use_causal_mask: false,
        }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.use_causal_mask = true;
        self
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            let mut attention_scores = vec![0.0f32; seq_len];
            let mut max_val = f32::MIN;

            for j in 0..seq_len {
                let q_idx = i * head_dim;
                let k_idx = j * head_dim;

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_idx + d] * k[k_idx + d];
                }
                score *= self.scale;

                if self.use_causal_mask && j > i {
                    score = f32::MIN;
                }

                attention_scores[j] = score;
                max_val = max_val.max(score);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..seq_len {
                attention_scores[j] = (attention_scores[j] - max_val).exp();
                sum_exp += attention_scores[j];
            }

            for j in 0..seq_len {
                attention_scores[j] /= sum_exp;
            }

            for d in 0..head_dim {
                let mut weighted_sum = 0.0f32;
                for j in 0..seq_len {
                    let v_idx = j * head_dim;
                    weighted_sum += attention_scores[j] * v[v_idx + d];
                }
                output[i * head_dim + d] = weighted_sum;
            }
        }

        output
    }

    pub fn launch_config(&self, batch_size: usize, seq_len: usize) -> KernelLaunchConfig {
        let block_size: usize = 256;
        let grid: usize = (batch_size * seq_len + block_size - 1) / block_size;

        KernelLaunchConfig {
            grid_x: grid as u32,
            grid_y: self.num_heads as u32,
            grid_z: 1,
            block_x: block_size as u32,
            block_y: 1,
            block_z: 1,
            shared_memory: (seq_len * 4) as u32,
        }
    }
}

pub struct SoftmaxKernel {
    stable: bool,
}

impl SoftmaxKernel {
    pub fn new() -> Self {
        Self { stable: true }
    }

    pub fn unstable(mut self) -> Self {
        self.stable = false;
        self
    }

    pub fn forward(&self, data: &[f32]) -> Vec<f32> {
        if self.stable {
            let max_val = data.iter().cloned().fold(f32::MIN, f32::max);

            let exps: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();

            let sum: f32 = exps.iter().sum();

            exps.iter().map(|x| x / sum).collect()
        } else {
            let exps: Vec<f32> = data.iter().map(|x| x.exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|x| x / sum).collect()
        }
    }

    pub fn launch_config(&self, element_count: usize) -> KernelLaunchConfig {
        let block_size: usize = 256;
        let grid: usize = (element_count + block_size - 1) / block_size;

        KernelLaunchConfig {
            grid_x: grid as u32,
            grid_y: 1,
            grid_z: 1,
            block_x: block_size as u32,
            block_y: 1,
            block_z: 1,
            shared_memory: (block_size * 4) as u32,
        }
    }
}

impl Default for SoftmaxKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_kernel_config() {
        let kernel = MatmulKernel::new();
        let config = kernel.launch(1024, 1024, 1024);

        assert!(config.total_threads() > 0);
    }

    #[test]
    fn test_relu_activation() {
        let kernel = ActivationKernel::new(ActivationType::ReLU);
        let mut data = vec![-1.0, 0.0, 1.0, 2.0];

        kernel.forward(&mut data);

        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 1.0);
        assert_eq!(data[3], 2.0);
    }

    #[test]
    fn test_softmax() {
        let kernel = SoftmaxKernel::new();
        let data = vec![1.0, 2.0, 3.0];

        let result = kernel.forward(&data);
        let sum: f32 = result.iter().sum();

        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        let kernel = NormalizationKernel::layer_norm(1e-5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = kernel.forward(&data, 6).unwrap();

        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_attention() {
        let kernel = AttentionKernel::new(4).with_causal_mask();

        let q = vec![0.1; 64];
        let k = vec![0.1; 64];
        let v = vec![0.1; 64];

        let output = kernel.forward(&q, &k, &v, 4, 16);

        assert_eq!(output.len(), 64);
    }
}
