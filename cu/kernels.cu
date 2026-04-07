// File: kernels.cu - CUDA kernels for AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     GPU kernel implementations for common ML operations.

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 16

// ============================================
// Activation Functions
// ============================================

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

__global__ void silu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        data[idx] = x * sig;
    }
}

// ============================================
// Matrix Multiplication (Tiled)
// ============================================

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {
        if (row < M && (phase * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + phase * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && (phase * TILE_SIZE + ty) < K) {
            Bs[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================
// Softmax
// ============================================

__global__ void softmax_kernel(float* data, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    
    int offset = idx * blockDim.x;
    
    if (offset + tid >= size) return;
    
    float val = data[offset + tid];
    sdata[tid] = val;
    __syncthreads();
    
    // Find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && offset + tid + s < size) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    __syncthreads();
    
    // Compute exp sum
    sdata[tid] = expf(data[offset + tid] - max_val);
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum = sdata[0];
    __syncthreads();
    
    data[offset + tid] = sdata[tid] / sum;
}

// ============================================
// Layer Normalization
// ============================================

__global__ void layer_norm_kernel(
    float* data,
    float* mean,
    float* var,
    int size,
    float eps
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    int offset = idx * size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum += data[offset + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float m = sdata[0] / size;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float diff = data[offset + i] - m;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float v = sdata[0] / size + eps;
    float inv_std = 1.0f / sqrtf(v);
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        data[offset + i] = (data[offset + i] - m) * inv_std;
    }
    
    if (tid == 0) {
        mean[idx] = m;
        var[idx] = v;
    }
}

// ============================================
// RMS Normalization
// ============================================

__global__ void rms_norm_kernel(
    float* data,
    int size,
    float eps
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    int offset = idx * size;
    
    // Compute sum of squares
    float ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = data[offset + i];
        ss += val * val;
    }
    sdata[tid] = ss;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float rms = sqrtf(sdata[0] / size + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        data[offset + i] *= inv_rms;
    }
}

// ============================================
// Attention (Simplified - QKV matmul + softmax)
// ============================================

__global__ void attention_score_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* scores,
    int seq_len,
    int head_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            sum += Q[row * head_dim + i] * K[col * head_dim + i];
        }
        scores[row * seq_len + col] = sum / sqrtf((float)head_dim);
    }
}

__global__ void attention_apply_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* output,
    int seq_len,
    int head_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < seq_len && col < head_dim) {
        float sum = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            sum += scores[row * seq_len + k] * V[k * head_dim + col];
        }
        output[row * head_dim + col] = sum;
    }
}

// ============================================
// FP32 to FP16 Conversion
// ============================================

__global__ void float_to_half_kernel(const float* input, short* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void half_to_float_kernel(const short* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// ============================================
// Launch Helpers
// ============================================

extern "C" {
    
void cuda_relu(float* data, int size, cudaStream_t stream) {
    int blocks = (size + 255) / 256;
    relu_kernel<<<blocks, 256, 0, stream>>>(data, size);
}
    
void cuda_gelu(float* data, int size, cudaStream_t stream) {
    int blocks = (size + 255) / 256;
    gelu_kernel<<<blocks, 256, 0, stream>>>(data, size);
}
    
void cuda_silu(float* data, int size, cudaStream_t stream) {
    int blocks = (size + 255) / 256;
    silu_kernel<<<blocks, 256, 0, stream>>>(data, size);
}
    
void cuda_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
    
void cuda_softmax(float* data, int size, cudaStream_t stream) {
    int blocks = 1;
    int threads = 256;
    softmax_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(data, size);
}
    
void cuda_layer_norm(
    float* data, float* mean, float* var,
    int batch_size, int size, float eps,
    cudaStream_t stream
) {
    int threads = 256;
    layer_norm_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        data, mean, var, size, eps
    );
}
    
void cuda_rms_norm(float* data, int batch_size, int size, float eps, cudaStream_t stream) {
    int threads = 256;
    rms_norm_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        data, size, eps
    );
}
    
void cuda_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim,
    cudaStream_t stream
) {
    int size = seq_len * seq_len;
    float* scores;
    cudaMalloc(&scores, size * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((seq_len + 15) / 16, (seq_len + 15) / 16);
    attention_score_kernel<<<grid, block, 0, stream>>>(Q, K, scores, seq_len, head_dim);
    
    // Apply softmax (simplified)
    int blocks = (size + 255) / 256;
    softmax_kernel<<<blocks, 256, 0, stream>>>(scores, size);
    
    dim3 block2(16, 16);
    dim3 grid2((seq_len + 15) / 16, (head_dim + 15) / 16);
    attention_apply_kernel<<<grid2, block2, 0, stream>>>(scores, V, output, seq_len, head_dim);
    
    cudaFree(scores);
}
    
void cuda_float_to_half(const float* input, short* output, int size, cudaStream_t stream) {
    int blocks = (size + 255) / 256;
    float_to_half_kernel<<<blocks, 256, 0, stream>>>(input, output, size);
}
    
void cuda_half_to_float(const short* input, float* output, int size, cudaStream_t stream) {
    int blocks = (size + 255) / 256;
    half_to_float_kernel<<<blocks, 256, 0, stream>>>(input, output, size);
}

} // extern "C"
