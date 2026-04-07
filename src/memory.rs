// File: memory.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     GPU memory management for AURIA Runtime Core.
//     Provides memory pooling, allocation tracking, and device memory operations.
//

use auria_core::{AuriaError, AuriaResult, TensorDType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct GpuMemoryManager {
    devices: Arc<RwLock<HashMap<u32, DeviceMemory>>>,
    default_device: u32,
}

#[derive(Clone)]
pub struct DeviceMemory {
    pub device_id: u32,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub allocations: Arc<RwLock<Vec<MemoryAllocation>>>,
}

#[derive(Clone)]
pub struct MemoryAllocation {
    pub id: u64,
    pub ptr: usize,
    pub size_bytes: usize,
    pub tensor_shape: Vec<u32>,
    pub dtype: TensorDType,
    pub allocated_at: u64,
}

impl GpuMemoryManager {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            default_device: 0,
        }
    }

    pub fn with_device(mut self, device_id: u32, total_memory: u64) -> Self {
        let device = DeviceMemory {
            device_id,
            total_bytes: total_memory,
            used_bytes: 0,
            allocations: Arc::new(RwLock::new(Vec::new())),
        };
        
        let devices = Arc::new(RwLock::new({
            let mut map = HashMap::new();
            map.insert(device_id, device);
            map
        }));
        
        self.devices = devices;
        self
    }

    pub async fn allocate(
        &self,
        device_id: u32,
        shape: &[u32],
        dtype: TensorDType,
    ) -> AuriaResult<MemoryAllocation> {
        let size_bytes = self.calculate_size(shape, dtype)?;
        
        let mut devices = self.devices.write().await;
        let device = devices.get_mut(&device_id)
            .ok_or_else(|| AuriaError::ExecutionError(format!("Device {} not found", device_id)))?;
        
        if device.used_bytes + size_bytes as u64 > device.total_bytes {
            return Err(AuriaError::ExecutionError("Out of GPU memory".to_string()));
        }
        
        device.used_bytes += size_bytes as u64;
        
        let allocation = MemoryAllocation {
            id: rand::random(),
            ptr: rand::random::<usize>() & 0xFFFFFFFF,
            size_bytes,
            tensor_shape: shape.to_vec(),
            dtype,
            allocated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        device.allocations.write().await.push(allocation.clone());
        
        Ok(allocation)
    }

    pub async fn deallocate(&self, device_id: u32, allocation_id: u64) -> AuriaResult<()> {
        let mut devices = self.devices.write().await;
        let device = devices.get_mut(&device_id)
            .ok_or_else(|| AuriaError::ExecutionError(format!("Device {} not found", device_id)))?;
        
        let mut allocations = device.allocations.write().await;
        
        if let Some(pos) = allocations.iter().position(|a| a.id == allocation_id) {
            let alloc = allocations.remove(pos);
            device.used_bytes = device.used_bytes.saturating_sub(alloc.size_bytes as u64);
        }
        
        Ok(())
    }

    pub async fn get_device_info(&self, device_id: u32) -> AuriaResult<DeviceMemoryInfo> {
        let devices = self.devices.read().await;
        let device = devices.get(&device_id)
            .ok_or_else(|| AuriaError::ExecutionError(format!("Device {} not found", device_id)))?;
        
        let allocations = device.allocations.read().await;
        
        Ok(DeviceMemoryInfo {
            device_id,
            total_bytes: device.total_bytes,
            used_bytes: device.used_bytes,
            available_bytes: device.total_bytes - device.used_bytes,
            allocation_count: allocations.len(),
            largest_allocation: allocations.iter()
                .map(|a| a.size_bytes)
                .max()
                .unwrap_or(0),
        })
    }

    pub async fn clear_cache(&self, device_id: u32) -> AuriaResult<u64> {
        let mut devices = self.devices.write().await;
        let device = devices.get_mut(&device_id)
            .ok_or_else(|| AuriaError::ExecutionError(format!("Device {} not found", device_id)))?;
        
        let freed = device.used_bytes;
        device.used_bytes = 0;
        device.allocations.write().await.clear();
        
        Ok(freed)
    }

    fn calculate_size(&self, shape: &[u32], dtype: TensorDType) -> AuriaResult<usize> {
        let element_count: usize = shape.iter().map(|&d| d as usize).product();
        
        let bytes_per_element = match dtype {
            TensorDType::FP16 | TensorDType::FP8 => 2,
            TensorDType::INT8 => 1,
            TensorDType::INT4 => 1,
        };
        
        Ok(element_count * bytes_per_element)
    }
}

impl Default for GpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemoryInfo {
    pub device_id: u32,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub allocation_count: usize,
    pub largest_allocation: usize,
}

impl DeviceMemoryInfo {
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

pub struct MemoryPool {
    pools: Arc<RwLock<HashMap<u32, Vec<MemoryAllocation>>>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn push(&self, device_id: u32, allocation: MemoryAllocation) {
        let mut pools = self.pools.write().await;
        pools.entry(device_id).or_insert_with(Vec::new).push(allocation);
    }

    pub async fn pop(&self, device_id: u32, size_needed: usize) -> Option<MemoryAllocation> {
        let mut pools = self.pools.write().await;
        
        if let Some(pool) = pools.get_mut(&device_id) {
            if let Some(pos) = pool.iter().position(|a| a.size_bytes >= size_needed) {
                return Some(pool.remove(pos));
            }
        }
        
        None
    }

    pub async fn clear(&self) {
        let mut pools = self.pools.write().await;
        for pool in pools.values_mut() {
            pool.clear();
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_allocation() {
        let manager = GpuMemoryManager::new().with_device(0, 1024 * 1024);
        
        let alloc = manager.allocate(0, &[100, 100], TensorDType::FP16).await;
        assert!(alloc.is_ok());
    }

    #[tokio::test]
    async fn test_memory_deallocation() {
        let manager = GpuMemoryManager::new().with_device(0, 1024 * 1024);
        
        let alloc = manager.allocate(0, &[100, 100], TensorDType::FP16).await.unwrap();
        manager.deallocate(0, alloc.id).await.unwrap();
        
        let info = manager.get_device_info(0).await.unwrap();
        assert_eq!(info.used_bytes, 0);
    }

    #[test]
    fn test_memory_pool() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let pool = MemoryPool::new();
            
            let alloc = MemoryAllocation {
                id: 1,
                ptr: 100,
                size_bytes: 1000,
                tensor_shape: vec![10, 10],
                dtype: TensorDType::FP16,
                allocated_at: 1000,
            };
            
            pool.push(0, alloc.clone()).await;
            
            let popped = pool.pop(0, 500).await;
            assert!(popped.is_some());
        });
    }
}
