// File: build.rs - Build script for CUDA kernels
// Copyright (c) 2026 AURIA Developers and Contributors

use std::env;
use std::path::Path;

fn main() {
    let cuda_sdk = env::var("CUDA_PATH").ok();

    if cuda_sdk.is_some() {
        println!(
            "cargo:rustc-link-search=native={}/lib/x64",
            cuda_sdk.unwrap()
        );
        println!("cargo:rustc-link-lib=cudart");
    }

    println!("cargo:rerun-if-changed=cu/kernels.cu");
}
