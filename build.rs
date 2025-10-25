use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    //部分路径
    let cuda_cpp = "src/reference/cuda/rwkv7_state_fwd_fp16.cpp";
    let cuda_cu = "src/reference/cuda/rwkv7_state_fwd_fp16.cu";
    let libtorch_path = "/home/astur/libtorch";
    let python_include = "/usr/include/python3.10";

    // 生成共享库
    let so_path = out_dir.join("librwkv7_state_fwd_fp16.so");

    // nvcc 编译
    let status = Command::new("nvcc")
        .args([
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "-shared",
            cuda_cpp,
            cuda_cu,
            "-o",
            so_path.to_str().unwrap(),
            "-O3",
            "--use_fast_math",
            "-D_N_=64",
            &format!("-I{}/include", libtorch_path),
            &format!("-I{}/include/torch/csrc/api/include", libtorch_path),
            "-I/usr/local/cuda/include",
            &format!("-I{}", python_include),
        ])
        .status()
        .expect("Failed to compile CUDA code");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=rwkv7_state_fwd_fp16");
}
