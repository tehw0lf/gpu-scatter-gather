use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Generate C bindings
    generate_c_bindings();

    // Compile CUDA kernels
    compile_cuda_kernels();
}

fn generate_c_bindings() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let config =
        cbindgen::Config::from_file("cbindgen.toml").expect("Unable to find cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/wordlist_generator.h");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}

fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=kernels/wordlist_poc.cu");

    // Check if CUDA is available
    let nvcc_path = which::which("nvcc").ok();

    if nvcc_path.is_none() {
        println!("cargo:warning=CUDA toolkit not found. GPU features will be disabled.");
        println!("cargo:warning=Install CUDA toolkit to enable GPU acceleration.");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile CUDA kernel to PTX for multiple compute capabilities
    let compute_capabilities = vec![
        "70", // Volta (V100)
        "75", // Turing (RTX 20xx)
        "80", // Ampere (A100)
        "86", // Ampere (RTX 30xx)
        "89", // Ada Lovelace (RTX 40xx)
        "90", // Hopper (H100)
    ];

    for cc in &compute_capabilities {
        let ptx_file = out_dir.join(format!("wordlist_poc_sm_{cc}.ptx"));

        let status = Command::new("nvcc")
            .arg("kernels/wordlist_poc.cu")
            .arg("-ptx")
            .arg(format!("-arch=sm_{cc}"))
            .arg("-o")
            .arg(&ptx_file)
            .arg("--use_fast_math")
            .arg("-O3")
            .status();

        match status {
            Ok(status) if status.success() => {
                println!("cargo:warning=Compiled CUDA kernel for sm_{cc}");
            }
            Ok(status) => {
                println!(
                    "cargo:warning=Failed to compile CUDA kernel for sm_{}: exit code {:?}",
                    cc,
                    status.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=Failed to run nvcc for sm_{cc}: {e}");
            }
        }
    }

    // Also try to compile a CUBIN for the default architecture (faster loading)
    let cubin_file = out_dir.join("wordlist_poc.cubin");
    let _ = Command::new("nvcc")
        .arg("kernels/wordlist_poc.cu")
        .arg("-cubin")
        .arg("-arch=sm_89") // Default to RTX 4070 architecture
        .arg("-o")
        .arg(&cubin_file)
        .arg("--use_fast_math")
        .arg("-O3")
        .status();

    println!("cargo:rustc-env=CUDA_KERNELS_DIR={}", out_dir.display());
}
