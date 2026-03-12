# cuda-cnn-accelerator

Why use a proprietary solution like `CUDA` over the FOSS alternative like `OpenCL`? 

I generally try to use `FOSS` where possible. For the scope of this project `CUDA` made sense for a of couple reasons:

1. Hardware

My PC and servers at home have `NVIDIA` GPUs. Sshing into my servers or running it on my PC was not an issue. The real tradeoff is the hardware limitation (yes translation layers like
`SCALE` and `ZLUDA` do exist) 

2. Performance

`CUDA` generally has better raw performance on `NVIDIA` hardware

3. Ease of Use 

`CUDA` is simply easier. Supports `C` / `C++` syntax. `OpenCL` requires more boiler plate and is tedious. `CUDA` has better tooling / libraries. `CUDA` has better documentation, both official and community. 

4. Project Specific Considerations

The existence of `RustaCUDA` and there not being a real alternative for `OpenCL` was a big part since this was Rust project from the beginning. Another small difference is the convolution neural network dimensions are fixed and can be hardcoded. `CUDA` kernels compiled to PTX at build time fit this workflow better, whereas `OpenCL` kernels are typically compiled at runtime. We can use this ahead of time compilation to our advantage with `build.rs` and `cargo`'s cache to only recompile when the kernel actually changes. 

Rust based CNN inference engine that offloads compute-heavy operations to the GPU using `CUDA-C` kernels. Using `RustaCUDA` bindings, supports both CPU and GPU backends for benchmarking and comparison.

For information on `RustaCUDA`: 

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

`NVIDIA` source on parallel reductions (dividing and conquering) in `CUDA`: 

https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

