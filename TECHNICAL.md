# CUDA-Accelerated CNN Inference Engine

A Rust-based convolutional neural network inference engine that offloads compute-heavy matrix operations to the GPU via custom CUDA-C kernels. The host application uses [RustaCUDA](https://docs.rs/rustacuda/) bindings to manage device memory, kernel launches, and synchronization, while supporting both CPU and GPU execution backends for correctness verification and performance comparison.

## Motivation

CNN inference is dominated by large matrix operations — convolutions, element-wise activations, and dense dot products — all of which map naturally to GPU parallelism. This project explores how much of that compute can be pushed to CUDA kernels while keeping the host-side logic in Rust, leveraging Rust's safety guarantees for everything outside the GPU boundary.

## CNN Architecture

The network is a simplified 3-layer CNN that transforms a `100 x 100` input matrix (representing a grayscale image) into a 10-element output vector (representing classification probabilities).

### Layer 1 — Convolution

- 10 neurons, each with a `5 x 5` filter (weights matrix)
- Each neuron slides its filter across the input in `5 x 5` sections, computing a dot product at every position
- Produces a `20 x 20` output matrix per neuron (10 total)

### Layer 2 — ReLU Activation

- Element-wise activation applied to all convolution outputs
- Clamps negative values to zero: `f(x) = max(0, x)`

### Layer 3 — Fully Connected Output

- 10 output neurons, each with a `4000 x 1` weight vector
- All `20 x 20` matrices from the previous layer are flattened and concatenated into a single `4000 x 1` vector
- Each output neuron computes a dot product between this vector and its weights, producing one scalar
- Final output is a 10-element vector

## CUDA Kernel Design

All compute is implemented in GPU kernels (`kernel/kernel.cu`), compiled ahead-of-time to PTX.

### Convolution Kernel

Each thread handles one or more output elements across the 10 filters. The input matrix, filter weights, and output buffer are passed as device pointers. Multidimensional indexing is handled by casting raw pointers to sized array types in CUDA-C (e.g., `double input_data[100][100]`), which avoids manual 1D offset arithmetic.

### ReLU Kernel

Trivially parallelizable — each thread clamps a single element. The kernel operates in-place on the convolution output buffer.

### Output Layer Kernel (Parallel Reduction)

The fully connected layer requires computing 10 dot products over 4000-element vectors. This is the most interesting kernel from a parallelism standpoint.

A naive approach would assign one thread per dot product, but that leaves most of the GPU idle. Instead, a divide-and-conquer parallel reduction is used:

1. **Element-wise multiply**: 4000 threads compute `C[n] = A[n] * B[n]` in parallel
2. **Segmented reduction**: The 4000-element product vector is split into equal segments, each assigned to a thread block. Threads within each block cooperatively sum their segment using shared memory reduction
3. **Final reduction**: The partial sums are reduced to produce the final scalar dot product

This pattern is applied for each of the 10 output neurons, maximizing thread occupancy across the GPU.

### Kernel Launch Configuration

Kernels are launched with a block size of 256 threads. Grid size is calculated as `ceil(num_elements / 256)` to cover the full output domain. Shared memory is not explicitly allocated at launch (`0` bytes) — static shared memory is declared within kernels where needed.

## Host-Side Architecture (Rust)

### Execution Modes

The application supports two backends selected at runtime:

```
cargo run --release -- cpu <cnn_file> <input_file> <output_file>
cargo run --release -- cuda <cnn_file> <input_file> <output_file>
```

- **CPU mode**: Sequential reference implementation for correctness verification
- **CUDA mode**: GPU-accelerated path using RustaCUDA

### Data Pipeline

1. Parse CNN description from CSV (filter weights, output weights)
2. Parse input matrices from CSV (batch of `100 x 100` images)
3. For each input matrix:
   - Allocate device memory and copy input + weights to GPU
   - Launch convolution kernel
   - Launch ReLU kernel (operates on convolution output in-place)
   - Launch output layer kernel
   - Copy 10-element result vector back to host
4. Write all output vectors to CSV
5. Report elapsed computation time (excludes I/O and GPU initialization overhead)

### RustaCUDA Integration

RustaCUDA provides safe(r) Rust wrappers around the CUDA Driver API. Key abstractions used:

- `CudaContext` / `CudaModule` — GPU initialization and PTX module loading
- `DeviceBuffer` / `DeviceBox` — typed device memory allocation
- `Stream` — asynchronous kernel launch and synchronization
- `launch!` macro — type-safe kernel invocation with grid/block configuration

Since calling a CUDA kernel from Rust is fundamentally an unsafe operation (no compiler guarantees across the FFI boundary), care is taken to ensure buffer sizes and pointer types match between the Rust host code and CUDA-C kernel signatures.

## Build System

The CUDA kernel is compiled separately from the Rust source:

1. `kernel/kernel.cu` is compiled to PTX via `nvcc -ptx`
2. A `build.rs` build script automates this, integrating with Cargo's build cache so the kernel is only recompiled when `kernel.cu` changes
3. The resulting `kernel.ptx` is loaded at runtime by RustaCUDA

This ahead-of-time compilation model avoids runtime compilation overhead (unlike OpenCL's typical runtime compilation approach) and allows `cargo build` to produce a fully self-contained binary + PTX artifact.

## Input Format

All data files use CSV format:

- **CNN description** (`cnn.csv`): Serialized filter weights (10 filters x 5 x 5) followed by output layer weights (10 neurons x 4000)
- **Input images** (`in.csv`): One or more `100 x 100` matrices, flattened row-major
- **Output** (`out.csv` / `out_cuda.csv`): 10-element output vectors, one per input image

## Helper Scripts

Two Python helper scripts are included:

- **`generate.py`** — Generates random CNN descriptions and input matrices, writing to `input/cnn.csv` and `input/in.csv` by default
- **`compare.py`** — Compares two output files (`output/out.csv` and `output/out_cuda.csv`) to verify they are numerically close enough, used for correctness testing between CPU and GPU implementations

Run with `python3 generate.py` or `python3 compare.py`.

### Typical Workflow

```
python3 generate.py
cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv
cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv
python3 compare.py
```

## Performance

The timing measurement captures only the core computation (matrix operations across all input images), excluding file I/O and one-time GPU initialization costs. This gives a fair comparison between CPU and GPU execution paths.

The GPU implementation targets full utilization of the GPU's parallelism — no single-threaded bottlenecks on the device side. The parallel reduction strategy for the output layer's dot products is critical here, as a naive single-thread-per-dot-product approach would underutilize the hardware by orders of magnitude.

## Dependencies

- **Rust** (2024 edition)
- **NVIDIA CUDA Toolkit** (`nvcc` compiler)
- **RustaCUDA** — Rust bindings for the CUDA Driver API
- **NVIDIA GPU** with CUDA support
