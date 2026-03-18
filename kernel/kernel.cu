#include <stdio.h>

#define INPUT_DIM 100
#define FILTER_DIM 5
#define CONV_OUT_DIM 20
#define CONV_LAYER_SIZE 10
#define OUT_NEURON_DIM 4000
#define OUT_LAYER_SIZE 10

// Fused convolution + ReLU kernel.
// Each thread computes one element of the 10x20x20 output.
// Total threads 10 * 20 * 20 = 4000
extern "C" __global__ void conv_relu(
    double input[INPUT_DIM][INPUT_DIM],
    double filters[CONV_LAYER_SIZE][FILTER_DIM][FILTER_DIM],
    double output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= CONV_LAYER_SIZE * CONV_OUT_DIM * CONV_OUT_DIM) return;

    int f = tid / (CONV_OUT_DIM * CONV_OUT_DIM);
    int rem = tid % (CONV_OUT_DIM * CONV_OUT_DIM);
    int out_r = rem / CONV_OUT_DIM;
    int out_c = rem % CONV_OUT_DIM;

    int in_r = out_r * FILTER_DIM;
    int in_c = out_c * FILTER_DIM;

    double sum = 0.0;
    for (int x = 0; x < FILTER_DIM; x++) {
        for (int y = 0; y < FILTER_DIM; y++) {
            sum += input[in_r + x][in_c + y] * filters[f][x][y];
        }
    }

    // ReLU clamp negative values to zero
    output[f][out_r][out_c] = (sum > 0.0) ? sum : 0.0;
}

// Output layer kernel with shared-memory divide-and-conquer reduction.
// Each block computes the dot product for one output neuron.
// Launch with grid=10, block=256.
extern "C" __global__ void output_layer(
    double* conv_output,
    double weights[OUT_LAYER_SIZE][OUT_NEURON_DIM],
    double* result
) {
    __shared__ double partial[256];

    int neuron = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1 each thread accumulates partial sum over strided elements
    double sum = 0.0;
    for (int i = tid; i < OUT_NEURON_DIM; i += blockDim.x) {
        sum += conv_output[i] * weights[neuron][i];
    }
    partial[tid] = sum;
    __syncthreads();

    // Phase 2 tree reduction in shared memory (divide-and-conquer)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[neuron] = partial[0];
    }
}
