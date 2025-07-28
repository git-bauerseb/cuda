#include <cstdio>
#include "common.hu"

// Stencil: Each output element is sum of input elements
// Implementing stencil without padding on CPU is hard, since
// we need to differentiate between border/in-border in kernel.
const int N = 64;
const int BLOCK_SIZE = 1024;
const int RADIUS = 3;

// in: pointing to memory that is offsetted by RADIUS
//     therefore, g_idx - RADIUS < 0 is possible
__global__ void stencil(int *in, int *out, int actual_block_size) {
    __shared__ int tmp[BLOCK_SIZE + 2 * RADIUS];

    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx = threadIdx.x + RADIUS;

    tmp[l_idx] = in[g_idx];
    if (threadIdx.x < RADIUS) {
        tmp[l_idx - RADIUS] = in[g_idx - RADIUS];
        tmp[l_idx + actual_block_size] = in[g_idx + actual_block_size];
    }
    __syncthreads();

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
        result += tmp[l_idx + offset];
    }

    out[g_idx] = result;
}

int32_t main() {

    int *h_in, *h_out;
    int *d_in, *d_out;

    int size = (N+2*RADIUS) * sizeof(int);
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);
    fillOnes(h_in, N+2*RADIUS);
    fillOnes(h_out, N+2*RADIUS);


    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);


    int ACTUAL_BLOCK_SIZE = min(N, BLOCK_SIZE);

    stencil<<<(N+ACTUAL_BLOCK_SIZE-1)/ACTUAL_BLOCK_SIZE, ACTUAL_BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS, ACTUAL_BLOCK_SIZE);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N + 2*RADIUS; i++) {
        if (i<RADIUS || i>=N+RADIUS){
        if (h_out[i] != 1)
            printf("index %d, was: %d, should be: %d\n", i, h_out[i], 1);
        } else {
        if (h_out[i] != 1 + 2*RADIUS)
            printf("index %d, was: %d, should be: %d\n", i, h_out[i], 1 + 2*RADIUS);
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}