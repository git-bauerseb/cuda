#include <stdio.h>
#include "common.hu"

// Prescan
// Perform scan in a singlem thread block 
// Bleloch scan (down/up-sweep)
// Assume n is 2^k
__global__ void prescan(const float* inp_, float *out_, float *bsum, int n) {
    extern __shared__ float tmp[];

    const float* inp = inp_ + n * blockIdx.x;
    float* out = out_ + n * blockIdx.x;

    uint idx = threadIdx.x;
    tmp[2*idx] = inp[2*idx];
    tmp[2*idx+1] = inp[2*idx+1];
    
    
    // UP SWEEP
    // for d = 0 to log2 n – 1 do
    //   for all k = 0 to n – 1 by 2 d+1 in parallel do
    //     x[k + 2 d+1 – 1] = x[k + 2 d – 1] + x[k + 2 d +1 – 1]

    // Number of active threads: n/2, n/4, ..., 1
    // d = 3 -> 4
    // d = 2 -> 2
    // d = 1 -> 1
    // --> Therefore, not d == 0
    uint offset = 1;

    for (uint d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (idx < d) {
            uint f_idx = offset * (2*idx + 1) - 1;
            uint s_idx = offset * (2*idx + 1 + 1)  - 1;
            tmp[s_idx] += tmp[f_idx];
        }

        offset <<= 1;
    }

    __syncthreads();

    // DOWN SWEEP
    // x[n - 1] := 0
    // for d := log2n – 1 down to 0 do
    //   for k from 0 to n – 1 by 2 d + 1 in parallel do
    //     t := x[k + 2d- 1]
    //     x[k + 2d- 1] := x [k + 2d + 1- 1]
    //     x[k + 2d + 1- 1] := t + x [k + 2d + 1- 1]
    if (idx == 0) {
        bsum[blockIdx.x] = tmp[n-1];
        tmp[n-1] = 0;
    }

    for (uint d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            int ai = offset*(2*idx+1)-1;
            int bi = offset*(2*idx+2)-1;
            float t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t;
        }
    }

    __syncthreads();

    out[2*idx] = tmp[2*idx];
    out[2*idx+1] = tmp[2*idx+1];
}

__global__ void offsetAdd(const float* inp_, float* out_, float *bsum, int n) {
    float* out = out_ + n * blockIdx.x;
    const float* inp = inp_ + n * blockIdx.x;
    if (blockIdx.x > 0) {
        out[2*threadIdx.x] += bsum[blockIdx.x-1];
        out[2*threadIdx.x+1] += bsum[blockIdx.x-1];

    }

    __syncthreads();

    // DIfference between include/exclusive scan
    out[2*threadIdx.x] += inp[2*threadIdx.x];
    out[2*threadIdx.x+1] += inp[2*threadIdx.x+1];

}


void scan_pow2(const float* inp, float *out, int N) {

    // Number of elements per block
    const int B = 8;
    const int BLK_CNT = (N+B-1)/B;

    float *d_bsum;
    cudaMalloc(&d_bsum, BLK_CNT*sizeof(float));

    dim3 block(B/2,1,1);
    dim3 grid(BLK_CNT,1,1);

    uint mem = sizeof(float) * B;
    prescan<<<grid,block,mem>>>(inp, out, d_bsum, B);


    // Debug
    
    float *h_bsum = new float[BLK_CNT];
    cudaMemcpy(h_bsum, d_bsum, sizeof(float)*BLK_CNT, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLK_CNT; ++i) {
        printf("%f\n", h_bsum[i]);
    }
    


    offsetAdd<<<grid,block>>>(inp, out, d_bsum, B);
   
    cudaFree(d_bsum);
}

// Simple scan extended to arbitrary array lengths 
// by getting to next power of 2
void scan(const float* inp, float* out, int N) {
    int p = 1;
    while (p < N) {p <<= 1;}

    size_t mem = sizeof(float) * p;
    
    float *d_inpPow, *d_outPow;
    cudaMalloc((void**)&d_inpPow, mem);
    cudaMalloc((void**)&d_outPow, mem);
    
    cudaMemset(d_inpPow, 0, mem);
    cudaMemcpy(d_inpPow, inp, sizeof(float)*N, cudaMemcpyDeviceToDevice);

    // Perform scan on array of length power of 2
    scan_pow2(d_inpPow, d_outPow, p);

    cudaMemcpy(out, d_outPow, sizeof(float)*N, cudaMemcpyDeviceToDevice);

    cudaFree(d_inpPow);
    cudaFree(d_outPow);
}


int main(){
    srand(time(NULL));

    const int N = 31;

    float *h_a, *h_b;
    float *d_a, *d_b;

    h_a = new float[N];
    h_b = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1.0;
    }

    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    scan(d_a, d_b, N);
    cudaMemcpy(h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);


    float csum = 0;
    for (int i = 0; i < N; ++i) {
        csum += h_a[i];
        printf("%f %f\n", h_b[i], csum);
    }



    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}