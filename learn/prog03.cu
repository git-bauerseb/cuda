#include <stdio.h>
#include "common.hu"

const int N = 2048*2048;
const int BLOCK_SIZE = 1024;

__global__ void vadd(const float *a, const float *b, float *c, int ds){
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < ds) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    srand(time(NULL));

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = rand()/(float)RAND_MAX;
        h_b[i] = rand()/(float)RAND_MAX;
    }

    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    
    vadd<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
   
    printf("%f\n", h_c[1025] - (h_a[1025] + h_b[1025]));


    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}