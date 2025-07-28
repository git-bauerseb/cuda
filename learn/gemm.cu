#include <cstdio>
#include "common.hu"
#include "cuda_runtime.h"
#include "cublas_v2.h"

void cublasGEMMFP32(cublasHandle_t& handle, float alpha, float beta,
    float *a, float *b, float *c, int m, int n, int k) {
    cudaDataType_t data_type = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F;
    // cublasGemmEx
    // C = alpha * op(A)*op(B) + beta*C
    // op(A) = A if CUBLAS_OP_N
    //       = A^T if CUBLAS_OP_T
    //
    // a := mxk
    // b := kxn

    // leading dimension
    // column-major (like in cuBLAS): number of rows
    // row-major: number of columns
    int lda = m;
    int ldb = k;
    int ldc = m;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,        // op(A) is A (no transpose)
        CUBLAS_OP_N,        // op(B) is B (no transpose)
        m,                  // rows of op(A) and C
        n,                  // columns of op(B) and C
        k,                  // columns of op(A) and rows of op(B)
        &alpha,             // scalar alpha
        a,                  // pointer to A on device
        data_type,          // data type of A
        lda,                // leading dimension of A
        b,                  // pointer to B on device
        data_type,          // data type of B
        ldb,                // leading dimension of B
        &beta,              // scalar beta
        c,                  // pointer to C on device (input and output)
        data_type,          // data type of C
        ldc,                // leading dimension of C
        compute_type,       // data type for internal computation
        CUBLAS_GEMM_DEFAULT // algorithm to use
    ));
}


// Version 0: Naive implementation, no coalescing memory read/write
__global__ void gemm_v00(int M, int N, int K, const float *a, 
    const float *b, float *c, float alpha, float beta) {
    // a  -> Mat(m,k)
    // b  -> Mat(k,n)
    // c  -> Mat(m,n)

    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += a[y*K+i] * b[i*K+x];
        }

        c[y*N+x] = alpha * tmp + beta * c[y*N+x];
    }
}

// Version 1: Implementation with coalescing read/write
// Changes: Change x,y
__global__ void gemm_v01(int M, int N, int K, float alpha, float beta,
    const float *a, const float *b, float *c) {
    // a  -> Mat(m,k)
    // b  -> Mat(k,n)
    // c  -> Mat(m,n)
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += a[y*K+i] * b[i*K+x];
        }

        c[y*N+x] = alpha * tmp + beta * c[y*N+x];
    }
}

// Version 2: Coalescing read/write + shared memory
// Each 2D block with index (b_m, b_n) is responsible for computing one matrix
template<size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_M,
         size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02kernel(const float *a, const float *b, float *c) {

    const uint col_idx = blockIdx.x * blockDim.x + blockIdx.x;
    const uint row_idx = blockIdx.y * blockDim.y + blockIdx.y;
    
    __shared__ float A[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    __shared__ float B[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];


}

void gemm_v02(int M, int N, int K, float alpha, float beta,
    const float *a, const float *b, float *c) {
    
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};  

    dim3 block_dim(BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U);

    // Each block responsible for one block of size (d_m, d_n)
    // for C
    unsigned int GRID_X_DIM{(static_cast<unsigned int>(N) + block_dim.x - 1U) / block_dim.x};
    unsigned int GRID_Y_DIM{(static_cast<unsigned int>(M) + block_dim.y - 1U) / block_dim.y};
    dim3 grid_dim(GRID_X_DIM,GRID_Y_DIM,1U);

    gemm_v02kernel<<<grid_dim, block_dim>>>(a, b, c);

}




int32_t main() {

    int deviceIdx = 0;
    CUDA_CHECK(cudaSetDevice(deviceIdx));

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        exit(EXIT_FAILURE);
    }

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    float *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    const size_t N = 512;


    h_a = (float *)malloc(sizeof(float) * N * N);
    h_b = (float *)malloc(sizeof(float) * N * N);
    h_c = (float *)malloc(sizeof(float) * N * N);
    
    fillRandomized(h_a, N * N);
    fillRandomized(h_b, N * N);

    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * N * N));
    CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(float) * N * N));
    CUDA_CHECK(cudaMalloc((void**)&d_c, sizeof(float) * N * N));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * N * N, cudaMemcpyHostToDevice));



    // Launch kernel
    int REP = 50;
    
    cudaEventRecord(beg);



    // MATMUL kernel
    dim3 gridDim((N+32-1)/32,(N+32-1)/32);
    dim3 blockDim(32,32);

    for (int rep = 0; rep < REP; ++rep) {
        gemm_v02(N, N, N, 1.0f, 0.0f, d_a, d_b, d_c);
        // cublasGEMMFP32(handle, 1.0, 0.0, d_a, d_b, d_c, N, N, N);
    }


    // for (int rep = 0; rep < REP; ++rep) {
    //    cublasGEMMFP32(handle, 1.0, 0.0, d_a, d_b, d_c, N, N, N);
    // }
    

    
    
    cudaEventRecord(end);

    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);

    float time = 0.0f;
    cudaEventElapsedTime(&time, beg, end);
    printf("Avg time: (%7.6f) ms\n", time / REP);
    

    CUDA_CHECK(cudaMemcpy(h_c, d_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost));


    int errors = 0;
    if (N < 1024) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {

                float v = 0.0f;
                for (int i = 0; i < N; ++i) {
                    int idx0 = y * N + i;
                    int idx1 = i * N + x;
                    // c[y][x] = sum_{i} a[y][i]*b[i][x]
                    v += h_a[idx0] * h_b[idx1];
                }

                if (errors < 10 && abs(h_c[y*N +x] - v) > 0.01f) {
                    printf("Error at (%d,%d)\n", y, x);
                }

                errors += abs(h_c[y*N +x] - v) > 0.01f;
            }
        }
    }

    printf("Total errors: %d\n", errors);

    free(h_a);
    free(h_b);
    free(h_c);
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}