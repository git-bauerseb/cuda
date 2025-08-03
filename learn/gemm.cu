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
    const float *A, const float *B, float *C) {
    // a  -> Mat(m,k)
    // b  -> Mat(k,n)
    // c  -> Mat(m,n)
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += A[y*K+i] * B[i*K+x];
        }

        C[y*N+x] = alpha * tmp + beta * C[y*N+x];
    }
}

// Version 2: Coalescing read/write + shared memory
// Each 2D block with index (b_m, b_n) is responsible for computing one matrix
// C = alpha*AB + beta*C
template <size_t dbn, size_t dbm, size_t dbk, size_t num_threads>
__device__ void load_to_shared_memory(float const* A,
                                           float const* B,
                                           float A_thread_block_tile[dbm][dbk],
                                           float B_thread_block_tile[dbk][dbn],
                                           size_t tile_idx,             // Index 
                                           size_t linear_idx,
                                           size_t m, size_t n,
                                           size_t k) {

    // We have dbn * dbm threads
    // Load tile of A and B into shared memory
    // For A
    //  need to load dbm * dbk elements
    //  every thread needs to load (dbm * dbk + num_threads - 1) / num_threads many elements
    uint elem_per_thread_a = (dbm * dbk + num_threads - 1) / num_threads;

    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (uint load_idx = 0; load_idx < elem_per_thread_a; ++load_idx) {

        // Since we launch 1 thread for 1 element in output matrix,
        // we can use the global index (linear_idx) for calculating
        // position inside a (same for b)
        uint a_block_tile_row_idx = (linear_idx + load_idx * num_threads) / dbk;
        uint a_block_tile_col_idx = (linear_idx + load_idx * num_threads) % dbk;
        
        uint a_row_idx = blockIdx.y * dbm + a_block_tile_row_idx;
        uint a_col_idx = tile_idx * dbk + a_block_tile_col_idx;
        
        float val = 0.0f;
        if (a_row_idx < m && a_col_idx < k) {
            val = A[a_row_idx * k + a_col_idx];
        }

        A_thread_block_tile[a_block_tile_row_idx][a_block_tile_col_idx] = val;
    }


    uint elem_per_thread_b = (dbk * dbn + num_threads - 1) / num_threads;
#pragma unroll
    for (uint load_idx = 0; load_idx < elem_per_thread_b; ++load_idx) {
        uint b_block_tile_row_idx = (linear_idx + load_idx * num_threads) / dbn;
        uint b_block_tile_col_idx = (linear_idx + load_idx * num_threads) % dbn;
        
        uint b_row_idx = tile_idx * dbk + b_block_tile_row_idx;
        uint b_col_idx = blockIdx.x * dbn + b_block_tile_col_idx;
        
        float val = 0.0f;
        if (b_row_idx < k && b_col_idx < n) {
            val = B[b_row_idx * n + b_col_idx];
        }

        B_thread_block_tile[b_block_tile_row_idx][b_block_tile_col_idx] = val;
    }
}


template<uint dbn, uint dbm, uint dbk>
__global__ void gemm_v02_kernel(uint m, uint n, uint k, float alpha, float beta,
    float const* A, float const* B, float* C) {
    
    constexpr uint num_threads = dbn * dbm;
    uint thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;


    // Output index for matrix C
    uint c_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint c_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float A_block[dbm][dbk];
    __shared__ float B_block[dbk][dbn];

    // Number of times to multiply A and B blocks for result
    uint num_sum_blocks = (k + dbk-1) / dbk;


    float sum = 0.0f;
    for (uint tile_idx = 0; tile_idx < num_sum_blocks; ++tile_idx) {
        // bm = blockDim.y
        // bn = blockDim.x

        // From A want to load block with index
        //   bm
        //   tile_idx
        load_to_shared_memory<dbn, dbm, dbk, num_threads>(A,B,A_block,B_block,tile_idx,thread_linear_idx,m,n,k);
        __syncthreads();

#pragma unroll
        for (uint ki = 0; ki < dbk; ++ki) {
            sum += A_block[threadIdx.y][ki] * B_block[ki][threadIdx.x];
        }
        __syncthreads();
    }

    if (c_col_idx < n && c_row_idx < m) {
        C[c_row_idx * n + c_col_idx] = alpha * sum + beta * C[c_row_idx * n + c_col_idx];
    }
}

void gemm_v02(uint m, uint n, uint k, float alpha, float beta,
            float const* A, float const* B, float* C) {

    constexpr uint dbn = 32;
    constexpr uint dbm = 32;
    constexpr uint dbk = 32;

    dim3 const block_dim{dbn, dbm, 1};
    dim3 const grid_dim{(n+block_dim.x-1u) / block_dim.x, (m+block_dim.y-1u) / block_dim.y, 1};


    gemm_v02_kernel<dbn, dbm, dbk>
        <<<grid_dim, block_dim>>>(m, n, k, alpha, beta, A, B, C);
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

    const size_t N = 1024;


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
    for (int rep = 0; rep < REP; ++rep) {
        gemm_v02(N,N,N,1.0f,0.0f,d_a,d_b,d_c);
    }

    // for (int rep = 0; rep < REP; ++rep) {
        // gemm_v02(N, N, N, 1.0f, 0.0f, d_a, d_b, d_c);
        // cublasGEMMFP32(handle, 1.0, 0.0, d_a, d_b, d_c, N, N, N);
    // }
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