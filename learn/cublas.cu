#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <numeric> // For std::iota

// Macro for checking CUDA runtime API errors
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Macro for checking cuBLAS API errors
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: ", __FILE__, __LINE__);      \
            if (status == CUBLAS_STATUS_NOT_INITIALIZED) fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n"); \
            else if (status == CUBLAS_STATUS_ALLOC_FAILED) fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n"); \
            else if (status == CUBLAS_STATUS_INVALID_VALUE) fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n"); \
            else if (status == CUBLAS_STATUS_ARCH_MISMATCH) fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH\n"); \
            else if (status == CUBLAS_STATUS_MAPPING_ERROR) fprintf(stderr, "CUBLAS_STATUS_MAPPING_ERROR\n"); \
            else if (status == CUBLAS_STATUS_EXECUTION_FAILED) fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n"); \
            else if (status == CUBLAS_STATUS_INTERNAL_ERROR) fprintf(stderr, "CUBLAS_STATUS_INTERNAL_ERROR\n"); \
            else if (status == CUBLAS_STATUS_NOT_SUPPORTED) fprintf(stderr, "CUBLAS_STATUS_NOT_SUPPORTED\n"); \
            else if (status == CUBLAS_STATUS_LICENSE_ERROR) fprintf(stderr, "CUBLAS_STATUS_LICENSE_ERROR\n"); \
            else fprintf(stderr, "Unknown cuBLAS error\n");                      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)


// Function to print a matrix (for verification)
template <typename T>
void printMatrix(const std::vector<T>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Assuming column-major storage for cuBLAS output, adjust for row-major print
            // If matrix is stored column-major, element (i,j) is at index i + j*rows
            // If matrix is stored row-major, element (i,j) is at index i*cols + j
            // cuBLAS uses column-major by default, so we print assuming column-major input/output
            std::cout << matrix[i + j * rows] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Define matrix dimensions
    const int M = 3; // Number of rows of C and A
    const int N = 2; // Number of columns of C and B
    const int K = 4; // Number of columns of A and rows of B

    // Host matrices
    std::vector<float> h_A(M * K); // M x K matrix
    std::vector<float> h_B(K * N); // K x N matrix
    std::vector<float> h_C(M * N); // M x N matrix (initial C, then result)

    // Initialize host matrices
    // A (M x K)
    // Example: A = [[1, 2, 3, 4],
    //                [5, 6, 7, 8],
    //                [9, 10, 11, 12]]
    // Stored in column-major: 1,5,9,2,6,10,3,7,11,4,8,12
    std::iota(h_A.begin(), h_A.end(), 1.0f);

    // B (K x N)
    // Example: B = [[1, 2],
    //                [3, 4],
    //                [5, 6],
    //                [7, 8]]
    // Stored in column-major: 1,3,5,7,2,4,6,8
    std::iota(h_B.begin(), h_B.end(), 1.0f);

    // C (M x N) - Initialize to zeros for simplicity, will be overwritten
    // Example: C = [[0, 0],
    //                [0, 0],
    //                [0, 0]]
    std::fill(h_C.begin(), h_C.end(), 0.0f);

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create a cuBLAS handle
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    // Define scalar coefficients alpha and beta
    // C = alpha * A * B + beta * C
    float alpha = 1.0f; // For A*B
    float beta = 0.0f;  // To overwrite C, set beta to 0.0f. If you want to accumulate, set to 1.0f.

    // Leading dimensions (lda, ldb, ldc)
    // For column-major matrices: leading dimension is the number of rows.
    // For A (M x K), lda = M
    // For B (K x N), ldb = K
    // For C (M x N), ldc = M
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    // Data types for A, B, C and for computation
    // For single-precision float, use CUDA_R_32F
    cudaDataType_t A_type = CUDA_R_32F;
    cudaDataType_t B_type = CUDA_R_32F;
    cudaDataType_t C_type = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F; // Type for internal computation

    // Perform matrix multiplication using cublasGemmEx
    // Note: CUBLAS_GEMM_DEFAULT is a generic algorithm. For Tensor Cores,
    // you might use CUBLAS_GEMM_DEFAULT_TENSOR_OP if applicable for your GPU and data types.
    CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, // op(A) is A (no transpose)
        CUBLAS_OP_N, // op(B) is B (no transpose)
        M,           // rows of op(A) and C
        N,           // columns of op(B) and C
        K,           // columns of op(A) and rows of op(B)
        &alpha,      // scalar alpha
        d_A,         // pointer to A on device
        A_type,      // data type of A
        lda,         // leading dimension of A
        d_B,         // pointer to B on device
        B_type,      // data type of B
        ldb,         // leading dimension of B
        &beta,       // scalar beta
        d_C,         // pointer to C on device (input and output)
        C_type,      // data type of C
        ldc,         // leading dimension of C
        compute_type, // data type for internal computation
        CUBLAS_GEMM_DEFAULT // algorithm to use
    ));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Input Matrices (Column-Major):" << std::endl;
    printMatrix(h_A, M, K, "A");
    printMatrix(h_B, K, N, "B");
    printMatrix(h_C, M, N, "C (Result)"); // This will show the result after computation
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}