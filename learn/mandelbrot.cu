#include <fstream>
#include "common.hu"

constexpr int THRESH = 512;

__host__ __device__ int dwell(int w, int h, complex cmin, complex cmax,
                            int x, int y) {
    complex dc = cmax - cmin;
    float fx = (float)x / w, fy = (float)y / h;
    complex c = cmin + complex(fx * dc.m_re, fy * dc.m_im);
    int dwell = 0;
    complex z = c;
    while(dwell < THRESH && abs2(z) < 2 * 2) {
        z = z * z + c;
        dwell++;
    }
    return dwell;
}

__global__ void mandelbrot(int *dwells, int w, int h,
                    complex cmin, complex cmax) {
    int x = threadIdx.x + blockDim.x * blockIdx. x;
    int y = threadIdx.y + blockDim.y * blockIdx. y;
    
    if (x < w && y < h) {
        dwells[x + y * w] = dwell(w, h, cmin, cmax, x, y);
    }
}





void dwell_color(int* r, int* g, int* b, int dwell) {
    constexpr int CUTOFF = THRESH / 4;
    if (dwell >= THRESH) {
        *r = 0, *g = 0, *b = 0;
    } else {
        if(dwell < 0) dwell = 0;
		if(dwell <= CUTOFF) {
			*r = *g = 0;
			*b = 128 + dwell * 127 / (CUTOFF);
		} else {
			*b = 255;
			*r = *g = (dwell - CUTOFF) * 255 / (THRESH - CUTOFF);
		}
    }
}

void createPPM(const char* filename, int* data, int width, int height) {
    std::ofstream file(filename, std::ios::out);
    file << "P3\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; ++i) {
        int r, g, b;
        dwell_color(&r, &g, &b, data[i]);
        file << r << ' ' << g << ' ' << b << '\n';
    }
    file.close();
}


int32_t main() {

    int w = 4096;
    int h = 4096;
    dim3 block_size(32,32);
    dim3 grid_size((w+block_size.x-1)/block_size.x, (h+block_size.y-1)/block_size.y);

    int* d_ptr;
    int* h_ptr;

    size_t tsize = sizeof(int) * w * h;

    CUDA_CHECK(cudaMalloc((void**)&d_ptr, tsize));
    h_ptr = (int*)malloc(tsize);

    complex mn(-1.5, -1);
    complex mx(0.5, 1);


    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);


    cudaEventRecord(beg);
    mandelbrot<<<grid_size,block_size>>>(d_ptr, w, h, mn, mx);
    CUDA_CHECK(cudaThreadSynchronize());
    cudaEventRecord(end);

    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);

    float time = 0.0f;
    cudaEventElapsedTime(&time, beg, end);
    printf("Took %7.6f ms\n", time);


    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, tsize, cudaMemcpyDeviceToHost));
    
    // Write as image
    createPPM("output.ppm", h_ptr, w, h);
    
    
    CUDA_CHECK(cudaFree(d_ptr));
    free(h_ptr);
    
    return 0;
}