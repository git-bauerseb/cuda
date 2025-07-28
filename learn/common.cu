#include "common.hu"
#include <cstdio>

void fillOnes(float *vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = 1.0f;
    }
}

void fillOnes(int *vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = 1;
    }
}

void fillRandomized(float *vec, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        vec[i] = float(rand()) / RAND_MAX;
    }
}


void printMatrix(float *mat, int m, int n) {
    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%3.2f ", mat[y*n + x]);
        }
        printf("\n");
    }
}