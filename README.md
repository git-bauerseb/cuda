# CUDA Playground

![](./notebooks/fft_comp.png)

```
nvidia-smi --query-gpu=compute_cap && \
nvcc gemm.cu common.cu -o p \
    -gencode=arch=compute_75,code=sm_75 \
    -lcublas
```


## Implementation - Benchmarks

GEMM (C = alpha*A*B + beta*C)
for A, (1024x1024)

```
V00 (Naive)            105.289154 ms
V01 (Coalescing)       10.251516 ms
V02 (Shared Memory)    6.875430 ms
V0x (cuBLAS)           1.742542  ms
```

## Tricks/Readings

### FFT Bit Reversal

Recursive FFT selects even/odd elements. When arriving at base case,
this yields a permutation that is sorted by reversing bits.
This allows for in-place operations:
```
0   -> 000  -> 000  -> 0
1   -> 100  -> 001  -> 4
2   -> 010  -> 010  -> 2
3   -> 110  -> 011  -> 6
4   -> 001  -> 100  -> 1
5   -> 101  -> 101  -> 5
6   -> 011  -> 110  -> 3
7   -> 111  -> 111  -> 7
```