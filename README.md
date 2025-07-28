# CUDA Playground

![](./notebooks/fft_comp.png)



## Tricks


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