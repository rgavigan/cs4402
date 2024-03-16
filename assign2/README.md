### Compilation and Testing
```sh
# Make sure your PATH is properly set first
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64

make clean && make && ./PolynomialMultiplication
```

The code currently has tests running for question 2 (n = 2^14, 2^16 and B = 32, 64, 128, 256, 512). It validates the results against the serial implementation and calculates the speedup in time.
