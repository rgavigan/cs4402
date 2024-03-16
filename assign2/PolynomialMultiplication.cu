#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define GPU_DEVICE 0
#define EPSILON 1e-6

/**
 * Univariate Polynomial Multiplication With ceil((2n+1)/B) thread blocks of B threads each
*/
__global__ void polynomialMultiplication(int* A, int* B, int* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid <= 2 * n) {
        for (int i = 0; i <= tid; i++) {
            C[tid] += A[i] * B[tid - i];
        }
    }
}

void polynomialMultiplicationSerial(int* A, int* B, int* C, int N) {
    for (int i = 0; i <= 2 * N; i++) {
        C[i] = 0;
        for (int j = 0; j <= i; j++) {
            C[i] += A[j] * B[i - j];
        }
    }
}

bool verifyResults(int* C1, int* C2, int N) {
    for (int i = 0; i < 2 * N; i++) {
        if (fabs(C1[i] - C2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

int main() {
    int B_values[] = {32, 64, 128, 256, 512};
    int N_values[] = {(int)pow(2, 14), (int)pow(2, 16)};

    for (int i = 0; i < sizeof(B_values) / sizeof(int); i++) {
        for (int j = 0; j < sizeof(N_values) / sizeof(int); j++) {
            int B1 = B_values[i];
            int N = N_values[j];

            // Allocate memory on the host
            int* A = (int*)malloc((2 * N + 1) * sizeof(int));
            int* B = (int*)malloc((2 * N + 1) * sizeof(int));
            int* C = (int*)malloc((2 * N + 1) * sizeof(int));

            // Allocate memory for the CPU result
            int* C_cpu = (int*)malloc((2 * N + 1) * sizeof(int));

            // Initialize A and B arrays with random values of {-1, 0, 1}
            for (int i = 0; i <= 2 * N; i++) {
                A[i] = (rand() % 3) - 1;
                B[i] = (rand() % 3) - 1;
            }

            // Allocate memory on the device
            int* d_A, *d_B, *d_C;
            cudaMalloc((void**)&d_A, (2 * N + 1) * sizeof(int));
            cudaMalloc((void**)&d_B, (2 * N + 1) * sizeof(int));
            cudaMalloc((void**)&d_C, (2 * N + 1) * sizeof(int));

            // Copy input data from host to device
            cudaMemcpy(d_A, A, (2 * N + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, (2 * N + 1) * sizeof(int), cudaMemcpyHostToDevice);

            // Launch kernel
            int numBlocks = ceil((2 * N + 1) / B1);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            polynomialMultiplication<<<numBlocks, B1>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            // Run the serial version
            clock_t start_cpu = clock();
            // Run the serial version
            polynomialMultiplicationSerial(A, B, C_cpu, N);
            clock_t end_cpu = clock();
            double cpu_time = ((double) (end_cpu - start_cpu)) * 1000 / CLOCKS_PER_SEC;

            // Copy result from device to host
            cudaMemcpy(C, d_C, (2 * N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Verify the results
            bool resultsMatch = verifyResults(C, C_cpu, N);
            printf("B = %d, N = %d, GPU Runtime: %f ms, CPU Runtime: %f ms, Results Match: %s\n", B1, N, milliseconds, cpu_time, resultsMatch ? "Yes" : "No");

            // Free device memory
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);

            // Free host memory
            free(A);
            free(B);
            free(C);
            free(C_cpu);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    return 0;
}