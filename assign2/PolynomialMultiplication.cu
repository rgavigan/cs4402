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

// CUDA Parallel Polynomial Multiplication
__global__ void polynomialMultiplication(int* A, int* B, int* C, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p <= 2 * n) {
        C[p] = 0;
        for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
            C[p] += A[t + p - n] * B[n - t];
        }
    }
}

void polynomialMultiplicationSerial(int* A, int* B, int* C, int n) {
    for (int p = 0; p <= 2 * n; p++) {
        C[p] = 0;
        for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
            C[p] += A[t + p - n] * B[n - t];
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

    printf("\\begin{table}[ht]\n");
    printf("\\centering\n");
    printf("\\begin{tabular}{|c|c|c|c|c|}\n");
    printf("\\hline\n");
    printf("N & B & GPU Time (ms) & CPU Time (ms) & Speedup \\\\\n");
    printf("\\hline\n");

    // Run the kernel for different values of B and N and print out GPU vs CPU time
    for (int j = 0; j < 2; j++) {
        int N = N_values[j];
        for (int i = 0; i < 5; i++) {
            int B = B_values[i];
            int* A = (int*)malloc((N + 1) * sizeof(int));
            int* Br = (int*)malloc((N + 1) * sizeof(int));
            int* C = (int*)malloc((2 * N + 1) * sizeof(int));
            int* C_serial = (int*)malloc((2 * N + 1) * sizeof(int));
            
            // Random values from {-1, 0, 1}
            for (int i = 0; i < N + 1; i++) {
                A[i] = rand() % 3 - 1;
                Br[i] = rand() % 3 - 1;
            }

            int* d_A;
            int* d_B;
            int* d_C;
            cudaMalloc(&d_A, (N + 1) * sizeof(int));
            cudaMalloc(&d_B, (N + 1) * sizeof(int));
            cudaMalloc(&d_C, (2 * N + 1) * sizeof(int));

            cudaMemcpy(d_A, A, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, Br, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);

            struct timeval start, end;
            gettimeofday(&start, NULL);
            polynomialMultiplication<<<(2 * N + B - 1) / B, B>>>(d_A, d_B, d_C, N);
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            float gpu_time = (end.tv_sec - start.tv_sec) * 1e6 * 1000;
            gpu_time = (gpu_time + (end.tv_usec - start.tv_usec)) * 1e-6;

            cudaMemcpy(C, d_C, (2 * N + 1) * sizeof(int), cudaMemcpyDeviceToHost);

            gettimeofday(&start, NULL);
            polynomialMultiplicationSerial(A, Br, C_serial, N);
            gettimeofday(&end, NULL);
            float time = (end.tv_sec - start.tv_sec) * 1e6 * 1000;
            time = (time + (end.tv_usec - start.tv_usec)) * 1e-6;
            bool valid = verifyResults(C, C_serial, N);
            if (valid) {
                printf("%d & %d & %.1e & %.1e & %.3f \\\\\n", N, B, gpu_time, time, time / gpu_time);
            }
            else {
                printf("%d & %d & %.1e & %.1e & %.3f* \\\\\n", N, B, gpu_time, time, time / gpu_time);
            }

            free(A);
            free(Br);
            free(C);
            free(C_serial);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
    }

    printf("\\hline\n");
    printf("\\end{tabular}\n");
    printf("\\caption{GPU vs CPU Time Comparison}\n");
    printf("\\label{tab:comparison}\n");
    printf("\\end{table}\n");
}