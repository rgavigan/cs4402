#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Helper that generates a random lower-triangular matrix for testing
*/
std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> M(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j <= i; j++) {
            M[i][j] = (rand() % 100 + 1) / 10.0;
        }
    }
    return M;
}

/**
 * Helper that prints out a matrix
*/
void printMatrix(std::vector<std::vector<double>> M) {
    for (int i = 0; i < M.size(); i++) {
        for (int j = 0; j < M[0].size(); j++) {
            if (M[i][j] == -0 || M[i][j] - 0 < 0.000001) {
                M[i][j] = 0;
            }
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void mm_loop_serial1(int* C, int* A, int* B, int n)
{
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            for (int k = 0; k < n; ++k){
                C[i * n + j] = C[i * n + j] + (A[i * n + k] * B[k * n + j]);
            }
        }
    }
}

void mm_loop_serial2(int* C, int k0, int k1, int* A, int i0, int i1, int* B, int j0, int j1,  int n)
{
	for (int i = i0; i < i1; i++){
		for (int j = j0; j < j1; j++) {
            for (int k = k0; k < k1; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void serial_dandc(int i0, int i1, int j0, int j1, int k0, int k1, int* A, int lda, int* B, int ldb, int* C, int ldc, int X)
{
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;

    if (di >= dj && di >= dk && di >= X) {
        int mi = i0 + di / 2;
        serial_dandc(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        serial_dandc(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
    } 
    else if (dj >= dk && dj >= X) {
        int mj = j0 + dj / 2;
        serial_dandc(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc,X);
        serial_dandc(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
    } 
    else if (dk >= X) {    
        int mk = k0 + dk / 2;
        serial_dandc(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc,X);
        serial_dandc(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc,X);
    } 
    else {
        mm_loop_serial2(C, k0, k1,  A, i0, i1, B, j0, j1, lda)  ;
    }
}

void parallel_dandc(int i0, int i1, int j0, int j1, int k0, int k1, int* A, int lda, int* B, int ldb, int* C, int ldc, int X)
{
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;

    if (di >= dj && di >= dk && di >= X) {
        int mi = i0 + di / 2;
        cilk_spawn parallel_dandc(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        parallel_dandc(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        cilk_sync;
    } 
    else if (dj >= dk && dj >= X) {
        int mj = j0 + dj / 2;
        cilk_spawn parallel_dandc(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc,X);
        parallel_dandc(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        cilk_sync;
    } 
    else if (dk >= X) {    
        int mk = k0 + dk / 2;
        parallel_dandc(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc,X);
        parallel_dandc(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc,X);
    } 
    else {
        mm_loop_serial2(C, k0, k1,  A, i0, i1, B, j0, j1, lda)  ;
    }
}

void invertMatrix(int* A, int n, int B, int i0, int i1, int j0, int j1) {
    if (i1 - i0 <= B) {
        serial_dandc(i0, i1, j0, j1, 0, n, A, n, A, n, A, n, B);
    } else {
        int mi = i0 + (i1 - i0) / 2;
        cilk_spawn invertMatrix(A, n, B, i0, mi, j0, j1);
        invertMatrix(A, n, B, mi, i1, j0, j1);
        cilk_sync;
    }
}

void testInverse(int n, int B) {
    std::vector<std::vector<double>> A = generateRandomMatrix(n);
    invertMatrix(A, n, B, 0, n, 0, n);
    std::cout << "n = " << n << ", B = " << B << std::endl;
    printMatrix(A);
}

/**
 * Correctness Tests
 * n = 4
 * B = 4, 2, 1
*/
void correctnessTests() {
    std::cout << "Correctness Tests:" << std::endl;
    testInverse(4, 4);
    testInverse(4, 2);
    testInverse(4, 1);
}

/**
 * Performance Tests
*/
void performanceTests() {
    std::vector<int> n = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> B = {32, 64, 128};

    // Create LaTeX table for performance tests
    std::cout << "\\begin{tabular}{|c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "n & B & Time & Success \\\\" << std::endl;
    std::cout << "\\hline" << std::endl;
    for (int i = 0; i < n.size(); i++) {
        std::vector<std::vector<double>> A = generateRandomMatrix(n[i]);
        for (int j = 0; j < B.size(); j++) {
            clock_t start = clock();
            invertMatrix(A, n[i], B[j], 0, n[i], 0, n[i]);
            clock_t end = clock();
            double time = (double)(end - start) / CLOCKS_PER_SEC;
            std::cout << n[i] << " & " << B[j] << " & " << time << " & ";
        }
    }
    std::cout << "\\hline" << std::endl;
    std::cout << "\\end{tabular}" << std::endl;
}

int main() {
    std::vector<std::vector<double>> A1 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {-1, -1, 1, 0}, {-1, -1, -1, 1}};
    std::vector<std::vector<double>> A2 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {1, 1, -1, 1}};
    std::vector<std::vector<double>> A3 = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}};

    srand(time(0));
    correctnessTests();
    performanceTests();
}