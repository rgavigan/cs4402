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

/**
 * Parallel Divide and Conquer Matrix Multiplication
 * i0, i1: Starting and ending row of A
 * j0, j1: Starting and ending column of B
 * k0, k1: Starting and ending column of A
 * A: Matrix A containing double values
 * lda: Leading dimension of A
 * B: Matrix B containing double values
 * ldb: Leading dimension of B
 * C: Matrix C containing double values
 * ldc: Leading dimension of C
 * X: Base case size
*/
void matrixMultiplication(int i0, int i1, int j0, int j1, int k0, int k1, double* A, int lda, double* B, int ldb, double* C, int ldc, int X)
{
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;

    if (di >= dj && di >= dk && di >= X) {
        int mi = i0 + di / 2;
        cilk_spawn matrixMultiplication(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X); // Remove cilk_spawn to run serially
        matrixMultiplication(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        cilk_sync;
    } 
    else if (dj >= dk && dj >= X) {
        int mj = j0 + dj / 2;
        cilk_spawn matrixMultiplication(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc,X); // Remove cilk_spawn to run serially
        matrixMultiplication(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        cilk_sync;
    } 
    else if (dk >= X) {    
        int mk = k0 + dk / 2;
        matrixMultiplication(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc,X);
        matrixMultiplication(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc,X);
    } 
    else {
        for (int i = i0; i < i1; i++){
            for (int j = j0; j < j1; j++) {
                for (int k = k0; k < k1; k++) {
                    C[i * lda + j] += A[i * lda + k] * B[k * lda + j];
                }
            }
        }
    }
}

/**
 * Divide-and-Conquer Matrix Inversion of Lower-Triangular Matrices
 * Calculations: Top Left Inverse (A1), Bottom Right Inverse (A3), Bottom Left Multiplication (A2 = -A3^-1 * A2 * A1^-1)
 * A: Lower-Triangular n*n matrix
 * n: Size of the matrix
 * B: Base case size
*/
void matrixInversion() {
    
}

/**
 * Function to invert a matrix in-place in serial
 * Matrix: Lower-Triangular n*n matrix
 * n: Size of the matrix
*/
void serialInvertMatrix(std::vector<std::vector<double>> &A, int n) {
    // Invert the top-left quarter of A
    for (int i = 0; i < n; i++) {
        A[i][i] = 1 / A[i][i];
        for (int j = i + 1; j < n; j++) {
            double sum = 0;
            for (int k = i; k < j; k++) {
                sum += A[j][k] * A[k][i];
            }
            A[j][i] = -sum / A[j][j];
        }
    }

    // Invert the bottom-right quarter of A
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double sum = 0;
            for (int k = j + 1; k <= i; k++) {
                sum += A[j][k] * A[k][i];
            }
            A[j][i] = -sum;
        }
    }

    // Multiply bottom-right by bottom-left using serial_dandc
    

    // Multiply bottom-left by top-left and store it in bottom-left
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0;
            for (int k = j; k < n; k++) {
                sum += A[j][k] * A[k][i];
            }
            A[j][i] = sum;
        }
    }

    // Multiply bottom-left by -1 and store it in bottom-left
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j][i] = -A[j][i];
        }
    }
}

/**
 * Function to invert a matrix in-place in parallel
 * Uses divide-and-conquer algorithm and uses a parallel matrix multiplication algorithm (matrixMultiplication)
 * Matrix: Lower-Triangular n*n matrix
 * n: Size of the matrix
 * startRow: Starting row of the matrix
 * startCol: Starting column of the matrix
 * B: Base case size
*/
void parallelInvertMatrix(std::vector<std::vector<double>> &A, int startRow, int startCol, int n, int B) {
    if (n <= B) {
        serialInvertMatrix(A, n);
    } else {
        int mid = n / 2;
        cilk_spawn parallelInvertMatrix(A, startRow, startCol, mid, B); // Top-Left
        parallelInvertMatrix(A, startRow + mid, startCol + mid, n - mid, B); // Bottom-Right
        cilk_sync;
    }
}

void testInverse(int n, int B) {
    std::vector<std::vector<double>> A = generateRandomMatrix(n);
    parallelInvertMatrix(A, 0, 0, n, B);
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
            parallelInvertMatrix(A, 0, 0, n[i], B[j]);
            clock_t end = clock();
            double time = (double)(end - start) / CLOCKS_PER_SEC;
            std::cout << n[i] << " & " << B[j] << " & " << time << " & ";
        }
    }
    std::cout << "\\hline" << std::endl;
    std::cout << "\\end{tabular}" << std::endl;
}

int main() {
    // Unimodular Correctness Matrices
    std::vector<std::vector<double>> A1 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {-1, -1, 1, 0}, {-1, -1, -1, 1}};
    std::vector<std::vector<double>> A2 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {1, 1, -1, 1}};
    std::vector<std::vector<double>> A3 = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}};
    int B = 4;
    int n = 4;
    serialInvertMatrix(A1, n);
    std::cout << "n = " << n << ", B = " << B << std::endl;
    printMatrix(A1);
    serialInvertMatrix(A2, n);
    std::cout << "n = " << n << ", B = " << B << std::endl;
    printMatrix(A2);
    serialInvertMatrix(A3, n);
    std::cout << "n = " << n << ", B = " << B << std::endl;
    printMatrix(A3);

    srand(time(0));
    correctnessTests();
    //performanceTests();
}