// Cilk Parallelism
#include <cilk/cilk.h>

// 
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Generates random matrix of floats from 0.1 -> 10.0
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
 * Prints out a matrix
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
    // Difference between start/end of i, j, k
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;

    if (di >= dj && di >= dk && di >= X) {
        int mi = i0 + di / 2;
        cilk_spawn matrixMultiplication(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc, X); // Divide for i
        matrixMultiplication(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc,X);
        cilk_sync;
    } 
    else if (dj >= dk && dj >= X) {
        int mj = j0 + dj / 2;
        cilk_spawn matrixMultiplication(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc, X); // Divide for j
        matrixMultiplication(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
        cilk_sync;
    } 
    else if (dk >= X) {    
        int mk = k0 + dk / 2;
        matrixMultiplication(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc, X); // Divide for k
        matrixMultiplication(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc, X);
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
void matrixInversion(std::vector<std::vector<double>> &A, int startRow, int endRow, int startCol, int endCol, int B) {
    // Base Case: Forward Substitution
    if (endRow - startRow <= B) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < i; j++) {
                for (int k = j; k < i; k++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
            A[i][i] = 1.0 / A[i][i];
        }
        return;
    }
    // Divide-and-Conquer Parallelism
    else {
        // Formula: A = [A11 A12; A21 A22]
        // A^-1 = [A11^-1 0; -1 * A22^-1 * A21 * A11^-1 A22^-1]
        int mid = (startRow + endRow) / 2;
        cilk_spawn matrixInversion(A, startRow, mid, startCol, mid, B); // A11
        matrixInversion(A, mid, endRow, mid, endCol, B); // A22
        cilk_sync;

        // A21:
        std::vector<std::vector<double>> C(endRow - mid, std::vector<double>(mid - startCol, 0));
        for (int i = mid; i < endRow; i++) {
            for (int j = startCol; j < mid; j++) {
                C[i - mid][j - startCol] = A[i][j];
            }
        }
        // C = -A22^-1 * A21
        matrixMultiplication(0, endRow - mid, 0, mid - startCol, 0, mid - startCol, A[mid][mid], mid - startCol, A[mid][mid], mid - startCol, C[0].data(), mid - startCol, B);

        // C = -A22^-1 * A21 * A11^-1
        matrixMultiplication(0, endRow - mid, 0, mid - startCol, 0, mid - startCol, C[0].data(), mid - startCol, A[startRow][startRow], mid - startCol, C[0].data(), mid - startCol, B);

        // Set A21 = C
        for (int i = mid; i < endRow; i++) {
            for (int j = startCol; j < mid; j++) {
                A[i][j] = C[i - mid][j - startCol];
            }
        }
    }
}

/**
 * Tests matrix inversion with random data for size n and base case size B
*/
void testInverse(int n, int B) {
    std::vector<std::vector<double>> A = generateRandomMatrix(n);
    matrixInversion(A, 0, n, 0, n, B);
    std::cout << "n = " << n << ", B = " << B << std::endl;
    printMatrix(A);
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
    matrixInversion(A1, 0, n, 0, n, B);
    matrixInversion(A2, 0, n, 0, n, B);
    matrixInversion(A3, 0, n, 0, n, B);
    printMatrix(A1);
    printMatrix(A2);
    printMatrix(A3);

    srand(time(0));

    std::cout << "Correctness Tests:" << std::endl;
    testInverse(4, 4);
    testInverse(4, 2);
    testInverse(4, 1);

    performanceTests();
}