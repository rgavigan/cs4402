#include <cilk/cilk.h>
#include <iostream>
#include <vector>
/*
Assumptions:
- A is a square matrix: n x n, n is a power of 2
- A is invertible: every diagonal element is non-zero
- A is a lower-triangular matrix
*/

/**
 * @brief Inverts a square matrix A
 * Divide and Conquer: A divided into 4 sub-matrices. 
 * If n <= B, then compute the inverse of A without further dividing it.
 * Multi-threading done with Cilk
 * Returns the 2D vector of float values
*/
std::vector<std::vector<float>> inverseMatrix(std::vector<std::vector<float>> A, int n, int B) {
    std::vector<std::vector<float>> A_inv(n, std::vector<float>(n, 0));
    if (n <= B) {
        for (int i = 0; i < n; i++) {
            A_inv[i][i] = 1 / A[i][i];
            for (int j = i + 1; j < n; j++) {
                float sum = 0;
                for (int k = i; k < j; k++) {
                    sum += A[j][k] * A_inv[k][i];
                }
                A_inv[j][i] = -sum / A[j][j];
            }
        }
    } else {
        // Divide A
        int n2 = n / 2;
        std::vector<std::vector<float>> A1(n2, std::vector<float>(n2, 0));
        std::vector<std::vector<float>> A2(n2, std::vector<float>(n2, 0));
        std::vector<std::vector<float>> A3(n2, std::vector<float>(n2, 0));
        std::vector<std::vector<float>> A1_inv(n2, std::vector<float>(n2, 0));
        std::vector<std::vector<float>> A2_inv(n2, std::vector<float>(n2, 0));
        std::vector<std::vector<float>> A3_inv(n2, std::vector<float>(n2, 0));
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                A1[i][j] = A[i][j];
                A3[i][j] = A[i + n2][j + n2];
                A2[i][j] = A[i + n2][j];
            }
        }

        // Compute the inverse of the 4 sub-matrices
        cilk_spawn A1_inv = inverseMatrix(A1, n2, B);
        cilk_spawn A3_inv = inverseMatrix(A3, n2, B);
        cilk_sync;

        // Compute inverse of A2: A2_inv = -A3_inv * A2 * A1_inv
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                A2_inv[i][j] = 0;
                for (int k = 0; k < n2; k++) {
                    A2_inv[i][j] += -A3_inv[i][k] * A2[k][j];
                }
                for (int k = 0; k < n2; k++) {
                    A2_inv[i][j] += A2_inv[i][k] * A1_inv[k][j];
                }
            }
        }
        
        // Fill A_inv with A1_inv, A2_inv, A3_inv
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                A_inv[i][j] = A1_inv[i][j];
                A_inv[i + n2][j] = A2_inv[i][j];
                A_inv[i][j + n2] = 0;
                A_inv[i + n2][j + n2] = A3_inv[i][j];
            }
        }
    }
    return A_inv;
}

int main() {
    std::vector<std::vector<float>> A1 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {-1, -1, 1, 0}, {-1, -1, -1, 1}};
    int n = 4;
    int B = 1;
    std::vector<std::vector<float>> A1_inv = inverseMatrix(A1, n, B);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A1_inv[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}