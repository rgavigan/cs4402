#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Function to perform matrix multiplication
*/
std::vector<std::vector<double>> multiplyMatrices(std::vector<std::vector<double>> M1, std::vector<std::vector<double>> M2) {
    int rows1 = M1.size(), cols1 = M1[0].size();
    int rows2 = M2.size(), cols2 = M2[0].size();

    // Error Handling: Ensure the matrices can be multiplied
    if (cols1 != rows2) {
        std::cout << "The matrices cannot be multiplied" << std::endl;
        return {};
    }

    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2, 0));

    // Multiplication
    cilk_for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result[i][j] += M1[i][k] * M2[k][j];
            }
        }
    }
    return result;
}

/**
 * Function to get the inverse of a n*n lower triangular matrix
 * Used to compute A_1^-1 and A_3^-1 to then obtain A^-1
*/
std::vector<std::vector<double>> getLowerTriangularInverse(std::vector<std::vector<double>> M, int B) {
    int rows = M.size(), cols = M[0].size();

    // Base Case: Ensure the matrix is square
    if (rows != cols) {
        std::cout << "The matrix is not square" << std::endl;
        return {};
    }

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0));

    // Base Case: Sequentially compute inverse when the matrix is small enough
    if (rows <= B) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    result[i][j] = 1 / M[i][j];
                } else {
                    double sum = 0;
                    for (int k = j; k < i; k++) {
                        sum += M[i][k] * result[k][j];
                    }
                    result[i][j] = -sum / M[i][i];
                }
            }
        }
        return result;
    }

    // Top-left quarter of the matrix
    std::vector<std::vector<double>> A_1(B, std::vector<double>(B, 0));
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < B; j++) {
            A_1[i][j] = M[i][j];
        }
    }

    // Bottom-left quarter of the matrix
    std::vector<std::vector<double>> A_2(rows - B, std::vector<double>(B, 0));
    for (int i = B; i < rows; i++) {
        for (int j = 0; j < B; j++) {
            A_2[i - B][j] = M[i][j];
        }
    }

    // Bottom-right quarter of the matrix
    std::vector<std::vector<double>> A_3(rows - B, std::vector<double>(rows - B, 0));
    for (int i = B; i < rows; i++) {
        for (int j = B; j < rows; j++) {
            A_3[i - B][j - B] = M[i][j];
        }
    }

    // Computing A1 and A3 inverses
    std::vector<std::vector<double>> A_1_inv = cilk_spawn getLowerTriangularInverse(A_1, B);
    std::vector<std::vector<double>> A_3_inv = getLowerTriangularInverse(A_3, B);
    cilk_sync;

    // Computing A2 inverse
    std::vector<std::vector<double>> bottom_left = multiplyMatrices(A_3_inv, A_2);
    bottom_left = multiplyMatrices(bottom_left, A_1_inv);

    for (int i = 0; i < bottom_left.size(); i++) {
        for (int j = 0; j < bottom_left[0].size(); j++) {
            bottom_left[i][j] = -bottom_left[i][j];
        }
    }

    // Top Left: A_1^-1
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < B; j++) {
            result[i][j] = A_1_inv[i][j];
        }
    }
    // Top Right: 0
    for (int i = 0; i < B; i++) {
        for (int j = B; j < cols; j++) {
            result[i][j] = 0;
        }
    }
    // Bottom Right: A_3^-1
    for (int i = B; i < rows; i++) {
        for (int j = B; j < cols; j++) {
            result[i][j] = A_3_inv[i - B][j - B];
        }
    }
    // Bottom Left: -A_3^-1 * A_2 * A_1^-1
    for (int i = B; i < rows; i++) {
        for (int j = 0; j < B; j++) {
            result[i][j] = bottom_left[i - B][j];
        }
    }
    
    return result;
}

/**
 * Generate a random 4x4 lower-triangular matrix with values between 1/10 and 10 using rand
*/
std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> result(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j <= i; j++) {
            result[i][j] = (rand() % 100 + 1) / 10.0;
        }
    }
    return result;
}

/**
 * Print out a matrix
*/
void printMatrix(std::vector<std::vector<double>> M) {
    for (int i = 0; i < M.size(); i++) {
        for (int j = 0; j < M[0].size(); j++) {
            // Handle -0 and epsilon value for 0
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
 * Correctness Tests
 * n = 4
 * B = 4, 2, 1
*/
void correctnessTests() {
    std::cout << "First Correctness Check, B = 4:" << std::endl;
    std::vector<std::vector<double>> A1c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A1c_inv = getLowerTriangularInverse(A1c, 4);
    std::vector<std::vector<double>> I = multiplyMatrices(A1c, A1c_inv);
    std::cout << "A1c * A1c_inv Result:" << std::endl;
    printMatrix(I);

    std::cout << "Second Correctness Check, B = 2:" << std::endl;
    std::vector<std::vector<double>> A2c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A2c_inv = getLowerTriangularInverse(A2c, 2);
    std::vector<std::vector<double>> I_2 = multiplyMatrices(A2c, A2c_inv);
    std::cout << "A2c * A2c_inv Result:" << std::endl;
    printMatrix(I_2);

    std::cout << "Third Correctness Check, B = 1:" << std::endl;
    std::vector<std::vector<double>> A3c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A3c_inv = getLowerTriangularInverse(A3c, 1);
    std::vector<std::vector<double>> I_3 = multiplyMatrices(A3c, A3c_inv);
    std::cout << "A3c * A3c_inv Result:" << std::endl;
    printMatrix(I_3);
}

/**
 * Performance Tests
 * n = 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
 * B = 32, 64, 128
*/
void performanceTests() {
    std::vector<int> n = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> B = {32, 64, 128};

    for (int i = 0; i < n.size(); i++) {
        std::vector<std::vector<double>> A = generateRandomMatrix(n[i]);
        for (int j = 0; j < B.size(); j++) {
            clock_t start = clock();
            std::vector<std::vector<double>> A_inv = getLowerTriangularInverse(A, B[j]);
            clock_t end = clock();
            double time = (double)(end - start) / CLOCKS_PER_SEC;
            std::cout << "n = " << n[i] << ", B = " << B[j] << ", Time = " << time << ", ";

            // Check that M * M^-1 = I and print success if so
            std::vector<std::vector<double>> I = multiplyMatrices(A, A_inv);
            bool success = true;
            for (int k = 0; k < I.size(); k++) {
                for (int l = 0; l < I[0].size(); l++) {
                    if (k == l) {
                        if (I[k][l] - 1 > 0.000001) {
                            success = false;
                            break;
                        }
                    }
                }
            }
            if (success) {
                std::cout << "Success" << std::endl;
            } else {
                std::cout << "Failure" << std::endl;
            }
        }
    }
}

int main() {
    std::cout << "Inverse of A1 Example:" << std::endl;
    std::vector<std::vector<double>> A1 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {-1, -1, 1, 0}, {-1, -1, -1, 1}};
    std::vector<std::vector<double>> A1_inv = getLowerTriangularInverse(A1, 4);
    printMatrix(A1_inv);

    std::cout << "Inverse of A2 Example:" << std::endl;
    std::vector<std::vector<double>> A2 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {1, 1, -1, 1}};
    std::vector<std::vector<double>> A2_inv = getLowerTriangularInverse(A2, 4);
    printMatrix(A2_inv);

    std::cout << "Inverse of A3 Example:" << std::endl;
    std::vector<std::vector<double>> A3 = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}};
    std::vector<std::vector<double>> A3_inv = getLowerTriangularInverse(A3, 4);
    printMatrix(A3_inv);

    srand(time(0));
    correctnessTests();
    performanceTests();
}