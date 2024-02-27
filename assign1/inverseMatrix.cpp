#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iomanip>
#include <chrono>

/**
 * Print out a matrix
*/
void printMatrix(const std::vector<std::vector<double>>& M, std::string title = "") {
    std::cout << title << std::endl;
    for (int i = 0; i < M.size(); i++) {
        for (int j = 0; j < M[0].size(); j++) {
            // Epsilon and Negative Removal for Clean Printing - Floating Point Stuff
            if (M[i][j] == -0 || M[i][j] - 0 < 0.000001) {
                std::cout << std::setw(5) << std::setprecision(2) << 0 << " ";
            }
            else {
                std::cout << std::setw(5) << std::setprecision(2) << M[i][j] << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * Function to perform matrix multiplication
 * M1: The first matrix - this matrix will be updated with the result
 * M2: The second matrix
 * Computation: M1 * M2
*/
void multiplyMatricesParallel(std::vector<std::vector<double>>& M1, std::vector<std::vector<double>>& M2, int startRowM1, int startColM1, int startRowM2, int startColM2, int size) {
    const int blockSize = 16; // Adjust this value to match your CPU's cache line size
    std::vector<std::vector<double>> temp(size, std::vector<double>(size, 0));

    cilk_for (int i = 0; i < size; i += blockSize) {
        cilk_for (int j = 0; j < size; j += blockSize) {
            for (int k = 0; k < size; k += blockSize) {
                for (int ii = i; ii < std::min(i + blockSize, size); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, size); ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < std::min(k + blockSize, size); ++kk) {
                            sum += M1[startRowM1 + ii][startColM1 + kk] * M2[startRowM2 + kk][startColM2 + jj];
                        }
                        temp[ii][jj] += sum;
                    }
                }
            }
        }
    }

    cilk_for (int i = 0; i < size; i++) {
        cilk_for (int j = 0; j < size; j++) {
            M1[startRowM1 + i][startColM1 + j] = temp[i][j];
        }
    }
}

/**
 * Method to divide-and-conquer on multiplyMatricesParallel
*/
void multiplyMatrices(std::vector<std::vector<double>>& M1, std::vector<std::vector<double>>& M2, int startRowM1, int startColM1, int startRowM2, int startColM2, int size) {
    // Base Case: When the matrix is small enough
    if (size <= 128) {
        multiplyMatricesParallel(M1, M2, startRowM1, startColM1, startRowM2, startColM2, size);
        return;
    }
    int mid = size / 2;

    // Divide and Conquer
    cilk_spawn multiplyMatrices(M1, M2, startRowM1, startColM1, startRowM2, startColM2, mid);                    // A11 * B11
    cilk_spawn multiplyMatrices(M1, M2, startRowM1, startColM1 + mid, startRowM2 + mid, startColM2, mid);        // A12 * B21
    cilk_spawn multiplyMatrices(M1, M2, startRowM1, startColM1, startRowM2, startColM2 + mid, mid);              // A11 * B12
    multiplyMatrices(M1, M2, startRowM1, startColM1 + mid, startRowM2 + mid, startColM2 + mid, mid);  // A12 * B22
    cilk_sync;
    cilk_spawn multiplyMatrices(M1, M2, startRowM1 + mid, startColM1, startRowM2, startColM2, mid);              // A21 * B11
    cilk_spawn multiplyMatrices(M1, M2, startRowM1 + mid, startColM1 + mid, startRowM2 + mid, startColM2, mid);  // A22 * B21
    cilk_spawn multiplyMatrices(M1, M2, startRowM1 + mid, startColM1, startRowM2, startColM2 + mid, mid);        // A21 * B12
    multiplyMatrices(M1, M2, startRowM1 + mid, startColM1 + mid, startRowM2 + mid, startColM2 + mid, mid);      // A22 * B22
    cilk_sync;                                                                                                    // Syncing Parallelism
}

/**
 * Function to get the inverse of a n*n lower triangular matrix
 * Used to compute A_1^-1 and A_3^-1 to then obtain A^-1
*/
void getLowerTriangularInverseParallel(std::vector<std::vector<double>> &M, int B, int startRow, int startCol, int endRow, int endCol) {
    // Base Case: Forward substitution when the matrix is small enough
    if (endRow - startRow <= B) {
        cilk_for (int i = startRow; i < endRow; i++) {
            cilk_for (int j = startCol; j <= i; j++) {
                if (i == j) {
                    M[i][j] = 1 / M[i][j];                                                                // Diagonal Elements
                }
                else {
                    double sum = 0;
                    cilk_for (int k = j; k < i; k++) {
                        sum += M[i][k] * M[k][j];                                                         // Summation
                    }
                    M[i][j] = -sum / M[i][i];                                                             // Off-Diagonal Elements
                }
            }
        }
        return;
    }
    int mid = (endRow - startRow) / 2;

    // Computing A1^-1 and A3^-1
    cilk_spawn getLowerTriangularInverseParallel(M, B, startRow, startCol, mid, mid);                     // A1 inverse
    cilk_spawn getLowerTriangularInverseParallel(M, B, startRow + mid, startCol + mid, endRow, endCol);   // A3 inverse
    cilk_sync;                                                                                            // Syncing Parallelism

    // Computing the multiplication for A2 (A_2 * A_1^-1)
    cilk_spawn multiplyMatrices(M, M, startRow + mid, startCol, startRow, startCol, mid);                    // A_2 * A_1^-1

    // Create a temporary matrix to hold A3
    std::vector<std::vector<double>> M3(mid, std::vector<double>(mid, 0));

    // Fill the temporary matrix with A3^-1
    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            M3[i][j] = M[startRow + mid + i][startCol + mid + j];
        }
    }
    cilk_sync;
    multiplyMatrices(M3, M, 0, 0, startRow + mid, startCol, mid);                                // M3 = A3^-1 * (A_2 * A_1^-1)
    cilk_for (int i = 0; i < mid; i++) {
        cilk_for (int j = 0; j < mid; j++) {
            M[startRow + mid + i][startCol + j] = -M3[i][j];                                             // A3^-1 * (A_2 * A_1^-1)
        }
    }
}

/**
 * Function to perform matrix multiplication
 * M1: The first matrix - this matrix will be updated with the result
 * M2: The second matrix
 * Computation: M1 * M2
*/
void multiplyMatricesSerial(std::vector<std::vector<double>>& M1, std::vector<std::vector<double>>& M2, int startRowM1, int startColM1, int startRowM2, int startColM2, int size) {
    // Parallelized matrix multiplication
    std::vector<std::vector<double>> temp(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += M1[startRowM1 + i][startColM1 + k] * M2[startRowM2 + k][startColM2 + j];
            }
            temp[i][j] = sum;
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            M1[startRowM1 + i][startColM1 + j] = temp[i][j];
        }
    }
    return;
}

/**
 * Function to get the inverse of a n*n lower triangular matrix
 * Used to compute A_1^-1 and A_3^-1 to then obtain A^-1
*/
void getLowerTriangularInverseSerial(std::vector<std::vector<double>> &M, int B, int startRow, int startCol, int endRow, int endCol) {
    // Base Case: Forward substitution when the matrix is small enough
    if (endRow - startRow <= B) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j <= i; j++) {
                if (i == j) {
                    M[i][j] = 1 / M[i][j];                                                                // Diagonal Elements
                }
                else {
                    double sum = 0;
                    for (int k = j; k < i; k++) {
                        sum += M[i][k] * M[k][j];                                                         // Summation
                    }
                    M[i][j] = -sum / M[i][i];                                                             // Off-Diagonal Elements
                }
            }
        }
        return;
    }
    int mid = (endRow - startRow) / 2;

    // Computing A1^-1 and A3^-1
    getLowerTriangularInverseSerial(M, B, startRow, startCol, mid, mid);                     // A1 inverse
    getLowerTriangularInverseSerial(M, B, startRow + mid, startCol + mid, endRow, endCol);   // A3 inverse                                                                                        // Syncing Parallelism

    // Computing the multiplication for A2
    multiplyMatricesSerial(M, M, startRow + mid, startCol, startRow, startCol, mid);                    // A_2 * A_1^-1

    // Create a temporary matrix to hold A3
    std::vector<std::vector<double>> M3(mid, std::vector<double>(mid, 0));

    // Fill the temporary matrix with A3^-1
    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            M3[i][j] = M[startRow + mid + i][startCol + mid + j];
        }
    }

    multiplyMatricesSerial(M3, M, 0, 0, startRow + mid, startCol, mid);                                // M3 = A3^-1 * (A_2 * A_1^-1)
    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            M[startRow + mid + i][startCol + j] = -M3[i][j];                                             // A3^-1 * (A_2 * A_1^-1)
        }
    }
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

void correctnessTests() {
    std::cout << "First Correctness Check, B = 4:" << std::endl;
    std::vector<std::vector<double>> A1c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A1c_copy = A1c;
    getLowerTriangularInverseParallel(A1c, 4, 0, 0, 4, 4);
    multiplyMatricesParallel(A1c, A1c_copy, 0, 0, 0, 0, 4);
    printMatrix(A1c);

    std::cout << "Second Correctness Check, B = 2:" << std::endl;
    std::vector<std::vector<double>> A2c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A2c_copy = A2c;
    getLowerTriangularInverseParallel(A2c, 2, 0, 0, 4, 4);
    multiplyMatricesParallel(A2c, A2c_copy, 0, 0, 0, 0, 4);
    printMatrix(A2c);

    std::cout << "Third Correctness Check, B = 1:" << std::endl;
    std::vector<std::vector<double>> A3c = generateRandomMatrix(4);
    std::vector<std::vector<double>> A3c_copy = A3c;
    getLowerTriangularInverseParallel(A3c, 1, 0, 0, 4, 4);
    multiplyMatricesParallel(A3c, A3c_copy, 0, 0, 0, 0, 4);
    printMatrix(A3c);
}

void performanceTests() {
    std::vector<int> n = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> B = {32, 64, 128};

    // Open CSV file for writing results
    FILE *fp = fopen("results.csv", "w");
    fprintf(fp, "n,B,Parallel Time,Serial Time,Speedup\n");

    // Create LaTeX table for performance tests
    std::cout << "LaTeX Table for Performance Tests:" << std::endl;
    std::cout << "\\begin{table}[H]" << std::endl;
    std::cout << "\\caption{Parallel vs Serial Performance}" << std::endl;
    std::cout << "\\vspace{2mm}" << std::endl << "\\centering" << std::endl;
    std::cout << "\\begin{tabular}{|c|c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "n & B & Parallel Time & Serial Time & Speedup \\\\" << std::endl;
    std::cout << "\\hline" << std::endl;
    for (int i = 0; i < n.size(); i++) {
        std::vector<std::vector<double>> A = generateRandomMatrix(n[i]);
        for (int j = 0; j < B.size(); j++) {
            auto start = std::chrono::high_resolution_clock::now();
            getLowerTriangularInverseParallel(A, B[j], 0, 0, n[i], n[i]);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;

            start = std::chrono::high_resolution_clock::now();
            getLowerTriangularInverseSerial(A, B[j], 0, 0, n[i], n[i]);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diffSerial = end - start;
            std::cout << n[i] << " & " << B[j] << " & " << diff.count() << " & " << diffSerial.count() << " & " << diffSerial.count() / diff.count() << " \\\\" << std::endl;
            fprintf(fp, "%d,%d,%f,%f,%f\n", n[i], B[j], diff.count(), diffSerial.count(), diffSerial.count() / diff.count());
        }
    }
    std::cout << "\\hline" << std::endl;
    std::cout << "\\end{tabular}" << std::endl;
    std::cout << "\\end{table}" << std::endl;
}

void unimodularExamples() {
    std::cout << "Inverse of A1 Example:" << std::endl;
    std::vector<std::vector<double>> A1 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {-1, -1, 1, 0}, {-1, -1, -1, 1}};
    getLowerTriangularInverseParallel(A1, 1, 0, 0, 4, 4);
    printMatrix(A1);

    std::cout << "Inverse of A2 Example:" << std::endl;
    std::vector<std::vector<double>> A2 = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {1, 1, -1, 1}};
    getLowerTriangularInverseParallel(A2, 2, 0, 0, 4, 4);
    printMatrix(A2);

    std::cout << "Inverse of A3 Example:" << std::endl;
    std::vector<std::vector<double>> A3 = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}};
    getLowerTriangularInverseParallel(A3, 4, 0, 0, 4, 4);
    printMatrix(A3);
}

int main() {
    srand(time(0));
    unimodularExamples();
    correctnessTests();
    performanceTests();
}