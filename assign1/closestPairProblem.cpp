#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define the number of points
#define n 128

/**
 * Helper to calculate the distance between two points
*/
double distance(int x1, int y1, int x2, int y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/** Function to find the closest pair of points in a 2D plane
 *  Input: 2d array of points (x, y) - sorted by their x values
 *  Output: the pair of points that are closest to each other [(x1, y1), (x2, y2)]
 */
void closestPairs(int points[][2]) {
    // Base case: if there are only two points, return them
    if (n == 2) {
        printf("Closest pair of points: (%d, %d), (%d, %d)\n", points[0][0], points[0][1], points[1][0], points[1][1]);
        return;
    }

    // Split the array into two halves
    double left[n / 2][2];
    double right[n / 2][2];
    for (int i = 0; i < n / 2; i++) {
        left[i][0] = points[i][0];
        left[i][1] = points[i][1];
        right[i][0] = points[i + n / 2][0];
        right[i][1] = points[i + n / 2][1];
    }

    // Recursively find the closest pair in each half
    double res1[2][2];
    double res2[2][2];
    findClosestPair(left, res1);
    findClosestPair(right, res2);

    // Print out res1 and res2
    printf("Closest pair of points in left half: (%d, %d), (%d, %d)\n", res1[0][0], res1[0][1], res1[1][0], res1[1][1]);
    printf("Closest pair of points in right half: (%d, %d), (%d, %d)\n", res2[0][0], res2[0][1], res2[1][0], res2[1][1]);
}

void findClosestPair(double points[][2], double res[2][3]) {
    // Print the points
    printf("Points: ");
    for (int i = 0; i < n; i++) {
        printf("(%d, %d) ", points[i][0], points[i][1]);
    }


    // Find the closest pair of points in the array recursively with sqrt((xj - xi)^2 + (yj - yi)^2)
    double minDistance = distance(points[0][0], points[0][1], points[1][0], points[1][1]);
    int minIndex1 = 0;
    int minIndex2 = 1;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dist = distance(points[i][0], points[i][1], points[j][0], points[j][1]);
            if (dist < minDistance) {
                minDistance = dist;
                minIndex1 = i;
                minIndex2 = j;
            }
        }
    }
}

/**
 * Helper to sort the array of points by their x-values before calling the closestPairs function
 * Necessary for the closestPairs function to work with medians
*/
void sortArrayByX(double points[][2]) {
    // Sort the points by their x-values
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (points[i][0] > points[j][0]) {
                int tempX = points[i][0];
                int tempY = points[i][1];
                points[i][0] = points[j][0];
                points[i][1] = points[j][1];
                points[j][0] = tempX;
                points[j][1] = tempY;
            }
        }
    }
}

int main() {
    // Random initialization
    srand(time(0));

    // Create an array of points
    double points[n][2];
    for (int i = 0; i < n; i++) {
        points[i][0] = i + rand() % 1000;
        points[i][1] = i + rand() % 1000;
    }

    // Sort points by their x-values
    sortArrayByX(points);

    // Call closest pairs on the points
    closestPairs(points);
}