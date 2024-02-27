#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <math.h>

#define basecase 4096

/**
 * Point structure to represent a point in 2D space
*/
struct Point {
    double x, y;

    bool operator<(const Point &p) const {
        return y < p.y;
    }

    bool operator>(const Point &p) const {
        return y > p.y;
    }

    bool operator<=(const Point &p) const {
        return y <= p.y;
    }

    bool operator>=(const Point &p) const {
        return y >= p.y;
    }
};

/**
 * Calculates the distance between two points, pi and pj
*/
double distance(Point pi, Point pj) {
    return sqrt(pow(pj.x - pi.x, 2) + pow(pj.y - pi.y, 2));
}

/**
 * Sort an array of points by y-coordinate
*/
std::vector<Point> sortY(std::vector<Point> points) {
    std::sort(points.begin(), points.end(), [](Point a, Point b) {
        return a.y < b.y;
    });
    return points;
}

/**
 * Closest Pair Problem: Given n points, divide and conquer algorithm to find closest pair of points
 * Param: points - Array of points where all coordinates are pairwise distinct (unique)
 * Returns: Array of two points that are closest to each other
*/
std::vector<Point> closestPair(std::vector<Point>& points, int start, int end) {
    // Base Case: If there are only two points, return them
    std::vector<Point> pair;
    if (end - start == 2) {
        pair.push_back(points[start]);
        pair.push_back(points[end]);
        return pair;
    }

    // Find value x that divides the set of points into two equal halves
    int half = (start + end) / 2;
    double x = (points[half - 1].x + points[half].x) / 2;

    // Recursively find closest pair in left and right side (L and R)
    std::vector<Point> L = closestPair(points, start, half);
    std::vector<Point> R = closestPair(points, half, end);
    double dL = distance(L[0], L[1]);
    double dR = distance(R[0], R[1]);

    // Check if L or R is the closest pair
    double d;
    if (dL < dR) {
        pair = L;
        d = dL;
    } else {
        pair = R;
        d = dR;
    }

    // Discard all points with xi < x - d or xi > x + d
    std::vector<Point> S;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].x >= x - d && points[i].x <= x + d) {
            S.push_back(points[i]);
        }
    }

    // Sort S by y-coordinate
    S = sortY(S);

    // Go through S and for each point, compare it to the next 6 points. Save the closest pair as a point
    for (int i = 0; i < S.size(); i++) {
        for (int j = i + 1; j < i + 7 && j < S.size(); j++) {
            if (distance(S[i], S[j]) < d) {
                d = distance(S[i], S[j]);
                pair[0] = S[i];
                pair[1] = S[j];
            }
        }
    }
    return pair;
}

/**
 * Closest Pair Problem: Given n points, divide and conquer algorithm to find closest pair of points
 * Param: points - Array of points where all coordinates are pairwise distinct (unique)
 * Returns: Array of two points that are closest to each other
*/
std::vector<Point> closestPairNaiveParallel(std::vector<Point> &points, int start, int end) {
    // Base Case: If there are only two points, return them
    std::vector<Point> pair;
    if (end - start == 2) {
        pair.push_back(points[start]);
        pair.push_back(points[end]);
        return pair;
    }

    // Find value x that divides the set of points into two equal halves
    int half = (start + end) / 2;
    double x = (points[half - 1].x + points[half].x) / 2;

    // Recursively find closest pair in left and right side (L and R)
    std::vector<Point> L = cilk_spawn closestPairNaiveParallel(points, start, half);
    std::vector<Point> R = cilk_spawn closestPairNaiveParallel(points, half, end);
    cilk_sync;
    double dL = distance(L[0], L[1]);
    double dR = distance(R[0], R[1]);

    // Check if L or R is the closest pair
    double d;
    if (dL < dR) {
        pair = L;
        d = dL;
    } else {
        pair = R;
        d = dR;
    }

    // Discard all points with xi < x - d or xi > x + d
    std::vector<Point> S;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].x >= x - d && points[i].x <= x + d) {
            S.push_back(points[i]);
        }
    }

    // Sort S by y-coordinate
    S = sortY(S);

    // Go through S and for each point, compare it to the next 6 points. Save the closest pair as a point
    for (int i = 0; i < S.size(); i++) {
        for (int j = i + 1; j < i + 7 && j < S.size(); j++) {
            if (distance(S[i], S[j]) < d) {
                d = distance(S[i], S[j]);
                pair[0] = S[i];
                pair[1] = S[j];
            }
        }
    }
    return pair;
}

void merge(Point* data, int istart, int mid, int iend) {
    int n = iend - istart + 1;
    int k = mid - istart + 1;
    int m = n - k;
    int indexA = 0, indexB = 0, indexC = 0;
    int i;
    Point* a = &data[istart];
    Point* b = &data[mid + 1];
    Point* c = (Point*)malloc(sizeof(Point) * n);

    while (indexA < k && indexB < m) {
        if (a[indexA].x <= b[indexB].x) {
            c[indexC] = a[indexA];
            indexA++;
            indexC++;
        } else {
            c[indexC] = b[indexB];
            indexB++;
            indexC++;
        }
    }

    if (indexA >= k) {
        for (i = 0; i < n - indexC; i++) {
            c[indexC + i] = b[indexB + i];
        }
    } else {
        for (i = 0; i < n - indexC; i++) {
            c[indexC + i] = a[indexA + i];
        }
    }

    for (i = 0; i < n; i++) {
        data[istart + i] = c[i];
    }

    free(c);
}

int binarysearch(Point* data, int low, int high, int key) {
    int middle;
    int istart = low, iend = high + 1;
    while (istart < iend) {
        middle = istart + (iend - istart) / 2;
        if (data[middle].x <= key)
            istart = middle + 1;
        else
            iend = middle;
    }
    return istart;
}

void parallel_submerge(Point* c, Point* data, int lowx, int highx, int lowy, int highy, int sp) {
    int k = highx - lowx + 1;
    int m = highy - lowy + 1;
    int indexA = 0, indexB = 0, indexC = sp;
    int i;
    Point* a = &data[lowx];
    Point* b = &data[lowy];

    while (indexA < k && indexB < m) {
        if (a[indexA].x <= b[indexB].x) {
            c[indexC] = a[indexA];
            indexA++;
            indexC++;
        } else {
            c[indexC] = b[indexB];
            indexB++;
            indexC++;
        }
    }

    if (indexA >= k) {
        for (i = 0; i < m - indexB; i++) {
            c[indexC + i] = b[indexB + i];
        }
    } else {
        for (i = 0; i < k - indexA; i++) {
            c[indexC + i] = a[indexA + i];
        }
    }
}


int parallel_merge(Point* c, Point* a, int lowx, int highx, int lowy, int highy, int sp) {
    int mx, my, p, lx, ly;
    lx = highx - lowx + 1;
    ly = highy - lowy + 1;
    if (ly <= basecase || lx <= basecase) {
        parallel_submerge(c, a, lowx, highx, lowy, highy, sp);
    } else if (lx < ly)
        parallel_merge(c, a, lowy, highy, lowx, highx, sp);
    else if (lx == 0) {
        return 0;
    } else {
        mx = (lowx + highx) / 2;
        my = binarysearch(a, lowy, highy, a[mx].x);
        p = sp + mx - lowx + my - lowy;
        c[p] = a[mx];
        // Assuming parallel_merge is Cilk Plus-style, adjust accordingly if using a different parallelization method
        cilk_spawn parallel_merge(c, a, lowx, mx - 1, lowy, my - 1, sp);
        parallel_merge(c, a, mx + 1, highx, my, highy, p + 1);
    }
    return 0;
}

void serial_mergesort(Point* data, int istart, int iend) {
    if (istart < iend) {
        int mid = (istart + iend) / 2;
        serial_mergesort(data, istart, mid);
        serial_mergesort(data, mid + 1, iend);
        merge(data, istart, mid, iend);
    }
}

void parallel_mergesort(Point* c, Point* data, int istart, int iend, int BASE) {
    if (iend - istart <= BASE) {
        serial_mergesort(data, istart, iend);
    } else if (istart < iend) {
        int mid = (istart + iend) / 2;
        cilk_spawn parallel_mergesort(c, data, istart, mid, BASE);
        parallel_mergesort(c, data, mid + 1, iend, BASE);
        cilk_sync;
        parallel_merge(c, data, istart, mid, mid + 1, iend, istart);
        for (int i = istart; i <= iend; i++)
            data[i] = c[i];
    }
}

/**
 * Closest Pair Problem: Given n points, divide and conquer algorithm to find closest pair of points
 * Param: points - Array of points where all coordinates are pairwise distinct (unique)
 * Returns: Array of two points that are closest to each other
 * Parallelizes the merge to get to parallelism of O(n/logn)
*/
std::vector<Point> closestPairIdealParallel(std::vector<Point> &points, int start, int end) {
    if (end - start == 2) {
        return {points[start], points[end]};
    }

    // Find value x that divides the set of points into two equal halves
    int half = (start + end) / 2;
    double x = (points[half - 1].x + points[half].x) / 2;

    // Recursively find closest pair in left and right side (L and R)
    std::vector<Point> L = cilk_spawn closestPairIdealParallel(points, start, half);
    std::vector<Point> R = cilk_spawn closestPairIdealParallel(points, half, end);
    cilk_sync;
    double dL = sqrt(pow(L[0].x - L[1].x, 2) + pow(L[0].y - L[1].y, 2));
    double dR = sqrt(pow(R[0].x - R[1].x, 2) + pow(R[0].y - R[1].y, 2));

    // Check if L or R is the closest pair
    std::vector<Point> pair;
    double d;
    if (dL < dR) {
        pair = L;
        d = dL;
    } else {
        pair = R;
        d = dR;
    }
    int pointsSize = points.size();

    // Discard points with xi < x - d or xi > x + d
    std::vector<Point> S;
    auto lowerBoundIt = std::lower_bound(points.begin() + start, points.begin() + end, x - d, [](const Point& p, double val) {
        return p.x < val;
    });
    auto upperBoundIt = std::upper_bound(points.begin() + start, points.begin() + end, x + d, [](double val, const Point& p) {
        return val < p.x;
    });

    std::copy(lowerBoundIt, upperBoundIt, std::back_inserter(S));

    // Sort by y-coordinate using parallel mergesort
    parallel_mergesort(&S[0], &S[0], 0, S.size() - 1, basecase);

    // Go through S and for each point, compare it to the next 6 points. Save the closest pair as a point
    for (int i = 0; i < S.size(); i++) {
        for (int j = i + 1; j < std::min(i + 7, static_cast<int>(S.size())); j++) {
            double dist = distance(S[i], S[j]);
            if (dist < d) {
                d = dist;
                pair = {S[i], S[j]};
            }
        }
    }
    return pair;
}


/**
 * Sort an array of points by x-coordinate
*/
std::vector<Point> sortX(std::vector<Point> points) {
    std::sort(points.begin(), points.end(), [](Point a, Point b) {
        return a.x < b.x;
    });
    return points;
}

/**
 * Produce a vector of n random points with distinct x and y coordinates
 * Values from 0.1 -> 10.0
*/
std::vector<Point> randomPoints(int n) {
    std::vector<Point> points;
    for (int i = 0; i < n; i++) {
        Point p;
        p.x = (double)rand() / RAND_MAX * 10.0;
        p.y = (double)rand() / RAND_MAX * 10.0;
        points.push_back(p);
    }
    return points;
}

/**
 * Brute force computation of closest pair of points for testing answer
*/
std::vector<Point> bruteForce(std::vector<Point> points) {
    std::vector<Point> pair;
    pair.push_back(points[0]);
    pair.push_back(points[1]);
    double min = 1000000;
    for (int i = 0; i < points.size(); i++) {
        for (int j = i + 1; j < points.size(); j++) {
            double d = distance(points[i], points[j]);
            if (d < min) {
                min = d;
                pair[0] = points[i];
                pair[1] = points[j];
            }
        }
    }
    return pair;
}

/**
 * Performance testing for closest pair problem for different values of n (powers of 2), comparing time
*/
void performanceTest() {
    // Open CSV file for writing results
    FILE *fp = fopen("resultsClosest.csv", "w");
    fprintf(fp, "n,B,Serial Time,Parallel Time,Improved Parallel Time\n");

    // Create LaTeX Table for performance testing
    std::cout << "\\begin{table}[H]" << std::endl;
    std::cout << "\\centering" << std::endl;
    std::cout << "\\begin{tabular}{|c|c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "n & Serial Time (s) & Parallel Time (s) & Improved Parallel Time (s) & Speedup \\\\" << std::endl;
    for (int n = 16; n <= 65536; n *= 2) {
        std::vector<Point> points = sortX(randomPoints(n));
        auto start = std::chrono::high_resolution_clock::now();
        closestPair(points, 0, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        double closestPairTime = time.count();

        start = std::chrono::high_resolution_clock::now();
        closestPairNaiveParallel(points, 0, n);
        end = std::chrono::high_resolution_clock::now();
        time = end - start;
        double closestPairParallelTime = time.count();

        start = std::chrono::high_resolution_clock::now();
        closestPairIdealParallel(points, 0, n);
        end = std::chrono::high_resolution_clock::now();
        time = end - start;
        double closestPairIdealParallelTime = time.count();

        std::cout << n << " & " << std::fixed << std::setprecision(6) << closestPairTime << " & " << closestPairParallelTime << " & " << closestPairIdealParallelTime << " & " << int(closestPairTime / closestPairIdealParallelTime) << " \\\\" << std::endl;
        fprintf(fp, "%d,%d,%f,%f,%f\n", n, 32, closestPairTime, closestPairParallelTime, closestPairIdealParallelTime);
    }
    std::cout << "\\hline" << std::endl;
    std::cout << "\\end{tabular}" << std::endl;
    std::cout << "\\caption{Performance Testing for Closest Pair Problem}" << std::endl;
    std::cout << "\\end{table}" << std::endl;
}


int main() {
    srand(time(0));
    int n = 16;
    std::vector<Point> points = sortX(randomPoints(n));
    for (int i = 0; i < points.size(); i++) {
        std::cout << "Point " << i << ": (" << points[i].x << ", " << points[i].y << ")" << std::endl;
    }

    std::vector<Point> closest = closestPair(points, 0, n);
    std::cout << "Closest Pair: (" << closest[0].x << ", " << closest[0].y << ") and (" << closest[1].x << ", " << closest[1].y << ")" << std::endl;

    std::vector<Point> closestParallel = closestPairNaiveParallel(points, 0, n);
    std::cout << "Closest Pair Parallel: (" << closestParallel[0].x << ", " << closestParallel[0].y << ") and (" << closestParallel[1].x << ", " << closestParallel[1].y << ")" << std::endl;

    std::vector<Point> closestIdealParallel = closestPairIdealParallel(points, 0, n);
    std::cout << "Closest Pair Ideal Parallel: (" << closestIdealParallel[0].x << ", " << closestIdealParallel[0].y << ") and (" << closestIdealParallel[1].x << ", " << closestIdealParallel[1].y << ")" << std::endl;

    std::vector<Point> brute = bruteForce(points);
    std::cout << "Brute Force: (" << brute[0].x << ", " << brute[0].y << ") and (" << brute[1].x << ", " << brute[1].y << ")" << std::endl;

    performanceTest();
}
