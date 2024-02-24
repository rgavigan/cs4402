#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cilk/cilk.h>

using namespace std;

// Point structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Helper to compare points by x-coordinate
bool compareX(const Point& a, const Point& b) {
    return a.x < b.x;
}

// Helper to compare points by y-coordinate
bool compareY(const Point& a, const Point& b) {
    return a.y < b.y;
}

// Function to calculate the distance between two points
double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Brute force algorithm to find the closest pair of points
double bruteForce(const vector<Point>& points, int start, int end) {
    double minDist = numeric_limits<double>::max();
    for (int i = start; i < end; ++i) {
        for (int j = i + 1; j < end; ++j) {
            minDist = min(minDist, distance(points[i], points[j]));
        }
    }
    return minDist;
}

// Utility function to find the closest pair of points in the strip
double closestPairUtil(const vector<Point>& points, vector<Point>& strip, double d) {
    sort(strip.begin(), strip.end(), compareY);

    double minDist = d;
    for (size_t i = 0; i < strip.size(); ++i) {
        for (size_t j = i + 1; j < strip.size() && (strip[j].y - strip[i].y) < minDist; ++j) {
            minDist = min(minDist, distance(strip[i], strip[j]));
        }
    }

    return minDist;
}

double closestPairUtilParallel(const vector<Point>& points, int start, int end) {
    if (end - start <= 1) {
        return numeric_limits<double>::max();
    }

    if (end - start == 2) {
        return distance(points[start], points[start + 1]);
    }

    int mid = (start + end) / 2;
    double dL = cilk_spawn closestPairUtil(points, start, mid);
    double dR = closestPairUtil(points, mid, end);
    cilk_sync;

    double d = min(dL, dR);

    vector<Point> strip;
    for (int i = start; i < end; i++) {
        if (fabs(points[i].x - points[mid].x) < d) {
            strip.push_back(points[i]);
        }
    }

    sort(strip.begin(), strip.end(), compareY);

    for (size_t i = 0; i < strip.size(); ++i) {
        for (size_t j = i + 1; j < strip.size() && (strip[j].y - strip[i].y) < d; ++j) {
            d = min(d, distance(strip[i], strip[j]));
        }
    }

    return d;
}

// Divide-and-conquer algorithm to find the closest pair of points
double closestPair(vector<Point>& points, int start, int end) {
    if (end - start <= 3) {
        return bruteForce(points, start, end);
    }

    int mid = (start + end) / 2;
    double dL = closestPair(points, start, mid);
    double dR = closestPair(points, mid, end);
    double d = min(dL, dR);

    vector<Point> strip;
    for (int i = start; i < end; i++) {
        if (fabs(points[i].x - points[mid].x) < d) {
            strip.push_back(points[i]);
        }
    }

    return min(d, closestPairUtil(points, strip, d));
}

// Wrapper function to call the closestPair function
double closestPairWrapper(vector<Point>& points) {
    sort(points.begin(), points.end(), compareX);
    return closestPair(points, 0, points.size());
}

// Parallel divide-and-conquer algorithm to find the closest pair of points
double closestPairParallel(vector<Point>& points, int start, int end, int depth) {
    if (end - start <= 3) {
        return bruteForce(points, start, end);
    }

    if (depth == 0) {
        // Sequential algorithm for small subproblems
        return closestPair(points, start, end);
    }

    int mid = (start + end) / 2;
    double dL, dR;

    dL = cilk_spawn closestPairParallel(points, start, mid, depth - 1);
    dR = closestPairParallel(points, mid, end, depth - 1);
    cilk_sync;

    double d = min(dL, dR);

    vector<Point> strip;
    for (int i = start; i < end; i++) {
        if (fabs(points[i].x - points[mid].x) < d) {
            strip.push_back(points[i]);
        }
    }

    return min(d, closestPairUtil(points, strip, d));
}

double closestPairParallelBetter(vector<Point>& points, int start, int end, int depth) {
    if (end - start <= 3) {
        return bruteForce(points, start, end);
    }

    if (depth == 0) {
        // Sequential algorithm for small subproblems
        return closestPairUtil(points, start, end);
    }

    int mid = (start + end) / 2;
    double dL, dR;

    cilk_spawn dL = closestPairParallel(points, start, mid, depth - 1);
    dR = closestPairParallel(points, mid, end, depth - 1);
    cilk_sync;

    double d = min(dL, dR);

    return min(d, closestPairUtil(points, start, end));
}

// Wrapper function to call the closestPairParallel function
double closestPairWrapperParallel(vector<Point>& points, int num_threads) {
    sort(points.begin(), points.end(), compareX);
    return closestPairParallel(points, 0, points.size(), log2(num_threads));
}

// Wrapper function to call the closestPairParallelBetter function
double closestPairWrapperParallel(vector<Point>& points, int num_threads) {
    sort(points.begin(), points.end(), compareX);
    return closestPairParallelBetter(points, 0, points.size(), log2(num_threads));
}

int main() {
    // Example usage
    vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
    
    int num_threads = 8;
    double closestDistance = closestPairWrapper(points);
    double closestDistanceParallel = closestPairWrapperParallel(points, num_threads);
    double closestDistanceParallelBetter = closestPairWrapperParallelBetter(points, num_threads);

    cout << "Closest pair distance: " << closestDistance << endl;
    cout << "Closest pair distance (parallel): " << closestDistanceParallel << endl;

    return 0;
}
