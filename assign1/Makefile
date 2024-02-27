# Run inverseMatrix.cpp
inverseMatrix: inverseMatrix.cpp
	/opt/opencilk/bin/clang++ -fopencilk -O3 -g inverseMatrix.cpp -o inverseMatrix

# Run closestPairs.c
closestPairs: closestPairs.cpp
	/opt/opencilk/bin/clang++ -fopencilk -O3 -g closestPairs.cpp -o closestPairs

# Clean
clean:
	rm -f inverseMatrix closestPairs

# Test
testInverse: inverseMatrix
	CILK_NWORKERS=$(nproc) ./inverseMatrix

testClosest: closestPairs
	CILK_NWORKERS=$(nproc) ./closestPairs