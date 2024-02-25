# Run inverseMatrix.cpp
inverseMatrix: inverseMatrix.cpp
	/opt/opencilk/bin/clang++ -fopencilk -O2 -g inverseMatrix.cpp -o inverseMatrix

# Run closestPairs.c
closestPairs: closestPairs.cpp
	/opt/opencilk/bin/clang++ -fopencilk -O3 closestPairs.cpp -o closestPairs

# Clean
clean:
	rm -f inverseMatrix closestPairs

# Test
test: inverseMatrix
	CILK_NWORKERS=14 ./inverseMatrix