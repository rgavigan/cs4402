# Run inverseMatrix.cpp
inverseMatrix: inverseMatrix.cpp
	/opt/opencilk/bin/clang++ -fopencilk -O3 inverseMatrix.cpp -o inverseMatrix

# Run closestPairProblem.c
closestPairs: closestPairProblem.c
	/opt/opencilk/bin/clang -fopencilk -O3 closestPairProblem.c -o closestPairs

# Clean
clean:
	rm -f inverseMatrix closestPairs