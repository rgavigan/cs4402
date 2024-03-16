
GCC = g++

GCCFLAGS = -c

NVCC = nvcc

### NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin//gcc-4.8
NVCCFLAGS = -w -c -O2


RM = rm -f

OBJ = $(SRCCC:.c=.o) $(CUFILES:.cu=.o)

all: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXECUTABLE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

clean:
	$(RM) *.o *~ *.linkinfo a.out *.log $(EXECUTABLE)
