objects = PGaB.o kernels.o utils.o

all: $(objects)
	nvcc -g -lineinfo $(objects) -o PGaB -lm -std=c++11 -Xcompiler -Wall

%.o: ../src/%.cu
	nvcc -I ../src/ -dc $< -o $@ -std=c++11 -Xcompiler -Wall

clean:
	rm PGaB *.o
