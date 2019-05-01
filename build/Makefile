objects = PGaB.o kernels.o utils.o

all: $(objects)
	nvcc -g -G -lineinfo $(objects) -o PGaB -lm -std=c++11 -Xcompiler -Wall

%.o: ../src/%.cu ../src/*.cuh
	nvcc -g -G -I ../src/ -dc $< -o $@ -std=c++11 -Xcompiler -Wall

clean:
	rm PGaB *.o
