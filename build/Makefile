objects = PGaB.o kernels.o utils.o

all: $(objects)
	nvcc -g -lineinfo $(objects) -o PGaB -lm -std=c++11

%.o: ../src/%.cu
	nvcc -I ../src/ -dc $< -o $@ -std=c++11

clean:
	rm PGaB *.o
