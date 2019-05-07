objects = PGaB.o kernels.o utils.o

all: $(objects)
	nvcc -g -lineinfo $(objects) -o PGaB -lm -std=c++11 -Xcompiler -Wall -Xcompiler -fopenmp -lgomp -Xcompiler -lpthread

%.o: ../src/%.cu ../src/const.cuh
	nvcc -g -I ../src/ -dc $< -o $@ -std=c++11 -Xcompiler -Wall -Xcompiler -fopenmp -lgomp -Xcompiler -lpthread
	
serial: ../src/PGaB.c
	gcc -o PGaB_serial ../src/PGaB.c -lm

clean:
	rm PGaB *.o
