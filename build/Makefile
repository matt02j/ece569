all: PGaB GaB

PGaB: ../src/PGaB.cu
	nvcc -g -lineinfo -o PGaB ../src/PGaB.cu -lm -Xcompiler -fopenmp -lgomp
GaB: ../src/GaB.cu
	nvcc -g -lineinfo -o GaB ../src/GaB.cu -lm
