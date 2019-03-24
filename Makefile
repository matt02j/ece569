all: PGaB GaB

PGaB: PGaB.cu
	nvcc -g -lineinfo -o PGaB PGaB.cu -lm
GaB: GaB.cu
	nvcc -g -lineinfo -o GaB GaB.cu -lm
