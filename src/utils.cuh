///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matthew Johnson, Jeremy Sears, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : utils.cuh
//                   :
// Create Date:      : 8 May 2017
// Modified          : 26 March 2019
//                   :
// Description:      : Header for Utility functions for PGaB
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef UTIL_H 
#define UTIL_H

#include "const.cuh"

#ifdef PROFILE
#include <sys/time.h>
#endif

//cuda check
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

// free in 2d (int)
void free2d(unsigned** mem, const unsigned depth);

// free in 2d (unsigned)
void free2d(int** mem, const unsigned depth);

// 
unsigned GaussianElimination_MRB(unsigned* Perm, unsigned** MatOut, unsigned** Mat, unsigned M, unsigned N);


//#ifdef PROFILE
// 
unsigned long diff_time_usec(struct timeval start, struct timeval stop);
//#endif

// Initialize the NtoB matrix then unroll it into the interleaved matrix
// TODO could possibly due with an improvement in the NtoB initialization as the current method seems kinda hacky
// return num_branches
void initInterleaved(unsigned * h_interleaver, unsigned** data_matrix, const unsigned* rowRanks, const unsigned* histogram, const unsigned depth, const unsigned max_val);

// read in row rank matrix from local file
void readRowRanks(unsigned* rowRanks, const unsigned depth, const char* fileName);

// read in data matrix from local file
void readDataMatrix(unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const char* fileName);

// Histogram
void histogram(unsigned* histogram, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned max_val);

// unroll the data matrix
void unrollMatrix(unsigned* unrolledMatrix, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned num_branches);


void print_array_int(int* arr, int size);
void print_array_char(unsigned char* arr, int size);

#endif