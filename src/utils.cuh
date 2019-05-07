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
#include <pthread.h>
#include <cuda.h>
#include <curand_kernel.h>

#ifdef PROFILE
#include <sys/time.h>
#endif

typedef struct lots_of_pointers{
unsigned char* d_bit_stream;
unsigned char* d_MatG;
unsigned char* d_messageRecieved;
unsigned char* h_messageRecieved;
unsigned char* d_intermediate;
unsigned char* d_VtoC;
unsigned char* d_CtoV;
unsigned * d_interleaver;
unsigned char* d_decoded;
unsigned char* h_decoded;
unsigned char* d_synd;
unsigned char* h_synd;
unsigned * d_PermG;
unsigned char* d_varr;
curandState* devStates;
unsigned rank;
unsigned N;
unsigned M;
float alpha;
unsigned num_branches;
unsigned* NiterMax;
unsigned* NiterMoy;
unsigned* err_total_count;
unsigned* missed_error_count;
unsigned* Dmin;
unsigned* bit_error_count;
unsigned* nb; //frame count
int id;
cudaStream_t stream;
} myargs;




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
