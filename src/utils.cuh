///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
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

// free in 2d (int)
void free2d(unsigned** mem, const unsigned depth);

// free in 2d (unsigned)
void free2d(int** mem, const unsigned depth);

// 
unsigned GaussianElimination_MRB(int* Perm, int** MatOut, int** Mat, unsigned M, unsigned N);

// 
unsigned long diff_time_usec(struct timeval start, struct timeval stop);

// Initialize the NtoB matrix then unroll it into the interleaved matrix
// TODO could possibly due with an improvement in the NtoB initialization as the current method seems kinda hacky
// return num_branches
void initInterleaved(unsigned* h_interleaver, unsigned** data_matrix, const unsigned* rowRanks, const unsigned* histogram, const unsigned depth, const unsigned max_val);

// read in row rank matrix from local file
void readRowRanks(unsigned* rowRanks, const unsigned depth, const char* fileName);

// read in data matrix from local file
void readDataMatrix(unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const char* fileName);

// Histogram
void histogram(unsigned* histogram, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned max_val);

// unroll the data matrix
void unrollMatrix(unsigned* unrolledMatrix, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned num_branches);

#endif
