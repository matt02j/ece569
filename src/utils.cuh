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
#include <sys/time.h>

// 
unsigned GaussianElimination_MRB(int* Perm, int** MatOut, int** Mat, int M, int N);

// 
unsigned long diff_time_usec(struct timeval start, struct timeval stop);

#endif
