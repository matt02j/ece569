///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : const.cuh
//                   :
// Create Date:      : 8 May 2017
// Modified          : 26 March 2019
//                   :
// Description:      : Constants required for the program
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CONST_H
#define CONST_H

#include <cuda.h>
#include <math.h> 
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#define arrondi(x)((ceil(x) - x) < (x - floor(x)) ? (int) ceil(x) : (int) floor(x))
#define min(x, y)((x) < (y) ? (x) : (y))
#define signf(x)((x) >= 0 ? 0 : 1)
#define max(x, y)((x) < (y) ? (y) : (x))
#define SQR(A)((A) * (A))
#define BPSK(x)(1 - 2 * (x))

constexpr double PI = 3.1415926536;
constexpr unsigned M = 648;
constexpr unsigned N = 1296;

#endif
