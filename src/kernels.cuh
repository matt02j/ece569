///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : kernels.cuh
//                   :
// Create Date:      : 8 May 2017
// Modified          : 26 March 2019
//                   :
// Description:      : Header for kernels
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef KERNELS_H
#define KERNELS_H

#include "const.cuh"

#ifdef __linux
extern __constant__ unsigned d_matrix_flat[5184];
#else
__constant__ unsigned d_matrix_flat[5184];
#endif

// Message from channel copied into variable node to check node array.
__global__ void DataPassGB_0(int * VtoC, unsigned * Receivedword, unsigned * Interleaver, unsigned N, unsigned num_branches);

// for iterations between 1 and 15, this kernel launches to pass the message from variables nodes onto 
// the four check nodes it is connected to.
__global__ void DataPassGB_1(int* VtoC, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches);

// for iterations greater than 15, this kernel launches to pass the message from variables nodes onto the four 
// check nodes it is connected to.
__global__ void DataPassGB_2(int* VtoC, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches, unsigned varr);

// This kernel is launched to check if the CtoV copies the same information as VtoC depending upon the signe value
__global__ void CheckPassGB(int* CtoV, int* VtoC, unsigned M, unsigned num_branches);

// The following kernel is launched to decide each check node's decision whether the corresponding variable nodes 
// are in error or not.
__global__ void APP_GB(unsigned* Decide, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches);

//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(int * Synd, unsigned * Decide, unsigned M, unsigned num_branches, unsigned N);


__global__ void NestedFor(unsigned char* MatG_D, unsigned char* U_D, unsigned k, unsigned N);

__global__ void histogram_private_kernel(unsigned *bins, unsigned num_elements, unsigned num_bins);

#endif
