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

extern __constant__ int Mat_device[5184];

// Message from channel copied into variable node to check node array.
__global__ void DataPassGB_0(int * VtoC, int * Receivedword, int * Interleaver, int N, int NbBranch);

// for iterations between 1 and 15, this kernel launches to pass the message from variables nodes onto 
// the four check nodes it is connected to.
__global__ void DataPassGB_1(int* VtoC, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch);

// for iterations greater than 15, this kernel launches to pass the message from variables nodes onto the four 
// check nodes it is connected to.
__global__ void DataPassGB_2(int* VtoC, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch, int varr);

// This kernel is launched to check if the CtoV copies the same information as VtoC depending upon the signe value
__global__ void CheckPassGB(int* CtoV, int* VtoC, int M, int NbBranch);

// The following kernel is launched to decide each check node's decision whether the corresponding variable nodes 
// are in error or not.
__global__ void APP_GB(int* Decide, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch);

//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(int * Synd, int * Decide, int M, int NbBranch,int N);

__global__ void NestedFor(unsigned char* MatG_D, unsigned char* U_D, int k, int N);
#endif
