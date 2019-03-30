///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : kernels.cu
//                   :
// Create Date:      : 8 May 2017
// Modified          : 26 March 2019
//                   :
// Description:      : GPU kernels
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "kernels.cuh"

__constant__ int Mat_device[5184];

// Message from channel copied into variable node to check node array.
__global__ void DataPassGB_0(int * VtoC, int * Receivedword, int * Interleaver, int N, int NbBranch) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < N) {

      // 
      unsigned node_idx = 0;

      // 
      unsigned strides = (NbBranch / N);

      // 
      unsigned i = Receivedword[id];

      for (unsigned stride = 0; stride < strides; stride++) {
         
         // get node index from interleaver
         node_idx = Interleaver[id * strides + stride];
         
         VtoC[node_idx] = i;
      }
   }
}

// for iterations between 1 and 15, this kernel launches to pass the message from variables nodes onto 
// the four check nodes it is connected to.
__global__ void DataPassGB_1(int* VtoC, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < N) {

      // 
      int buf = 0;

      // 
      int i = Receivedword[id];

      // 
      int Global = (1 - 2 * i);

      // Used to index the CtoV and VtoC node arrays
      unsigned node_idx = 0;

      // 
      unsigned strides = (NbBranch / N);

      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         // get node index from interleaver
         node_idx = Interleaver[id * strides + stride];

         // 
         Global += (-2) * CtoV[node_idx] + 1;
      }

      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         // get node index from interleaver
         node_idx = Interleaver[id * strides + stride];

         // 
         // 
         buf = Global - ((-2) * CtoV[node_idx] + 1);
         
         // 
         VtoC[node_idx] = (buf < 0)? 1 : ((buf > 0)? 0 : i);
      }
   }
}

// for iterations greater than 15, this kernel launches to pass the message from variables nodes onto the four 
// check nodes it is connected to.
__global__ void DataPassGB_2(int* VtoC, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch, int varr) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < N) {

      // 
      int buf;
      
      // 
      int i = Receivedword[id];

      // 
      int Global = (1 - 2 * (varr ^ i));

      // 
      unsigned node_idx = 0;

      //
      unsigned strides = (NbBranch / N);

      //
      for (unsigned stride = 0; stride < strides; stride++) {

         // calculate node index
         node_idx = Interleaver[id * strides + stride];

         Global += (-2) * CtoV[node_idx] + 1;
      }

      // 
      for (unsigned stride = 0; stride < strides; stride++) {
         
         // calculate node index
         node_idx = Interleaver[id * strides + stride];

         // 
         // 
         buf = Global - ((-2) * CtoV[node_idx] + 1);

         // 
         VtoC[node_idx] = (buf < 0)? 1 : ((buf > 0)? 0 : i);
      }
   }
}

// This kernel is launched to check if the CtoV copies the same information as VtoC depending upon the signe value
__global__ void CheckPassGB(int* CtoV, int* VtoC, int M, int NbBranch) {
  
  	extern __shared__ int vtoc[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;


   for(int k=threadIdx.x; k<NbBranch; k+=blockDim.x){
	vtoc[k]=VtoC[k];
   }

   if (id < M) {

      int signe = 0;

      // For indexing the node arrays
      unsigned node_idx = 0;

      // 
      unsigned strides = (NbBranch / M);
      
      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         node_idx = stride + id * strides;
         signe ^= vtoc[node_idx];
      }
      
      // 
      for (unsigned stride = 0; stride < strides; stride++) {
         
         node_idx = stride + id * strides;
         CtoV[node_idx] = signe ^ vtoc[node_idx];
      }
   }
}

// The following kernel is launched to decide each check node's decision whether the corresponding variable nodes 
// are in error or not.
__global__ void APP_GB(int* Decide, int* CtoV, int* Receivedword, int* Interleaver, int N, int NbBranch) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < N) {

      // 
      int i = Receivedword[id];

      // 
      int Global = (1 - 2 * i);

      // Used to index the node array
      unsigned node_idx = 0;

      // 
      unsigned strides = (NbBranch / N);

      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         // TODO this is not coalesced at all
         node_idx = Interleaver[id * strides + stride];
         Global += (-2) * CtoV[node_idx] + 1;
      }
      
      // 
      Decide[id] = (Global < 0)? 1 : ((Global > 0)? 0 : i);
   }
}

//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(int * Synd, int * Decide, int M, int NbBranch,int N) {
	extern __shared__ int decide[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // intialize ___ regardless of bounds...
   int synd = 0;
   for(int k=threadIdx.x; k<N; k+=blockDim.x){
	decide[k]=Decide[k];
   }
   if (id < M) {
      
      unsigned strides = (NbBranch / M);
	int i=id * strides;
      // 
      for (unsigned stride = 0; stride < strides; stride++) {
         synd ^=decide[Mat_device[i + stride]];
      }
   }

   // NOTE write back regardless of thread
   Synd[id]=synd;
}

__global__ void NestedFor(int* MatG_D, int* U_D, int k, int N){
   	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = id*N;
	for(int i=k+1;i<N;i++){
		if(id <= k){
			//  0:k      0:k            0:k     k+1:N    k+1:N
			U_D[id] = U_D[id] ^ (MatG_D[stride + i] * U_D[i]);
		}
	}

	for(int i=k; i>0;i--){
		if(id < i){
			U_D[id] = U_D[id] ^ (MatG_D[stride + i] * U_D[i]);
		}
	}
}
