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

#ifdef __linux
__constant__ unsigned d_matrix_flat[5184];
#endif


// Message from channel copied into variable node to check node array.
__global__ void DataPassGB_0(int * VtoC, unsigned * Receivedword, unsigned * Interleaver, unsigned N, unsigned num_branches) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < N) {

      // 
      unsigned node_idx = 0;

      // 
      unsigned strides = (num_branches / N);

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
__global__ void DataPassGB_1(int* VtoC, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches) {
   
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
      unsigned strides = (num_branches / N);

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
__global__ void DataPassGB_2(int* VtoC, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches, unsigned varr) {
   
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
      unsigned strides = (num_branches / N);

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
__global__ void CheckPassGB(int* CtoV, int* VtoC, unsigned M, unsigned num_branches) {
  
  	extern __shared__ int vtoc[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;


   for(int k=threadIdx.x; k<num_branches; k+=blockDim.x){
	vtoc[k]=VtoC[k];
   }

   if (id < M) {

      int signe = 0;

      // For indexing the node arrays
      unsigned node_idx = 0;

      // 
      unsigned strides = (num_branches / M);
      
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
__global__ void APP_GB(unsigned* Decide, int* CtoV, unsigned* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches) {
   
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
      unsigned strides = (num_branches / N);

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
__global__ void ComputeSyndrome(int * Synd, unsigned * Decide, unsigned M, unsigned num_branches, unsigned N) {
	extern __shared__ int decide[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // intialize ___ regardless of bounds...
   int synd = 0;
   for(int k=threadIdx.x; k<N; k+=blockDim.x){
	decide[k]=Decide[k];
   }
   if (id < M) {
      
      unsigned strides = (num_branches / M);
	int i=id * strides;
      // 
      for (unsigned stride = 0; stride < strides; stride++) {
         synd ^=decide[d_matrix_flat[i + stride]];
      }
   }

   // NOTE write back regardless of thread
   Synd[id]=synd;
}

//assumes a single block is running // matg access is not coalesced

__global__ void NestedFor(unsigned char* MatG_D, unsigned char* U_D, unsigned k, unsigned N){

   	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = id*N;

	extern __shared__ unsigned char u[]; 
	
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		u[i]=U_D[i];
	}
	__syncthreads();
	for(int i=k+1;i<N;i++){
		if(id <= k){
			//  0:k      0:k            0:k     k+1:N    k+1:N
			u[id] = u[id] ^ (MatG_D[stride + i] * u[i]);
		}
	}
	__syncthreads();
	for(int i=k; i>0;i--){
		if(id < i){
			u[id] = u[id] ^ (MatG_D[stride + i] * u[i]);
		}
	}

	__syncthreads();
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		U_D[i]=u[i];
	}

}
