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
//TODO Interleaver shared
__global__ void DataPassGB_0(unsigned char * VtoC, unsigned char * Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches){
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

      unsigned strides = (num_branches / N);
	/*extern __shared__ unsigned interleave[]; //made it slower

	for(int i=0; i< strides; i+=1){
		interleave[threadIdx.x +i*blockDim.x] = Interleaver[id+i*N];
	}

	__syncthreads();*/
   if (id < N) { 

      // 
      unsigned node_idx = 0;

      // 
      unsigned i = Receivedword[id];

      for (int stride = 0; stride < strides; stride++) {
         
         // get node index from interleaver
         node_idx = Interleaver[N*stride+id];
         
         VtoC[node_idx] = i;
      }
   }
}

// for iterations between 1 and 15, this kernel launches to pass the message from variables nodes onto 
// the four check nodes it is connected to.
//TODO CtoV shared and Interleaver shared
__global__ void DataPassGB_1(unsigned char * VtoC, unsigned char * CtoV, unsigned char* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	/*extern __shared__ unsigned char data[];
	unsigned* interleave = (unsigned*)&data[num_branches];

	for(int i=threadIdx.x; i<num_branches; i+=blockDim.x){
		data[i]=CtoV[i];
	}
	for(int i=threadIdx.x; i<num_branches; i+=blockDim.x){
		interleave[i] = Interleaver[i];
	}

	__syncthreads();*/

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
      for (int stride = 0; stride < strides; stride++) {

         // get node index from interleaver
         node_idx = Interleaver[N*stride+id];

         // 
         Global += (-2) * CtoV[node_idx] + 1;
      }

      // 
      for (int stride = 0; stride < strides; stride++) {

         // get node index from interleaver
         node_idx = Interleaver[N*stride+id];

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
//TODO CtoV shared and Interleaver shared
__global__ void DataPassGB_2(unsigned char* VtoC, unsigned char* CtoV, unsigned char* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches, unsigned varr) {
   
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
      for (int stride = 0; stride < strides; stride++) {

         // calculate node index
         node_idx = Interleaver[N*stride+id];

         Global += (-2) * CtoV[node_idx] + 1;
      }

      // 
      for (int stride = 0; stride < strides; stride++) {
         
         // calculate node index
         node_idx = Interleaver[N*stride+id];

         // 
         // 
         buf = Global - ((-2) * CtoV[node_idx] + 1);

         // 
         VtoC[node_idx] = (buf < 0)? 1 : ((buf > 0)? 0 : i);
      }
   }
}

// This kernel is launched to check if the CtoV copies the same information as VtoC depending upon the signe value
__global__ void CheckPassGB(unsigned char* CtoV, unsigned char* VtoC, unsigned M, unsigned num_branches) {
  
  	//extern __shared__ unsigned char vtoc[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;


  // for(int k=threadIdx.x; k<num_branches; k+=blockDim.x){
//	vtoc[k]=VtoC[k];
   //}

   if (id < M) {

      int sign = 0;

      // For indexing the node arrays
      unsigned node_idx = 0;

      // 
      unsigned strides = (num_branches / M);
      int stride_idx =  id*strides;
      // 
      for (int stride = 0; stride < strides; stride++) {

         node_idx = stride + stride_idx;
         sign ^= VtoC[node_idx];
      }
      
      // 
      for (int stride = 0; stride < strides; stride++) {
         
         node_idx = stride + stride_idx;
         CtoV[node_idx] = sign ^ VtoC[node_idx];
      }
   }
}

// The following kernel is launched to decide each check node's decision whether the corresponding variable nodes 
// are in error or not.
__global__ void APP_GB(unsigned char* Decide, unsigned char* CtoV, unsigned char* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches) {
   
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
      for (int stride = 0; stride < strides; stride++) {

         // TODO this is not coalesced at all
         node_idx = Interleaver[N*stride+id];
         Global += (-2) * CtoV[node_idx] + 1;
      }
      
      // 
      Decide[id] = (Global < 0)? 1 : ((Global > 0)? 0 : i);
   }
}

//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(unsigned char * Synd, unsigned char * Decide, unsigned M, unsigned num_branches, unsigned N) {
	//extern __shared__ unsigned char decide[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // intialize ___ regardless of bounds...
   unsigned char synd = 0;
   //for(int k=threadIdx.x; k<N; k+=blockDim.x){
//	decide[k]=Decide[k];
   //}
   //__syncthreads();
   if (id < M) {
      
      unsigned strides = (num_branches / M);
	int stride_idx=id * strides;
      // 
      for (int stride = 0; stride < strides; stride++) {
         synd ^=Decide[d_matrix_flat[stride_idx + stride]];
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

__global__ void histogram_private_kernel(unsigned *bins, unsigned num_elements, unsigned num_bins){
   
   // Shared memory boxes
	extern __shared__ unsigned int boxs[];

   // size of grid
   unsigned grid_size = blockDim.x*gridDim.x;

   // Position in grid
   int myId = threadIdx.x + blockIdx.x*blockDim.x;

   int i = myId;

   // Initialize the gloabal bins to 0
   while(i < num_bins){
      bins[i] = 0;
      i+=grid_size;
   }
   __syncthreads();

   // Init shared bins using blocks
   for(int j=threadIdx.x; j<num_bins;j+=blockDim.x){
      boxs[j]=0;
   }
   __syncthreads();

   // reinitialize the index
   i = myId;

   // execute if we are within the input array
	if(i < num_elements){

      // for each of the elements strided by grid size
		while(i < num_elements){
			atomicAdd( &(boxs[d_matrix_flat[i]]),1);
			i+=grid_size;
		}   
	}
   __syncthreads();

   // Write back to global
   for(int j=threadIdx.x; j<num_bins; j+=blockDim.x){
      atomicAdd(&(bins[j]), boxs[j]);
   }
}

__global__ void setup_kernel (curandState* state, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 

__global__ void generate( curandState* globalState, unsigned char* randomArray, unsigned rank, unsigned N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = globalState[idx];
   float RANDOM = curand_uniform(&localState);
   RANDOM *= 1.999999; //https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
   unsigned char random = (unsigned char)truncf(RANDOM);

   if(idx >= rank && idx < N){
      randomArray[idx] = random;
      globalState[idx] = localState;
   }
}
