///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matthew Johnson, Jeremy Sears, Sebastian Thiem
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

	for(int b=0;b<BATCHSIZE;b++){
		   if (id < N) { 
		      int batch_offset=b*N;
		      // 
		      unsigned node_idx = 0;

		      // 
		      unsigned i = Receivedword[id+batch_offset];

		      for (int stride = 0; stride < strides; stride++) {
			 
			 // get node index from interleaver
			 node_idx = Interleaver[N*stride+id];
			 
			 VtoC[node_idx+b*num_branches] = i;
		      }
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

	for(int b=0;b<BATCHSIZE;b++){
	   if (id < N) { 
	      int batch_offset=b*num_branches;
         // 
         int buf = 0;

         // 
         int i = Receivedword[id+b*N];

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
            Global += (-2) * CtoV[node_idx+batch_offset] + 1;
         }

         // 
         for (int stride = 0; stride < strides; stride++) {

            // get node index from interleaver
            node_idx = Interleaver[N*stride+id];

            // 
            // 
            buf = Global - ((-2) * CtoV[node_idx+batch_offset] + 1);
            
            // 
            VtoC[node_idx+batch_offset] = (buf < 0)? 1 : ((buf > 0)? 0 : i);
         }
      }
   }
}

// for iterations greater than 15, this kernel launches to pass the message from variables nodes onto the four 
// check nodes it is connected to.
//TODO CtoV shared and Interleaver shared
__global__ void DataPassGB_2(unsigned char* VtoC, unsigned char* CtoV, unsigned char* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches, unsigned char* varr) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

 	for(int b=0;b<BATCHSIZE;b++){
	   if (id < N) { 
		   int batch_offset=b*num_branches;
	      // 
	      int buf;
	      
	      // 
	      int i = Receivedword[id+b*N];

	      // 
	      int Global = (1 - 2 * (varr[id] ^ i));

	      // 
	      unsigned node_idx = 0;

	      //
	      unsigned strides = (num_branches / N);
	      //
	      for (int stride = 0; stride < strides; stride++) {

            // calculate node index
            node_idx = Interleaver[N*stride+id];

            Global += (-2) * CtoV[node_idx+batch_offset] + 1;
	      }

	      // 
	      for (int stride = 0; stride < strides; stride++) {
            
            // calculate node index
            node_idx = Interleaver[N*stride+id];

            // 
            // 
            buf = Global - ((-2) * CtoV[node_idx+batch_offset] + 1);

            // 
            VtoC[node_idx+batch_offset] = (buf < 0)? 1 : ((buf > 0)? 0 : i);
	      }
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
   for(int b=0;b<BATCHSIZE;b++){
      if (id < M) { 
      int batch_offset=b*num_branches;
         int sign = 0;

         // For indexing the node arrays
         unsigned node_idx = 0;

         // 
         unsigned strides = (num_branches / M);
         int stride_idx =  id*strides;
         // 
         for (int stride = 0; stride < strides; stride++) {

            node_idx = stride + stride_idx;
            sign ^= VtoC[node_idx+batch_offset];
         }
         
         // 
         for (int stride = 0; stride < strides; stride++) {
            node_idx = stride + stride_idx;
            CtoV[node_idx+batch_offset] = sign ^ VtoC[node_idx+batch_offset];
         }
      }
	}
}

// The following kernel is launched to decide each check node's decision whether the corresponding variable nodes 
// are in error or not.
__global__ void APP_GB(unsigned char* Decide, unsigned char* CtoV, unsigned char* Receivedword, unsigned* Interleaver, unsigned N, unsigned num_branches) {
   
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  	for(int b=0;b<BATCHSIZE;b++){
	   if (id < N) { 
	int batch_offset=b*num_branches;
      // 
      int i = Receivedword[id+b*N];

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
         Global += (-2) * CtoV[node_idx+batch_offset] + 1;
      }
      
      // 
      Decide[id+b*N] = (Global < 0)? 1 : ((Global > 0)? 0 : i);
   }
}
}

//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(unsigned char * Synd, unsigned char * Decide, unsigned M, unsigned num_branches, unsigned N) {
	extern __shared__ unsigned char decide[];
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // intialize ___ regardless of bounds...
   unsigned char synd = 0;
   for(int k=threadIdx.x; k<N*BATCHSIZE; k+=blockDim.x){
	decide[k]=Decide[k];
   }
   __syncthreads();
	for(int b=0;b<BATCHSIZE;b++){
      synd = 0;
	   if (id < M) {
	      
	      unsigned strides = (num_branches / M);
	      // 
	      for (int stride = 0; stride < strides; stride++) {
		 synd ^=decide[d_matrix_flat[id + stride*M]+b*N];
	      }
	   

	   
	      Synd[id+M*b]=synd;
	    }
	}
}

//assumes a single block is running // matg access is not coalesced

__global__ void NestedFor(unsigned char* MatG_D, unsigned char* U_D, unsigned k, unsigned N,unsigned M){

   	int id = threadIdx.x;
	int batch_ofset=blockIdx.x*N;
	extern __shared__ unsigned char u[]; 
	//for(int b=0;b<BATCHSIZE;b++){
		for(int i=threadIdx.x; i<N; i+=blockDim.x){
			u[i]=U_D[i+batch_ofset];
		}
		__syncthreads();
		for(int i=k+1;i<N;i++){
			if(id <= k){
				//  0:k      0:k            0:k     k+1:N    k+1:N
				u[id] = u[id] ^ (MatG_D[id + i*M] * u[i]);
			}
		}
		__syncthreads();
		for(int i=k; i>0;i--){
			if(id < i){
				u[id] = u[id] ^ (MatG_D[id + i*M] * u[i]);
			}
		}

		__syncthreads();
		for(int i=threadIdx.x; i<N; i+=blockDim.x){
			U_D[i+batch_ofset]=u[i];
		}
	//}
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

   if(idx >= rank && idx < N){
      curandState localState = globalState[idx];
	for(int b=0;b<BATCHSIZE;b++){
      float RANDOM = curand_uniform(&localState);
      RANDOM *= 1.999999; //https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
      unsigned char random = (unsigned char)truncf(RANDOM);
      randomArray[idx+b*N] = random;
	}
      globalState[idx] = localState;
   }
}

__global__ void generate2( curandState* globalState, unsigned char* randomArray, unsigned N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if(idx < N){
      curandState localState = globalState[idx];
      float RANDOM = curand_uniform(&localState);
      unsigned char random = RANDOM < 0.19; //20% chance of a 1
      randomArray[idx] = random;
	
      globalState[idx] = localState;
   }
}

__global__ void simulateChannel(unsigned char* d_bit_stream, unsigned char* d_messageRecieved, unsigned* d_PermG, unsigned N, float alpha, curandState* globalState, unsigned char *d_intermediate){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//extern __shared__ unsigned char msgrcvd[];
	if(idx < N){
		for(int b=0;b<BATCHSIZE;b++){
		      	d_intermediate[d_PermG[idx]+b*N] = d_bit_stream[b*N+idx];
		}
	}
	__syncthreads();
	if(idx < N){
	  	curandState localState = globalState[idx];
		for(int b=0;b<BATCHSIZE;b++){
			int addr = b*N+idx;
	      		float RANDOM = curand_uniform(&localState);
			if (RANDOM < alpha) {
			      d_messageRecieved[addr] =1- d_intermediate[addr];
			}
			else{
			      d_messageRecieved[addr] = d_intermediate[addr];
			}
		}
      		globalState[idx] = localState;
	}
}

