///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matthew Johnson, Jeremy Sears, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : PGaB.cu
//                   :
// Create Date:      : 8 May 2017
// Modified          : 24 March 2019
//                   :
// Description:      : Probabalistic Gallager B
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <random>
#include <omp.h>
#include <pthread.h>

#include "const.cuh"
#include "utils.cuh"
#include "kernels.cuh"

// 
// TODO GaussianElimination_MRB as a kernel
//
// also uses 4 different input files, 3 of which are only dimensions





static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

void* frameloop(void* args);


int main(int argc, char * argv[]) {

   // need address of compressed data matrix
   if(argc > 1){

#ifdef PROFILE
      struct timeval start, stop;
      unsigned long diffTime = 0;
#endif

#ifdef QUIET
      std::ofstream outFile("results.log");
      std::cout.rdbuf(outFile.rdbuf());
#endif

#ifdef VERBOSE
      std::cout << "Starting." << std::endl << std::endl;
#endif

      // --------------------------------------------------------------------------
      // Parameters and memory
      // --------------------------------------------------------------------------

      //-------------------------Reading command line arguments-------------------------
      std::string matrixAddr(argv[1]);    // convert data addr to a string

      //-------------------------Channels probability for bit-flip-------------------------
      float alpha = 0.01;        // NOTE leave this here...

      float alpha_max = 0.06;    // max alpha val
      float alpha_min = 0.02;    // min alpha value
      float alpha_step = 0.01;   // step size in alpha for loop
      
      //--------------------------------Random Number Generation--------------------------------
#ifdef TRUERANDOM
      std::random_device rd;                          // boilerplate
      std::mt19937 e2(rd());                          // boilerplate 
#else
      unsigned seed = 1324;                           // Seed Initialization for consistent Simulations
      std::mt19937 e2(seed);                          // boilerplate

#endif
      std::uniform_real_distribution<> dist(0, 1);    // uni{} (use dist(e2) to generate)

      //-------------------------Host memory structures-------------------------
      unsigned* h_matrix_flat;      // unrolled matrix
      unsigned* h_interleaver;      // ...
      //unsigned char* h_messageRecieved;  // message after bit-flipping noise is applied
      //unsigned char* h_decoded;          // message decoded
      //unsigned char* h_synd;                  // ...
      //unsigned char* h_bit_stream;       // Randomly generated bit stream 
      unsigned** h_MatG;            // ...
      unsigned char* h_MatG_flat;        // MatG flattened    
      unsigned bin_size = 0;
	cudaStream_t streams[NUMSTREAMS];
	//cudaEvent_t start1, stop1, start2, stop2;
	unsigned char* h_messageRecieved[NUMSTREAMS];  // message after bit-flipping noise is applied
	unsigned char* h_decoded[NUMSTREAMS];          // message decoded
	unsigned char* h_synd[NUMSTREAMS];                  // ...
	unsigned char* h_bit_stream[NUMSTREAMS];       // Randomly generated bit stream 
	unsigned char* d_CtoV[NUMSTREAMS];                  // 
	unsigned char* d_VtoC[NUMSTREAMS];                  // 
	unsigned char* d_messageRecieved[NUMSTREAMS];  // message after bit-flipping noise is applied
	unsigned char* d_decoded[NUMSTREAMS];          // message decoded
	unsigned char* d_synd[NUMSTREAMS];                  // ...
	unsigned char* d_bit_stream[NUMSTREAMS];       // Randomly generated bit stream
	//unsigned char* message[NUMSTREAMS];         // test message {0,1,0,0,1,0,...} 
	unsigned char* d_varr[NUMSTREAMS];
	unsigned char *d_intermediate[NUMSTREAMS];
	myargs streamargs[NUMSTREAMS];
	pthread_t threads[NUMSTREAMS];
//pthread_mutex_t * lock=NULL;

      //-------------------------Device memory structures-------------------------
      // unsigned* d_matrix_flat;      // held as global in constant memory
      unsigned* d_interleaver;      // ...
      //unsigned char* d_CtoV;                  // 
      //unsigned char* d_VtoC;                  // 
      //unsigned char* d_messageRecieved;  // message after bit-flipping noise is applied
      //unsigned char* d_decoded;          // message decoded
      //unsigned char* d_synd;                  // ...
      //unsigned char* d_bit_stream;       // Randomly generated bit stream 
      unsigned char* d_MatG;             // MatG flattened
      unsigned int *d_Bins;

      //-------------------------Intermediate data structures-------------------------
      unsigned* rowRanks;        // list of codewords widths
      unsigned** data_matrix;    // matrix of codewords on the host
      unsigned* hist;            // histogram for <unk> // pinned
      //unsigned* message;         // test message {0,1,0,0,1,0,...}
      unsigned** sparse_matrix;  // uncompressed sparse data matrix
      unsigned* PermG;           // array to keep track of permutations in h_MatG 
      unsigned* d_PermG;

      //-------------------------Block and Grid dimensionality structures-------------------------
      dim3 GridDim1((N - 1) / BLOCK_DIM_1 + 1, 1);
      dim3 BlockDim1(BLOCK_DIM_1);
      dim3 GridDim2((M - 1) / BLOCK_DIM_2 + 1, 1);
      dim3 BlockDim2(BLOCK_DIM_2);
      dim3 NestedBlock(1024);
      dim3 NestedGrid(BATCHSIZE);

#ifdef VERBOSE
      std::cout << "Reading in test data..."<< std::endl;
#endif

      // Basically M*ColWidth but this code allows 
      // for their to be staggered columns, 
      // so the calculation is not as simple
      unsigned num_branches;
      // allocate and get row ranks
      rowRanks = (unsigned*)malloc(M * sizeof(unsigned));
      readRowRanks(rowRanks, M, (matrixAddr + "_RowDegree").c_str());
      // alocate and read in test data matrix from local file (also get num_branches while were in this loop)
      unsigned cols = 0;
      data_matrix = (unsigned**)malloc(M * sizeof(unsigned*));
      for (unsigned m = 0; m < M; m++) {
         cols = rowRanks[m];
         num_branches += cols;
         data_matrix[m] = (unsigned*)malloc(cols * sizeof(unsigned));
      }
      readDataMatrix(data_matrix, rowRanks, M, matrixAddr.c_str());

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Allocating memory...";
#endif

      //-------------------------Host Allocations-------------------------
      h_matrix_flat = (unsigned*)malloc(num_branches * sizeof(unsigned));
      h_interleaver = (unsigned*)malloc(num_branches * sizeof(unsigned));
      //h_synd = (unsigned char*)calloc(M, sizeof(unsigned char));
      //h_CtoV = (unsigned char*)calloc(num_branches, sizeof(unsigned char));
      //h_VtoC = (unsigned char*)calloc(num_branches, sizeof(unsigned char));
      //h_messageRecieved = (unsigned char*)calloc(N, sizeof(unsigned char));
      //h_decoded = (unsigned char*)calloc(N, sizeof(unsigned char));
      //h_bit_stream = (unsigned char *)calloc(N, sizeof(unsigned char));
      h_MatG = (unsigned **)calloc(M, sizeof(unsigned *));
      for (unsigned m = 0; m < M; m++) {
         h_MatG[m] = (unsigned *)calloc(N, sizeof(unsigned));
      }
      h_MatG_flat = (unsigned char*)malloc(M*N * sizeof(unsigned char));

      //-------------------------Device Allocations-------------------------
      cudaMalloc((void**)&d_interleaver, num_branches * sizeof(unsigned));
      //cudaMalloc((void**)&d_synd, M * sizeof(unsigned char));
      //cudaMalloc((void**)&d_CtoV, num_branches * sizeof(unsigned char));
      //cudaMalloc((void**)&d_VtoC, num_branches * sizeof(unsigned char));
      //cudaMalloc((void**)&d_messageRecieved, N * sizeof(unsigned char));
      //cudaMalloc((void**)&d_decoded, N * sizeof(unsigned char));
      //cudaMalloc((void **)&d_bit_stream, N * sizeof(unsigned char));
      cudaMalloc((void **)&d_MatG, M * N * sizeof(unsigned char));
      bin_size = sizeof(unsigned) * N;
      cudaMallocHost((void**)&hist, bin_size); // host pinned
      cudaMalloc((void**) &d_Bins, bin_size);
      cudaMalloc((void **) &d_PermG, N * sizeof(unsigned));
	//cudaEventCreate( &start1);
	//cudaEventCreate( &start2);
	for(int s=0;s<NUMSTREAMS;s++){
		cudaStreamCreate( &streams[s] );
		cudaMallocHost((void**)&h_messageRecieved[s],N*sizeof(unsigned char)*BATCHSIZE);
		cudaMallocHost((void**)&h_decoded[s],N*sizeof(unsigned char)*BATCHSIZE);
		cudaMallocHost((void**)&h_synd[s],M*sizeof(unsigned char)*BATCHSIZE);
		cudaMallocHost((void**)&h_bit_stream[s],N*sizeof(unsigned char)*BATCHSIZE);
	     	cudaMalloc((void**)&d_synd[s], M * sizeof(unsigned char)*BATCHSIZE);
	      	cudaMalloc((void**)&d_CtoV[s], num_branches * sizeof(unsigned char)*BATCHSIZE);
	     	cudaMalloc((void**)&d_VtoC[s], num_branches * sizeof(unsigned char)*BATCHSIZE);
	     	cudaMalloc((void**)&d_messageRecieved[s], N * sizeof(unsigned char)*BATCHSIZE);
	      	cudaMalloc((void**)&d_decoded[s], N * sizeof(unsigned char)*BATCHSIZE);
	      	cudaMalloc((void **)&d_bit_stream[s], N * sizeof(unsigned char)*BATCHSIZE);
		cudaMalloc((void **)&d_intermediate[s], N * sizeof(unsigned char)*BATCHSIZE);
		///message[s] = (unsigned char*)calloc(N, sizeof(unsigned char)*BATCHSIZE);
		cudaMalloc((void**)&d_varr[s],N*sizeof(unsigned char));
	}
      //message = (unsigned *)calloc(N, sizeof(unsigned));

      sparse_matrix = (unsigned **)calloc(M, sizeof(unsigned *));
      for (unsigned m = 0; m < M; m++) {
         sparse_matrix[m] = (unsigned *)calloc(N, sizeof(unsigned));
      }

      PermG = (unsigned *)calloc(N, sizeof(unsigned));


#ifdef VERBOSE
      std::cout << "Done." << std::endl; 

      std::cout << "Performing preliminary calulations...";
#endif

      // unroll host matrix into a flat host vector
      unrollMatrix(h_matrix_flat, data_matrix, rowRanks, M, num_branches);

      // Copying contents from the host to the device

      cudaMemcpyToSymbol(d_matrix_flat, h_matrix_flat, num_branches * sizeof(unsigned));
 
      // generate histogram on the data matrix

      histogram_private_kernel<<<GridDim1, BlockDim1,N * sizeof(unsigned int)>>>(d_Bins, num_branches, N);
      cudaMemcpy(hist, d_Bins, bin_size, cudaMemcpyDeviceToHost);

      // generate interleaver
      initInterleaved(h_interleaver, data_matrix, rowRanks, hist, M, N);
      cudaMemcpy(d_interleaver, h_interleaver, num_branches * sizeof(unsigned), cudaMemcpyHostToDevice);


      // free no longer needed structures
      cudaFree(hist);

      // init permutation matrix
      for (unsigned n = 0; n < N; n++) {
         PermG[n] = n;
      }

      // convert compressed data matrix to sparse matrix
      for (unsigned m = 0; m < M; m++) {
         for (unsigned k = 0; k < rowRanks[m]; k++) {
            sparse_matrix[m][data_matrix[m][k]] = 1;
         }
      }

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Gaussian Elimination...";
#endif


      unsigned rank;

      rank = GaussianElimination_MRB(PermG, h_MatG, sparse_matrix, M, N);
      // free no longer needed data structures
      free2d(sparse_matrix, M);
      free2d(data_matrix, M);
      free(rowRanks);

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Sim." << std::endl << std::endl;
#endif

      std::cout << "-------------------------------------------Gallager B-------------------------------------------" << std::endl;
      std::cout << "alpha\tNbEr(BER)\tNbFer(FER)\tNbtested\tIterAver(Itermax)\tNbUndec(Dmin)" << std::endl;

      // Variables for monitoring statistics
      unsigned err_total_count;
      unsigned bit_error_count;
      unsigned missed_error_count;
      //unsigned err_count;
      unsigned NiterMoy;
      unsigned NiterMax;
      unsigned Dmin;

      // add stochastic element to itteratcions past 16
      //unsigned varr = (dist(e2) <= 20) ? 1 : 0;

      // Flatten for memcpy // if we edit the gausian elimination function we can get rid of this
      for (unsigned m = 0; m < M; m++) {
         for (unsigned n = 0; n < N; n++) {
            h_MatG_flat[n * M + m] = (unsigned char)h_MatG[m][n];
         }
      } 
	unsigned nb=0;

      // copy h_MatG_flat to device only once
      cudaMemcpyAsync(d_MatG, h_MatG_flat, M * N * sizeof(unsigned char), cudaMemcpyHostToDevice);

      //-------------------------cuRand stuff--------------------------------------
      curandState* devStates;
      cudaMalloc((void **)&devStates, N * M * sizeof(curandState));
      setup_kernel<<<GridDim1,BlockDim1>>>(devStates, time(NULL));

      cudaMemcpyAsync(d_PermG, PermG, N * sizeof(unsigned), cudaMemcpyHostToDevice);

	//pthread_mutex_init(lock,NULL);
	for(int s=0; s<NUMSTREAMS; s++){
		streamargs[s].d_bit_stream=d_bit_stream[s];
		streamargs[s].d_MatG=d_MatG;
		streamargs[s].d_messageRecieved=d_messageRecieved[s];
		streamargs[s].h_messageRecieved=h_messageRecieved[s];
		streamargs[s].d_VtoC=d_VtoC[s];
		streamargs[s].d_CtoV=d_CtoV[s];
		streamargs[s].d_interleaver=d_interleaver;
		streamargs[s].d_decoded=d_decoded[s];
		streamargs[s].h_decoded=h_decoded[s];
		streamargs[s].d_synd=d_synd[s];
		streamargs[s].h_synd=h_synd[s];
		streamargs[s].d_PermG=d_PermG;
		streamargs[s].d_varr=d_varr[s];
		streamargs[s].devStates=devStates;
		streamargs[s].d_intermediate=d_intermediate[s];
		streamargs[s].rank=rank;
		streamargs[s].N=N;
		streamargs[s].M=M;
		streamargs[s].num_branches=num_branches;
		streamargs[s].NiterMax=&NiterMax;
		streamargs[s].NiterMoy=&NiterMoy;
		streamargs[s].err_total_count=&err_total_count;
		streamargs[s].missed_error_count=&missed_error_count;
		streamargs[s].Dmin=&Dmin;
		streamargs[s].bit_error_count=&bit_error_count;
		streamargs[s].nb=&nb;
		streamargs[s].id=s;
		streamargs[s].stream=streams[s];
	}

      // loop from alpha max to alpha min (increasing noise)
      for (alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {
            #ifdef PROFILE 
                  gettimeofday(&start, NULL); 
            #endif
            NiterMoy = 0;
            NiterMax = 0;
            Dmin = 1e5;
            err_total_count = 0;
            bit_error_count = 0;
            missed_error_count = 0;
            //err_count = 0;

            for(int s=0;s<NUMSTREAMS;s++){
            //these are both all 0s
		  streamargs[s].alpha=alpha;
                  cudaMemsetAsync(d_CtoV[s], 0, num_branches * sizeof(unsigned char)*BATCHSIZE,streams[s]);
                  cudaMemsetAsync(d_VtoC[s], 0, num_branches * sizeof(unsigned char)*BATCHSIZE,streams[s]);
            }

             nb = 0;

		for(int s=0; s<NUMSTREAMS;s++){
			pthread_create(&threads[s],NULL,frameloop,(void*)&streamargs[s]);
		}
		for(int s=0; s<NUMSTREAMS;s++){
			pthread_join(threads[s],NULL);
		}

#ifdef PROFILE  
         gettimeofday(&stop, NULL);
         diffTime = diff_time_usec(start, stop);
         fprintf(stderr, "time %lu \n", diffTime);
#endif 

         std::cout << alpha << "\t";
         std::cout << bit_error_count << "(" << (float)bit_error_count / N / nb << ")  ";
         std::cout << err_total_count << "(" << (float)err_total_count / nb << ")\t";
         std::cout << nb << "\t\t";
         std::cout << (float)NiterMoy / nb << "(" << NiterMax << ")\t\t";
         std::cout << missed_error_count << "(" << Dmin << ")\t" << std::endl;

      }//alpha loop

      //Freeing memory on the GPU
      cudaFree(d_interleaver);
      cudaFree(devStates);
      cudaFree(d_MatG);
      free(h_matrix_flat);
      free(h_interleaver);
      free(h_MatG_flat);
      cudaFree(d_PermG);
      cudaFree(d_Bins);
      
      
	for(int s=0;s<NUMSTREAMS;s++){
	      cudaFree(d_CtoV[s]);
	     	cudaFree(d_VtoC[s]);
            cudaFree(d_decoded[s]);
            cudaFree(d_CtoV[s]);
            cudaFree(d_varr[s]);
	      cudaFree(d_VtoC[s]);
            cudaFree(d_synd[s]);
            cudaFree(d_bit_stream[s]);
	      cudaFree(d_messageRecieved[s]);
		cudaFreeHost(h_bit_stream[s]);
		cudaFreeHost(h_synd[s]);
		cudaFreeHost(h_decoded[s]);
		cudaFreeHost(h_messageRecieved[s]);
	}
   }
   else {
      fprintf(stderr, "Usage: PGaB /Path/To/Data/File");
   }
   return 0;
}



void* frameloop(void* args){
myargs *arg = (myargs*)args;
      dim3 GridDim1((arg->N - 1) / BLOCK_DIM_1 + 1, 1);
      dim3 BlockDim1(BLOCK_DIM_1);
      dim3 GridDim2((arg->M - 1) / BLOCK_DIM_2 + 1, 1);
      dim3 BlockDim2(BLOCK_DIM_2);
      dim3 NestedBlock(1024);
      dim3 NestedGrid(BATCHSIZE);
	unsigned err_count=0;
	unsigned itter=0;
	unsigned char *message = (unsigned char*) malloc(arg->N*BATCHSIZE);
	unsigned batchitter[BATCHSIZE];
	while( *(arg->nb) < NbMonteCarlo && *(arg->err_total_count) < frame_count ){

            //--------------------------------------------Encode--------------------------------------------
		
           	 cudaMemsetAsync(arg->d_bit_stream, 0, arg->N * sizeof(unsigned char)*BATCHSIZE,arg->stream);

           	 generate<<< GridDim1,BlockDim1,0,arg->stream>>>(arg->devStates, arg->d_bit_stream, arg->rank, arg->N);

		 NestedFor <<<NestedGrid, NestedBlock, N * sizeof(unsigned char),arg->stream>>>(arg->d_MatG, arg->d_bit_stream, arg->rank - 1, arg->N,arg->M);

		 cudaMemcpyAsync(arg->h_messageRecieved, arg->d_bit_stream, arg->N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,arg->stream);

                 simulateChannel<<<GridDim1, BlockDim1, 0, arg->stream>>>(arg->d_bit_stream, arg->d_messageRecieved, arg->d_PermG, arg->N,arg->alpha,arg->devStates, arg->d_intermediate);
		 cudaMemcpyAsync(message, arg->d_intermediate, arg->N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,arg->stream);

            	itter = 0;

            	bool hasConverged[BATCHSIZE] = {false};

		bool hasConvergedStream=false;
		memset(batchitter,0,sizeof(unsigned)*BATCHSIZE);

            while(itter < itteration_count && !hasConvergedStream){

                              // Different itterations have different kernels

                              if (itter == 0) {

                                    DataPassGB_0<<<GridDim1,BlockDim1,0,arg->stream>>> (arg->d_VtoC, arg->d_messageRecieved, arg->d_interleaver, arg->N, arg->num_branches);

                              }

                              else if (itter < 15) {
                                    DataPassGB_1<<<GridDim1,BlockDim1,0,arg->stream>>> (arg->d_VtoC, arg->d_CtoV, arg->d_messageRecieved, arg->d_interleaver, arg->N, arg->num_branches);
                              }

                              else {
           	 		generate2<<<GridDim1, BlockDim1,0,arg->stream>>>(arg->devStates, arg->d_varr, arg->N);
                               	DataPassGB_2<<<GridDim1,BlockDim1,0,arg->stream>>>(arg->d_VtoC, arg->d_CtoV, arg->d_messageRecieved, arg->d_interleaver, arg->N, arg->num_branches, arg->d_varr);
                              }

                              CheckPassGB<<<GridDim2,BlockDim2,0,arg->stream>>>(arg->d_CtoV, arg->d_VtoC, arg->M, arg->num_branches);

                              APP_GB<<<GridDim1,BlockDim1,0,arg->stream>>>(arg->d_decoded,arg->d_CtoV, arg->d_messageRecieved,arg-> d_interleaver, arg->N, arg->num_branches);
                              
                              ComputeSyndrome<<<GridDim2, BlockDim2, arg->N * sizeof(unsigned char)*BATCHSIZE,arg->stream>>>(arg->d_synd, arg->d_decoded, arg->M, arg->num_branches, arg->N);
                              cudaMemcpyAsync(arg->h_synd, arg->d_synd, arg->M * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,arg->stream);


                              // check for convergence
				cudaStreamSynchronize(arg->stream);
				hasConvergedStream=true;
				for(int b=0;b<BATCHSIZE && hasConvergedStream;b++){
		                      hasConverged[b] = true;
				////if we do a reduction on d_synd we can get rid of this loop
		                      for (unsigned kk = 0; kk < arg->M; kk++) {
		                            if (arg->h_synd[kk+arg->M*b] == 1) {
		                                  hasConverged[b] = false;
						  hasConvergedStream = false;
		                                 break;
		                            }
		                      }
				}
                  itter++;
				
				for(int b=0;b<BATCHSIZE ;b++){
					if(hasConverged[b] && batchitter[b]==0){
						batchitter[b]=itter;
					}
				}
            }//itter loop

		if(!hasConvergedStream){
			for(int b=0;b<BATCHSIZE;b++){
		                      hasConverged[b] = true;
				////if we do a reduction on d_synd we can get rid of this loop
		                      for (unsigned kk = 0; kk < arg->M; kk++) {
		                            if (arg->h_synd[kk+arg->M*b] == 1) {
		                                  hasConverged[b] = false;
						  hasConvergedStream = false;
		                                 break;
		                            }
		                      }
			}
		}
		    cudaMemcpyAsync(arg->h_decoded, arg->d_decoded, arg->N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,arg->stream);
		
				cudaStreamSynchronize(arg->stream);
		    //============================================================================
		    // Compute Statistics
		    //============================================================================

		////maybe move error checking to gpu
			for(int b=0;b<BATCHSIZE;b++){
		    		err_count = 0;
			    // Calculate bit errors
			    for (unsigned k = 0; k < arg->N; k++) {
			       if (arg->h_decoded[(arg->N*b)+k] != message[(arg->N*b)+k]) { 
				  err_count++;
			       }
			    }
		
				pthread_mutex_lock(&lock);
			    *(arg->bit_error_count) += err_count;

			    // Case Divergence
			    if (!hasConverged[b]) {
			       *(arg->NiterMoy) = *(arg->NiterMoy) + 100;

			       *(arg->err_total_count)+=1;
			    }
			    // Case Convergence to Right message
			    else if (err_count == 0) {
			       *(arg->NiterMax) = max(*(arg->NiterMax), batchitter[b]);
			       *(arg->NiterMoy) = *(arg->NiterMoy) + batchitter[b];
			    }
			   // Case Convergence to Wrong message
			    else{
			       *(arg->NiterMax) = max(*(arg->NiterMax), batchitter[b]);
			       *(arg->NiterMoy) = *(arg->NiterMoy) + batchitter[b];
			       *(arg->err_total_count)+=1;
			       *(arg->missed_error_count)+=1;
			       *(arg->Dmin) = min(*(arg->Dmin), err_count);
			    }
				
				pthread_mutex_unlock(&lock);
			}
		

		pthread_mutex_lock(&lock2);
            	*(arg->nb)+= BATCHSIZE;
		pthread_mutex_unlock(&lock2);
         }//frame loop
	return 0;
}
