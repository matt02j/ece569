///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
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

#include "const.cuh"
#include "utils.cuh"
#include "kernels.cuh"

// TODO Verify stride paterns
// All of the following functions use this
// for (unsigned stride = 0; stride < (num_branches / N); stride++)
// pattern and this seems inherently a little weird that each
// thread is striding 1, 2, 3, 4... away from itself...
// 
// TODO The Data pass modules and their 2 for loops essentially 
// read CtoV[node_idx] across "strides" different elements, and 
// throws away data so the next for loop has to read global again. 
// Maybe find a way to keep these values in shared memory. This may 
// be weird because of the striding pattern, but if that can be 
// addressed this opens up for us
//
// TODO once we model what these functions are doing 
// COMMENT
// 
// TODO GaussianElimination_MRB as a kernel
//
// TODO Main is reading args in the most horendous way ive
// ever seen, us fstreams, arg parser, ect anything to make
// that not what it is
// also uses 4 different input files, 3 of which are only dimensions
//
// TODO replace calloc with mallocs where possible
//
// also uses 4 different input files, 3 of which are only dimensions
// TODO replace calloc with mallocs where possible
//
// TODO make sure all allocated memory gets freed at the apropriate time

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

      //-------------------------Simulation parameters for PGaB-------------------------
      unsigned NbMonteCarlo = 1000000; // Maximum number of codewords sent
      unsigned itteration_count = 100; // Maximum nb of iterations
      unsigned frames_tested = 0;      // NOTE dont move this
      unsigned frame_count = 100;      // Simulation stops when frame_count in error

      //-------------------------Channels probability for bit-flip-------------------------
      float alpha = 0.01;        // NOTE leave this here...
      float alpha_max = 0.07;    // max alpha val
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
	unsigned char* message[NUMSTREAMS];         // test message {0,1,0,0,1,0,...} 
	unsigned char* d_varr[NUMSTREAMS];

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
      std::cout << "Reading in test data...";
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
#ifdef PROFILE 
      gettimeofday(&start, NULL); 
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
		message[s] = (unsigned char*)calloc(N, sizeof(unsigned char)*BATCHSIZE);
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
      unsigned err_count;
      unsigned NiterMoy;
      unsigned NiterMax;
      unsigned Dmin;

      // add stochastic element to itteratcions past 16
      unsigned varr = (dist(e2) <= 20) ? 1 : 0;

      // Flatten for memcpy // if we edit the gausian elimination function we can get rid of this
      for (unsigned m = 0; m < M; m++) {
         for (unsigned n = 0; n < N; n++) {
            h_MatG_flat[m * N + n] = (unsigned char)h_MatG[m][n];
         }
      } 


      // copy h_MatG_flat to device only once
      cudaMemcpyAsync(d_MatG, h_MatG_flat, M * N * sizeof(unsigned char), cudaMemcpyHostToDevice);

      //-------------------------cuRand stuff--------------------------------------
      curandState* devStates;
      cudaMalloc((void **)&devStates, N * M * sizeof(curandState));
      setup_kernel<<<GridDim1,BlockDim1>>>(devStates, time(NULL));
      cudaMemcpy(d_PermG, PermG, N * sizeof(unsigned), cudaMemcpyHostToDevice);

      // loop from alpha max to alpha min (increasing noise)
      for (alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {
            NiterMoy = 0;
            NiterMax = 0;
            Dmin = 1e5;
            err_total_count = 0;
            bit_error_count = 0;
            missed_error_count = 0;
            err_count = 0;

            for(int s=0;s<NUMSTREAMS;s++){
            //these are both all 0s
                  cudaMemsetAsync(d_CtoV[s], 0, num_branches * sizeof(unsigned char)*BATCHSIZE,streams[s]);
                  cudaMemsetAsync(d_VtoC[s], 0, num_branches * sizeof(unsigned char)*BATCHSIZE,streams[s]);
            }

            frames_tested = 0;
            unsigned nb = 0;
            while (nb < NbMonteCarlo && err_total_count < frame_count) {

            //--------------------------------------------Encode--------------------------------------------
		for(int s=0;s<NUMSTREAMS;s++){
           	 cudaMemsetAsync(d_bit_stream[s], 0, N * sizeof(unsigned char)*BATCHSIZE,streams[s]);

           	 generate<<<BlockDim1, GridDim1,0,streams[s]>>>(devStates, d_bit_stream[s], rank, N);

            //cudaMemcpy(h_bit_stream, devRandomValues, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		    // randomly gerenates a uniform distribution of 0s and 1s
		    //for (unsigned k = rank; k < N; k++) {
		   //    h_bit_stream[s][k] = (unsigned char)floor(dist(e2) * 2);
		    //}

		    //replace that super long loop
		    //cudaMemcpyAsync(d_bit_stream[s], h_bit_stream[s], N * sizeof(unsigned char), cudaMemcpyHostToDevice,streams[s]);
		    NestedFor <<<NestedGrid, NestedBlock, N * sizeof(unsigned char),streams[s] >>>(d_MatG, d_bit_stream[s], rank - 1, N);
		   // cudaMemcpyAsync(h_bit_stream[s], d_bit_stream[s], N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,streams[s]); 
            }
            for(int s=0;s<NUMSTREAMS;s++){
                  cudaStreamSynchronize(streams[s]);
                  
                  simulateChannel<<<GridDim1, BlockDim1, 0, streams[s]>>>(d_bit_stream[s], d_messageRecieved[s], d_PermG, N);

                  cudaMemcpyAsync(message[s], d_messageRecieved[s], N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost);
                  //for (unsigned k = 0; k < N; k++) {
                  //      message[s][PermG[k]] = h_bit_stream[s][k];
                  //}
            
                  //---------------------------------------Simulate Channel---------------------------------------

                  // Flip the bits with the alpha percentage (noise over channel)
			
		 for(int i=0;i<BATCHSIZE;i++){
                  	for (unsigned n = 0; n < N; n++) {
		                if (dist(e2) < alpha) {
		                      h_messageRecieved[s][i*N+n] = 1 - message[s][i*N+n];
		                }
		                else {
		                      h_messageRecieved[s][i*N+n] = message[s][i*N+n];
		                }
			}
                  }
            
                  //-----------------------------------------------Decode-----------------------------------------------

                  cudaMemcpyAsync(d_messageRecieved[s], h_messageRecieved[s], N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyHostToDevice,streams[s]);
            } 
            unsigned itter = 0;
		unsigned it[NUMSTREAMS]={0};
            bool hasConverged[NUMSTREAMS][BATCHSIZE] = {{false}};
		bool hasConvergedStream[NUMSTREAMS]={false};
            while (itter < itteration_count) {
                  for(int s=0;s<NUMSTREAMS;s++){
                        if(!hasConvergedStream[s]){
                              // Different itterations have different kernels
                              if (itter == 0) {
                                    DataPassGB_0<<<GridDim1,BlockDim1,0,streams[s]>>> (d_VtoC[s], d_messageRecieved[s], d_interleaver, N, num_branches);
                              }
                              else if (itter < 15) {
                                    DataPassGB_1<<<GridDim1,BlockDim1,num_branches*sizeof(unsigned char)+num_branches*sizeof(unsigned),streams[s]>>>
                                                (d_VtoC[s],d_CtoV[s],d_messageRecieved[s],d_interleaver,N,num_branches);
                              }
                              else {
           	 			generate2<<<GridDim1, BlockDim1,0,streams[s]>>>(devStates, d_varr[s], N);
                                    DataPassGB_2<<<GridDim1,BlockDim1,0,streams[s]>>>(d_VtoC[s], d_CtoV[s], d_messageRecieved[s], d_interleaver, N, num_branches, d_varr[s]);
                              }
                              CheckPassGB<<<GridDim2,BlockDim2,num_branches * sizeof(unsigned char),streams[s]>>>(d_CtoV[s], d_VtoC[s], M, num_branches);

                              APP_GB<<<GridDim1,BlockDim1,0,streams[s]>>>(d_decoded[s],d_CtoV[s], d_messageRecieved[s], d_interleaver, N, num_branches);
                              
                              ComputeSyndrome<<<GridDim2, BlockDim2, N * sizeof(unsigned char),streams[s]>>>(d_synd[s], d_decoded[s], M, num_branches, N);
                              cudaMemcpyAsync(h_synd[s], d_synd[s], M * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,streams[s]);
                        }
                  }    
                  for(int s=0;s<NUMSTREAMS;s++){
			
                        if(!hasConvergedStream[s]){
                              // check for convergence
                              cudaStreamSynchronize(streams[s]);
				hasConvergedStream[s]=true;
				for(int b=0;b<BATCHSIZE;b++){
		                      hasConverged[s][b] = true;
		                      for (unsigned kk = 0; kk < M; kk++) {
		                            if (h_synd[s][kk+M*b] == 1) {
		                                  hasConverged[s][b] = false;
						hasConvergedStream[s]=false;
		                            break;
		                            }
		                      }
				}
                              it[s]++;
                        }
                  }
                  itter++;
            }//while


		for(int s=0;s<NUMSTREAMS;s++){
		    cudaMemcpyAsync(h_decoded[s], d_decoded[s], N * sizeof(unsigned char)*BATCHSIZE, cudaMemcpyDeviceToHost,streams[s]);
		}
		    //============================================================================
		    // Compute Statistics
		    //============================================================================
		    frames_tested+=(NUMSTREAMS*BATCHSIZE);

		for(int s=0;s<NUMSTREAMS;s++){
			cudaStreamSynchronize(streams[s]);
			for(int b=0;b<BATCHSIZE;b++){
		    		err_count = 0;
			    // Calculate bit errors
			    for (unsigned k = 0; k < N; k++) {
			       if (h_decoded[s][(N*b)+k] != message[s][(N*b)+k]) { 
				  err_count++;
			       }
			    }
			    bit_error_count += err_count;

			    // Case Divergence
			    if (!hasConverged[s][b]) {
			       NiterMoy = NiterMoy + itteration_count;
			       err_total_count++;
			    }
			    // Case Convergence to Right message
			    else if (err_count == 0) {
			       NiterMax = max(NiterMax, it[s] + 1);
			       NiterMoy = NiterMoy + (it[s] + 1);
			    }
			   // Case Convergence to Wrong message
			    else{
			       NiterMax = max(NiterMax, it[s] + 1);
			       NiterMoy = NiterMoy + (it[s] + 1);
			       err_total_count++;
			       missed_error_count++;
			       Dmin = min(Dmin, err_count);
			    }
			}
		}

            nb+= NUMSTREAMS * BATCHSIZE;
         }//frame loop

#ifdef PROFILE  
         gettimeofday(&stop, NULL);
         diffTime = diff_time_usec(start, stop);
         fprintf(stderr, "time %lu \n", diffTime);
#endif 

         std::cout << alpha << "\t";
         std::cout << bit_error_count << "(" << (float)bit_error_count / N / frames_tested << ")  ";
         std::cout << err_total_count << "(" << (float)err_total_count / frames_tested << ")\t";
         std::cout << frames_tested << "\t\t";
         std::cout << (float)NiterMoy / frames_tested << "(" << NiterMax << ")\t\t";
         std::cout << missed_error_count << "(" << Dmin << ")\t" << std::endl;

      }//alpha loop

      //Freeing memory on the GPU
      cudaFree(d_interleaver);
      cudaFree(devStates);
	for(int s=0;s<NUMSTREAMS;s++){
	      cudaFree(d_CtoV[s]);
	     	cudaFree(d_VtoC[s]);
            cudaFree(d_decoded[s]);
            cudaFree(d_CtoV[s]);
            cudaFree(d_varr[s]);
	      cudaFree(d_VtoC[s]);
            cudaFree(d_synd[s]);
            cudaFree(d_PermG);
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
