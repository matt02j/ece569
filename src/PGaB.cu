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
#include <iostream>
#include <fstream>
#include <random>

#ifdef OMP
#include <omp.h>
#endif

#include "const.cuh"
#include "utils.cuh"
#include "kernels.cuh"

/*
TODO Verify stride paterns
All of the following functions use this
for (unsigned stride = 0; stride < (num_branches / N); stride++)
pattern and this seems inherently a little weird that each
thread is striding 1, 2, 3, 4... away from itself...

TODO The Data pass modules and their 2 for loops essentially 
read CtoV[node_idx] across "strides" different elements, and 
throws away data so the next for loop has to read global again. 
Maybe find a way to keep these values in shared memory. This may 
be weird because of the striding pattern, but if that can be 
addressed this opens up for us

TODO once we model what these functions are doing 
COMMENT

TODO GaussianElimination_MRB as a kernel

TODO Main is reading args in the most horendous way ive
ever seen, us fstreams, arg parser, ect anything to make
that not what it is
also uses 4 different input files, 3 of which are only dimensions

TODO replace calloc with mallocs where possible
also uses 4 different input files, 3 of which are only dimensions

TODO replace calloc with mallocs where possible

TODO make sure all allocated memory gets freed at the apropriate time


TODO read in data like this instead of fscans
std::ifstream deg_file(matrixAddr + "_RowDegree", 'r');
deg_file.read(rowRanks, M * sizeof(int));
deg_file.close();

*/

int main(int argc, char * argv[]) {

#ifdef QUIET
   std::ofstream outFile("results.log");
   std::cout.rdbuf(outFile.rdbuf());
#endif

#ifdef VERBOSE
   std::cout << "Starting." << std::endl << std::endl;
#endif

   // expect: ./PGaB data_matrix
   if(argc > 1){

      // read args
      std::string matrixAddr(argv[1]);    // convert data addr to a string

      // --------------------------------------------------------------------------
      // Parameters and memory
      // --------------------------------------------------------------------------

      //-------------------------Simulation parameters for PGaB-------------------------
      float NbMonteCarlo = 1000000.0f;   // Maximum number of codewords sent
      unsigned itteration_count = 100; // Maximum nb of iterations
      unsigned frame_count = 100;      // Simulation stops when frame_count in error
      unsigned seed = 42;               // Seed Initialization for consistent Simulations

      //----------------------------------Noise Parameters----------------------------------
      // Channel Crossover Probability Max and Min
      // Basically the noise, but in terms of probability of bit flip
      float alpha_max = 0.03f;
      float alpha_min = 0.03f;
      float alpha_step = 0.01f;
      
      std::random_device rd;                          // boilerplate
      std::mt19937 e2(rd());                          // boilerplate 
      std::uniform_real_distribution<> dist(0, 1);    // uni{} (use dist(e2) to generate)

      //-------------------------Host memory structures-------------------------
      unsigned* h_interleaver;
      int* h_CtoV;
      int* h_VtoC;
      int* h_messageRecieved;
      int* h_decide;
      int* h_synd;

      //-------------------------Device memory structures-------------------------
      unsigned* d_interleaver;
      int* d_CtoV;
      int* d_VtoC;
      int* d_receivedWord;
      int* d_decide;
      int* d_synd;

      //-------------------------Intermediate data structures-------------------------
      unsigned* rowRanks;        // list of codewords widths
      unsigned** data_matrix;    // matrix of codewords on the host
      unsigned* h_matrix_flat;   // unrolled matrix
      unsigned* hist;            // histogram for <unk>
      unsigned* bit_stream;      // Randomly generated bit stream
      int** MatFull;             // ...
      int** MatG;                // ...
      unsigned* PermG;           // array to keep track of permutations in MatG 
      unsigned* message;         // test message {0,1,0,0,1,0,...}
      
      //------------------------------------Miscellenious Variables------------------------------------
      unsigned varr = (dist(e2) <= 0.2)? 1 : 0;    // Probabalisic element
      unsigned num_branches = 0;          // total number of elements in the test data matrix
      unsigned rank;                      // returned from Gaussian Elimination

      // Variables for Statistics
      unsigned frames_tested = 0;   // Keep track of how many frames run at an alpha level
	   unsigned err_total_count;
	   unsigned bit_error_count;
	   unsigned missed_error_count;
	   unsigned err_count;
      unsigned NiterMoy;
      unsigned NiterMax;
      unsigned Dmin;

      unsigned itter;            // keep track of itterations run (kernels run)
      bool hasConverged;         // monitor kernel launches for convergence

      // Initialize grid and block dimensions
      dim3 GridDim1((N - 1) / BLOCK_DIM_1 + 1, 1);
      dim3 BlockDim1(BLOCK_DIM_1);
      dim3 GridDim2((M - 1) / BLOCK_DIM_2 + 1, 1);
      dim3 BlockDim2(BLOCK_DIM_2);

#ifdef PROFILE
      // Used for timing CPU
      struct timeval start;
      struct timeval stop;
      unsigned long diffTime = 0;
#endif

#ifdef VERBOSE
      std::cout << "Reading in test data...";
#endif

      // allocate and get row ranks
      rowRanks = (unsigned*) malloc(M * sizeof(unsigned));
      readRowRanks(rowRanks, M, (matrixAddr + "_RowDegree").c_str());

      //alocate and read in test data matrix from local file (also get num_branches while were in this loop)
      unsigned cols = 0;
      data_matrix = (unsigned**)malloc(M * sizeof(unsigned*));
      for (unsigned m = 0; m < M; m++) {
         cols = rowRanks[m];
         num_branches += cols;
         data_matrix[m] = (unsigned*) malloc(cols * sizeof(unsigned));
      }
      readDataMatrix(data_matrix, rowRanks, M, matrixAddr.c_str());

#ifdef VERBOSE
	   std::cout << "Done." << std::endl;

	   std::cout << "Allocating memory...";
#endif
      //-------------------------Host Allocations-------------------------
	   h_matrix_flat = (unsigned*)malloc(num_branches * sizeof(unsigned));
	   h_interleaver = (unsigned*)malloc(num_branches * sizeof(unsigned));
      h_synd = (int*)calloc(M, sizeof(int));
      h_CtoV = (int*) calloc(num_branches, sizeof(int));
      h_VtoC = (int*) calloc(num_branches, sizeof(int));
      h_messageRecieved = (int*) calloc(N, sizeof(int));
      h_decide = (int*) calloc(N, sizeof(int));

      //-------------------------Device Allocations-------------------------
      cudaMalloc((void**) &d_interleaver, num_branches * sizeof(unsigned));
      cudaMalloc((void**) &d_synd, M * sizeof(int));
      cudaMalloc((void**) &d_CtoV, num_branches * sizeof(int));
      cudaMalloc((void**) &d_VtoC, num_branches * sizeof(int));
      cudaMalloc((void**) &d_receivedWord, N * sizeof(int));
      cudaMalloc((void**) &d_decide, N * sizeof(int));

      //-------------------------Other Allocations-------------------------
	   hist = (unsigned*)calloc(N, sizeof(unsigned));
      bit_stream = (unsigned*)calloc(N, sizeof(unsigned));
      message = (unsigned*)calloc(N, sizeof(unsigned));

      MatG = (int**)malloc(M * sizeof(int*));
      for (unsigned m = 0; m < M; m++) {
         MatG[m] = (int*)calloc(N, sizeof(int));
      }

      MatFull = (int**)malloc(M * sizeof(int * ));
      for (unsigned m = 0; m < M; m++) {
         MatFull[m] = (int*)malloc(N * sizeof(int));
      }

      PermG = (unsigned*)malloc(N * sizeof(unsigned));
    
#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Performing preliminary calulations...";
#endif

      // allocate and generate histogram on the data
      histogram(hist, data_matrix, rowRanks, M, N);

      // allocate and generate interleaver (allocation done in method)
      initInterleaved(h_interleaver, data_matrix, rowRanks, hist, M, N);

      // allocate and unroll host matrix into a flat host vector
      unrollMatrix(h_matrix_flat, data_matrix, rowRanks, M, num_branches);

      // free no longer needed structures
      free(hist);

      // uh....
      for (unsigned n = 0; n < N; n++) {
	      PermG[n] = n;
      }

      // okay.
      for (unsigned m = 0; m < M; m++) {
	      for (unsigned k = 0; k < rowRanks[m]; k++) {
		      MatFull[m][data_matrix[m][k]] = 1;
	      }
      }

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Gaussian Elimination...";
#endif

      rank = GaussianElimination_MRB(PermG, MatG, MatFull, M, N);

      // free no longer needed data structures
      free2d(MatFull, M);
      free2d(data_matrix, M);
      free(rowRanks);

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Sim." << std::endl << std::endl;
#endif

      std::cout << "--------------------------------------------------Gallager B--------------------------------------------------" << std::endl;
      std::cout << "alpha\tNbEr(BER)\t\tNbFer(FER)\tNbtested\tIterAver(Itermax)\tNbUndec(Dmin)" << std::endl;

      // loop from alpha max to alpha min
      // run sim on each alpha level
      for (float alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {

         // reinitialize all the parameters for the next run
         NiterMoy = 0;
         NiterMax = 0;
         Dmin = 100000;
         err_total_count = 0;
         bit_error_count = 0;
         missed_error_count = 0;
         err_count = 0;

         // encoding
#ifdef PROFILE 
		 gettimeofday(&start,NULL);
#endif 
         frames_tested = 0;
         for (unsigned nb = 0; nb < NbMonteCarlo; nb++) {
            
#ifdef ZERO_CODE
            // All zero codeword
            for (n = 0; n < N; n++) {
               message[n] = 0;
            }
#else
            // randomly gerenate a uniform distribution of 0s and 1s
            for (unsigned k = rank; k < N; k++) {
               bit_stream[k] = (int)floor(dist(e2) * 2);
            }

            // apply organized chaos to bit stream
            // TODO this is what takes ~60% of the whole program
            for (int k = rank - 1; k >= 0; k--) {
               for (unsigned l = k + 1; l < N; l++) {
                  bit_stream[k] = bit_stream[k] ^ (MatG[k][l] * bit_stream[l]);
               }
            }

            // assign altered bit stream codewords array,
            // based on order determined by the permutation array
            for (unsigned k = 0; k < N; k++) {
               message[PermG[k]] = bit_stream[k];
            }
#endif

            // no longer needed
            free(bit_stream);

            // Flip bits based on alpha
            for (unsigned n = 0; n < N; n++){
               if (dist(e2) < alpha){
                  h_messageRecieved[n] = 1 - message[n];
               } 
               else {
                  h_messageRecieved[n] = message[n];
               }
            }

            //---------------------------------------Decoder---------------------------------------
            
            // Initialize CtoV to 0
            memset(h_CtoV,0,num_branches*sizeof(int));

            // make... a second recieved message vector... for reasons.
            memmove(h_decide,h_messageRecieved,N*sizeof(int));

            //----------------------------------------Send data to device----------------------------------------
            cudaMemcpy(d_interleaver, h_interleaver, num_branches * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(d_matrix_flat, h_matrix_flat, num_branches * sizeof(unsigned));
            cudaMemcpy(d_CtoV, h_CtoV, num_branches * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_VtoC, h_VtoC, num_branches * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_receivedWord, h_messageRecieved, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_decide, h_decide, N * sizeof(int), cudaMemcpyHostToDevice);

            // loop through each itteration, and break on convergence
            itter = 0;
            hasConverged = false;
            while(itter < itteration_count && !hasConverged) {
               
               // Different itterations have different kernels
               if (itter == 0) {
                  DataPassGB_0<<<GridDim1, BlockDim1>>>(d_VtoC, d_receivedWord, d_interleaver, N, num_branches);
               }
               else if (itter < 15) {
                  DataPassGB_1<<<GridDim1, BlockDim1>>>(d_VtoC, d_CtoV, d_receivedWord, d_interleaver, N, num_branches);
               }
               else {
                  DataPassGB_2<<<GridDim1, BlockDim1>>>(d_VtoC, d_CtoV, d_receivedWord, d_interleaver, N, num_branches, varr);
               }

               CheckPassGB<<<GridDim2, BlockDim2>>>(d_CtoV, d_VtoC, M, num_branches);

               APP_GB<<<GridDim1, BlockDim1>>>(d_decide, d_CtoV, d_receivedWord, d_interleaver, N, num_branches);

               ComputeSyndrome<<<GridDim2, BlockDim2>>>(d_synd, d_decide, M, num_branches);

               cudaMemcpy(h_synd, d_synd, M * sizeof(int), cudaMemcpyDeviceToHost);

               // check for convergence
               hasConverged = true;
               for (unsigned kk = 0; kk < M; kk++) {
                  if (h_synd[kk] == 1) {
                     hasConverged = false;
                     break;
                  }
               }

               itter++;
            }

            cudaMemcpy(h_decide, d_decide, N * sizeof(int), cudaMemcpyDeviceToHost);

            //============================================================================
            // Compute Statistics
            //============================================================================
            frames_tested++;
            err_count = 0;

            //
            for (unsigned k = 0; k < N; k++) {
               if (h_decide[k] != message[k]) {
                  ++err_count;
               }
            }

            // 
            bit_error_count = bit_error_count + err_count;
            
            // Case Divergence
            if (!hasConverged) {
               NiterMoy = NiterMoy + itteration_count;
               err_total_count++;
            }

            // Case Convergence to Right message
            if ((hasConverged) && (err_count == 0)) {
               NiterMax = max(NiterMax, itter + 1);
               NiterMoy = NiterMoy + (itter + 1);
            }

            // Case Convergence to Wrong message
            if ((hasConverged) && (err_count != 0)) {
               NiterMax = max(NiterMax, itter + 1);
               NiterMoy = NiterMoy + (itter + 1);
               err_total_count++;
               missed_error_count++;
               Dmin = min(Dmin, err_count);
            }

            // Stopping Criterion
            if (err_total_count == frame_count) {
               break;
            }
         }

#ifdef PROFILE  
         gettimeofday(&stop,NULL);  
         diffTime = diff_time_usec(start,stop);  
         fprintf(stderr,"time for loops in MicroSec: %lu \n",diffTime);
#endif 

         std::cout << alpha << "\t";
         std::cout << bit_error_count << " (" << (float)bit_error_count / N / frames_tested << ")\t";
         std::cout << err_total_count << " (" << (float)err_total_count / frames_tested << ")\t";
         std::cout << frames_tested << "\t\t";
         std::cout << (float)NiterMoy / frames_tested << "(" << NiterMax << ")\t\t\t";
         std::cout << missed_error_count << "(" << Dmin << ")\t" << std::endl;
         
         // TODO go ahead and keep this^ but this data should be sent to a CSV
      }

      // Freeing memory on the GPU
      cudaFree(d_CtoV);
      cudaFree(d_VtoC);
      cudaFree(d_interleaver);
      cudaFree(d_synd);
      cudaFree(d_receivedWord);
      cudaFree(d_decide);
   }
   else{
      fprintf(stderr,"Usage: PGaB /Path/To/Data/File Path/to/output/file");
   }

   

   return 0;
}