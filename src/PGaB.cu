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
// #include <omp.h>

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
   if(argc > 1){

#ifdef VERBOSE
      std::cout << "Starting." << std::endl << std::endl;
#endif
      // --------------------------------------------------------------------------
      // Parameters and memory
      // --------------------------------------------------------------------------

      //-------------------------Simulation parameters for PGaB-------------------------
      float NbMonteCarlo = 1000000.0f;   // Maximum number of codewords sent
      unsigned itteration_count = 100; // Maximum nb of iterations
      unsigned frame_count = 100;      // Simulation stops when frame_count in error
      unsigned seed = 42;               // Seed Initialization for consistent Simulations

      // Channel Crossover Probability Max and Min
      float alpha = 0.01f;     // Channel probability of error
      float alpha_max = 0.03f; // shortend for testing purposes, was alpha_max=0.06
      float alpha_min = 0.03f; 
      float alpha_step = 0.01f;

      //-------------------------Host memory structures-------------------------
      unsigned* h_interleaver;
      int* h_CtoV;
      int* h_VtoC;
      int* h_receivedWord;
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
      unsigned* unrolledMatrix;  // unrolled matrix
      unsigned* hist;            // histogram for <unk>
      int* U;                    // 
      int* Codeword;             // 
      int** MatFull;             // 
      int** MatG;                // 
      int* PermG;                // 
      
      //------------------------------------Miscellenious Variables------------------------------------
      unsigned varr = (rand() % 100 >= 80)? 1 : 0;    // stochastic semi-boolean variable, 
                                                      // integer value is used so keep numerical value
      FILE* f;                            // file IO, tbTerminated 
      std::string matrixAddr(argv[1]);    // convert data addr to a string
      unsigned num_branches;              // total number of elements in the test data matrix
      unsigned rank;                      // returned from Gaussian Elimination

      // Variables for Statistics
      int IsCodeword;
      int NiterMoy;
      int NiterMax;
      int Dmin;
      unsigned NbTotalErrors;
      unsigned NbBitError;
      unsigned NbUnDetectedErrors;
      unsigned NbError;

#ifdef PROFILE
      // Used for timing CPU
      struct timeval start;
      struct timeval stop;
      unsigned long diffTime = 0;
#endif

#ifdef VERBOSE
      std::cout << "Running preliminary calculations...";
#endif

      //------------------------------------Preliminary Calculations------------------------------------
      // Before allocating most of the memory, some 
      // preliminary calculations and file reads must be done

      // seed the RNG
      // srand48(time(0) + seed * 31 + 113); // ignore seed, be random
      srand48(seed * 31 + 113);

      // Initialize grid and block dimensions
      dim3 GridDim1((N - 1) / BLOCK_DIM_1 + 1, 1);
      dim3 BlockDim1(BLOCK_DIM_1);
      dim3 GridDim2((M - 1) / BLOCK_DIM_2 + 1, 1);
      dim3 BlockDim2(BLOCK_DIM_2);

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

      // allocate and generate histogram on the data
      hist = (unsigned*)malloc(N * sizeof(unsigned));
      histogram(hist, data_matrix, rowRanks, M, N);

      // allocate and generate interleaver (allocation done in method)
      h_interleaver = (unsigned*)malloc(num_branches * sizeof(unsigned));
      initInterleaved(h_interleaver, rowRanks, hist, M, N);

      // allocate and unroll host matrix into a flat host vector
      unrolledMatrix = (unsigned*)malloc(num_branches * sizeof(unsigned));
      unrollMatrix(unrolledMatrix, data_matrix, rowRanks, M, num_branches);

      // free no longer needed structures
      free(hist);

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Allocating memory...";
#endif
      //-------------------------Host Allocations-------------------------
      h_synd = (int*)calloc(M, sizeof(int));
      h_CtoV = (int*) calloc(num_branches, sizeof(int));
      h_VtoC = (int*) calloc(num_branches, sizeof(int));
      h_receivedWord = (int*) calloc(N, sizeof(int));
      h_decide = (int*) calloc(N, sizeof(int));

      //-------------------------Device Allocations-------------------------
      cudaMalloc((void**) &d_interleaver, num_branches * sizeof(unsigned));
      cudaMalloc((void**) &d_synd, M * sizeof(int));
      cudaMalloc((void**) &d_CtoV, num_branches * sizeof(int));
      cudaMalloc((void**) &d_VtoC, num_branches * sizeof(int));
      cudaMalloc((void**) &d_receivedWord, N * sizeof(int));
      cudaMalloc((void**) &d_decide, N * sizeof(int));

      //-------------------------Other Allocations-------------------------
      U = (int*)calloc(N, sizeof(int));
      Codeword = (int*)calloc(N, sizeof(int));

      MatG = (int**)malloc(M * sizeof(int*));
      for (unsigned m = 0; m < M; m++) {
         MatG[m] = (int*)calloc(N, sizeof(int));
      }

      MatFull = (int**)malloc(M * sizeof(int * ));
      for (unsigned m = 0; m < M; m++) {
         MatFull[m] = (int*)malloc(N * sizeof(int));
      }

      PermG = (int*)malloc(N * sizeof(int));
    
#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Gaussian Elimination...";
#endif
      // -------------------------------------------------------------------
      // Gaussian Elimination for the Encoding Matrix (Full Representation)
      // -------------------------------------------------------------------

      // uh....
      for (unsigned n = 0; n < N; n++) {
         PermG[n] = n;
      }

      // okay...
      for (unsigned m = 0; m < M; m++) {
         for (unsigned k = 0; k < rowRanks[m]; k++) {
            MatFull[m][data_matrix[m][k]] = 1;
         }
      }
      rank = GaussianElimination_MRB(PermG, MatG, MatFull, M, N);

      // free no longer needed data structures
      free2d(MatFull, M);
      free2d(data_matrix, M);
      free(rowRanks);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      f = fopen((matrixAddr+"_Res").c_str(), "w");
      fprintf(f, "-------------------------Gallager B--------------------------------------------------\n");
      fprintf(f, "alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");

      printf("-------------------------Gallager B--------------------------------------------------\n");
      printf("alpha\t\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");

#ifdef VERBOSE
      std::cout << "Done." << std::endl;

      std::cout << "Running Gaussian Elimination...";
#endif
      // loop from alpha max to alpha min
      for (unsigned alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {

         NiterMoy = 0;
         NiterMax = 0;
         Dmin = 1e5;
         NbTotalErrors = 0;
         NbBitError = 0;
         NbUnDetectedErrors = 0;
         NbError = 0;

         // Copying contents from the host to the device
         cudaMemcpy(d_interleaver, h_interleaver, num_branches * sizeof(int), cudaMemcpyHostToDevice);
         cudaMemcpyToSymbol(Mat_device, unrolledMatrix, num_branches * sizeof(unsigned));
         cudaMemcpy(d_CtoV, h_CtoV, num_branches * sizeof(int), cudaMemcpyHostToDevice);
         cudaMemcpy(d_VtoC, h_VtoC, num_branches * sizeof(int), cudaMemcpyHostToDevice);

         // encoding
   #ifdef PROFILE 
     gettimeofday(&start,NULL);
   #endif 
         unsigned nbtestedframes = 0;
         for (unsigned nb = 0; nb < NbMonteCarlo; nb++) {
            
            //
            memset(U,0,rank*sizeof(int));
            
            // randomly gerenates a uniform distribution of 0s and 1s
            for (unsigned k = rank; k < N; k++) {
               U[k] = floor(drand48() * 2);
            }

            // TODO this is what takes ~60% of the whole program
            for (int k = rank - 1; k >= 0; k--) {
               for (unsigned l = k + 1; l < N; l++) {
                  U[k] = U[k] ^ (MatG[k][l] * U[l]);
               }
            }

            //
            for (unsigned k = 0; k < N; k++) {
               Codeword[PermG[k]] = U[k];
            }
            
            // All zero codeword
            //for (n=0;n<N;n++) { Codeword[n]=0; }

            // Add Noise 
            for (unsigned n = 0; n < N; n++){
               if (drand48() < alpha){
                  h_receivedWord[n] = 1 - Codeword[n];
               } 
               else {
                  h_receivedWord[n] = Codeword[n];
               }
            }

            //============================================================================
            // Decoder
            //============================================================================
            
            //
            memset(h_CtoV,0,num_branches*sizeof(int));

            //
            memmove(h_decide,h_receivedWord,N*sizeof(int));

            cudaMemcpy(d_receivedWord, h_receivedWord, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_decide, h_decide, N * sizeof(int), cudaMemcpyHostToDevice);

            // loop through each itteraction, and break on convergence
            unsigned itter = 0;
            bool hasConverged = false;
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
            nbtestedframes++;
            NbError = 0;

            //
            for (unsigned k = 0; k < N; k++) {
               if (h_decide[k] != Codeword[k]) {
                  ++NbError;
               }
            }

            // 
            NbBitError = NbBitError + NbError;
            
            // Case Divergence
            if (!IsCodeword) {
               NiterMoy = NiterMoy + itteration_count;
               NbTotalErrors++;
            }

            // Case Convergence to Right Codeword
            if ((IsCodeword) && (NbError == 0)) {
               NiterMax = max(NiterMax, itter + 1);
               NiterMoy = NiterMoy + (itter + 1);
            }

            // Case Convergence to Wrong Codeword
            if ((IsCodeword) && (NbError != 0)) {
               NiterMax = max(NiterMax, itter + 1);
               NiterMoy = NiterMoy + (itter + 1);
               NbTotalErrors++;
               NbUnDetectedErrors++;
               Dmin = min(Dmin, NbError);
            }

            // Stopping Criterion
            if (NbTotalErrors == frame_count) {
               break;
            }
         }

   #ifdef PROFILE  
     gettimeofday(&stop,NULL);  
     diffTime = diff_time_usec(start,stop);  
     fprintf(stderr,"time for loops in MicroSec: %lu \n",diffTime);
   #endif 

         printf("%1.5f\t\t", alpha);
         printf("%10d (%1.6f)\t\t", NbBitError, (float) NbBitError / N / nbtestedframes);
         printf("%4d (%1.6f)\t\t", NbTotalErrors, (float) NbTotalErrors / nbtestedframes);
         printf("%10d\t\t", nbtestedframes);
         printf("%1.2f(%d)\t\t", (float) NiterMoy / nbtestedframes, NiterMax);
         printf("%d(%d)\n", NbUnDetectedErrors, Dmin);

         fprintf(f, "%1.5f\t\t", alpha);
         fprintf(f, "%10d (%1.8f)\t\t", NbBitError, (float) NbBitError / N / nbtestedframes);
         fprintf(f, "%4d (%1.8f)\t\t", NbTotalErrors, (float) NbTotalErrors / nbtestedframes);
         fprintf(f, "%10d\t\t", nbtestedframes);
         fprintf(f, "%1.2f(%d)\t\t", (float) NiterMoy / nbtestedframes, NiterMax);
         fprintf(f, "%d(%d)\n", NbUnDetectedErrors, Dmin);
      }

      //Freeing memory on the GPU
      cudaFree(d_CtoV);
      cudaFree(d_VtoC);
      cudaFree(d_interleaver);
      cudaFree(d_synd);
      cudaFree(d_receivedWord);
      cudaFree(d_decide);
      fclose(f);
   }
   else{
      fprintf(stderr,"Usage: PGaB /Path/To/Data/File Path/to/output/file");
   }

   

   return 0;
}