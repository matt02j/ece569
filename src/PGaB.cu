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

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 

#include <unistd.h> 
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
// #include <omp.h>

#define arrondi(x)((ceil(x) - x) < (x - floor(x)) ? (int) ceil(x) : (int) floor(x))
#define min(x, y)((x) < (y) ? (x) : (y))
#define signf(x)((x) >= 0 ? 0 : 1)
#define max(x, y)((x) < (y) ? (y) : (x))
#define SQR(A)((A) * (A))
#define BPSK(x)(1 - 2 * (x))
#define PI 3.1415926536
   
#define PROFILE

__constant__ int Mat_device[5184];

// TODO Verify stride paterns
// All of the following functions use this
// for (unsigned stride = 0; stride < (NbBranch / N); stride++)
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
  
   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   if (id < M) {

      int signe = 0;

      // For indexing the node arrays
      unsigned node_idx = 0;

      // 
      unsigned strides = (NbBranch / M);
      
      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         node_idx = stride + id * strides;
         signe ^= VtoC[node_idx];
      }
      
      // 
      for (unsigned stride = 0; stride < strides; stride++) {
         
         node_idx = stride + id * strides;
         CtoV[node_idx] = signe ^ VtoC[node_idx];
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
__global__ void ComputeSyndrome(int * Synd, int * Decide, int M, int NbBranch) {

   // calculate the current index on the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // intialize ___ regardless of bounds...
   int synd = 0;
   
   if (id < M) {
      
      unsigned strides = (NbBranch / M);

      // 
      for (unsigned stride = 0; stride < strides; stride++) {

         __syncthreads();

         synd ^=Decide[Mat_device[id * strides + stride]];
      }
   }

   // NOTE write back regardless of thread
   Synd[id]=synd;
}

// 
unsigned GaussianElimination_MRB(int* Perm, int** MatOut, int** Mat, int M, int N) {
   
   // 
   int buf;
   
   // 
   // used in a for loop with break
   // TODO bad practice, fix for loop if possible
   unsigned ind;

   // 
   unsigned Rank;

   // 
   int* Index;
   Index = (int*) calloc(N, sizeof(int));

   // Triangularization
   int indColumn = 0;
   int nb = 0;
   int dep = 0;

   //
   for (unsigned m = 0; m < M; m++) {

      // 
      if (indColumn == N) {
         dep = M - m;
         break;
      }

      // 
      for (ind = m; ind < M; ind++) {
         if (Mat[ind][indColumn] != 0) {
            break;
         }
      }

      // If a "1" is found on the column, permutation of rows
      if (ind < M) {

         // swap row "m" with row "ind" from "indColumn" to the end of the row
         for (unsigned n = indColumn; n < N; n++) {
            buf = Mat[m][n];
            Mat[m][n] = Mat[ind][n];
            Mat[ind][n] = buf;
         }

         // bottom of the column ==> 0
         for (unsigned m1 = m + 1; m1 < M; m1++) {

            // 
            if (Mat[m1][indColumn] == 1) {

               // XOR row "m1" with row "m" from "indColumn" to the end of the row
               for (unsigned n = indColumn; n < N; n++) {
                  Mat[m1][n] = Mat[m1][n] ^ Mat[m][n];
               }
            }
         }

         Perm[m] = indColumn;
      }
      else { 

         // else we "mark" the column.
         Index[nb++] = indColumn;
         m--;
      }

      indColumn++;
   }

   // 
   Rank = M - dep;

   // 
   for (unsigned n = 0; n < nb; n++) {
      Perm[Rank + n] = Index[n];
   }

   // Permutation of the matrix
   for (unsigned m = 0; m < M; m++) {
      for (unsigned n = 0; n < N; n++) {
         MatOut[m][n] = Mat[m][Perm[n]];
      }
   }

   // Diagonalization
   for (unsigned m = 0; m < (Rank - 1); m++) {
      for (unsigned n = m + 1; n < Rank; n++) {
         
         //
         if (MatOut[m][n] == 1) {
            for (unsigned k = n; k < N; k++) {
               MatOut[m][k] = MatOut[n][k] ^ MatOut[m][k];
            }
         }
      }
   }

   free(Index);

   return Rank;
}

unsigned long diff_time_usec(struct timeval start, struct timeval stop){
  unsigned long diffTime;
  if(stop.tv_usec < start.tv_usec){
   diffTime = 1000000 + stop.tv_usec-start.tv_usec;
        diffTime += 1000000 * (stop.tv_sec - 1 - start.tv_sec);
  }
  else{
   diffTime = stop.tv_usec - start.tv_usec;
        diffTime += 1000000 * (stop.tv_sec - start.tv_sec);
  }
  return diffTime;
}

int main(int argc, char * argv[]) {
   if(argc < 3 ){
      fprintf(stderr,"Usage: PGaB /Path/To/Data/File Path/to/output/file");
   }

   struct timeval start,stop;
   unsigned long diffTime=0;

   // 
   FILE * f;

   // 
   int Graine;

   unsigned NbIter;

   unsigned nbtestedframes;

   unsigned NBframes;

   // 
   float alpha_max, alpha_min, alpha_step, alpha, NbMonteCarlo;

   // ----------------------------------------------------
   // read command line params
   // ----------------------------------------------------
   char* FileName;
   char* FileMatrix;
   char* FileResult;
   FileName = (char * ) malloc(200);
   FileMatrix = (char * ) malloc(200);
   FileResult = (char * ) malloc(200);

   strcpy(FileMatrix, argv[1]); // Matrix file
   strcpy(FileResult, argv[2]); // Results file
   //--------------Simulation input for GaB BF-------------------------
   NbMonteCarlo = 1000000; // Maximum nb of codewords sent
   NbIter = 100; // Maximum nb of iterations
   alpha = 0.01; // Channel probability of error
   NBframes = 100; // Simulation stops when NBframes in error
   Graine = 1; // Seed Initialization for Multiple Simulations

   // shortend for testing purposes, was alpha_max=0.06
   alpha_max = 0.03; //Channel Crossover Probability Max and Min
   alpha_min = 0.03;
   alpha_step = 0.01;

   // ----------------------------------------------------
   // Load Matrix
   // ----------------------------------------------------
   int * ColumnDegree, * RowDegree, ** Mat_host, * Mat_host1;
   int M, N, m, n, k;
   strcpy(FileName, FileMatrix);
   strcat(FileName, "_size");
   f = fopen(FileName, "r");
   fscanf(f, "%d", & M);
   fscanf(f, "%d", & N);
   ColumnDegree = (int * ) calloc(N, sizeof(int));
   RowDegree = (int * ) calloc(M, sizeof(int));
   fclose(f);
   strcpy(FileName, FileMatrix);
   strcat(FileName, "_RowDegree");
   f = fopen(FileName, "r");

   for (m = 0; m < M; m++) {
      fscanf(f, "%d", & RowDegree[m]);
   }
   fclose(f);

   Mat_host = (int ** ) calloc(M, sizeof(int * ));

   for (m = 0; m < M; m++) {
      Mat_host[m] = (int * ) calloc(RowDegree[m], sizeof(int));
   }

   //changes made
   strcpy(FileName, FileMatrix);

   f = fopen(FileName, "r");
   for (m = 0; m < M; m++) {
      for (k = 0; k < RowDegree[m]; k++) {
         fscanf(f, "%d", & Mat_host[m][k]);
      }
   }
   fclose(f);

   for (m = 0; m < M; m++) {
      for (k = 0; k < RowDegree[m]; k++){
         ColumnDegree[Mat_host[m][k]]++;
      }
   }
   //TODO free filename and filematrix
   printf("Matrix Loaded \n");

   // ----------------------------------------------------
   // Build Graph
   // ----------------------------------------------------
   int NbBranch, ** NtoB, * Interleaver_host, * ind, numColumn, numBranch;
   
   NbBranch = 0;
   
   for (m = 0; m < M; m++) {
      NbBranch = NbBranch + RowDegree[m];
   }
   
   NtoB = (int ** ) calloc(N, sizeof(int * ));
   
   for (n = 0; n < N; n++) {
      NtoB[n] = (int * ) calloc(ColumnDegree[n], sizeof(int));
   }
   
   Interleaver_host = (int * ) calloc(NbBranch, sizeof(int));
   ind = (int * ) calloc(N, sizeof(int));
   numBranch = 0;
   
   for (m = 0; m < M; m++) {
      for (k = 0; k < RowDegree[m]; k++) {
         numColumn = Mat_host[m][k];
         NtoB[numColumn][ind[numColumn]++] = numBranch++;
      }
   }

   free(ind);
   numBranch = 0;

   for (n = 0; n < N; n++) {
      for (k = 0; k < ColumnDegree[n]; k++) {
         Interleaver_host[numBranch++] = NtoB[n][k];
      }
   }
   
   Mat_host1 = (int * ) calloc(NbBranch, sizeof(int));
   
   for (m = 0; m < M; m++) {
      for (n = 0; n < 8; n++) {
         Mat_host1[m * 8 + n] = Mat_host[m][n];
      }
   }

   printf("Graph Build \n");

   // ----------------------------------------------------
   // Decoder
   // ----------------------------------------------------
   int * CtoV_host, * VtoC_host, * Codeword, * Receivedword_host, * Decide_host, * U, l, kk, * CtoV_device, * VtoC_device, * Receivedword_device, * Decide_device;
   int iter;
   int * Synd_host, * Synd_device, * Interleaver_device;
   int Synd_host1 = 0;
   Synd_host = (int * ) calloc(M, sizeof(int));
   int varr;
   
   if (rand() % 100 >= 80) {
      varr = 1;
   } 
   else {
      varr = 0;
   }

   //Allocating memory for variables on device as well as the host
   cudaMalloc((void ** ) & Synd_device, M * sizeof(int));
   CtoV_host = (int * ) calloc(NbBranch, sizeof(int));
   cudaMalloc((void ** ) & CtoV_device, NbBranch * sizeof(int));
   VtoC_host = (int * ) calloc(NbBranch, sizeof(int));
   cudaMalloc((void ** ) & VtoC_device, NbBranch * sizeof(int));
   Codeword = (int * ) calloc(N, sizeof(int));
   Receivedword_host = (int * ) calloc(N, sizeof(int));
   cudaMalloc((void ** ) & Receivedword_device, N * sizeof(int));
   Decide_host = (int * ) calloc(N, sizeof(int));
   cudaMalloc((void ** ) & Decide_device, N * sizeof(int));
   cudaMalloc((void ** ) & Interleaver_device, NbBranch * sizeof(int));
   U = (int * ) calloc(N, sizeof(int));
   srand48(time(0) + Graine * 31 + 113);

   //Initializing grid and block dimensions

   dim3 GridDim1((N - 1) / 1024 + 1, 1);
   dim3 BlockDim1(1024);
   dim3 GridDim2((M - 1) / 1024 + 1, 1);
   dim3 BlockDim2(1024);

   // ----------------------------------------------------
   // Gaussian Elimination for the Encoding Matrix (Full Representation)
   // ----------------------------------------------------
   int ** MatFull, ** MatG, * PermG;
   int rank;

   MatG = (int ** ) calloc(M, sizeof(int * ));
   

   for (m = 0; m < M; m++) {
      MatG[m] = (int * ) calloc(N, sizeof(int));
   }
   
   MatFull = (int ** ) calloc(M, sizeof(int * ));
   
   for (m = 0; m < M; m++) {
      MatFull[m] = (int * ) calloc(N, sizeof(int));
   }

   PermG = (int * ) calloc(N, sizeof(int));

   for (n = 0; n < N; n++) {
      PermG[n] = n;
   }

   for (m = 0; m < M; m++) {
      for (k = 0; k < RowDegree[m]; k++) {
         MatFull[m][Mat_host[m][k]] = 1;
      }
   }
   rank = GaussianElimination_MRB(PermG, MatG, MatFull, M, N);

   // Variables for Statistics
   int IsCodeword, nb;
   int NiterMoy, NiterMax;
   int Dmin;
   int NbTotalErrors, NbBitError;
   int NbUnDetectedErrors, NbError;

   strcpy(FileName, FileResult);
   f = fopen(FileName, "w");
   fprintf(f, "-------------------------Gallager B--------------------------------------------------\n");
   fprintf(f, "alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");

   printf("-------------------------Gallager B--------------------------------------------------\n");
   printf("alpha\t\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");

   // 
   for (alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {

      NiterMoy = 0;
      NiterMax = 0;
      Dmin = 1e5;
      NbTotalErrors = 0;
      NbBitError = 0;
      NbUnDetectedErrors = 0;
      NbError = 0;

      // Copying contents from the host to the device
      cudaMemcpy(Interleaver_device, Interleaver_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(Mat_device, Mat_host1, NbBranch * sizeof(int));
      cudaMemcpy(CtoV_device, CtoV_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(VtoC_device, VtoC_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);

      // encoding
#ifdef PROFILE 
  gettimeofday(&start,NULL);
#endif 
      for (nb = 0, nbtestedframes = 0; nb < NbMonteCarlo; nb++) {
         
         //
         memset(U,0,rank*sizeof(int));
         
         // randomly gerenates a uniform distribution of 0s and 1s
         for (k = rank; k < N; k++) {
            U[k] = floor(drand48() * 2);
         }

         // TODO this is what takes ~60% of the whole program
         for (k = rank - 1; k >= 0; k--) {
            for (l = k + 1; l < N; l++) {
               U[k] = U[k] ^ (MatG[k][l] * U[l]);
            }
         }

         //
         for (k = 0; k < N; k++) {
            Codeword[PermG[k]] = U[k];
         }
         
         // All zero codeword
         //for (n=0;n<N;n++) { Codeword[n]=0; }

         // Add Noise 
         for (n = 0; n < N; n++){
            if (drand48() < alpha){
               Receivedword_host[n] = 1 - Codeword[n];
            } 
            else {
               Receivedword_host[n] = Codeword[n];
            }
         }

         //============================================================================
         // Decoder
         //============================================================================
         
         //
         memset(CtoV_host,0,NbBranch*sizeof(int));

         //
         memmove(Decide_host,Receivedword_host,N*sizeof(int));

         cudaMemcpy(Receivedword_device, Receivedword_host, N * sizeof(int), cudaMemcpyHostToDevice);
         cudaMemcpy(Decide_device, Decide_host, N * sizeof(int), cudaMemcpyHostToDevice);

         for (iter = 0; iter < NbIter; iter++) {
            
            // Different itterations have different kernels
            if (iter == 0) {
               DataPassGB_0<<<GridDim1, BlockDim1>>>(VtoC_device, Receivedword_device, Interleaver_device, N, NbBranch);
            }
            else if (iter < 15) {
               DataPassGB_1<<<GridDim1, BlockDim1>>>(VtoC_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch);
            }
            else {
               DataPassGB_2<<<GridDim1, BlockDim1>>>(VtoC_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch, varr);
            }

            CheckPassGB<<<GridDim2, BlockDim2>>>(CtoV_device, VtoC_device, M, NbBranch);

            APP_GB<<<GridDim1, BlockDim1>>>(Decide_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch);

            ComputeSyndrome<<<GridDim2, BlockDim2>>>(Synd_device, Decide_device, M, NbBranch);

            cudaMemcpy(Synd_host, Synd_device, M * sizeof(int), cudaMemcpyDeviceToHost);

            // 
            int count1 = 0;
            for (kk = 0; kk < M; kk++) {
               if (Synd_host[kk] == 1) {
                  count1++;
                  break;
               }
            }

            // 
            if (count1 > 0) {
               Synd_host1 = 0;
            }           
            else {
               Synd_host1 = 1;
            }

            // if (IsCodeword) algorithm has converged and we are done, exit the loop
            IsCodeword = Synd_host1;
            if (IsCodeword) {
               break;
            }
         }

         cudaMemcpy(Decide_host, Decide_device, N * sizeof(int), cudaMemcpyDeviceToHost);

         //============================================================================
         // Compute Statistics
         //============================================================================
         nbtestedframes++;
         NbError = 0;

         //
         for (k = 0; k < N; k++) {
            if (Decide_host[k] != Codeword[k]) {
               ++NbError;
            }
         }

         // 
         NbBitError = NbBitError + NbError;
         
         // Case Divergence
         if (!IsCodeword) {
            NiterMoy = NiterMoy + NbIter;
            NbTotalErrors++;
         }

         // Case Convergence to Right Codeword
         if ((IsCodeword) && (NbError == 0)) {
            NiterMax = max(NiterMax, iter + 1);
            NiterMoy = NiterMoy + (iter + 1);
         }

         // Case Convergence to Wrong Codeword
         if ((IsCodeword) && (NbError != 0)) {
            NiterMax = max(NiterMax, iter + 1);
            NiterMoy = NiterMoy + (iter + 1);
            NbTotalErrors++;
            NbUnDetectedErrors++;
            Dmin = min(Dmin, NbError);
         }

         // Stopping Criterion
         if (NbTotalErrors == NBframes) {
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
   cudaFree(CtoV_device);
   cudaFree(VtoC_device);
   cudaFree(Interleaver_device);
   cudaFree(Synd_device);
   cudaFree(Receivedword_device);
   cudaFree(Decide_device);
   fclose(f);

   return 0;
}
