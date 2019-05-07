///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matthew Johnson, Jeremy Sears, Sebastian Thiem
//                   :
// Organization:     : The University of Arizona
//                   :
// Project Name:     : OPTIMIZATIONS OF LDPC DECODERS IN CUDA
// File Name:        : utils.cu
//                   :
// Create Date:      : 8 May 2017
// Modified          : 26 March 2019
//                   :
// Description:      : Utility functions for PGaB
//                   :
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "utils.cuh"

#define CUDA_CHECK(ans)                                                   \
   { gpuAssert((ans), __FILE__, __LINE__); }
   
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
                  file, line);
      if (abort)
         exit(code);
   }
}


// free in 2d (int)
void free2d(unsigned** mem, const unsigned depth) {

   for (unsigned i = 0; i < depth; i++) {
      free(mem[i]);
   }
   free(mem);
}

// free in 2d (unsigned)
void free2d(int** mem, const unsigned depth) {

   for (unsigned i = 0; i < depth; i++) {
      free(mem[i]);
   }
   free(mem);
}

// 
unsigned GaussianElimination_MRB(unsigned* Perm, unsigned** MatOut, unsigned** Mat, unsigned M, unsigned N) {
   
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
   unsigned indColumn = 0;
   unsigned nb = 0;
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

//#ifdef PROFILE
//
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
//#endif

// Initialize the NtoB matrix then unroll it into the interleaved matrix
// TODO could possibly due with an improvement in the NtoB initialization as the current method seems kinda hacky 
// return num_branches
void initInterleaved(unsigned * h_interleaver, unsigned** data_matrix, const unsigned* rowRanks, const unsigned* hist, const unsigned depth, const unsigned max_val) {

   /*******
   * NtoB *
   *******/

   // temp array the length of max_val in the input matrix
   unsigned* ind;
   ind = (unsigned*)calloc(max_val, sizeof(unsigned));

   // allocate another matrix 
   // where col width is based on the hist results
   unsigned** NtoB;
   unsigned histy;
   NtoB = (unsigned**)malloc(max_val * sizeof(unsigned*));
   for (unsigned i = 0; i < max_val; i++) {
      histy = hist[i];
      NtoB[i] = (unsigned*)malloc(histy * sizeof(unsigned));
   }

   //
   unsigned col = 0;
   unsigned branch = 0;
   unsigned ind_loc = 0;   // local ind
   for (unsigned i = 0; i < depth; i++) {
      for (unsigned j = 0; j < rowRanks[i]; j++) {

         // read host matrix element
         col = data_matrix[i][j];

         // read from the ind, given the host value
         ind_loc = ind[col];

         // set NtoB element
         NtoB[col][ind_loc] = branch;

         // increment the local ind and branch counter
         ind_loc++;
         branch++;

         // update ind in memory
         ind[col] = ind_loc;
      }
   }

   // dont need this anymore
   free(ind);

   /**************
   * Interleaver *
   **************/

   // unroll NtoB into interleaver vector
   unsigned i = 0;

//TODO verify why we need a histogram when they are all the same
   for (unsigned k = 0; k < hist[0]; k++) {
      for (unsigned n = 0; n < max_val; n++) {

         h_interleaver[i] =NtoB[n][k];
         i++;
      }
   }

   // Free NtoB matrix
   free2d(NtoB, max_val);
}

// read in row rank matrix from local file
void readRowRanks(unsigned* rowRanks, const unsigned depth, const char* fileName) {

   // read from file
   // TODO use streaming, even consider reformatting data vectors to allow one liner read
   FILE* f;
   f = fopen(fileName, "r");
   for (unsigned m = 0; m < depth; m++) {
      fscanf(f, "%d", &rowRanks[m]);
   }
   fclose(f);
}

// read in data matrix from local file
void readDataMatrix(unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const char* fileName) {

   // read in from matrix data file to the host matrix in memory
   // TODO use streaming, even consider reformatting data vectors to allow one liner read
   FILE* f;
   f = fopen(fileName, "r");
   for (unsigned m = 0; m < depth; m++) {
      for (unsigned k = 0; k < rowRanks[m]; k++) {
         fscanf(f, "%d", &data_matrix[m][k]);
      }
   }
   fclose(f);
}

// histogram
void histogram(unsigned* hist, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned max_val) {

   // do hist
   for (unsigned m = 0; m < depth; m++) {
      for (unsigned k = 0; k < rowRanks[m]; k++) {
         hist[data_matrix[m][k]]++;
      }
   }
}

// unroll the data matrix
void unrollMatrix(unsigned* unrolledMatrix, unsigned** data_matrix, const unsigned* rowRanks, const unsigned depth, const unsigned num_branches) {
	
   // unroll the memory

   unsigned i = 0;
   for (unsigned n = 0; n < 8; n++) {
      for (unsigned m = 0; m < depth; m++) {
         unrolledMatrix[i] = data_matrix[m][n];
         i++;
      }
   }
	
}

void print_array_int(int* arr, int size){
	for(int i=0; i<size; i++){
		printf("%d ",arr[i]);
	}
	printf("\n");
}
void print_array_char(unsigned char* arr, int size){
	for(int i=0; i<size; i++){
		printf("%d ",arr[i]);
	}
	printf("\n");
}
