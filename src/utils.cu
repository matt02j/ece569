///////////////////////////////////////////////////////////////////////////////////////////////////
// Created By        : Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
// Modified By       : Matt <LastName>, Jeremy Seers, Sebastian Thiem
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