/* ###########################################################################################################################
## Organization         : The University of Arizona
##                      :
## File name            : PGaB.cu
## Language             : CUDA C
## Short description    : Probablistic Gallager-B algorithm
##                      :
##                      :
##                      :
## History              : Modified 05/08/2017, Created by Adithya Madhava Rao, Harshil Pankaj Kakaiya, Priyanka Devashish Goswami
##                      :
## COPYRIGHT            : amrao@email.arizona.edu, harshilpk@email.arizona.com, priyankag@email.arizona.edu
## ######################################################################################################################## */
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include <stdio.h> 
#include <unistd.h> 
#include <cuda.h>

#define arrondi(x)((ceil(x) - x) < (x - floor(x)) ? (int) ceil(x) : (int) floor(x))
#define min(x, y)((x) < (y) ? (x) : (y))
#define signf(x)((x) >= 0 ? 0 : 1)
#define max(x, y)((x) < (y) ? (y) : (x))
#define SQR(A)((A) * (A))
#define BPSK(x)(1 - 2 * (x))
#define PI 3.1415926536

__constant__ int Mat_device[5184];

//#####################################################################################################
//Message from channel copied into variable node to check node array
__global__ void DataPassGBIter0(int * VtoC, int * Receivedword, int * Interleaver, int N, int NbBranch) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride;

  if (id < N) {

    for (stride = 0; stride < (NbBranch / N); stride++) {
      VtoC[Interleaver[id * (NbBranch / N) + stride]] = Receivedword[id];

    }

  }
}

//#####################################################################################################
//for iterations between 1 and 15, this kernel launches to pass the message from variables nodes onto the four check nodes it is connected to
__global__ void DataPassGB(int * VtoC, int * CtoV, int * Receivedword, int * Interleaver, int N, int NbBranch) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride, buf;
  int Global;
  int buf_intermediate = 0;

  if (id < N) {
    Global = (1 - 2 * Receivedword[id]);
    int i = Receivedword[id];

    for (stride = 0; stride < (NbBranch / N); stride++) {
      Global += (-2) * CtoV[Interleaver[id * (NbBranch / N) + stride]] + 1;
    }
    for (stride = 0; stride < (NbBranch / N); stride++) {
      buf = Global - ((-2) * CtoV[Interleaver[id * (NbBranch / N) + stride]] + 1);
      buf_intermediate = ((buf < 0) ? 1 : ((buf > 0) ? 0 : i));
      VtoC[Interleaver[id * (NbBranch / N) + stride]] = buf_intermediate;
    }
  }

}
//#####################################################################################################
//for iterations greater than 15, this kernel launches to pass the message from variables nodes onto the four check nodes it is connected to
__global__ void DataPassGB2(int * VtoC, int * CtoV, int * Receivedword, int * Interleaver, int N, int NbBranch, int varr) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride, buf;
  int Global;
  int buf_intermediate = 0;

  if (id < N) {
    Global = (1 - 2 * (varr ^ Receivedword[id]));
    int i = Receivedword[id];

    for (stride = 0; stride < (NbBranch / N); stride++) {
      Global += (-2) * CtoV[Interleaver[id * (NbBranch / N) + stride]] + 1;
    }
    for (stride = 0; stride < (NbBranch / N); stride++) {
      buf = Global - ((-2) * CtoV[Interleaver[id * (NbBranch / N) + stride]] + 1);
      buf_intermediate = ((buf < 0) ? 1 : ((buf > 0) ? 0 : i));
      VtoC[Interleaver[id * (NbBranch / N) + stride]] = buf_intermediate;
    }
  }
}

//##################################################################################################
//this kernel is launched to check if the CtoV copies the same information as VtoC depending upon the signe value
__global__ void CheckPassGB(int * CtoV, int * VtoC, int M, int NbBranch) {
  int stride, signe;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < M) {
    signe = 0;
    for (stride = 0; stride < (NbBranch / M); stride++) {
      signe ^= VtoC[stride + id * (NbBranch / M)];
    }
    for (stride = 0; stride < (NbBranch / M); stride++) {
      CtoV[stride + id * (NbBranch / M)] = signe ^ VtoC[stride + id * (NbBranch / M)];
    }
  }
}
//#####################################################################################################
//The following kernel is launched to decide each check node's decision whether the corresponding variable nodes are in error or not
__global__ void APP_GB(int * Decide, int * CtoV, int * Receivedword, int * Interleaver, int N, int NbBranch) {
  int stride, Global;
  int buf_intermediate = 0;
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < N) {
    int i = Receivedword[id];
    Global = (1 - 2 * Receivedword[id]);

    for (stride = 0; stride < (NbBranch / N); stride++) {
      Global += (-2) * CtoV[Interleaver[stride + id * (NbBranch / N)]] + 1;
    }
    buf_intermediate = ((Global > 0) ? 0 : ((Global < 0) ? 1 : i));
    Decide[id] = buf_intermediate;
  }
}
//#####################################################################################################
//Here a cumulative decision is made on the variable node error depending upon all the four check nodes to which the variable node is connected to 
__global__ void ComputeSyndrome(int * Synd, int * Decide, int M, int NbBranch) {

  int stride;
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  Synd[id] = 0;

  if (id < M) {
    for (stride = 0; stride < (NbBranch / M); stride++) {
      __syncthreads();
      Synd[id] = Synd[id] ^ Decide[Mat_device[id * (NbBranch / M) + stride]];
    }
  }
}
//#####################################################################################################
int GaussianElimination_MRB(int * Perm, int ** MatOut, int ** Mat, int M, int N) {
  int k, n, m, m1, buf, ind, indColumn, nb, * Index, dep, Rank;

  Index = (int * ) calloc(N, sizeof(int));

  // Triangularization
  indColumn = 0;
  nb = 0;
  dep = 0;
  for (m = 0; m < M; m++) {
    if (indColumn == N) {
      dep = M - m;
      break;
    }

    for (ind = m; ind < M; ind++) {
      if (Mat[ind][indColumn] != 0) break;
    }
    // If a "1" is found on the column, permutation of rows
    if (ind < M) {
      for (n = indColumn; n < N; n++) {
        buf = Mat[m][n];
        Mat[m][n] = Mat[ind][n];
        Mat[ind][n] = buf;
      }
      // bottom of the column ==> 0
      for (m1 = m + 1; m1 < M; m1++) {
        if (Mat[m1][indColumn] == 1) {
          for (n = indColumn; n < N; n++) Mat[m1][n] = Mat[m1][n] ^ Mat[m][n];
        }
      }
      Perm[m] = indColumn;
    }
    // else we "mark" the column.
    else {
      Index[nb++] = indColumn;
      m--;
    }

    indColumn++;
  }

  Rank = M - dep;

  for (n = 0; n < nb; n++) Perm[Rank + n] = Index[n];

  // Permutation of the matrix
  for (m = 0; m < M; m++) {
    for (n = 0; n < N; n++) MatOut[m][n] = Mat[m][Perm[n]];
  }

  // Diagonalization
  for (m = 0; m < (Rank - 1); m++) {
    for (n = m + 1; n < Rank; n++) {
      if (MatOut[m][n] == 1) {
        for (k = n; k < N; k++) MatOut[m][k] = MatOut[n][k] ^ MatOut[m][k];
      }
    }
  }
  free(Index);
  return (Rank);
}

//#####################################################################################################

int main(int argc, char * argv[]) {
  // Variables Declaration
  FILE * f;
  int Graine, NbIter, nbtestedframes, NBframes;
  float alpha_max, alpha_min, alpha_step, alpha, NbMonteCarlo;
  // ----------------------------------------------------
  // lecture des param de la ligne de commande
  // ----------------------------------------------------
  char * FileName, * FileMatrix, * FileResult;
  FileName = (char * ) malloc(200);
  FileMatrix = (char * ) malloc(200);
  FileResult = (char * ) malloc(200);

  strcpy(FileMatrix, argv[1]); // Matrix file
  strcpy(FileResult, argv[2]); // Results file
  //--------------Simulation input for GaB BF-------------------------
  //NbMonteCarlo=1000000;	    // Maximum nb of codewords sent
  NbMonteCarlo = 1000000; // Maximum nb of codewords sent
  NbIter = 100; // Maximum nb of iterations
  alpha = 0.01; // Channel probability of error
  NBframes = 100; // Simulation stops when NBframes in error
  Graine = 1; // Seed Initialization for Multiple Simulations

  // brkunl
  alpha_max = 0.06; //Channel Crossover Probability Max and Min
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
  for (m = 0; m < M; m++)
    fscanf(f, "%d", & RowDegree[m]);
  fclose(f);
  Mat_host = (int ** ) calloc(M, sizeof(int * ));
  for (m = 0; m < M; m++)
    Mat_host[m] = (int * ) calloc(RowDegree[m], sizeof(int));

  //changes made
  // printf("%d", Mat_host[m]);
  strcpy(FileName, FileMatrix);

  f = fopen(FileName, "r");
  for (m = 0; m < M; m++) {
    for (k = 0; k < RowDegree[m]; k++)
      fscanf(f, "%d", & Mat_host[m][k]);
  }
  fclose(f);
  for (m = 0; m < M; m++) {
    for (k = 0; k < RowDegree[m]; k++)
      ColumnDegree[Mat_host[m][k]]++;
  }

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
    for (k = 0; k < ColumnDegree[n]; k++)
      Interleaver_host[numBranch++] = NtoB[n][k];
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
  } else {
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
  //for (m=0;m<N;m++) printf("%d\t",PermG[m]);printf("\n");

  // Variables for Statistics
  int IsCodeword, nb;
  int NiterMoy, NiterMax;
  int Dmin;
  int NbTotalErrors, NbBitError;
  int NbUnDetectedErrors, NbError;

  strcpy(FileName, FileResult);
  f = fopen(FileName, "w");
  fprintf(f, "-------------------------Gallager B--------------------------------------------------\n");
  fprintf(f, "alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");

  printf("-------------------------Gallager B--------------------------------------------------\n");
  printf("alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");

  for (alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step) {

    NiterMoy = 0;
    NiterMax = 0;
    Dmin = 1e5;
    NbTotalErrors = 0;
    NbBitError = 0;
    NbUnDetectedErrors = 0;
    NbError = 0;

    //Copying contents from the host to th device
    cudaMemcpy(Interleaver_device, Interleaver_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mat_device, Mat_host1, NbBranch * sizeof(int));
    cudaMemcpy(CtoV_device, CtoV_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(VtoC_device, VtoC_host, NbBranch * sizeof(int), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------
    for (nb = 0, nbtestedframes = 0; nb < NbMonteCarlo; nb++) {
      //encoding
      for (k = 0; k < rank; k++) U[k] = 0;
      for (k = rank; k < N; k++) U[k] = floor(drand48() * 2);
      for (k = rank - 1; k >= 0; k--) {
        for (l = k + 1; l < N; l++) U[k] = U[k] ^ (MatG[k][l] * U[l]);
      }
      for (k = 0; k < N; k++) Codeword[PermG[k]] = U[k];
      // All zero codeword
      //for (n=0;n<N;n++) { Codeword[n]=0; }

      // Add Noise
      for (n = 0; n < N; n++)
        if (drand48() < alpha) Receivedword_host[n] = 1 - Codeword[n];
        else Receivedword_host[n] = Codeword[n];
      //============================================================================
      // Decoder
      //============================================================================
      for (k = 0; k < NbBranch; k++) {
        CtoV_host[k] = 0;
      }
      for (k = 0; k < N; k++) Decide_host[k] = Receivedword_host[k];

      cudaMemcpy(Receivedword_device, Receivedword_host, N * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(Decide_device, Decide_host, N * sizeof(int), cudaMemcpyHostToDevice);

      for (iter = 0; iter < NbIter; iter++) {
        if (iter == 0) DataPassGBIter0 << < GridDim1, BlockDim1 >>> (VtoC_device, Receivedword_device, Interleaver_device, N, NbBranch);
        else if (iter < 15) DataPassGB << < GridDim1, BlockDim1 >>> (VtoC_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch);
        //else if ((iter>=15) && (iter<=16)) DataPassGB2(VtoC,CtoV,Receivedword,Decide,Interleaver,ColumnDegree,N,NbBranch,Decide);
        //else DataPassGB(VtoC,CtoV,Receivedword,Decide,Interleaver,ColumnDegree,N,NbBranch);

        else {
          DataPassGB2 << < GridDim1, BlockDim1 >>> (VtoC_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch, varr);
        }
        CheckPassGB << < GridDim2, BlockDim2 >>> (CtoV_device, VtoC_device, M, NbBranch);
        APP_GB << < GridDim1, BlockDim1 >>> (Decide_device, CtoV_device, Receivedword_device, Interleaver_device, N, NbBranch);

        ComputeSyndrome << < GridDim2, BlockDim2 >>> (Synd_device, Decide_device, M, NbBranch);

        cudaMemcpy(Synd_host, Synd_device, M * sizeof(int), cudaMemcpyDeviceToHost);

        int count1 = 0;
        for (kk = 0; kk < M; kk++) {
          if (Synd_host[kk] == 1) {
            count1++;
            break;
          }
        }

        if (count1 > 0) {
          Synd_host1 = 0;
        } else {
          Synd_host1 = 1;
        }

        IsCodeword = Synd_host1;

        if (IsCodeword) break;
      }

      cudaMemcpy(Decide_host, Decide_device, N * sizeof(int), cudaMemcpyDeviceToHost);

      //============================================================================
      // Compute Statistics
      //============================================================================
      nbtestedframes++;
      NbError = 0;
      for (k = 0; k < N; k++) {
        if (Decide_host[k] != Codeword[k]) {
          NbError++;
        }
      }
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
      if (NbTotalErrors == NBframes) break;
    }
    printf("%1.5f\t\t", alpha);
    printf("%10d (%1.16f)\t\t", NbBitError, (float) NbBitError / N / nbtestedframes);
    printf("%4d (%1.16f)\t\t", NbTotalErrors, (float) NbTotalErrors / nbtestedframes);
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
  return (0);
}