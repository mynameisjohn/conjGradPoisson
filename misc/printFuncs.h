#include <stdio.h>

//Print matrix function
void printMat(float * M, int n){
   int i,j;
   printf("{\n");
   for (i=0;i<n;i++){
      for (j=0;j<n;j++)
         printf("%2.2f,",M[i*n+j]);
      printf("\n");
   }
   printf("}\n\n");
}

//Print a vector of length n
void printVec(float * x, int n){
   int i=0;
   printf("[");
   for (i=0;i<n-1;i++)
      printf("%lf, ",x[i]);
   printf("%lf]\n",x[n-1]);
   return;
}
