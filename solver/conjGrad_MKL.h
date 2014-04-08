#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>

#define EPS 0.00001

//Initialize the boundary conditions
int initBC(float * v, int size){
  int x,y;
  float x0=(float)0.25, x1=(float)0.75, y0=(float)0.25, y1=(float)0.35,
    xc=(float)0.5, yc=(float)0.6, r=(float)0.125, xd=(float)0.35;
  float v1,v2,v3,v4,v5,v6,xa,ya;

  v1=0.008f;
  v2=0.008f;
  v3=-20.0f;
  v4=20.0f;
  v5=-20.0f;
  v6=20.0f;
  
  for (y=0;y<size;y++){
    ya=((float)y)/((float)size);
    for (x=0;x<size;x++){
      xa=((float)x)/((float)size);
      if (y==0) v[y*size+x]=v5;
      if (y==size-1) v[y*size+x]=v3;
      if (x==0) v[y*size+x]=v6;
      if (x==size-1) v[y*size+x]=v4;
      if (x0<=xa && xa<=x1 && y0<=ya && ya<=y1)  v[y*size+x]=v2;
      if (pow((xa-xc),2)+pow((ya-yc),2)<=r*r) v[y*size+x]=-v1;
      if (pow((xa-xd),2)+pow((ya-yc),2)<=r*r) v[y*size+x]=v1;
    }
  }

  return 1;
}

//Laplacian convolution
void convolve(float * x, float * xN, int n){
  int i,j;
  #pragma omp parallel for private (i,j) shared (xN,x)
  for (i=0;i<n;i++){
    for (j=0;j<n;j++){
	xN[i*n+j]=
	  4.0f*x[i*n+j]-(
			((j==0) ? 0.0f : x[(i*n+j-1)])+
			((j==n-1) ? 0.0f : x[(i*n+j+1)])+
			((i==0) ? 0.0f : x[((i-1)*n+j)])+
			((i==n-1) ? 0.0f : x[((i+1)*n+j)]));
    }
  }
}

//Laplacian convolution with scaling
void convolve_A(float * x, float * xN, float alpha, int n){
  int i,j;
   float a = alpha;
  #pragma omp parallel for shared (xN,x) private(i,j)
  for (i=0;i<n;i++){
    for (j=0;j<n;j++){
      xN[i*n+j]=(
	  4.0f*x[i*n+j]-(
			((j==0) ? 0.0f : x[(i*n+j-1)])+
			((j==n-1) ? 0.0f : x[(i*n+j+1)])+
			((i==0) ? 0.0f : x[((i-1)*n+j)])+
			((i==n-1) ? 0.0f : x[((i+1)*n+j)])));
      xN[i*n+j]*=a;
    }
  }
}

//Residue function
void residue(float * x, float * b, float * r, int n){
  int i,j;
  #pragma omp parallel for private (i,j) shared (x,b,r)
  for (i=0;i<n;i++){
    for (j=0;j<n;j++){
	r[i*n+j]=
	  4.0f*x[i*n+j]-(
			((j==0) ? 0.0f : x[(i*n+j-1)])+
			((j==n-1) ? 0.0f : x[(i*n+j+1)])+
			((i==0) ? 0.0f : x[((i-1)*n+j)])+
			((i==n-1) ? 0.0f : x[((i+1)*n+j)]));
	r[i*n+j]=b[i*n+j]-r[i*n+j];
    }
  }
}

//Calculate quadratic form
float vTxMxv(float *v, int n){
   float result=0;
   float * tmp = (float *)mkl_malloc(sizeof(float)*n*n,64);

   convolve(v,tmp,n);
   result=cblas_sdot(n*n,v,1,tmp,1);
   
   mkl_free(tmp);
   return result;
}

//Solve Poisson's eqn, store result in x
int conjGradMKL(float * x, int N){
  static int initialized=0;
  static int step=0,max=2000,nstp=1;
  static float *r;
  static double start,finish,duration;
  int i,j,k;
  int iter=0;
   
  if (!initialized){
    omp_set_num_threads(4);
    mkl_set_num_threads(4);
    r=(float *)mkl_malloc(sizeof(float)*N*N,64); 
    float * b = (float *)calloc(N*N,sizeof(float));

    initBC(b,N);
    start=omp_get_wtime();
    residue(x,b,r,N);
    free(b);

    initialized=1;
  }
   //p,pN,r,rN
  if (step<=max){
    duration = omp_get_wtime();
    static float * rN, * p, * pN;
    static float alpha;
    float beta,error;
    if (step==0){
	    rN = (float *)mkl_malloc(sizeof(float)*N*N,64);
      p = (float *)mkl_malloc(sizeof(float)*N*N,64); 
      pN = (float *)mkl_malloc(sizeof(float)*N*N,64);
      error=cblas_sdot(N*N,r,1,r,1);
      cblas_scopy(N*N,r,1,p,1);
      alpha=cblas_sdot(N*N,r,1,r,1)/vTxMxv(p,N);
      cblas_scopy(N*N,r,1,rN,1);
    }
    while (iter<nstp){
      cblas_saxpy(N*N,alpha,p,1,x,1);
      convolve_A(p,pN,-alpha,N);
      cblas_saxpy(N*N,1.0,pN,1,rN,1);
      error=cblas_sdot(N*N,rN,1,rN,1);
      if (step>=max || error<EPS){
        finish=omp_get_wtime();
        printf("Solution found in %lf seconds with %d steps. \n",finish-start,step);
        mkl_free(p);
        mkl_free(rN);
        mkl_free(pN);
        mkl_free(r);
        return 1;
      }
      beta=cblas_sdot(N*N,rN,1,rN,1)/cblas_sdot(N*N,r,1,r,1);
      cblas_scopy(N*N,rN,1,pN,1);
      cblas_saxpy(N*N,beta,p,1,pN,1);
      alpha=cblas_sdot(N*N,rN,1,rN,1)/vTxMxv(pN,N);
      cblas_scopy(N*N,rN,1,r,1);
      cblas_scopy(N*N,pN,1,p,1);
      step++; 
      iter++;
      }
    iter=0;
   }

   return 0;
}
