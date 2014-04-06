#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>

#define N 1000

//No longer needed
void v2pxa(int * pxa, float * v, int * inside, int size){
        int x,y;

#pragma omp parallel for shared(pxa,v,inside) private (x,y)
        for (y=0;y<size;y++){
                for (x=0;x<size;x++){
                        float t;
                        int r,g,b;

                        t=(5*(M_PI/(float)2-atan(v[y*size+x])))/(float)3;
                        r = 128*(2*cos(t)+1);
                        if (r<0) r=0; if (r>255) r=255;
                        g = 128*(2*cos(t-((float)2)*M_PI/((float)3))+1);
                        if (g<0) g=0; if (g>255) g=255;
                        b = 128*(2*cos(t+((float)2)*M_PI/(float)3)+1);
                        if (b<0) b=0; if (b>255) b=255;
                        if (inside[y*size+x])
                                pxa[y*size+x]=0xFF<<24 | b<<16 | g<<8 | r;
                        else
                                pxa[y*size+x]=0;
                }
        }
        return;
}

//Initialize the boundary conditions
int initialize(float * v, int * inside, int size){
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
      if (y==0){
   v[y*size+x]=v5;
   inside[y*size+x]=0;
      }
      if (y==size-1){
   v[y*size+x]=v3;
   inside[y*size+x]=0;
      }
      if (x==0){
   v[y*size+x]=v6;
   inside[y*size+x]=0;
      }
      if (x==size-1){
   v[y*size+x]=v4;
   inside[y*size+x]=0;
   }
      if (x0<=xa && xa<=x1 && y0<=ya && ya<=y1){
   v[y*size+x]=v2;
   inside[y*size+x]=0;
      }
      if (pow((xa-xc),2)+pow((ya-yc),2)<=r*r){
   v[y*size+x]=-v1;
   inside[y*size+x]=0;
      }
      if (pow((xa-xd),2)+pow((ya-yc),2)<=r*r){
   v[y*size+x]=v1;
   inside[y*size+x]=0;
      }
    }
  }

  return 1;
}

//Print matrix function
void printMat(float * M, int n){
   int i,j;
   printf("{\n");
   for (i=0;i<n;i++){
      for (j=0;j<n;j++)
         printf("%6.2f,",M[i*n+j]);
      printf("\n");
   }
   printf("}\n\n");
}

//Print vector function
void print(float * x, int n){
  int i=0;
  printf("[");
  for (i=0;i<n-1;i++)
    printf("%lf, ",x[i]);
  printf("%lf]\n",x[n-1]);
  return;
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

//No longer needed
float vTxMxv_alt(float *M, float *v, int n){
   float result=0;
   float * tmp = (float *)mkl_malloc(sizeof(float)*n,64);

   cblas_sgemv(CblasRowMajor,CblasNoTrans,n,n,1.0,M,n,v,1,0.0,tmp,1);
   result=cblas_sdot(n,v,1,tmp,1);
   
   mkl_free(tmp);
   return result;
}

//Solve Poisson's eqn, store result in x
int conjGrad(float * x){

   //mkl_set_num_threads(mkl_get_max_threads());

   static int initialized=0;
   static int step=0;
   static float *r;// = (float *)mkl_malloc(sizeof(float)*N*N,64); 
   //static float * r = (float *)mkl_malloc(sizeof(float)*N*N,64);
   static int * inside;// = (int *)malloc(sizeof(int)*N*N);
   //float alpha=0,beta=0,error;
   static double start,finish,duration;
   int i,j,k;
   int nstp=1,npr=200,max=2000,iter=0;;
   
   if (!initialized){
      omp_set_num_threads(4);
      mkl_set_num_threads(4);
      //x=(float *)mkl_malloc(sizeof(float)*N*N,64); 
      r=(float *)mkl_malloc(sizeof(float)*N*N,64); 
      inside=(int *)malloc(sizeof(int)*N*N);
      float * b = (float *)calloc(N*N,sizeof(float));

      for (i=0;i<N*N;i++) inside[i]=1;

      initialized=1;
      //for (i=0;i<N*N;i++) x[i]=1;
      //convolve(x,b,N);
      //for (i=0;i<N*N;i++) x[i]=N*N-i;
      //printMat(b,N);
      //printf("fuck you\n"); 
      //initialize(x,inside,N);
      initialize(b,inside,N);
      //convolve_A(x,b,-1.0f,N);
      //for (i=0;i<N*N;i++) x[i]=b[i];
      start=omp_get_wtime();
      residue(x,b,r,N);
      free(b);
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
	 //create arrays
      }
      while (iter<nstp){
      cblas_saxpy(N*N,alpha,p,1,x,1);
      convolve_A(p,pN,-alpha,N);
      //cblas_sgemv(CblasRowMajor,CblasNoTrans,N*N,N*N,-alpha,A,N*N,p,1,0.0,pN,1);
      cblas_scopy(N*N,r,1,rN,1);
      cblas_saxpy(N*N,1.0,pN,1,rN,1);
      error=cblas_sdot(N*N,rN,1,rN,1);
      //if (error<0.00001) break;
      if (step==max||error<0.00001){
	 finish=omp_get_wtime();
	 printf("Solution found in %lf seconds with %d steps. \n",finish-start,step);
         mkl_free(p);
         mkl_free(rN);
         mkl_free(pN);
	 //mkl_free(x);
	 mkl_free(r);
	 free(inside);
         return 1;
      }
      beta=cblas_sdot(N*N,rN,1,rN,1)/cblas_sdot(N*N,r,1,r,1);
      cblas_scopy(N*N,rN,1,pN,1);
      cblas_saxpy(N*N,beta,p,1,pN,1);
      alpha=cblas_sdot(N*N,rN,1,rN,1)/vTxMxv(pN,N);
      cblas_scopy(N*N,rN,1,r,1);
      cblas_scopy(N*N,pN,1,p,1);
      //printf("%lf\n",error);
      step++; 
      iter++;
      }
      iter=0;
      //printf("%d iteration(s) to %lf seconds. \n",nstp,omp_get_wtime()-duration);

      //if j is max, free arrays
   }
   //finish=omp_get_wtime();
						     
   //printMat(x,N);
   //printMat(b,N);
   //printf("%d, %lf\n",step,finish-start);
   //double colorTime =  omp_get_wtime();

   //v2pxa(pxa,x,inside,N);
   //printf("%lf\n",omp_get_wtime()-colorTime);
   //mkl_free(x);
   //mkl_free(r);
   return 0;
}
