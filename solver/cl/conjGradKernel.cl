//Simple Divid Kernel (divides two scalars)
__kernel void divide(__global float * scalars)
{
   scalars[0]=scalars[1]/scalars[2];
}

//Simple scale kernel
__kernel void scale(__global float * v0, __global float * vN, __global float * a)
{
   int idx = get_global_id(0);
   vN[idx]=a[0]*v0[idx];
}

//Laplacian Convolution
__kernel void convolve(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   vN[i*n+j]=
      4.0f*v0[i*n+j]-(
      ((j==0)   ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0)   ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)]));
}

//Laplacian convolution using local memory
__kernel void convolveLocal(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]=4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]);
}

//Laplacian Convolution with negative scale using local memory
__kernel void convolveLocal_a(__global float * v0, __global float * vN, __global float * a)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]=-a[0]*(4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]));
}

//Laplacian convolution that subtracts, using local memory
__kernel void convolveLocal_b(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]-=4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]);
}

//Laplacian convolution with negative scale
__kernel void convolve_a(__global float * v0, __global float * vN, __global float * a)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(1);

   vN[i*n+j]=-a[0]*(
      4.0f*v0[i*n+j]-(
      ((j==0) ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0) ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)])));

}

//Laplacian convolution that subtracts
__kernel void convolve_b(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(1);
/*   float left = (float)(j!=0);
   float right = (float)(j!=n);
   float bottom = (float)(i!=0);
   float top = (float)(i!=n);

   vN[i*n+j]-=
      4.0f*v0[i*n+j]-(
         left*v0[(i*n+j-1)]+
         right*v0[(i*n+j+1)]+
         bottom*v0[((i-1)*n+j)]+
         top*v0[((i+1)*n+j)]);
  */ 
   vN[i*n+j]-=
      4.0f*v0[i*n+j]-(
      ((j==0) ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0) ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)]));
}

//Laplacian convolution that adds with positive scale
__kernel void convolve_c(__global float * v0, __global float * vN, global float * a)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(1);

   vN[i*n+j]+=a[0]*(
      4.0f*v0[i*n+j]-(
      ((j==0) ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0) ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)])));

}
