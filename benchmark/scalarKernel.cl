//Used to divide two numbers
__kernel void divide(__global float * scalars)
{
   scalars[0]=scalars[1]/scalars[2];
}

//A simple scale operation
__kernel void scale(__global float * v0, __global float * vN, __global float * a)
{
   int idx = get_global_id(0);
   vN[idx]=a[0]*v0[idx];
}

__kernel void convolve(__global float * v0, __global float * vN)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   int n = get_global_size(0);

   vN[i*n+j]=
      4.0f*v0[i*n+j]-(
      ((j==0)   ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0)   ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)]));
}

//Laplacian Convolution with a negative scale factor
__kernel void convolve_a(__global float * v0, __global float * vN, global float * a)
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

//Laplacian Convolution
__kernel void convolve_b(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(1);
   
   vN[i*n+j]-=
      4.0f*v0[i*n+j]-(
      ((j==0) ? 0.0f : v0[(i*n+j-1)])+
      ((j==n-1) ? 0.0f : v0[(i*n+j+1)])+
      ((i==0) ? 0.0f : v0[((i-1)*n+j)])+
      ((i==n-1) ? 0.0f : v0[((i+1)*n+j)]));
}

//Laplacian Convolution with a positive scale factor
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

