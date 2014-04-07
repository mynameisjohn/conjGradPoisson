/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
__kernel void helloworld(__global char* in, __global char* out)
{
	int num = get_global_id(0);
	out[num] = in[num] + 1;
}

__kernel void divide(__global float * scalars)
{
   scalars[0]=scalars[1]/scalars[2];
}

__kernel void scale(__global float * v0, __global float * vN, __global float * a)
{
   int idx = get_global_id(0);
   vN[idx]=a[0]*v0[idx];
}

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

__kernel void convolveLocal(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]=4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]);
}

__kernel void convolveLocal_a(__global float * v0, __global float * vN, __global float * a)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]=-a[0]*(4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]));
}
__kernel void convolveLocal_b(__global float * v0, __global float * vN)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int n = get_global_size(0);

   float cached[4]= {((j==0)   ? 0.0f : v0[(i*n+j-1)]), ((j==n-1) ? 0.0f : v0[(i*n+j+1)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)]), ((i==0)   ? 0.0f : v0[((i-1)*n+j)])};
   vN[i*n+j]-=4.0f*v0[i*n+j]-(cached[0]+cached[1]+cached[2]+cached[3]);
}

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

