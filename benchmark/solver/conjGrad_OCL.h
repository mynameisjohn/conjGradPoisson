/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTstatusUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

// For clarity,statusor checking has been omitted.

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <clAmdBlas.h>


#define SUCCESS 0
#define FAILURE 1

//#define N 1024

using namespace std;
/*
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
*/
//Initialize the boundary conditions
int initBC(float * v, int * inside, int size){
  int x,y;
  float x0=(float)0.25, x1=(float)0.75, y0=(float)0.25, y1=(float)0.35,
    xc=(float)0.5, yc=(float)0.6, r=(float)0.125, xd=(float)0.35;
  float v1,v2,v3,v4,v5,v6,xa,ya;

  v1=0.008f;
  v2=0.008f;
  v3=-21.0f;
  v4=21.0f;
  v5=-21.0f;
  v6=21.0f;

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

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
   size_t size;
   char*  str;
   std::fstream f(filename, (std::fstream::in | std::fstream::binary));

   if(f.is_open())
   {
      size_t fileSize;
      f.seekg(0, std::fstream::end);
      size = fileSize = (size_t)f.tellg();
      f.seekg(0, std::fstream::beg);
      str = new char[size+1];
      if(!str)
      {
	 f.close();
	 return 0;
      }

      f.read(str, fileSize);
      f.close();
      str[size] = '\0';
      s = str;
      delete[] str;
      return 0;
   }
   cout<<"statusor: failed to open file\n:"<<filename<<endl;
   return FAILURE;
}
/*
void print(float * x, int n){
   int i=0;
   printf("[");
   for (i=0;i<n-1;i++)
      printf("%lf, ",x[i]);
   printf("%lf]\n",x[n-1]);
   return;
}

void print(double * x, int n){
   int i=0;
   printf("[");
   for (i=0;i<n-1;i++)
      printf("%lf, ",x[i]);
   printf("%lf]\n",x[n-1]);
   return;
}
*/
double timeIt(cl_command_queue commandQueue, cl_event event){
   cl_ulong time_start, time_end;
   clFinish(commandQueue);
   clWaitForEvents(1 , &event);

   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
   return time_end - time_start;
}


double conjGradOCL(float * x, int N)
{
   cl_mem buf_b, buf_x, buf_r0, buf_rN, buf_p0, buf_pN, buf_tmp, buf_scratch;
   cl_mem buf_scalars;
   cl_float scalars [3];
   cl_float * b = (cl_float *)malloc(sizeof(cl_float)*N*N);
   int * inside = (int *)malloc(sizeof(int)*N*N);

   size_t divideGWs[1] = {1};
   size_t scaleGWs[1] = {N*N};
   size_t convolveGWs[2] = {N,N};

   cl_event event = NULL;
   int ret = 0;

   double total_time=0;

   /*Step1: Getting platforms and choose an available one.*/
   cl_uint numPlatforms;   //the NO. of platforms
   cl_platform_id platform = NULL;  //the chosen platform
   cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
   if (status != CL_SUCCESS)
   {
      cout << "statusor: Getting platforms!" << endl;
      return FAILURE;
   }

   /*For clarity, choose the first available platform. */
   if(numPlatforms > 0)
   {
      cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
      status = clGetPlatformIDs(numPlatforms, platforms, NULL);
      platform = platforms[0];
      free(platforms);
   }
   
   /*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
   cl_uint           numDevices = 0;
   cl_device_id        *devices;
   status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);  
   if (numDevices == 0) //no GPU available.
   {
      cout << "No GPU device available." << endl;
      cout << "Choose CPU as default device." << endl;
      status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);  
      devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
      status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
   }  
   else
   {
      devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
      status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
   }

   /*Step 3: Create context.*/
   cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);
   
   /*Step 4: Creating command queue associate with the context.*/
   cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);

   /* Setup clAmdBlas. */
   status = clAmdBlasSetup();
   if (status != CL_SUCCESS) {
      printf("clAmdBlasSetup() failed with %d\n", status);
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      return 1;
   }

   static int initialized = 0;

   scalars[1]=1;

   /* Prepare OpenCL memory objects and place matrices inside them. */
   buf_b       = clCreateBuffer(context, CL_MEM_READ_ONLY, (N*N*sizeof(cl_float)), NULL, &status);
   buf_x       = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_r0      = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_rN      = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_p0      = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_pN      = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_tmp     = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);
   buf_scratch = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*N*sizeof(cl_float)), NULL, &status);   
   
   buf_scalars = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*sizeof(cl_float), NULL, &status);

   /*Step 5: Create program object */
   const char *filename = "solver/scalarKernel.cl";
   string sourceStr;
   status = convertToString(filename, sourceStr);
   const char *source = sourceStr.c_str();
   size_t sourceSize[] = {strlen(source)};
   cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
   
   /*Step 6: Build program. */
   status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);
   
   cl_kernel divideKernel = clCreateKernel(program,"divide",NULL);
   status = clSetKernelArg(divideKernel,0,sizeof(cl_mem),(void *)&buf_scalars);
   
   cl_kernel scaleKernel = clCreateKernel(program,"scale",NULL);
   status = clSetKernelArg(scaleKernel,0,sizeof(cl_mem),(void *)&buf_p0);
   status = clSetKernelArg(scaleKernel,1,sizeof(cl_mem),(void *)&buf_pN);
   status = clSetKernelArg(scaleKernel,2,sizeof(cl_mem),(void *)&buf_scalars);

   cl_kernel convolveKernel = clCreateKernel(program,"convolve",NULL);
   cl_kernel convolveKernel_a = clCreateKernel(program,"convolve_a",NULL);
   cl_kernel convolveKernel_b = clCreateKernel(program,"convolve_b",NULL);
   cl_kernel convolveKernel_c = clCreateKernel(program,"convolve_c",NULL);

   int step=0,max=2000;

   clFinish(commandQueue);

   while (step<=max){
      if (!initialized){
         //Initialize Boundary Conditions
         initBC(b,inside,N);
         status = clEnqueueWriteBuffer(commandQueue, buf_b, CL_TRUE, 0, (N*N*sizeof(cl_float)), b, 0, NULL, &event);
         total_time+=timeIt(commandQueue,event);
         status = clEnqueueWriteBuffer(commandQueue, buf_x, CL_TRUE, 0, (N*N*sizeof(cl_float)), x, 0, NULL, &event);
         total_time+=timeIt(commandQueue,event);

         //r0=b-Ax
         status = clAmdBlasScopy(N*N,buf_b,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);  
         total_time+=timeIt(commandQueue,event); 
         status = clSetKernelArg(convolveKernel_b,0,sizeof(cl_mem),(void *)&buf_x);
         status = clSetKernelArg(convolveKernel_b,1,sizeof(cl_mem),(void *)&buf_r0);
         status = clEnqueueNDRangeKernel(commandQueue, convolveKernel_b, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
         total_time+=timeIt(commandQueue,event);

         //p0=r0      
         status = clAmdBlasScopy(N*N,buf_r0,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);
         total_time+=timeIt(commandQueue,event);

         //buf_scalars[1]=numerator of alpha
         status = clAmdBlasSdot(N*N,buf_scalars,1,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
         total_time+=timeIt(commandQueue,event);
         
         //tmp=A*p0
         status = clSetKernelArg(convolveKernel,0,sizeof(cl_mem),(void *)&buf_p0);
         status = clSetKernelArg(convolveKernel,1,sizeof(cl_mem),(void *)&buf_tmp);
         status = clEnqueueNDRangeKernel(commandQueue, convolveKernel, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
         total_time+=timeIt(commandQueue,event);

         status = clAmdBlasSdot(N*N,buf_scalars,2,buf_p0,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
         total_time+=timeIt(commandQueue,event);
         
         //buf_scalars[0]=alpha
         status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);
         total_time+=timeIt(commandQueue,event);

         status = clAmdBlasScopy(N*N,buf_r0,0,1,buf_rN,0,1,1,&commandQueue,0,NULL,&event);
         total_time+=timeIt(commandQueue,event);

         initialized=1;
      }

      //pN=alpha*p0
      status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);

      //x+=pN
      status = clAmdBlasSaxpy(N*N,1.0,buf_pN,0,1,buf_x,0,1,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);

      //rN=r0-a*A*p0
      //status = clAmdBlasScopy(N*N,buf_r0,0,1,buf_rN,0,1,1,&commandQueue,0,NULL,&event);
      status = clSetKernelArg(convolveKernel_b,0,sizeof(cl_mem),(void *)&buf_pN);
      status = clSetKernelArg(convolveKernel_b,1,sizeof(cl_mem),(void *)&buf_rN);
      status = clEnqueueNDRangeKernel(commandQueue, convolveKernel_b, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);
      
      //buf_scalars[1]=numerator of beta
      status = clAmdBlasSdot(N*N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);  
      total_time+=timeIt(commandQueue,event);
      //buf_scalars[2]=denominator of beta
      status = clAmdBlasSdot(N*N,buf_scalars,2,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);
      //buf_scalars[0]=beta
      status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);

      //pN=beta*p0
      status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);
      //pN+=rN
      status = clAmdBlasSaxpy(N*N,1.0,buf_rN,0,1,buf_pN,0,1,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);

      //buf_scalars[1]=numerator of alpha
      status = clAmdBlasSdot(N*N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);

      //buf_scalars[2]=denominator of alpha
      status = clSetKernelArg(convolveKernel,0,sizeof(cl_mem),(void *)&buf_pN);
      status = clSetKernelArg(convolveKernel,1,sizeof(cl_mem),(void *)&buf_tmp);
      status = clEnqueueNDRangeKernel(commandQueue, convolveKernel, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);
      
      status = clAmdBlasSdot(N*N,buf_scalars,2,buf_pN,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);
      
      //buf_scalars[0]=alpha
      status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);
      total_time+=timeIt(commandQueue,event);

      //r0=rN
      status = clAmdBlasScopy(N*N,buf_rN,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);
      //p0=pN
      status = clAmdBlasScopy(N*N,buf_pN,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);
      total_time+=timeIt(commandQueue,event);

      step++;
   }
   
   
   //Read back x
   status = clEnqueueReadBuffer(commandQueue, buf_x, CL_TRUE, 0, N*N*sizeof(cl_float), x, 0, NULL, &event);
   total_time+=timeIt(commandQueue,event);
   /*
   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
   total_time = time_end - time_start;
   printf("\nExecution time in milliseconds = %0.3f ms\n", (total_time / 1000000000000.0) );
*/

   total_time /= 1000000.0;
   
   free(b);
   free(inside);

   /* Release OpenCL memory objects. */
   clReleaseMemObject(buf_x);
   clReleaseMemObject(buf_b);
   clReleaseMemObject(buf_r0);
   clReleaseMemObject(buf_rN);
   clReleaseMemObject(buf_p0);
   clReleaseMemObject(buf_pN);
   clReleaseMemObject(buf_tmp);
   clReleaseMemObject(buf_scalars);
   clReleaseMemObject(buf_scratch);
   
    /*Step 12: Clean the resources.*/		
   status = clReleaseKernel(divideKernel);
   status = clReleaseKernel(scaleKernel);
   status = clReleaseKernel(convolveKernel);
   status = clReleaseKernel(convolveKernel_a);
   status = clReleaseKernel(convolveKernel_b);
   status = clReleaseKernel(convolveKernel_c);
   status = clReleaseProgram(program);				//Release the program object.
   status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
   status = clReleaseContext(context);				//Release context.
   
   /* Finalize work with clAmdBlas. */
   clAmdBlasTeardown();
	
   if (devices != NULL)
   {
      free(devices);
      devices = NULL;
   }
   return total_time;
}

int benchmarkOCL(){
   int n,i; float * x;
   double * OCL_time = (double *)calloc(4,sizeof(double)),navg=5;
   
   //Test 1
   n=1024;
   x = (float *)malloc(sizeof(float)*n*n);
   for (i=0;i<(int)navg;i++)
      OCL_time[0]+=conjGradOCL(x,n);
   OCL_time[0]/=navg;
   free(x);

   n=512;
   x = (float *)malloc(sizeof(float)*n*n);
   for (i=0;i<(int)navg;i++)
      OCL_time[1]+=conjGradOCL(x,n);
   OCL_time[1]/=navg;
   free(x);

   n=256;
   x = (float *)malloc(sizeof(float)*n*n);
   for (i=0;i<(int)navg;i++)
      OCL_time[2]+=conjGradOCL(x,n);
   OCL_time[2]/=navg;
   free(x);

   n=128;
   x = (float *)malloc(sizeof(float)*n*n);
   for (i=0;i<(int)navg;i++)
      OCL_time[3]+=conjGradOCL(x,n);
   OCL_time[3]/=navg;
   free(x);

   print(OCL_time,4);
   free(OCL_time);
   return 1;
}