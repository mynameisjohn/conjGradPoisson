/**
conjGrad_OCL.h - A file containing functions capable of solving Poisson's Equation in 2-D using the method of Conjugate Gradients. The method is implemented using the clAmdBlas subroutines for linear algebra operations alongside a few kernels for performing convolutions and scalar operations. 

References:
http://en.wikipedia.org/wiki/Conjugate_gradient_method
http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

This code was essentially built off of the example_saxpy.cpp file from the clAmdBlas samples and the HelloWorld.cpp file from the OpenCL samples. 

A special thanks to Claudio Rebbi, my professor for Intermediate Mechanics and Computational Physics. 

   - John Joseph, 4/4/2014
**/

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <clAmdBlas.h>

#include "../misc/printFuncs.h"

#define SUCCESS 0
#define FAILURE 1

using namespace std;

//Initialize the boundary conditions
int initBC(float * v, int * inside, int size){
  int x,y;
  float x0=(float)0.25, x1=(float)0.75, y0=(float)0.25, y1=(float)0.35,
    xc=(float)0.5, yc=(float)0.6, r=(float)0.125, xd=(float)0.35;
  float v1,v2,v3,v4,v5,v6,xa,ya;

  v1=0.008f; //charge density of square and cirlce 
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

//Method used for timing OpenCL events
double timeIt(cl_command_queue commandQueue, cl_event event){
   cl_ulong time_start, time_end;
   clFinish(commandQueue);
   clWaitForEvents(1 , &event);

   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
   return time_end - time_start;
}

//Solve Poisson's equation via conjugate gradient using OpenCL and clBlas
int conjGradOCL(float * x, int N)
{
   //OpenCL buffers (used to store information on the device)
   cl_mem buf_b, buf_x, buf_r0, buf_rN, buf_p0, buf_pN, buf_tmp, buf_scratch;
   cl_mem buf_scalars;

   //Boundary Conditions (host side)
   cl_float * b = (cl_float *)malloc(sizeof(cl_float)*N*N);
   int * inside = (int *)malloc(sizeof(int)*N*N);

   //Initialization condition
   static int initialized = 0;

   //Global Work groups for the three kernels
   size_t divideGWs[1] = {1};
   size_t scaleGWs[1] = {N*N};
   size_t convolveGWs[2] = {N,N};

   //Event, used mainly for profiling
   cl_event event = NULL;

   //Total execution time
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
   cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, &status);

   /* Setup clAmdBlas. */
   status = clAmdBlasSetup();
   if (status != CL_SUCCESS) {
      printf("clAmdBlasSetup() failed with %d\n", status);
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      return 1;
   }

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
   cl_kernel convolveKernel_b = clCreateKernel(program,"convolve_b",NULL);

   //Current and max iteration step
   int step=0,max=2000;

   //Sync up the command queue
   clFinish(commandQueue);

   //Iteration loop
   while (step<=max){
      //Initialization (only called once)
      if (!initialized){
         //Initialize Boundary Conditions
         initBC(b,inside,N);
         status = clEnqueueWriteBuffer(commandQueue, buf_b, CL_TRUE, 0, (N*N*sizeof(cl_float)), b, 0, NULL, &event);
         status = clEnqueueWriteBuffer(commandQueue, buf_x, CL_TRUE, 0, (N*N*sizeof(cl_float)), x, 0, NULL, &event);

         //r0=b-Ax
         status = clAmdBlasScopy(N*N,buf_b,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);  
         status = clSetKernelArg(convolveKernel_b,0,sizeof(cl_mem),(void *)&buf_x);
         status = clSetKernelArg(convolveKernel_b,1,sizeof(cl_mem),(void *)&buf_r0);
         status = clEnqueueNDRangeKernel(commandQueue, convolveKernel_b, 2, NULL, convolveGWs, NULL, 0, NULL, &event);

         //p0=r0      
         status = clAmdBlasScopy(N*N,buf_r0,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);

         //buf_scalars[1]=numerator of alpha
         status = clAmdBlasSdot(N*N,buf_scalars,1,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
         
         //tmp=A*p0
         status = clSetKernelArg(convolveKernel,0,sizeof(cl_mem),(void *)&buf_p0);
         status = clSetKernelArg(convolveKernel,1,sizeof(cl_mem),(void *)&buf_tmp);
         status = clEnqueueNDRangeKernel(commandQueue, convolveKernel, 2, NULL, convolveGWs, NULL, 0, NULL, &event);

         //buf_scalars[2]=denominator of alpha
         status = clAmdBlasSdot(N*N,buf_scalars,2,buf_p0,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
         
         //buf_scalars[0]=alpha
         status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);

         //r0=rN
         status = clAmdBlasScopy(N*N,buf_r0,0,1,buf_rN,0,1,1,&commandQueue,0,NULL,&event);

         initialized=1;
      }

      //pN=alpha*p0
      status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, &event);

      //x+=pN
      status = clAmdBlasSaxpy(N*N,1.0,buf_pN,0,1,buf_x,0,1,1,&commandQueue,0,NULL,&event);

      //rN-=A*pN
      status = clSetKernelArg(convolveKernel_b,0,sizeof(cl_mem),(void *)&buf_pN);
      status = clSetKernelArg(convolveKernel_b,1,sizeof(cl_mem),(void *)&buf_rN);
      status = clEnqueueNDRangeKernel(commandQueue, convolveKernel_b, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
      
      //buf_scalars[1]=numerator of beta
      status = clAmdBlasSdot(N*N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);

      //buf_scalars[2]=denominator of beta
      status = clAmdBlasSdot(N*N,buf_scalars,2,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);

      //buf_scalars[0]=beta
      status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);

      //pN=beta*p0
      status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, &event);

      //pN+=rN
      status = clAmdBlasSaxpy(N*N,1.0,buf_rN,0,1,buf_pN,0,1,1,&commandQueue,0,NULL,&event);

      //buf_scalars[1]=numerator of alpha
      status = clAmdBlasSdot(N*N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);

      //buf_scalars[2]=denominator of alpha
      status = clSetKernelArg(convolveKernel,0,sizeof(cl_mem),(void *)&buf_pN);
      status = clSetKernelArg(convolveKernel,1,sizeof(cl_mem),(void *)&buf_tmp);
      status = clEnqueueNDRangeKernel(commandQueue, convolveKernel, 2, NULL, convolveGWs, NULL, 0, NULL, &event);
      
      status = clAmdBlasSdot(N*N,buf_scalars,2,buf_pN,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
      
      //buf_scalars[0]=alpha
      status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, &event);

      //r0=rN
      status = clAmdBlasScopy(N*N,buf_rN,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);

      //p0=pN
      status = clAmdBlasScopy(N*N,buf_pN,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);

      step++;
   }
   
   //Read back x
   status = clEnqueueReadBuffer(commandQueue, buf_x, CL_TRUE, 0, N*N*sizeof(cl_float), x, 0, NULL, &event);

   //Free host arrays
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
   status = clReleaseKernel(convolveKernel_b);
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
   
   return 1;
}
