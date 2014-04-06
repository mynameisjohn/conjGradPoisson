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
#include <iostream>
#include <string>
#include <fstream>
#include <clAmdBlas.h>


#define SUCCESS 0
#define FAILURE 1

#define N 2

using namespace std;

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

void print(float * x, int n){
   int i=0;
   printf("[");
   for (i=0;i<n-1;i++)
      printf("%lf, ",x[i]);
   printf("%lf]\n",x[n-1]);
   return;
}


int main(int argc, char* argv[])
{
   cl_mem buf_A, buf_b, buf_x, buf_r0, buf_rN, buf_p0, buf_pN,buf_tmp,buf_scratch;
   cl_mem buf_scalars;//buf_alpha,buf_beta,buf_zeta;
   cl_float scalars [3];//alpha,beta,zeta;

   cl_float A[] = {4, 1, 1, 3};
   cl_float b[] = {1, 2};
   cl_float x[] = {2, 1};

   size_t divideGWs[1] = {1};
   size_t scaleGWs[1] = {N};

   cl_event event = NULL;
   int ret = 0;

   /*Step1: Getting platforms and choose an available one.*/
   cl_uint numPlatforms;	//the NO. of platforms
   cl_platform_id platform = NULL;	//the chosen platform
   cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
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
   cl_uint				numDevices = 0;
   cl_device_id        *devices;
   status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	
   if (numDevices == 0)	//no GPU available.
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
   cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	
   /* Setup clAmdBlas. */
   status = clAmdBlasSetup();
   if (status != CL_SUCCESS) {
      printf("clAmdBlasSetup() failed with %d\n", status);
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      return 1;
   }
	
   /* Prepare OpenCL memory objects and place matrices inside them. */
   buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY, (N*N*sizeof(cl_float)), NULL, &status);
   buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY, (N*sizeof(cl_float)), NULL, &status);
   
   buf_x = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_r0 = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_rN = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_p0 = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_pN = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);
   buf_scratch = clCreateBuffer(context, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &status);   
   buf_scalars = clCreateBuffer(context,CL_MEM_READ_WRITE, 3*sizeof(cl_float),NULL,&status);
   
   status = clEnqueueWriteBuffer(commandQueue, buf_A, CL_TRUE, 0, (N*N*sizeof(cl_float)), A, 0, NULL, NULL);
   status = clEnqueueWriteBuffer(commandQueue, buf_x, CL_TRUE, 0, (N*sizeof(cl_float)), x, 0, NULL, NULL);
   status = clEnqueueWriteBuffer(commandQueue, buf_b, CL_TRUE, 0, (N*sizeof(cl_float)), b, 0, NULL, NULL);
   
	
   /*Step 5: Create program object */
   const char *filename = "scalarKernel.cl";
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

   /* Call clAmdBlas functions. */
   //r0=b
   status = clAmdBlasScopy(N,buf_b,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);
   //r0-=Ax
   status = clAmdBlasSgemvEx(clAmdBlasRowMajor,clAmdBlasNoTrans,N,N,-1.0,buf_A,0,N,buf_x,0,1,1.0,buf_r0,0,1,1,&commandQueue,0,NULL,&event);
   //p0=r0
   status = clAmdBlasScopy(N,buf_r0,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);
   //buf_scalars[1]=numerator of alpha
   status = clAmdBlasSdot(N,buf_scalars,1,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
   //A*p0
   status = clAmdBlasSgemvEx(clAmdBlasRowMajor,clAmdBlasNoTrans,N,N,1.0,buf_A,0,N,buf_p0,0,1,0.0,buf_tmp,0,1,1,&commandQueue,0,NULL,&event);
   //buf_scalars[2]=denominator of alpha
   status = clAmdBlasSdot(N,buf_scalars,2,buf_p0,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
   //buf_scalars[0]=alpha
   status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, NULL);
   //status = clEnqueueReadBuffer(commandQueue, buf_scalars, CL_TRUE, 0, 3*sizeof(cl_float), scalars, 0, NULL, NULL);	
   //pN=alpha*p0
   status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, NULL);
   //x+=pN
   status = clAmdBlasSaxpy(N,1.0,buf_pN,0,1,buf_x,0,1,1,&commandQueue,0,NULL,&event);//0->N
   //rN=-A*p0
   status = clAmdBlasSgemvEx(clAmdBlasRowMajor,clAmdBlasNoTrans,N,N,-1.0,buf_A,0,N,buf_pN,0,1,0.0,buf_rN,0,1,1,&commandQueue,0,NULL,&event);
   //rN+=r0
   status = clAmdBlasSaxpy(N,1.0,buf_r0,0,1,buf_rN,0,1,1,&commandQueue,0,NULL,&event);
   //buf_scalars[1]=numerator of beta
   status = clAmdBlasSdot(N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);	
   //buf_scalars[2]=denominator of beta
   status = clAmdBlasSdot(N,buf_scalars,2,buf_r0,0,1,buf_r0,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
   //buf_scalars[0]=beta
   status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, NULL);
   //status = clEnqueueReadBuffer(commandQueue, buf_scalars, CL_TRUE, 0, 3*sizeof(cl_float), scalars, 0, NULL, NULL);	
   //pN=beta*p0
   status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, NULL);
   //rN+=pN
   status = clAmdBlasSaxpy(N,1.0,buf_rN,0,1,buf_pN,0,1,1,&commandQueue,0,NULL,&event);
   //buf_scalars[1]=numerator of alpha
   status = clAmdBlasSdot(N,buf_scalars,1,buf_rN,0,1,buf_rN,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
   //A*p0
   status = clAmdBlasSgemvEx(clAmdBlasRowMajor,clAmdBlasNoTrans,N,N,1.0,buf_A,0,N,buf_pN,0,1,0.0,buf_tmp,0,1,1,&commandQueue,0,NULL,&event);
   //buf_scalars[2]=denominator of alpha
   status = clAmdBlasSdot(N,buf_scalars,2,buf_pN,0,1,buf_tmp,0,1,buf_scratch,1,&commandQueue,0,NULL,&event);
   //buf_scalars[0]=alpha
   status = clEnqueueNDRangeKernel(commandQueue, divideKernel, 1, NULL, divideGWs, NULL, 0, NULL, NULL);
   //status = clEnqueueReadBuffer(commandQueue, buf_scalars, CL_TRUE, 0, 3*sizeof(cl_float), scalars, 0, NULL, NULL);	
   //r0=rN
   status = clAmdBlasScopy(N,buf_rN,0,1,buf_r0,0,1,1,&commandQueue,0,NULL,&event);
   //p0=pN
   status = clAmdBlasScopy(N,buf_pN,0,1,buf_p0,0,1,1,&commandQueue,0,NULL,&event);
   //pN=alpha*p0
   status = clEnqueueNDRangeKernel(commandQueue, scaleKernel, 1, NULL, scaleGWs, NULL, 0, NULL, NULL);
   //x+=pN
   status = clAmdBlasSaxpy(N,1.0,buf_pN,0,1,buf_x,0,1,1,&commandQueue,0,NULL,&event);

   //Read back x and print
   status = clEnqueueReadBuffer(commandQueue, buf_x, CL_TRUE, 0, N*sizeof(cl_float), x, 0, NULL, NULL);	
   print(x,N);
   
   /* Release OpenCL memory objects. */
   clReleaseMemObject(buf_A);
   clReleaseMemObject(buf_x);
   clReleaseMemObject(buf_b);
   clReleaseMemObject(buf_r0);
   clReleaseMemObject(buf_rN);
   clReleaseMemObject(buf_p0);
   clReleaseMemObject(buf_pN);
   clReleaseMemObject(buf_tmp);
   //clReleaseMemObject(buf_alpha);
   clReleaseMemObject(buf_scalars);
   clReleaseMemObject(buf_scratch);
   
    /*Step 12: Clean the resources.*/		
   status = clReleaseKernel(divideKernel);
   status = clReleaseProgram(program);				//Release the program object.
   //status = clReleaseMemObject(inputBuffer);		//Release mem object.
   //status = clReleaseMemObject(outputBuffer);
   status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
   status = clReleaseContext(context);				//Release context.
   
   /* Finalize work with clAmdBlas. */
   clAmdBlasTeardown();
	
   if (devices != NULL)
   {
      free(devices);
      devices = NULL;
   }

   std::cout<<"Passed!\n";
   return SUCCESS;
}
