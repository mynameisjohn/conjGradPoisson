#Change MKLHOME / OCL directorie as needed
#make (or make all): make a tool that chooses between MKL and OCL
#make MKL_G: make MKL version of the tool
#make OCL_G: Make OCL version of the tool

MKLHOME = /opt/intel/mkl

OCLHOME = /opt/AMDAPP
CLBLASHOME = /opt/clAmdBlas-1.10.321

GL_FLAGS = -lGL -lGLU -lglut -lGLEW

MKL_LINK = -Wl,--start-group $(MKLHOME)/lib/intel64/libmkl_intel_lp64.a $(MKLHOME)/lib/intel64/libmkl_core.a $(MKLHOME)/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group

FLAGS = -O3 -fopenmp -m64 -ldl -lpthread -lm -Wno-format-extra-args

OBJ = shader/shader.o

all: shader/shader.o ; g++ drivers/driver.cpp $(FLAGS) -I$(MKLHOME)/include $(MKL_LINK) $(GL_FLAGS) $(OBJ) -lOpenCL -lclAmdBlas -I$(OCLHOME)/include/ -I$(CLBLASHOME)/include/ -L$(CLBLASHOME)/lib64/ -L$(OCLHOME)/lib/x86_64/ -o conjGradPoisson

MKL_G: shader/shader.o ; g++ drivers/mkl_graphics.cpp $(FLAGS) -I$(MKLHOME)/include $(MKL_LINK) $(GL_FLAGS) $(OBJ) -o MKLconjGrad

OCL_G: shader/shader.o; g++ drivers/ocl_graphics.cpp $(FLAGS) $(GL_FLAGS) $(OBJ) -o OCLconjGrad -lOpenCL -lclAmdBlas -I$(OCLHOME)/include/ -I$(CLBLASHOME)/include/  -L$(CLBLASHOME)/lib64/ -L$(OCLHOME)/lib/x86_64/ -lm

shader.o: shader/shader.cpp ; g++ -c shader/shader.cpp -o shader/shader.o
