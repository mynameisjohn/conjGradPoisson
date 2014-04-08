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

BENCHMARK: shader/shader.o ; g++ drivers/benchmarkdriver.cpp $(FLAGS) -I$(MKLHOME)/include $(MKL_LINK) $(GL_FLAGS) $(OBJ) -lOpenCL -lclAmdBlas -I$(OCLHOME)/include/ -I$(CLBLASHOME)/include/  -L$(CLBLASHOME)/lib64/ -L$(OCLHOME)/lib/x86_64/ -o benchmarker

shader.o: shader/shader.cpp ; g++ -c shader/shader.cpp -o shader/shader.o
