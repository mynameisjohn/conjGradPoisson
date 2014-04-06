MKLHOME = /opt/intel/mkl

GL_FLAGS = -lGL -lGLU -lglut -lGLEW

MKL_LINK = -Wl,--start-group $(MKLHOME)/lib/intel64/libmkl_intel_lp64.a $(MKLHOME)/lib/intel64/libmkl_core.a $(MKLHOME)/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group

FLAGS = -O3 -fopenmp -m64 -ldl -lpthread -lm

OBJ = shader/shader.o

MKL_G: shader/shader.o ; g++ drivers/mkl_graphics.cpp $(FLAGS) -I$(MKLHOME)/include $(MKL_LINK) $(GL_FLAGS) $(OBJ) -o MKLconjGrad

shader.o: shader/shader.cpp ; g++ -c shader/shader.cpp -o shader/shader.o
