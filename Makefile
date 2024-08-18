
TARGET = gemmEx_test
CUDA_PATH = /usr/local/cuda-12.5
INCS = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcublas -lcudart -lcuda
#CXX = g++
CXXFLAGS = -std=c++17 
NVCC = $(CUDA_PATH)/bin/nvcc
NVCCFLAGS = -std=c++17 -O3 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_89,code=sm_89 \
-gencode arch=compute_87,code=sm_87 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_80,code=sm_80 \
-Xcompiler="${CXXFLAGS}"

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(NVCC) $< $(INCS) $(LIBS) $(NVCCFLAGS) -o $@

clean:
	rm $(TARGET)
