
CUDA_VER?= 11.6

ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

OPENCV?=
ifeq ($(OPENCV),)
	OPENCV=0
endif

CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
#PKGS:= 

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include 
 
CFLAGS+= -I/home/ek/ResearchProjects/deepStreamProjects/sources/apps/my_ds_apps/plateDetectorYolo/lib/plateInfo/include
#CFLAGS+=$(shell pkg-config --cflags $(PKGS))


ifeq ($(OPENCV), 1)
	COMMON= -DOPENCV
	CFLAGS+= $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
	LIBS+= $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

LIBS+= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LIBS+= -L/home/ek/ResearchProjects/deepStreamProjects/sources/apps/my_ds_apps/plateDetectorYolo/lib/plateInfo/lib/plateInfo.a

#LIBS+=$(shell pkg-config --libs $(PKGS))

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= $(filter-out calibrator.cpp, $(wildcard *.cpp))


ifeq ($(OPENCV), 1)
	SRCFILES+= calibrator.cpp
endif

SRCFILES+= $(wildcard layers/*.cpp)
SRCFILES+= $(wildcard *.cu)


TARGET_LIB:= nvds_custom_impl_Yolo_plate.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
