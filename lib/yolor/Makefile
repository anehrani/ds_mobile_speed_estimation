################################################################################
# YOLOR
################################################################################

CUDA_VER = 11.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

OPENCV?=
ifeq ($(OPENCV),)
  OPENCV=0
endif

CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include -I/home/ek/install/TensorRT-8.4.1.5/include

ifeq ($(OPENCV), 1)
COMMON= -DOPENCV
CFLAGS+= $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
LIBS+= $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

LIBS+= -L/home/ek/install/TensorRT-8.4.1.5/lib -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvdsinfer_yolo_engine.cpp \
           nvdsparsebbox_Yolo.cpp \
           yoloPlugins.cpp \
           layers/convolutional_layer.cpp \
           layers/batchnorm_layer.cpp \
           layers/implicit_layer.cpp \
           layers/channels_layer.cpp \
           layers/shortcut_layer.cpp \
           layers/route_layer.cpp \
           layers/upsample_layer.cpp \
           layers/pooling_layer.cpp \
           layers/activation_layer.cpp \
           layers/reorg_layer.cpp \
           layers/reduce_layer.cpp \
           layers/shuffle_layer.cpp \
           layers/softmax_layer.cpp \
           layers/cls_layer.cpp \
           layers/reg_layer.cpp \
           utils.cpp \
           yolo.cpp \
           yoloForward_r.cu \
           sortDetections.cu
           #yoloForward.cu \
           #yoloForward_v2.cu \
           #yoloForward_nc.cu \
           #yoloForward_r.cu \
           #yoloForward_e.cu \
           #sortDetections.cu

ifeq ($(OPENCV), 1)
SRCFILES+= calibrator.cpp
endif

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

ifeq ($(CUDA_VER), 10.2)
CUB=-I/usr/local/cuda-$(CUDA_VER)/include/thrust/system/cuda/detail
endif

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(CUB) --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
