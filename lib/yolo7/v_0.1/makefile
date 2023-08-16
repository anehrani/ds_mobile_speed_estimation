CUDA_VER:=11.6
SRCFILES:=nvdsparsebbox_yolo_plt.cpp
TARGET_LIB:=libnvdsinfer_custom_bbox_yolo_plt.so
DEEPSTREAM_PATH:=/opt/nvidia/deepstream/deepstream-6.1

ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
#CC:= g++

CC=g++
CFLAGS=-c -fPIC

CFLAGS+= -I$(DEEPSTREAM_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include



LDFLAGS = -L$(DEEPSTREAM_PATH)/lib  -lnvdsgst_meta -lnvds_meta -lnvds_inferutils \
          -lnvds_utils -lm -lstdc++
LDFLAGS+=-shared


SOURCES=nvdsparsebbox_yolo_plt.cpp
OBJECTS=$(SOURCES:.cpp=.o)



all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(SHARED_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $(OBJECTS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(TARGET_LIB)
















#CFLAGS:= -Wall -Werror -std=c++11 -shared -fPIC
#CFLAGS+= -I$(DEEPSTREAM_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
#CFLAGS += -I ../plateInfo/include
#CFLAGS += -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include
#
#
#LIBS:= -lnvinfer -lnvparsers -L$(DEEPSTREAM_PATH)/lib  -lnvdsgst_meta -lnvds_meta -lnvds_inferutils \
#       -lnvds_utils -lm -lstdc++ -L ../plateInfo/lib/plateInfo.a
##LIBS += ../plateInfo/lib/plateInfo.a
#LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group -lglib-2.0
#
#all: $(TARGET_LIB)
#
#$(TARGET_LIB) : $(SRCFILES)
#	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
#
#clean:
#	rm -rf $(TARGET_LIB)