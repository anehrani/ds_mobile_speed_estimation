# Makefile for building a library


LIBRARY_NAME = plateInfo
SRC_DIR = src
INC_DIR = include
OBJ_DIR = bin
# Library directory
LIB_DIR = lib

CC = gcc
CXX = g++


CXXFLAGS = -c -I$(INC_DIR)
AR = ar
ARFLAGS = rcs




# Library and object files
LIBRARY = $(LIB_DIR)/lib$(LIBRARY_NAME).a
OBJECTS = $(OBJ_DIR)/$(LIBRARY_NAME).o



# Source files
SRCS = $(SRC_DIR)/*.cpp


# Default target
all: library



# Build object files target
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CXX) $(CXXFLAGS) -o $@ $<

library: $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBRARY) $(OBJECTS)
	
	

# Clean up
clean:
	rm -f $(OBJECTS) $(LIBRARY)

