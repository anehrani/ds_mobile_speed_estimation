


CC=g++

CFLAGS=-c -fPIC -I./include

LDFLAGS=-shared

SOURCES=src/plateInfo.cpp
OBJECTS=$(SOURCES:.cpp=.o)
SHARED_LIB=lib/libplateInfo.so

all: $(SHARED_LIB)

$(SHARED_LIB): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm -rf  $(OBJECTS) $(SHARED_LIB)
