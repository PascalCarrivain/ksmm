CXX=nvcc #g++
SRC_DIR=$(PWD)
# CXXFLAGS=-O3 -arch=sm_60
CXXFLAGS=-O3 -arch=sm_86 -Xcompiler -fopenmp
LDFLAGS=-lgomp

COMMON_SOURCES=
COMMON_OBJECTS=$(COMMON_SOURCES:.cpp=.o)

SOURCES0=src/sparse_mm.cu
OBJECTS0=$(SOURCES0:.cu=.o)
EXECUTABLE0=$(SOURCES0:.cu=)

.PHONY: all test0

all: test0

test0: $(EXECUTABLE0)


$(EXECUTABLE0): $(COMMON_OBJECTS) $(OBJECTS0)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

mrproper:
	rm -f *.o $(EXECUTABLE0)
