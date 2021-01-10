PROGNAME = tpu-list
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

CC=g++
CXXFLAGS = -I ../include -I. 
LDFLAGS  = -L ../lib/ -l:libedgetpu.so.1

SRC := $(wildcard *.cpp)
DEP := $(patsubst %.cpp,%.d,$(SRC))
OBJ := $(patsubst %.cpp,%.o,$(SRC))


.PHONY: all
all: $(PROGNAME)

$(PROGNAME): $(OBJ) 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^  common.o ../lib/libcommon.lo -o $@

.PHONY: clean
clean:
	$(RM) $(DEP) $(OBJ) $(PROGNAME)

