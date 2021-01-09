PROGNAME = tpu-list
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

CXXFLAGS = -I../include -I.

SRC := $(wildcard *.cc)
DEP := $(patsubst %.cc,%.d,$(SRC))
OBJ := $(patsubst %.cc,%.o,$(SRC))

LINKFLAGS=-l:libedgetpu.so.1

all: $(PROGNAME)

$(PROGNAME): $(OBJ) 
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(DEP) $(OBJ) $(PROGNAME)
