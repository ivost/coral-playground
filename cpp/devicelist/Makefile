PROGNAME = tpu-list
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

CXXFLAGS = -I ../include -I.
LDFLAGS=-L ../lib/ -l:libedgetpu.so.1

SRC := $(wildcard *.cc)
DEP := $(patsubst %.cc,%.d,$(SRC))
OBJ := $(patsubst %.cc,%.o,$(SRC))


all: $(PROGNAME)

$(PROGNAME): $(OBJ) 
	$(CXX) $(CXXFLAGS) $(LDDFLAGS) $^ -o $@

clean:
	$(RM) $(DEP) $(OBJ) $(PROGNAME)
