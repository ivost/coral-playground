SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
# Allowed CPU values: k8, armv7a, aarch64
CPU ?= k8
# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= opt

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --crosstool_top=@crosstool//:toolchains \
                     --compilation_mode=$(COMPILATION_MODE) \
                     --compiler=gcc \
                     --cpu=$(CPU) \
                     --linkopt=-L$(shell bazel info output_base)/external/edgetpu/libedgetpu/direct/$(CPU) \
                     --linkopt=-l:libedgetpu.so.1

lstpu:
	bazel build $(BAZEL_BUILD_FLAGS) //:lstpu

clean:
	bazel clean



cmake_minimum_required(VERSION 3.17)
project(tf1)

set(CMAKE_CXX_STANDARD 14)

add_executable(tf1 main.cpp)

