# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

JLM_ROOT ?= .
JIVE_ROOT ?= $(JLM_ROOT)/external/jive

LLVMCONFIG ?= llvm-config

CPPFLAGS += -I$(JIVE_ROOT)/include -I$(shell $(LLVMCONFIG) --includedir)
CXXFLAGS += -Wall -Wpedantic -Wextra -Wno-unused-parameter --std=c++14 -Wfatal-errors
LDFLAGS  += $(shell $(LLVMCONFIG) --libs core irReader) $(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs) -L$(JIVE_ROOT)

all: libjlm libjlc jlm-print jlm-opt jlc check

include $(JIVE_ROOT)/Makefile.sub
include libjlm/Makefile.sub
include libjlc/Makefile.sub
include jlm-print/Makefile.sub
include jlm-opt/Makefile.sub
include tests/Makefile.sub

%.la: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.la: %.c
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.a:
	rm -f $@
	ar clqv $@ $^
	ranlib $@

.PHONY: clean
clean: jive-clean libjlm-clean libjlc-clean jlmopt-clean jlmprint-clean jlmtest-clean
	@rm -rf $(JLM_ROOT)/bin

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
