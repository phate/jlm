# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

LLVMCONFIG = llvm-config

CPPFLAGS += -Iexternal/jive/include -I$(shell $(LLVMCONFIG) --includedir)
CXXFLAGS += -Wall -Wpedantic -Wextra -Wno-unused-parameter --std=c++14 -Wfatal-errors
LDFLAGS += $(shell $(LLVMCONFIG) --libs core irReader) $(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs) -Lexternal/jive/

all: create-folders libjlm.a libjlc.a jlm-print jlm-opt jlc check

include libjlm/Makefile.sub
include libjlc/Makefile.sub
include jlm-print/Makefile.sub
include jlm-opt/Makefile.sub
include tests/Makefile.sub

.PHONY: create-folders
create-folders:
	mkdir -p bin

%.la: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.a:
	rm -f $@
	ar clqv $@ $^
	ranlib $@

.PHONY: clean
clean:
	find . -name "*.o" -o -name "*.la" -o -name "*.a" | grep -v external | xargs rm -rf
	rm -rf tests/test-runner
	rm -rf bin

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
