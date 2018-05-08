# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

LLVMCONFIG = llvm-config

CPPFLAGS += -Iinclude -Iexternal/jive/include -I$(shell $(LLVMCONFIG) --includedir)
CXXFLAGS += -Wall --std=c++14 -Wfatal-errors
LDFLAGS += $(shell $(LLVMCONFIG) --libs core irReader) $(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs) -Lexternal/jive/

LIBJLM_SRC = \
	src/libjlm/ir/aggregation/aggregation.cpp \
	src/libjlm/ir/aggregation/annotation.cpp \
	src/libjlm/ir/aggregation/structure.cpp \
	src/libjlm/ir/basic-block.cpp \
	src/libjlm/ir/cfg.cpp \
	src/libjlm/ir/cfg-structure.cpp \
	src/libjlm/ir/cfg-node.cpp \
	src/libjlm/ir/ipgraph.cpp \
	src/libjlm/ir/module.cpp \
	src/libjlm/ir/operators/alloca.cpp \
	src/libjlm/ir/operators/call.cpp \
	src/libjlm/ir/operators/delta.cpp \
	src/libjlm/ir/operators/getelementptr.cpp \
	src/libjlm/ir/operators/lambda.cpp \
	src/libjlm/ir/operators/load.cpp \
	src/libjlm/ir/operators/operators.cpp \
	src/libjlm/ir/operators/sext.cpp \
	src/libjlm/ir/operators/store.cpp \
	src/libjlm/ir/ssa.cpp \
	src/libjlm/ir/tac.cpp \
	src/libjlm/ir/types.cpp \
	src/libjlm/ir/variable.cpp \
	src/libjlm/ir/view.cpp \
	\
	src/libjlm/jlm2llvm/instruction.cpp \
	src/libjlm/jlm2llvm/jlm2llvm.cpp \
	src/libjlm/jlm2llvm/type.cpp \
	\
	src/libjlm/jlm2rvsdg/module.cpp \
	src/libjlm/jlm2rvsdg/restructuring.cpp \
	\
	src/libjlm/llvm2jlm/constant.cpp \
	src/libjlm/llvm2jlm/instruction.cpp \
	src/libjlm/llvm2jlm/module.cpp \
	src/libjlm/llvm2jlm/type.cpp \
	\
	src/libjlm/rvsdg2jlm/rvsdg2jlm.cpp \
	\
	src/libjlm/opt/cne.cpp \
	src/libjlm/opt/dne.cpp \
	src/libjlm/opt/inlining.cpp \
	src/libjlm/opt/invariance.cpp \
	src/libjlm/opt/inversion.cpp \
	src/libjlm/opt/optimization.cpp \
	src/libjlm/opt/pull.cpp \
	src/libjlm/opt/push.cpp \
	src/libjlm/opt/reduction.cpp \
	src/libjlm/opt/unroll.cpp \

LIBJLC_SRC = \
	src/jlc/cmdline.cpp \
	src/jlc/command.cpp \

JLMPRINT_SRC = \
	src/jlm-print.cpp \

JLMOPT_SRC = \
	src/jlm-opt/jlm-opt.cpp \

JLC_SRC = \
	src/jlc/jlc.cpp \

all: libjlm.a libjlc.a jlm-print jlm-opt jlc check

libjlc.a: $(patsubst %.cpp, %.la, $(LIBJLC_SRC))

libjlm.a: $(patsubst %.cpp, %.la, $(LIBJLM_SRC))

jlm-print: LDFLAGS+=-L. -ljlm -ljive
jlm-print: $(patsubst %.cpp, %.o, $(JLMPRINT_SRC)) libjlm.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

jlm-opt: LDFLAGS+=-L. -ljlm -ljive
jlm-opt: $(patsubst %.cpp, %.o, $(JLMOPT_SRC)) libjlm.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

jlc: LDFLAGS+=-L. -ljlc -ljlm -ljive
jlc: $(patsubst %.cpp, %.o, $(JLC_SRC)) libjlc.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

include tests/Makefile.sub

%.la: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.a:
	rm -f $@
	ar clqv $@ $^
	ranlib $@

clean:
	find . -name "*.o" -o -name "*.la" -o -name "*.a" | grep -v external | xargs rm -rf
	rm -rf tests/test-runner
	rm -rf jlc
	rm -rf jlm-opt
	rm -rf jlm-print

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
