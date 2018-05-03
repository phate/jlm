# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

LLVMCONFIG = llvm-config

CPPFLAGS += -Iinclude -Iexternal/jive/include -I$(shell $(LLVMCONFIG) --includedir)
CXXFLAGS += -Wall --std=c++14 -Wfatal-errors
LDFLAGS += $(shell $(LLVMCONFIG) --libs core irReader) $(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs) -Lexternal/jive/

LIBJLM_SRC = \
	src/ir/aggregation/aggregation.cpp \
	src/ir/aggregation/annotation.cpp \
	src/ir/aggregation/structure.cpp \
	src/ir/basic_block.cpp \
	src/ir/callgraph.cpp \
	src/ir/cfg.cpp \
	src/ir/cfg-structure.cpp \
	src/ir/cfg_node.cpp \
	src/ir/data.cpp \
	src/ir/lambda.cpp \
	src/ir/module.cpp \
	src/ir/operators/alloca.cpp \
	src/ir/operators/call.cpp \
	src/ir/operators/getelementptr.cpp \
	src/ir/operators/load.cpp \
	src/ir/operators/operators.cpp \
	src/ir/operators/sext.cpp \
	src/ir/operators/store.cpp \
	src/ir/ssa.cpp \
	src/ir/tac.cpp \
	src/ir/types.cpp \
	src/ir/variable.cpp \
	src/ir/view.cpp \
	\
	src/jlm2llvm/instruction.cpp \
	src/jlm2llvm/jlm2llvm.cpp \
	src/jlm2llvm/type.cpp \
	\
	src/jlm2rvsdg/module.cpp \
	src/jlm2rvsdg/restructuring.cpp \
	\
	src/llvm2jlm/constant.cpp \
	src/llvm2jlm/instruction.cpp \
	src/llvm2jlm/module.cpp \
	src/llvm2jlm/type.cpp \
	\
	src/rvsdg2jlm/rvsdg2jlm.cpp \
	\
	src/opt/cne.cpp \
	src/opt/dne.cpp \
	src/opt/inlining.cpp \
	src/opt/invariance.cpp \
	src/opt/inversion.cpp \
	src/opt/pull.cpp \
	src/opt/push.cpp \
	src/opt/unroll.cpp \

JLMPRINT_SRC = \
	src/jlm-print.cpp \

JLMOPT_SRC = \
	src/jlm-opt.cpp \

JLM_SRC = \
	src/jlm.cpp

all: libjlm.a jlm-print jlm-opt jlm check

libjlm.a: $(patsubst %.cpp, %.la, $(LIBJLM_SRC))

jlm-print: LDFLAGS+=-L. -ljlm -ljive
jlm-print: $(patsubst %.cpp, %.o, $(JLMPRINT_SRC)) libjlm.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

jlm-opt: LDFLAGS+=-L. -ljlm -ljive
jlm-opt: $(patsubst %.cpp, %.o, $(JLMOPT_SRC)) libjlm.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

jlm: LDFLAGS+=-L. -ljlm -ljive
jlm: $(patsubst %.cpp, %.o, $(JLM_SRC)) libjlm.a
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
	rm -rf jlm
	rm -rf jlm-opt
	rm -rf jlm-print

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
