# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

LLVMCONFIG = llvm-config-3.7

CPPFLAGS += -Iinclude -Iexternal/jive/include -I$(shell $(LLVMCONFIG) --includedir)
CXXFLAGS += -Wall -Werror --std=c++14 -Wfatal-errors -g -DJLM_DEBUG
LDFLAGS += $(shell $(LLVMCONFIG) --libs core IRReader) $(shell $(LLVMCONFIG) --ldflags) $(shell $(LLVMCONFIG) --system-libs) -Lexternal/jive/

LIBJLM_SRC = \
	src/IR/aggregation/aggregation.cpp \
	src/IR/aggregation/annotation.cpp \
	src/IR/aggregation/node.cpp \
	src/IR/aggregation/structure.cpp \
	src/IR/basic_block.cpp \
	src/IR/cfg.cpp \
	src/IR/cfg-structure.cpp \
	src/IR/cfg_node.cpp \
	src/IR/clg.cpp \
	src/IR/module.cpp \
	src/IR/operators.cpp \
	src/IR/ssa.cpp \
	src/IR/tac.cpp \
	src/IR/variable.cpp \
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

JLM_SRC = \
	src/jlm.cpp \

all: libjlm.a jlm check

libjlm.a: $(patsubst %.cpp, %.la, $(LIBJLM_SRC))

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
