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
	src/IR/cfg_node.cpp \
	src/IR/clg.cpp \
	src/IR/module.cpp \
	src/IR/operators.cpp \
	src/IR/ssa.cpp \
	src/IR/tac.cpp \
	src/IR/variable.cpp \
	\
	src/construction/constant.cpp \
	src/construction/instruction.cpp \
	src/construction/module.cpp \
	src/construction/type.cpp \
	\
	src/destruction/destruction.cpp \
	src/destruction/restructuring.cpp \


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
	find . -name "*.o" -o -name "*.la" -o -name "*.a" -path .external -prune | xargs rm -rf
	rm -rf tests/test-runner
	rm -rf jlm
