# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

LLVMCONFIG = llvm-config-3.3

CPPFLAGS += -Iinclude -I/home/reissman/Documents/jive/include -I/usr/include/llvm-3.3
CXXFLAGS += -Wall -Werror --std=c++0x -Wfatal-errors -g -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LDFLAGS += $(shell $(LLVMCONFIG) --libs) $(shell $(LLVMCONFIG) --ldflags) -L/home/reissman/Documents/jive

LIBJLM_SRC = \
	src/IR/basic_block.cpp \
	src/IR/cfg.cpp \
	src/IR/cfg_node.cpp \
	src/IR/clg.cpp \
	src/IR/tac/operators.cpp \
	src/IR/tac/tac.cpp \
	\
	src/construction/binops.cpp \
	src/construction/constant.cpp \
	src/construction/jlm.cpp \
	src/construction/instruction.cpp \
	src/construction/type.cpp \
	\
	src/destruction/destruction.cpp \
	src/destruction/restructuring.cpp \


JLM_SRC = \
	src/main.cpp \

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
	find . -name "*.o" -o -name "*.la" -o -name "*.a" | xargs rm -rf
	rm -rf tests/test-runner
	rm -rf jlm
