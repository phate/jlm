# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

define HELP_TEXT
clear
echo "Makefile for the JLM compiler"
echo "Version 1.1 - 2019-11-26"
endef
.PHONY: help
help:
	@$(HELP_TEXT)
	@$(HELP_TEXT_JLM)
	@echo ""
	@echo "all                    Compile jlm in release mode, and run unit and C tests"
	@echo "release                Alias for jlm-release"
	@echo "debug                  Alias for jlm-debug and check"
	@echo "docs                   Generate doxygen documentation."
	@echo "clean                  Alias for jlm-clean"
	@$(HELP_TEXT_JIVE)

JLM_ROOT ?= .

include $(JLM_ROOT)/Makefile.sub
include $(JLM_ROOT)/tests/Makefile.sub

# LLVM related variables
LLVMCONFIG ?= llvm-config
CLANG_BIN=$(shell $(LLVMCONFIG) --bindir)
CC=$(CLANG)
CXX=$(CLANG_BIN)/clang++

.PHONY: all
all: jlm-release check

.PHONY: release
release: jlm-release

.PHONY: debug
debug: jlm-debug check

.PHONY: docs
docs: jlm-docs-build

%.la: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.a:
	rm -f $@
	ar cqv $@ $^
	ranlib $@

.PHONY: clean
clean: jive-clean jlm-clean

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
