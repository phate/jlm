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
	@echo "submodule              Initializes all the dependent git submodules"
	@echo "all                    Compile jlm in release mode, and run unit and C tests"
	@echo "release                Alias for jlm-release"
	@echo "debug                  Alias for jlm-debug and check"
	@echo "clean                  Alias for jlm-clean"
	@echo "purge                  Alias for jlm-clean and jive-clean"
	@$(HELP_TEXT_JIVE)

JLM_ROOT ?= .
JIVE_ROOT ?= $(JLM_ROOT)/external/jive

include $(JLM_ROOT)/Makefile.sub
include $(JLM_ROOT)/tests/Makefile.sub
ifneq ("$(wildcard $(JIVE_ROOT)/Makefile.sub)","")
include $(JIVE_ROOT)/Makefile.sub
endif

LLVMCONFIG ?= llvm-config

.PHONY: all
all: jlm-release check

.PHONY: release
release: jlm-release

.PHONY: debug
debug: jlm-debug check

.PHONY: submodule
submodule:
	git submodule update --init --recursive

%.la: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

%.a:
	rm -f $@
	ar clqv $@ $^
	ranlib $@

.PHONY: clean
clean: jlm-clean

.PHONY: purge 
purge: jive-clean jlm-clean

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
