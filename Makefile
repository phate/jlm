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
	@echo "JLM Aliases"
	@echo "--------------------------------------------------------------------------------"
	@echo "all                    Compile jlm in release mode, and run unit and C tests"
	@echo "release                Alias for jlm-release"
	@echo "debug                  Alias for jlm-debug and check"
	@echo "docs                   Generate doxygen documentation."
	@echo "clean                  Alias for jlm-clean"

JLM_ROOT ?= .

include $(JLM_ROOT)/Makefile.sub

# LLVM related variables
CLANG_BIN=$(shell $(LLVMCONFIG) --bindir)
CC=$(CLANG)
CXX=$(CLANG_BIN)/clang++

.PHONY: all
all: jlm-release check

.PHONY: release
release: jlm-release

.PHONY: debug
debug: jlm-debug check

.PHONY: check
check: jlm-check

.PHONY: check-ctests
check-ctests: jlm-check-ctests

.PHONY: check-utests
check-utests: jlm-check-utests

.PHONY: valgrind-check
valgrind-check: jlm-valgrind-check

.PHONY: docs
docs: jlm-docs-build

.PHONY: clean
clean: jlm-clean

ifeq ($(shell if [ -e .Makefile.override ] ; then echo yes ; fi),yes)
include .Makefile.override
endif
