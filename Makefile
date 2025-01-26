# Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
# See COPYING for terms of redistribution.

BUILD_OUT_PREFIX = build/

default_target: all

ifeq ($(shell if [ -e Makefile.config ] ; then echo yes ; else echo no; fi),no)
$(error No build configuration set. Please run ./configure.sh before building.)
endif

include Makefile.config

include Makefile.macros

include jlm/rvsdg/Makefile.sub
include jlm/util/Makefile.sub
include jlm/llvm/Makefile.sub
include jlm/tooling/Makefile.sub
include tests/Makefile.sub
include tools/Makefile.sub

ifdef ENABLE_HLS
include jlm/hls/Makefile.sub
include tools/jhls/Makefile.sub
include tools/jlm-hls/Makefile.sub
endif

ifdef ENABLE_MLIR
include jlm/mlir/Makefile.sub
endif

include Makefile.rules

# Provide support for custom make targets
ifneq ("$(wildcard Makefile.custom)","")
include Makefile.custom
endif
