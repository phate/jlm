.SUFFIXES:

# Assume rvsdg-opt is in PATH or provide the correct path

RVSDG_OPT = ../build-mlir/build/bin/rvsdg-opt
JLM_OPT = ../build/jlm-opt

LLVM_MODULES = core jit native
LLVMCONFIG ?= llvm-config-18
LDFLAGS = $(shell $(LLVMCONFIG) --ldflags)
LIBS = $(shell $(LLVMCONFIG) --libs)

POLYBENCH_ROOT ?= ../usr/polybench

MLIR_POLYBENCH_ROOT ?= ../build-mlir/mlir_rvsdg.git/polybench_mlir

POLYBENCH_BUILD = build
POLYBENCH_RESULTS = results
# POLYBENCH_JLMSTATS = $(POLYBENCH_RESULTS)/jlmstats.log
# POLYBENCH_ALL_TIMINGS_FILE = $(POLYBENCH_RESULTS)/all_bench_timings.csv

JLM_OPT_FLAGS = --FunctionInlining \
                --InvariantValueRedirection \
                --NodeReduction \
                --DeadNodeElimination \
                --ThetaGammaInversion \
                --InvariantValueRedirection \
                --DeadNodeElimination \
                --NodePushOut \
                --InvariantValueRedirection \
                --DeadNodeElimination \
                --NodeReduction \
                --CommonNodeElimination \
                --DeadNodeElimination \
                --NodePullIn \
                --InvariantValueRedirection \
                --DeadNodeElimination \
                --LoopUnrolling \
                --InvariantValueRedirection \
                --IfConversion \
                --CommonNodeElimination \
                --DeadNodeElimination
 

CGEIST_FLAGS =  -I $(POLYBENCH_ROOT) \
				-I $(POLYBENCH_ROOT)/utilities/ \
				-I /usr/lib/gcc/aarch64-linux-gnu/11/include \
				-I /usr/include \
				-lm \
				--raise-scf-to-affine 


POLYBENCH_SRC = \
	linear-algebra/blas/gesummv/gesummv.c \
	linear-algebra/blas/gemver/gemver.c \
	linear-algebra/blas/trmm/trmm.c \
	linear-algebra/kernels/atax/atax.c \
	linear-algebra/kernels/mvt/mvt.c \
	medley/floyd-warshall/floyd-warshall.c \
	stencils/fdtd-2d/fdtd-2d.c \
	stencils/jacobi-2d/jacobi-2d.c \

.PHONY: build
# build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.polygeist, $(POLYBENCH_SRC))
# build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.polygeist_no_polymer, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.jlm, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.polygeist_jlm, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.optimized_rvsdg.mlir, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.optimized_rvsdg_no_polymer.mlir, $(POLYBENCH_SRC))
build: $(patsubst %.c, $(POLYBENCH_BUILD)/%.jlm_optimized.mlir, $(POLYBENCH_SRC))

# # Rule to compile C to executable with cgeist (Polygeist)
# # Possibly add more flags
# $(POLYBENCH_BUILD)/%.polygeist: $(POLYBENCH_ROOT)/%.c
# 	@mkdir -p $(dir $@)
# 	/usr/local/bin/cgeist \
# 		$< \
# 		$(CGEIST_FLAGS) \
# 		--polyhedral-opt \
# 		-O3 \
# 		-o $@ 2> /dev/null

# # Rule to compile C to executable with cgeist without polymer (Polygeist)
# # Possibly add more flags
# $(POLYBENCH_BUILD)/%.polygeist_no_polymer: $(POLYBENCH_ROOT)/%.c
# 	@mkdir -p $(dir $@)
# 	/usr/local/bin/cgeist \
# 		$< \
# 		$(CGEIST_FLAGS) \
# 		-O3 \
# 		-o $@ 2> /dev/null

# # Rule to compile C to MLIR with cgeist (Polygeist)
# # Possibly add more flags
# .PRECIOUS: $(POLYBENCH_BUILD)/%.polygeist.mlir
# $(POLYBENCH_BUILD)/%.polygeist.mlir: $(POLYBENCH_ROOT)/%.c
# 	@mkdir -p $(dir $@)
# 	/usr/local/bin/cgeist \
# 		$< \
# 		$(CGEIST_FLAGS) \
# 		--polyhedral-opt \
# 		-S -o $@ 2> /dev/null

# # Rule to compile C to MLIR with cgeist without polymer (Polygeist)
# # Possibly add more flags
# .PRECIOUS: $(POLYBENCH_BUILD)/%.polygeist_no_polymer.mlir
# $(POLYBENCH_BUILD)/%.polygeist_no_polymer.mlir: $(POLYBENCH_ROOT)/%.c
# 	@mkdir -p $(dir $@)
# 	/usr/local/bin/cgeist \
# 		$< \
# 		$(CGEIST_FLAGS) \
# 		-O0 \
# 		-S -o $@ 2> /dev/null

# Rule to convert Polygeist MLIR to RVSDG MLIR with rvsdg-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.rvsdg.mlir
$(POLYBENCH_BUILD)/%.rvsdg.mlir: $(MLIR_POLYBENCH_ROOT)/%.polygeist.mlir | $(dir $(POLYBENCH_BUILD)/%.rvsdg.mlir)
	$(RVSDG_OPT) --rvsdg-import-polygeist $< > $@

# Rule to optimize RVSDG MLIR with jlm-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.optimized_rvsdg.mlir
$(POLYBENCH_BUILD)/%.optimized_rvsdg.mlir: $(POLYBENCH_BUILD)/%.rvsdg.mlir | $(dir $(POLYBENCH_BUILD)/%.optimized_rvsdg.mlir)
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=mlir --output-format=mlir -o $@

# Rule to convert RVSDG MLIR to LLVM with jlm-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.polygeist_jlm.ll
$(POLYBENCH_BUILD)/%.polygeist_jlm.ll: $(POLYBENCH_BUILD)/%.rvsdg.mlir | $(dir $(POLYBENCH_BUILD)/%.polygeist_jlm.ll)
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=mlir --output-format=llvm -o $@

# Rule to convert RVSDG MLIR to dot with jlm-opt
$(POLYBENCH_BUILD)/%.polygeist_jlm.dot: $(POLYBENCH_BUILD)/%.rvsdg.mlir | $(dir $(POLYBENCH_BUILD)/%.polygeist_jlm.dot)
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=mlir --output-format=dot -o $@

# Rule to compile RVSDG LLVM to object files with llc
$(POLYBENCH_BUILD)/%.polygeist_jlm.o: $(POLYBENCH_BUILD)/%.polygeist_jlm.ll | $(dir $(POLYBENCH_BUILD)/%.polygeist_jlm.o)
	llc-18 -O3 --relocation-model=static -filetype=obj -o $@ $<

# Rule to compile RVSDG object files to executables with clang
$(POLYBENCH_BUILD)/%.polygeist_jlm: $(POLYBENCH_BUILD)/%.polygeist_jlm.o | $(dir $(POLYBENCH_BUILD)/%)
	clang-18 -no-pie -O0 -I$(POLYBENCH_ROOT) -I$(POLYBENCH_ROOT)/utilities -DPOLYBENCH_USE_C99_PROTO $< $(LIBS) $(LDFLAGS) -o $@ $(POLYBENCH_ROOT)/utilities/polybench.c -lm

### No polymer

# Rule to convert Polygeist MLIR to RVSDG MLIR with rvsdg-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.rvsdg_no_polymer.mlir
$(POLYBENCH_BUILD)/%.rvsdg_no_polymer.mlir: $(MLIR_POLYBENCH_ROOT)/%.polygeist_no_polymer.mlir | $(dir $(POLYBENCH_BUILD)/%.rvsdg_no_polymer.mlir)
	$(RVSDG_OPT) --rvsdg-import-polygeist $< > $@

# Rule to optimize RVSDG MLIR with jlm-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.optimized_rvsdg_no_polymer.mlir
$(POLYBENCH_BUILD)/%.optimized_rvsdg_no_polymer.mlir: $(POLYBENCH_BUILD)/%.rvsdg_no_polymer.mlir | $(dir $(POLYBENCH_BUILD)/%.optimized_rvsdg_no_polymer.mlir)
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=mlir --output-format=mlir -o $@

# Rule to convert RVSDG MLIR to LLVM with jlm-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.ll
$(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.ll: $(POLYBENCH_BUILD)/%.rvsdg_no_polymer.mlir | $(dir $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.ll)
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=mlir --output-format=llvm -o $@

# Rule to compile RVSDG LLVM to object files with llc
$(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.o: $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.ll | $(dir $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.o)
	llc-18 -O3 --relocation-model=static -filetype=obj -o $@ $<

# Rule to compile RVSDG object files to executables with clang
$(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer: $(POLYBENCH_BUILD)/%.polygeist_jlm_no_polymer.o | $(dir $(POLYBENCH_BUILD)/%)
	clang-18 -no-pie -O0 -I$(POLYBENCH_ROOT) -I$(POLYBENCH_ROOT)/utilities -DPOLYBENCH_USE_C99_PROTO $< $(LIBS) $(LDFLAGS) -o $@ $(POLYBENCH_ROOT)/utilities/polybench.c -lm


# JLM

# Rule to compile c to llvm with clang 
$(POLYBENCH_BUILD)/%.jlm.ll: $(POLYBENCH_ROOT)/%.c
	@mkdir -p $(dir $@)
	clang-18 -O0 -Xclang -disable-O0-optnone -I$(POLYBENCH_ROOT) -I$(POLYBENCH_ROOT)/utilities -emit-llvm -DPOLYBENCH_USE_C99_PROTO $< -S -o $@

# Rule to optimize llvm with opt
$(POLYBENCH_BUILD)/%.jlm_mem2reg.ll: $(POLYBENCH_BUILD)/%.jlm.ll
	opt-18 -passes=mem2reg -S $< -o $@

# Rule to optimize llvm with jlm-opt
.PRECIOUS: $(POLYBENCH_BUILD)/%.jlm_optimized.ll
$(POLYBENCH_BUILD)/%.jlm_optimized.ll: $(POLYBENCH_BUILD)/%.jlm_mem2reg.ll
	$(JLM_OPT) $< $(JLM_OPT_FLAGS) --input-format=llvm --output-format=llvm -o $@

# Rule generate MLIR from optimized llvm
.PRECIOUS: $(POLYBENCH_BUILD)/%.jlm_optimized.mlir
$(POLYBENCH_BUILD)/%.jlm_optimized.mlir: $(POLYBENCH_BUILD)/%.jlm_optimized.ll
	$(JLM_OPT) $< --input-format=llvm --output-format=mlir -o $@

# Rule to compile llvm to object files with llc
$(POLYBENCH_BUILD)/%.jlm_optimized.o: $(POLYBENCH_BUILD)/%.jlm_optimized.ll
	llc-18 -O3 --relocation-model=static -filetype=obj -o $@ $<

# Rule to compile object files to executables with clang
$(POLYBENCH_BUILD)/%.jlm: $(POLYBENCH_BUILD)/%.jlm_optimized.o
	clang-18 -no-pie -O0 -I$(POLYBENCH_ROOT) -I$(POLYBENCH_ROOT)/utilities -DPOLYBENCH_USE_C99_PROTO $< $(LIBS) $(LDFLAGS) -o $@ $(POLYBENCH_ROOT)/utilities/polybench.c -lm

# Rule for comparison: execute both binaries and check output diff
# Requires compiling with -DPOLYBENCH_DUMP_ARRAYS
$(POLYBENCH_BUILD)/%: $(POLYBENCH_BUILD)/%.jlm $(POLYBENCH_BUILD)/%.polygeist_jlm
	@echo "Running and comparing $*..."
	@$@.jlm 2> $@-jlm.log
	@$@.polygeist_jlm 2> $@-polygeist_jlm.log
	@echo "$@" ; \
	python3 float-diff.py $@-jlm.log $@-polygeist_jlm.log

.PHONY: compare
compare: $(patsubst %.c, $(POLYBENCH_BUILD)/%, $(POLYBENCH_SRC))