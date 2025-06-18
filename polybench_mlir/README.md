# Compiling polybench with P-JLM 
This directory contains a makefile that compiles polybench with JLM, P-JLM-Pluto and P-JLM-Baseline, 
as described in my masters thesis.
To avoid having to install Polygeist, 
the commit in the mlir_rvsdg repository pointed to by the build-mlir.sh contains pre-compiled .mlir files built using Polygeist.
There are two .mlir files associated with each polybench benchmark, one for the P-JLM-Pluto and one for the P-JLM-Baseline.

## Compiling
Running `make` will compile three binaries per benchmark, one with the P-JLM-Pluto, one with the P-JLM-Baseline and one with the JLM.
The compiled binaries are placed in the `build` directory, together with a range of intermediate files, which might be of interest.
## Timing
To run the benchmarks, run `time_benchmarks.py` using any python3 installation.
The `--warmup-runs` and `--timed-runs` flags have to be specified to run the benchmarks.
The results are placed in the `results` directory.
## Verifying
Running `make compare` will verify the printed outputs from the benchmarks.
Note, that for this to work, the makefile needs to be altered to build polybench with the `-DPOLYBENCH_DUMP_ARRAYS` flag.