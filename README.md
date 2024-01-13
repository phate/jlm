# JLM: A research compiler based on the RVSDG IR
[![Tests](https://github.com/phate/jlm/actions/workflows/tests.yml/badge.svg)](https://github.com/phate/jlm/actions/workflows/tests.yml)

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies
* Clang/LLVM 16
* Doxygen 1.9.1

### HLS dependencies
* CIRCT
* Verilator 4.038

### Optional dependencies
* gcovr, for computing code coverage summary

## Bootstrap
```
./configure.sh
make all
```

This presumes that llvm-config-16 can be found in $PATH. If that is not the case,
you may need to explicitly configure it:

```
./configure.sh --llvm-config /path/to/llvm-config
make all
```

For working on the code, it is advisable to initialize header file
dependency tracking to ensure that dependent objects get recompiled when
header files are changed:

```
make depend
```

Additional information about supported build options is available via
`./configure.sh --help`. Useful options include specifying
`--target debug` to switch to a debug build target instead of the (default)
release target.

## Documentation
Invoke the following command to generate the doxygen documentation:
```
make docs
```
The documentation can then be found at `docs/html/index.html`

## Tests
To run unit tests and jlc C compilation tests, execute
```
make check
```

The tests can also be run instrumented under valgrind to validate absence
of detectable memory errors:
```
make valgrind-check
```

Lastly, when the build has been configured with coverage support (specifying
`--enable-coverage` configure flag for the build), then following build target
will compute unit test coverage for all files:
```
make coverage
```
The report will be available in build/coverage/coverage.html.

## High-level synthesis (HLS) backend
The HLS backend uses the MLIR FIRRTL dialect from CIRCT to convert llvm IR to FIRRTL code.

A compatible installation of CIRCT is needed to compile jlm with the capability to generate FIRRTL code
and the build has to be configured accordingly. A change of build configuration may require cleaning
stale intermediate files first, i.e., run 'make clean'.
```
./configure --circt-path=<path-to-CIRCT-installation> --llvm-config <path-to-CIRCT-installation>/bin/llvm-config
```

The LD_LIBRARY_PATH might also need to include CIRCT_LIB for the CIRCT tools to work.

### Manual CIRCT setup
Start by cloning the CIRCT git repository and checkout the compatible commit.
```
git clone git@github.com:circt/circt.git
cd circt
git checkout a0e883136331c4a05ac366d5b31962a9de8d803b
git submodule init
git submodule update
```

Then follow the instructions on "Setting this up" in circt/README.md, but skip 2) as it has already been performed with the above commands.

### Automated CIRCT setup
An automated CIRCT setup used to be provided by the jlm-eval-suite, but is temporarily broken
due to changes in the build system setup. The notes below document the intent, but will
```
git clone  git//github.com:phate/jlm-eval-suite.git
cd jlm-eval-suite
make submodule-circt
make circt-build
```

This will build llvm, mlir, and circt for you and install it in jlm-eval-suite/circt/local. The build of llvm requires at least 16 GiB of main memory (RAM), as well as ninja and cmake to be installed.
A complete list of dependencies can be found in the [getting started instrutions for LLVM/MLIR](https://mlir.llvm.org/getting_started/).

Not that the jlm-eval-suite has the jlm compiler as a submodule. To compile jlm with the newly installed CIRCT setup (assuming you are still in jlm-eval-suite):
```
make submodule
make jlm-release -j `nproc`
make jlm-check -j `nproc`
```

The jlm-eval-suite comes with a suite of HLS tests. The verilator simulator has to be installed To be able to run these. If the hls-test-suite is run using the provided make targets, e.g., 'make hls-test-run', then there is no need to set any of the environment variables mentioned above.

## Publications
An introduction to the RVSDG and the optimizations supported by jlm can be found in the
following articles:

N. Reissmann, J. C. Meyer, H. Bahmann, and M. Själander
*"RVSDG: An Intermediate Representation for Optimizing Compilers"*
ACM Transactions on Embedded Computing Systems (TECS), vol. 19, no. 6, Dec. 2020.
https://dl.acm.org/doi/abs/10.1145/3391902

H. Bahmann, N. Reissmann, M. Jahre, and J. C. Meyer
*"Perfect Reconstructability of Control Flow from Demand Dependence Graphs"*
ACM Transactions on Architecture and Code Optimization (TACO), no. 66, Jan. 2015.
https://dl.acm.org/doi/10.1145/2693261

N. Reissmann, J. C. Meyer, and M. Själander
*"RVSDG: An Intermediate Representation for the Multi-Core Era"*
Nordic Workshop on Multi-Core Computing (MCC), Nov. 2018.
https://www.sjalander.com/research/pdf/sjalander-mcc2018.pdf
