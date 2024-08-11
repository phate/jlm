# JLM: A research compiler based on the RVSDG IR

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies
* Clang/LLVM 17
* Doxygen 1.9.1

### HLS dependencies
* CIRCT that is built with LLVM/MLIR 17
* Verilator 4.038

### Optional dependencies
* gcovr, for computing code coverage summary

## Bootstrap
```
./configure.sh
make all
```

This presumes that llvm-config-17 can be found in $PATH. If that is not the case,
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

A compatible installation of CIRCT is needed to compile jlm with the capability to generate FIRRTL
and the build has to be configured accordingly. A change of build configuration may require cleaning
stale intermediate files first, i.e., run 'make clean'.
CIRCT and the HLS backend can be setup with the following commands:
```
./scripts/build-circt.sh --build-path <CIRCT-build-path> --install-path <path-to-CIRCT>

./configure --enable-hls <path-to-CIRCT>
```

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
