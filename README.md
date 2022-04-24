# JLM: A research compiler based on the RVSDG IR
[![Tests](https://github.com/phate/jlm/actions/workflows/tests.yml/badge.svg)](https://github.com/phate/jlm/actions/workflows/tests.yml)

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies
* Clang/LLVM 14
* Doxygen 1.9.1

## Bootstrap
```
export LLVMCONFIG=<path-to-llvm-config>
make all
```
Please ensure that `LLVMCONFIG` is set to the correct version of `llvm-config` as stated in
dependencies.

## Documentation
Invoke the following command to generate the doxygen documentation:
```
make docs
```
The documentation can then be found at `docs/html/index.html`

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
