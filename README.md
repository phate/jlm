# JLM: An experimental compiler/optimizer based on RVSDG IR
Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies
* Clang/LLVM 10

## Bootstrap
```
export LLVMCONFIG=<path-to-llvm-config>
make submodule
make all
```
Please ensure that `LLVMCONFIG` is set to the correct version of `llvm-config` as stated in
dependencies.

## More information
An introduction to RVSDG and the optimizations supported by jlm can be found in the following
article:

N. Reissmann, J. C. Meyer, H. Bahmann, and M. Själander
*"RVSDG: An Intermediate Representation for Optimizing Compilers"*
ACM Transactions on Embedded Computing Systems (TACO), vol. 19, no. 6, Dec. 2020.
https://arxiv.org/abs/1912.05036
