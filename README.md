# JLM: An experimental compiler/optimizer for LLVM IR

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies:
* Clang/LLVM 7

## Bootstrap:
```
export LLVMCONFIG=<path-to-llvm-config>
make submodule
make all
```
Please ensure that `LLVMCONFIG` is set to the correct version of `llvm-config` as stated in
dependencies.
