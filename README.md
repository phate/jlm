# JLM: An experimental compiler/optimizer for LLVM IR

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies:
* Clang/LLVM 7

## Bootstrap:
* make submodule
* make all
