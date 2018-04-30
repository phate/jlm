# JLM: An experimental compiler/optimizer for LLVM IR

Jlm is an experimental compiler/optimizer that consumes and produces LLVM IR. It uses the
Regionalized Value State Dependence Graph (RVSDG) as intermediate representation for optimizations.

## Dependencies:
* Clang/LLVM 4

## Bootstrap:
* git submodule init
* git submodule update
* cd external/jive && make
* cd ../../ && make
