/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMINSTRUCTIONCONVERSION_HPP
#define JLM_LLVM_FRONTEND_LLVMINSTRUCTIONCONVERSION_HPP

#include <jlm/llvm/ir/tac.hpp>

namespace llvm
{
class Constant;
class Instruction;
class Value;
}

namespace jlm::llvm
{

class context;
class Variable;

const Variable *
ConvertValue(::llvm::Value * v, tacsvector_t & tacs, context & ctx);

const Variable *
ConvertInstruction(
    ::llvm::Instruction * i,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx);

std::vector<std::unique_ptr<llvm::ThreeAddressCode>>
ConvertConstant(::llvm::Constant * constant, context & ctx);

const Variable *
ConvertConstant(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx);

}

#endif
