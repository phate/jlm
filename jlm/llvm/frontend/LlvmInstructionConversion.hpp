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

class Context;
class variable;

const variable *
ConvertValue(::llvm::Value * v, tacsvector_t & tacs, Context & ctx);

const variable *
ConvertInstruction(
    ::llvm::Instruction * i,
    std::vector<std::unique_ptr<llvm::tac>> & tacs,
    Context & ctx);

std::vector<std::unique_ptr<llvm::tac>>
ConvertConstant(::llvm::Constant * constant, Context & ctx);

const variable *
ConvertConstant(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::tac>> & tacs,
    Context & ctx);

}

#endif
