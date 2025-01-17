/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_JLM2LLVM_INSTRUCTION_HPP
#define JLM_LLVM_BACKEND_JLM2LLVM_INSTRUCTION_HPP

#include <jlm/llvm/ir/tac.hpp>

namespace llvm
{

class Constant;

}

namespace jlm::llvm
{

class cfg_node;
class tac;

namespace jlm2llvm
{

class context;

void
convert_instruction(const llvm::tac & tac, const cfg_node * node, context & ctx);

void
convert_tacs(const tacsvector_t & tacs, context & ctx);

}
}

#endif
