/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_JLM2LLVM_INSTRUCTION_HPP
#define JLM_LLVM_BACKEND_JLM2LLVM_INSTRUCTION_HPP

namespace llvm {

class Constant;

}

namespace jlm {

class tac;

namespace jlm2llvm {

class context;

void
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx);

void
convert_tacs(const tacsvector_t & tacs, context & ctx);

}}

#endif
