/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2LLVM_INSTRUCTION_HPP
#define JLM_JLM2LLVM_INSTRUCTION_HPP

namespace llvm {

class Constant;

}

namespace jlm {

class expr;
class tac;

namespace jlm2llvm {

class context;

void
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx);

llvm::Constant *
convert_expression(const jlm::expr & e, context & ctx);

}}

#endif
