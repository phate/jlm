/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_BINOPS_HPP
#define JLM_CONSTRUCTION_BINOPS_HPP

namespace llvm {
	class BinaryOperator;
	class CmpInst;
	class ICmpInst;
}

namespace jlm {
namespace frontend {
	class basic_block;
	class variable;
}

class context;

void
convert_binary_operator(const llvm::BinaryOperator & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx);

void
convert_comparison_instruction(const llvm::CmpInst & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx);

const jlm::frontend::variable *
convert_int_comparison_instruction(const llvm::ICmpInst & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx);

}

#endif
