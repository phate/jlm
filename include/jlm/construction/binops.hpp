/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_BINOPS_HPP
#define JLM_CONSTRUCTION_BINOPS_HPP

#include <jlm/construction/jlm.hpp>

namespace llvm {
	class BinaryOperator;
	class CmpInst;
	class ICmpInst;
}

namespace jlm {

void
convert_binary_operator(const llvm::BinaryOperator & i, jlm::frontend::basic_block * bb,
	value_map & vmap, const jlm::frontend::output * state);

void
convert_comparison_instruction(const llvm::CmpInst & i, jlm::frontend::basic_block * bb,
	value_map & vmap, const jlm::frontend::output * state);

const jlm::frontend::output *
convert_int_comparison_instruction(const llvm::ICmpInst & i, jlm::frontend::basic_block * bb,
	value_map & vmap, const jlm::frontend::output * state);

}

#endif
