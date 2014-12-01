/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BINOPS_HPP
#define JLM_BINOPS_HPP

#include <jlm/jlm.hpp>

namespace llvm {
	class BinaryOperator;
	class CmpInst;
	class ICmpInst;
}

namespace jlm {

void
convert_binary_operator(const llvm::BinaryOperator & i, jive::frontend::basic_block * bb,
	value_map & vmap, const jive::frontend::output ** state);

void
convert_comparison_instruction(const llvm::CmpInst & i, jive::frontend::basic_block * bb,
	value_map & vmap, const jive::frontend::output ** state);

const jive::frontend::output *
convert_int_comparison_instruction(const llvm::ICmpInst & i, jive::frontend::basic_block * bb,
	value_map & vmap, const jive::frontend::output ** state);

}

#endif
