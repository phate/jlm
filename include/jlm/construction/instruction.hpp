/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_INSTRUCTION_HPP
#define JLM_CONSTRUCTION_INSTRUCTION_HPP

#include <jlm/construction/jlm.hpp>

namespace jlm {
namespace frontend {
	class output;
}
}

namespace jlm  {

const jlm::frontend::output *
convert_value(const llvm::Value * v, jlm::frontend::basic_block * bb, value_map & vmap);

void
convert_instruction(
	const llvm::Instruction & i,
	jlm::frontend::basic_block * bb,
	const basic_block_map & bbmap,
	value_map & vmap,
	const jlm::frontend::output * state,
	const jlm::frontend::variable * result);

}

#endif
