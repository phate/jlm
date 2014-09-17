/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_INSTRUCTION_HPP
#define JLM_INSTRUCTION_HPP

#include <jlm/jlm.hpp>

namespace jlm  {

void
convert_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap);

}

#endif
