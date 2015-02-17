/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_INSTRUCTION_HPP
#define JLM_CONSTRUCTION_INSTRUCTION_HPP

namespace llvm {
	class Instruction;
	class Value;
}

namespace jlm  {
namespace frontend {
	class basic_block;
}

class context;

const jlm::frontend::variable *
convert_value(const llvm::Value * v, jlm::frontend::basic_block * bb, jlm::context & ctx);

void
convert_instruction(
	const llvm::Instruction & i,
	jlm::frontend::basic_block * bb,
	jlm::context & ctx);

}

#endif
