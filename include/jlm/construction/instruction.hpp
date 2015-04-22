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

class basic_block;
class context;
class variable;

const variable *
convert_value(
	const llvm::Value * v,
	const jlm::context & ctx);

const variable *
convert_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx);

}

#endif
