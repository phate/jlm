/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_INSTRUCTION_HPP
#define JLM_LLVM2JLM_INSTRUCTION_HPP

namespace llvm {
	class Constant;
	class Instruction;
	class Value;
}

namespace jlm  {

class context;
class variable;

const variable *
convert_value(
	llvm::Value * v,
	tacsvector_t & tacs,
	context & ctx);

const variable *
convert_instruction(
	llvm::Instruction * i,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);

std::vector<std::unique_ptr<jlm::tac>>
convert_constant(llvm::Constant * constant, context & ctx);

const variable *
convert_constant(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);


}

#endif
