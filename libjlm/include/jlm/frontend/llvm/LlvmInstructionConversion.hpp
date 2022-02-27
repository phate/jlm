/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_LLVM2JLM_LLVMINSTRUCTIONCONVERSION_HPP
#define JLM_FRONTEND_LLVM_LLVM2JLM_LLVMINSTRUCTIONCONVERSION_HPP

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
ConvertInstruction(
	llvm::Instruction * i,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);

std::vector<std::unique_ptr<jlm::tac>>
ConvertConstant(llvm::Constant * constant, context & ctx);

const variable *
ConvertConstant(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);


}

#endif
