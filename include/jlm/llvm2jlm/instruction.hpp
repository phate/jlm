/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_INSTRUCTION_HPP
#define JLM_LLVM2JLM_INSTRUCTION_HPP

namespace llvm {
	class Instruction;
	class Value;
}

namespace jlm  {

class basic_block_attribute;
class context;
class variable;

std::vector<std::unique_ptr<jlm::tac>>
convert_instruction(llvm::Instruction * i, context & ctx);

}

#endif
