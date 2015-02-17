/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONSTANT_HPP
#define JLM_CONSTRUCTION_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

namespace jlm {
namespace frontend {
	class basic_block;
	class variable;
}
}

namespace llvm {
	class APInt;
	class Constant;
	class Type;
}

namespace jlm {

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

const jlm::frontend::variable *
create_undef_value(const llvm::Type & type, jlm::frontend::basic_block * bb);

const jlm::frontend::variable *
convert_constant(const llvm::Constant & constant, jlm::frontend::basic_block * bb);

}

#endif
