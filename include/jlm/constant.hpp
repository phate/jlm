/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTANT_HPP
#define JLM_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

namespace jlm {
namespace frontend {
	class basic_block;
	class output;
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

const jlm::frontend::output *
create_undef_value(const llvm::Type & type, jlm::frontend::basic_block * bb);

const jlm::frontend::output *
convert_constant(const llvm::Constant & constant, jlm::frontend::basic_block * bb);

}

#endif
