/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTANT_HPP
#define JLM_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

namespace jive {
namespace frontend {
	class basic_block;
	class output;
}
}

namespace llvm {
	class APInt;
	class Constant;
}

namespace jlm {

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

const jive::frontend::output *
convert_constant(const llvm::Constant & constant, jive::frontend::basic_block * bb);

}

#endif
