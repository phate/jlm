/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONSTANT_HPP
#define JLM_CONSTRUCTION_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

namespace llvm {
	class APInt;
	class Constant;
	class Type;
}

namespace jlm {

class context;
class variable;

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

const variable *
create_undef_value(
	const llvm::Type * type,
	const context & ctx);

const variable *
convert_constant(
	const llvm::Constant * constant,
	const context & ctx);

}

#endif
