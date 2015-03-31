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

class basic_block;
class variable;

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

const jlm::variable *
create_undef_value(const llvm::Type * type, jlm::basic_block * bb);

const jlm::variable *
convert_constant(const llvm::Constant * constant, jlm::basic_block * bb);

}

#endif
