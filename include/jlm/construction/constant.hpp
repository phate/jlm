/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONSTANT_HPP
#define JLM_CONSTRUCTION_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

#include <memory>

namespace llvm {
	class APInt;
	class Constant;
	class Type;
}

namespace jlm {

class context;
class expr;

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

std::shared_ptr<const expr>
create_undef_value(const llvm::Type * type, context & ctx);

std::shared_ptr<const expr>
convert_constant(const llvm::Constant * constant, context & ctx);

}

#endif
