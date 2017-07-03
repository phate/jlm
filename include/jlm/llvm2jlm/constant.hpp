/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_CONSTANT_HPP
#define JLM_LLVM2JLM_CONSTANT_HPP

#include <jive/types/bitstring/value-representation.h>

#include <memory>

namespace llvm {
	class APFloat;
	class APInt;
	class Constant;
	class Type;
}

namespace jlm {

class context;
class expr;
class tac;
class variable;

jive::bits::value_repr
convert_apint(const llvm::APInt & value);

double
convert_apfloat(const llvm::APFloat & value);

std::vector<std::unique_ptr<jlm::tac>>
create_undef_value(const llvm::Type * type, context & ctx);

const variable *
convert_undef_value(
	const llvm::Type * type,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);

std::vector<std::unique_ptr<jlm::tac>>
convert_constant(llvm::Constant * constant, context & ctx);

const variable *
convert_constant(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx);

std::unique_ptr<const expr>
convert_constant_expression(llvm::Constant * constant, context & ctx);

}

#endif
