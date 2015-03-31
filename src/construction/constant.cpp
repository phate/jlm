/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/tac.hpp>

#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/value-representation.h>

#include <llvm/IR/Constants.h>

#include <unordered_map>

namespace jlm {

jive::bits::value_repr
convert_apint(const llvm::APInt & value)
{
	llvm::APInt v;
	if (value.isNegative())
		v = -value;

	std::string str = value.toString(2, false);
	std::reverse(str.begin(), str.end());

	jive::bits::value_repr vr(str.c_str());
	if (value.isNegative())
		vr = vr.sext(value.getBitWidth() - str.size());
	else
		vr = vr.zext(value.getBitWidth() - str.size());

	return vr;
}

const jlm::variable *
create_undef_value(
	const llvm::Type * type,
	jlm::basic_block * bb)
{
	if (type->getTypeID() == llvm::Type::IntegerTyID) {
		size_t nbits = static_cast<const llvm::IntegerType*>(type)->getBitWidth();
		jive::bits::constant_op op(jive::bits::value_repr::repeat(nbits, 'X'));
		return bb->append(op, {})->output(0);
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const jlm::variable *
convert_int_constant(
	const llvm::Constant * c,
	jlm::basic_block * bb)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantInt*>(c));
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bits::value_repr v = convert_apint(constant->getValue());
	return bb->append(jive::bits::constant_op(convert_apint(constant->getValue())), {})->output(0);
}

static const jlm::variable *
convert_undefvalue_instruction(
	const llvm::Constant * c,
	jlm::basic_block * bb)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UndefValue*>(c));
	const llvm::UndefValue * constant = static_cast<const llvm::UndefValue*>(c);

	return create_undef_value(constant->getType(), bb);
}

typedef std::unordered_map<
	std::type_index,
	const jlm::variable*(*)(const llvm::Constant *, jlm::basic_block*)
	> constant_map;

static constant_map cmap({
		{std::type_index(typeid(llvm::ConstantInt)), convert_int_constant}
	, {std::type_index(typeid(llvm::UndefValue)), convert_undefvalue_instruction}
});

const jlm::variable *
convert_constant(const llvm::Constant * c, jlm::basic_block * bb)
{
	JLM_DEBUG_ASSERT(cmap.find(std::type_index(typeid(*c))) != cmap.end());
	return cmap[std::type_index(typeid(*c))](c, bb);
}

}
