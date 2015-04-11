/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/instruction.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/tac.hpp>

#include <jive/arch/address.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/value-representation.h>
#include <jive/types/float/fltconstant.h>

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
	const context & ctx)
{
	basic_block * bb = ctx.entry_block();

	if (type->isIntegerTy()) {
		size_t nbits = type->getIntegerBitWidth();
		jive::bits::constant_op op(jive::bits::value_repr::repeat(nbits, 'X'));
		return bb->append(op, {})->output(0);
	}

	/* FIXME: differentiate between floating point types */
	if (type->isFloatingPointTy())
		return bb->append(jive::flt::constant_op(nan("")), {})->output(0);

	/* FIXME: adjust when we have a real unknown value */
	if (type->isPointerTy())
		return bb->append(jive::address::constant_op(jive::address::value_repr(0)), {})->output(0);

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const jlm::variable *
convert_int_constant(
	const llvm::Constant * c,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantInt*>(c));
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	basic_block * bb = ctx.entry_block();
	jive::bits::value_repr v = convert_apint(constant->getValue());
	return bb->append(jive::bits::constant_op(convert_apint(constant->getValue())), {})->output(0);
}

static const jlm::variable *
convert_undefvalue_instruction(
	const llvm::Constant * c,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UndefValue*>(c));
	const llvm::UndefValue * constant = static_cast<const llvm::UndefValue*>(c);

	return create_undef_value(constant->getType(), ctx);
}

static const variable *
convert_constantExpr(
	const llvm::Constant * constant,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantExpr*>(constant));
	const llvm::ConstantExpr * c = static_cast<const llvm::ConstantExpr*>(constant);

	return convert_instruction(const_cast<llvm::ConstantExpr*>(c)->getAsInstruction(),
		ctx.entry_block(), ctx);
}

static const variable *
convert_constantFP(
	const llvm::Constant * constant,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantFP*>(constant));

	/* FIXME: convert APFloat and take care of all types */
	basic_block * bb = ctx.entry_block();
	return bb->append(jive::flt::constant_op(nan("")), {})->output(0);
}

typedef std::unordered_map<
	std::type_index,
	const jlm::variable*(*)(const llvm::Constant *, const context & ctx)
	> constant_map;

static constant_map cmap({
		{std::type_index(typeid(llvm::ConstantInt)), convert_int_constant}
	, {std::type_index(typeid(llvm::UndefValue)), convert_undefvalue_instruction}
	, {std::type_index(typeid(llvm::ConstantExpr)), convert_constantExpr}
	,	{std::type_index(typeid(llvm::ConstantFP)), convert_constantFP}
});

const variable *
convert_constant(
	const llvm::Constant * c,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(cmap.find(std::type_index(typeid(*c))) != cmap.end());
	return cmap[std::type_index(typeid(*c))](c, ctx);
}

}
