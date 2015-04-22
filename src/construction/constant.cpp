/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/instruction.hpp>
#include <jlm/IR/expression.hpp>

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

std::shared_ptr<const expr>
create_undef_value(const llvm::Type * type)
{
	if (type->isIntegerTy()) {
		size_t nbits = type->getIntegerBitWidth();
		jive::bits::constant_op op(jive::bits::value_repr::repeat(nbits, 'X'));
		return std::shared_ptr<const expr>(new expr(op, {}));
	}

	/* FIXME: differentiate between floating point types */
	if (type->isFloatingPointTy())
		return std::shared_ptr<const expr>(new expr(jive::flt::constant_op(nan("")), {}));

	/* FIXME: adjust when we have a real unknown value */
	if (type->isPointerTy()) {
		jive::address::constant_op op(jive::address::value_repr(0));
		return std::shared_ptr<const expr>(new expr(op, {}));
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static std::shared_ptr<const expr>
convert_int_constant(const llvm::Constant * c)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantInt*>(c));
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bits::value_repr v = convert_apint(constant->getValue());
	return std::shared_ptr<const expr>(new expr(jive::bits::constant_op(v), {}));
}

static std::shared_ptr<const expr>
convert_undefvalue_instruction(const llvm::Constant * c)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UndefValue*>(c));
	const llvm::UndefValue * constant = static_cast<const llvm::UndefValue*>(c);

	return create_undef_value(constant->getType());
}

static std::shared_ptr<const expr>
convert_constantExpr(const llvm::Constant * constant)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantExpr*>(constant));
	/* const llvm::ConstantExpr * c = static_cast<const llvm::ConstantExpr*>(constant); */

	/* FIXME */

	return nullptr;
	//convert_instruction(const_cast<llvm::ConstantExpr*>(c)->getAsInstruction(),
	//	ctx.entry_block(), ctx);
}

static std::shared_ptr<const expr>
convert_constantFP(const llvm::Constant * constant)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantFP*>(constant));

	/* FIXME: convert APFloat and take care of all types */
	return std::shared_ptr<const expr>(new expr(jive::flt::constant_op(nan("")), {}));
}

typedef std::unordered_map<
	std::type_index,
	std::shared_ptr<const expr> (*)(const llvm::Constant *)
	> constant_map;

static constant_map cmap({
		{std::type_index(typeid(llvm::ConstantInt)), convert_int_constant}
	, {std::type_index(typeid(llvm::UndefValue)), convert_undefvalue_instruction}
	, {std::type_index(typeid(llvm::ConstantExpr)), convert_constantExpr}
	,	{std::type_index(typeid(llvm::ConstantFP)), convert_constantFP}
});

std::shared_ptr<const expr>
convert_constant(const llvm::Constant * c)
{
	JLM_DEBUG_ASSERT(cmap.find(std::type_index(typeid(*c))) != cmap.end());
	return cmap[std::type_index(typeid(*c))](c);
}

}
