/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/instruction.hpp>
#include <jlm/IR/expression.hpp>

#include <jive/arch/address.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/value-representation.h>
#include <jive/types/float/fltconstant.h>
#include <jive/types/record/rcdgroup.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalVariable.h>

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
create_undef_value(const llvm::Type * type, context & ctx)
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

	if (type->isStructTy()){
		std::vector<std::shared_ptr<const expr>> operands;
		for (size_t n = 0; n < type->getStructNumElements(); n++)
			operands.push_back(create_undef_value(type->getStructElementType(n), ctx));

		jive::rcd::group_op op(ctx.lookup_declaration(static_cast<const llvm::StructType*>(type)));
		return std::shared_ptr<const expr>(new expr(op, operands));
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static std::shared_ptr<const expr>
convert_int_constant(const llvm::Constant * c, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantInt*>(c));
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bits::value_repr v = convert_apint(constant->getValue());
	return std::shared_ptr<const expr>(new expr(jive::bits::constant_op(v), {}));
}

static std::shared_ptr<const expr>
convert_undefvalue_instruction(const llvm::Constant * c, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UndefValue*>(c));
	const llvm::UndefValue * constant = static_cast<const llvm::UndefValue*>(c);

	return create_undef_value(constant->getType(), ctx);
}

static std::shared_ptr<const expr>
convert_constantExpr(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantExpr*>(constant));
	/* const llvm::ConstantExpr * c = static_cast<const llvm::ConstantExpr*>(constant); */

	/* FIXME */
	return create_undef_value(constant->getType(), ctx);

	//convert_instruction(const_cast<llvm::ConstantExpr*>(c)->getAsInstruction(),
	//	ctx.entry_block(), ctx);
}

static std::shared_ptr<const expr>
convert_constantFP(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantFP*>(constant));

	/* FIXME: convert APFloat and take care of all types */
	return std::shared_ptr<const expr>(new expr(jive::flt::constant_op(nan("")), {}));
}

static std::shared_ptr<const expr>
convert_globalVariable(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GlobalVariable*>(constant));
	const llvm::GlobalVariable * c = static_cast<const llvm::GlobalVariable*>(constant);

	if (!c->hasInitializer())
		return create_undef_value(c->getType(), ctx);

	return convert_constant(c->getInitializer(), ctx);
}

static std::shared_ptr<const expr>
convert_constantPointerNull(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantPointerNull*>(constant));

	return std::shared_ptr<const expr>(new expr(jive::address::constant_op(0), {}));
}

static std::shared_ptr<const expr>
convert_blockAddress(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BlockAddress*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_constantAggregateZero(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantAggregateZero*>(constant));
	const llvm::ConstantAggregateZero * c = static_cast<const llvm::ConstantAggregateZero*>(constant);

	if (c->getType()->isStructTy()) {
		const llvm::StructType * type = static_cast<const llvm::StructType*>(c->getType());

		std::vector<std::shared_ptr<const expr>> operands;
		for (size_t n = 0; n < type->getNumElements(); n++)
			operands.push_back(convert_constant(c->getElementValue(n), ctx));

		jive::rcd::group_op op(ctx.lookup_declaration(type));
		return std::shared_ptr<const expr>(new expr(op, operands));
	}

	/* FIXME: */
	if (c->getType()->isArrayTy())
		return std::shared_ptr<const expr>(new expr(jive::address::constant_op(0), {}));

	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_constantArray(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantArray*>(constant));

	/* FIXME */
	return std::shared_ptr<const expr>(new expr(jive::address::constant_op(0), {}));
}

static std::shared_ptr<const expr>
convert_constantDataArray(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantDataArray*>(constant));

	/* FIXME */
	return std::shared_ptr<const expr>(new expr(jive::address::constant_op(0), {}));
}

static std::shared_ptr<const expr>
convert_constantDataVector(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantDataVector*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_constantStruct(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantStruct*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_constantVector(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantVector*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_globalAlias(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GlobalAlias*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
}

static std::shared_ptr<const expr>
convert_function(const llvm::Constant * constant, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::Function*>(constant));

	/* FIXME: */
	return std::shared_ptr<const expr>(new expr(jive::address::constant_op(0), {}));
}

std::shared_ptr<const expr>
convert_constant(const llvm::Constant * c, context & ctx)
{
	static std::unordered_map<
		unsigned,
		std::shared_ptr<const expr> (*)(const llvm::Constant*, context & ctx)
	> cmap({
		{llvm::Value::ConstantIntVal, convert_int_constant}
	,	{llvm::Value::UndefValueVal, convert_undefvalue_instruction}
	,	{llvm::Value::ConstantExprVal, convert_constantExpr}
	,	{llvm::Value::ConstantFPVal, convert_constantFP}
	,	{llvm::Value::GlobalVariableVal, convert_globalVariable}
	,	{llvm::Value::ConstantPointerNullVal, convert_constantPointerNull}
	,	{llvm::Value::BlockAddressVal, convert_blockAddress}
	,	{llvm::Value::ConstantAggregateZeroVal, convert_constantAggregateZero}
	,	{llvm::Value::ConstantArrayVal, convert_constantArray}
	,	{llvm::Value::ConstantDataArrayVal, convert_constantDataArray}
	,	{llvm::Value::ConstantDataVectorVal, convert_constantDataVector}
	,	{llvm::Value::ConstantStructVal, convert_constantStruct}
	,	{llvm::Value::ConstantVectorVal, convert_constantVector}
	,	{llvm::Value::GlobalAliasVal, convert_globalAlias}
	,	{llvm::Value::FunctionVal, convert_function}
	});

	JLM_DEBUG_ASSERT(cmap.find(c->getValueID()) != cmap.end());
	return cmap[c->getValueID()](c, ctx);
}

}
