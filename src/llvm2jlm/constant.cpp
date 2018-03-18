/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/constant.hpp>
#include <jlm/llvm2jlm/instruction.hpp>

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

const variable *
convert_constant(llvm::Constant*, std::vector<std::unique_ptr<jlm::tac>>&, context&);

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

double
convert_apfloat(const llvm::APFloat & value)
{
	using namespace llvm;

	if (APFloatBase::getSizeInBits(value.getSemantics()) == 32)
		return value.convertToFloat();

	JLM_DEBUG_ASSERT(APFloatBase::getSizeInBits(value.getSemantics()) == 64);
	return value.convertToDouble();
}

static const variable *
convert_int_constant(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantInt*>(c));
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bits::value_repr v = convert_apint(constant->getValue());
	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx), false);
	tacs.push_back(create_tac(jive::bits::constant_op(v), {}, {r}));
	return r;
}

static inline const variable *
convert_undefvalue(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::UndefValueVal);

	auto t = convert_type(c->getType(), ctx);
	auto r = ctx.module().create_variable(*t, false);
	tacs.push_back(create_undef_constant_tac(r));
	return r;
}

static const variable *
convert_constantExpr(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantExpr*>(constant));
	auto c = llvm::cast<llvm::ConstantExpr>(constant);

	/*
		FIXME: convert_instruction currently assumes that a instruction's result variable
					 is already added to the context. This is not the case for constants and we
					 therefore need to do some poilerplate checking in convert_instruction to
					 see whether a variable was already declared or we need to create a new
					 variable.
	*/

	/* FIXME: getAsInstruction is none const, forcing all llvm parameters to be none const */
	return convert_instruction(c->getAsInstruction(), tacs, ctx);
}

static const variable *
convert_constantFP(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::ConstantFPVal);
	auto c = llvm::cast<llvm::ConstantFP>(constant);

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx), false);
	tacs.push_back(create_fpconstant_tac(c->getValueAPF(), r));
	return r;
}

static const variable *
convert_globalVariable(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GlobalVariable*>(constant));
	auto c = static_cast<llvm::GlobalVariable*>(constant);

	if (!c->hasInitializer()) {
		auto t = convert_type(c->getType(), ctx);
		auto r = ctx.module().create_variable(*t, false);
		tacs.push_back(create_undef_constant_tac(r));
		return r;
	}

	return convert_constant(c->getInitializer(), tacs, ctx);
}

static const variable *
convert_constantPointerNull(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::ConstantPointerNull>(constant));
	auto & c = *llvm::cast<const llvm::ConstantPointerNull>(constant);

	auto t = convert_type(c.getType(), ctx);
	auto r = ctx.module().create_variable(*t, false);
	tacs.push_back(create_ptr_constant_null_tac(*t,r));
	return r;
}

static const variable *
convert_blockAddress(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BlockAddress*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_constantAggregateZero(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantAggregateZero*>(constant));
	auto c = static_cast<const llvm::ConstantAggregateZero*>(constant);

	if (c->getType()->isStructTy()) {
		const llvm::StructType * type = static_cast<const llvm::StructType*>(c->getType());

		std::vector<const variable*> operands;
		for (size_t n = 0; n < type->getNumElements(); n++)
			operands.push_back(convert_constant(c->getElementValue(n), tacs, ctx));

		jive::rcd::group_op op(ctx.lookup_declaration(type));
		auto r = ctx.module().create_variable(op.result(0).type(), false);
		tacs.push_back(create_tac(op, operands, {r}));
		return r;
	}

	if (c->getType()->isArrayTy()) {
		auto type = static_cast<const llvm::ArrayType*>(c->getType());

		std::vector<const variable*> operands;
		for (size_t n = 0; n < type->getNumElements(); n++)
			operands.push_back(convert_constant(c->getElementValue(n), tacs, ctx));

		auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx), false);
		tacs.push_back(create_data_array_constant_tac(operands, {r}));
		return r;
	}

	/* FIXME: */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_constantArray(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::ConstantArrayVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++) {
		auto operand = c->getOperand(n);
		JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::Constant>(operand));
		auto constant = llvm::cast<llvm::Constant>(operand);
		elements.push_back(convert_constant(constant, tacs, ctx));
	}

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx), false);
	tacs.push_back(create_constant_array_tac(elements, r));
	return r;
}

static const variable *
convert_constantDataArray(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<llvm::ConstantDataArray>(constant));
	const auto & c = *llvm::cast<const llvm::ConstantDataArray>(constant);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c.getNumElements(); n++)
		elements.push_back(convert_constant(c.getElementAsConstant(n), tacs, ctx));

	auto r = ctx.module().create_variable(*convert_type(c.getType(), ctx), false);
	tacs.push_back(create_data_array_constant_tac(elements, r));
	return r;
}

static const variable *
convert_constantDataVector(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantDataVector*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_constantStruct(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::ConstantStructVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++)
		elements.push_back(convert_constant(c->getAggregateElement(n), tacs, ctx));

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx), false);
	tacs.push_back(create_struct_constant_tac(elements, r));
	return r;
}

static const variable *
convert_constantVector(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantVector*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static inline const variable *
convert_globalAlias(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GlobalAlias*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static inline const variable *
convert_function(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::Function*>(constant));

	/* FIXME: */
	return nullptr;
}

const variable *
convert_constant(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	static std::unordered_map<
		unsigned,
		const variable*(*)(
			llvm::Constant*,
			std::vector<std::unique_ptr<jlm::tac>>&,
			context & ctx)
	> cmap({
		{llvm::Value::ConstantIntVal, convert_int_constant}
	,	{llvm::Value::UndefValueVal, convert_undefvalue}
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
	return cmap[c->getValueID()](c, tacs, ctx);
}

std::vector<std::unique_ptr<jlm::tac>>
convert_constant(llvm::Constant * c, context & ctx)
{
	std::vector<std::unique_ptr<jlm::tac>> tacs;
	convert_constant(c, tacs, ctx);
	return tacs;
}

}
