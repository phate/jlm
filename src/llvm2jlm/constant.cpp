/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/operators.hpp>
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

static inline std::unique_ptr<const expr>
tacs2expr(const std::vector<std::unique_ptr<jlm::tac>> & tacs)
{
	JLM_DEBUG_ASSERT(tacs.size() > 0 && tacs.back()->noutputs() == 1);

	std::unordered_map<const variable*, std::unique_ptr<const expr>> map;
	for (const auto & tac : tacs) {
		std::vector<std::unique_ptr<const expr>> operands;
		for (size_t n = 0; n < tac->ninputs(); n++) {
			auto v = tac->input(n);
			JLM_DEBUG_ASSERT(map.find(v) != map.end());
			operands.push_back(std::move(map[v]));
		}

		JLM_DEBUG_ASSERT(tac->noutputs() == 1);
		map[tac->output(0)] = std::make_unique<const expr>(tac->operation(), std::move(operands));
	}

	JLM_DEBUG_ASSERT(tacs.back()->noutputs() == 1);
	return std::unique_ptr<const expr>(map[tacs.back()->output(0)].release());
}

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

const variable *
create_undef_value(
	const llvm::Type * type,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	if (type->isIntegerTy()) {
		size_t nbits = type->getIntegerBitWidth();
		jive::bits::constant_op op(jive::bits::value_repr::repeat(nbits, 'X'));
		auto r = ctx.module().create_variable(op.result_type(0), false);
		tacs.push_back(create_tac(op, {}, {r}));
		return r;
	}

	/* FIXME: use proper undef floating point type; differentiate according to width */
	if (type->isFloatingPointTy()) {
		jive::flt::constant_op op(nan(""));
		auto r = ctx.module().create_variable(op.result_type(0), false);
		tacs.push_back(create_tac(op, {}, {r}));
		return r;
	}

	/* FIXME: adjust when we have a real unknown value */
	if (type->isPointerTy()) {
		auto t = convert_type(type, ctx);
		auto r = ctx.module().create_variable(*t, false);
		tacs.push_back(create_ptr_constant_null_tac(*t, r));
		return r;
	}

	if (type->isStructTy()){
		std::vector<const variable*> operands;
		for (size_t n = 0; n < type->getStructNumElements(); n++)
			operands.push_back(create_undef_value(type->getStructElementType(n), tacs, ctx));

		jive::rcd::group_op op(ctx.lookup_declaration(static_cast<const llvm::StructType*>(type)));
		auto r = ctx.module().create_variable(op.result_type(0), false);
		tacs.push_back(create_tac(op, operands, {r}));
		return r;
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

std::vector<std::unique_ptr<jlm::tac>>
create_undef_value(const llvm::Type * type, context & ctx)
{
	std::vector<std::unique_ptr<jlm::tac>> tacs;
	create_undef_value(type, tacs, ctx);
	return tacs;
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

static const variable *
convert_undefvalue_instruction(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UndefValue*>(c));
	return create_undef_value(c->getType(), tacs, ctx);
}

static const variable *
convert_constantExpr(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantExpr*>(constant));
	auto c = llvm::cast<llvm::ConstantExpr>(constant);

	/* FIXME: getAsInstruction is none const, forcing all llvm parameters to be none const */
	auto tmp = convert_instruction(c->getAsInstruction(), ctx);
	for (size_t n = 0; n < tmp.size(); n++)
		tacs.push_back(std::move(tmp[n])); 

	JLM_DEBUG_ASSERT(tacs.back()->noutputs() == 1);
	return tacs.back()->output(0);
}

static const variable *
convert_constantFP(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantFP*>(constant));

	/* FIXME: convert APFloat and take care of all types */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_globalVariable(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GlobalVariable*>(constant));
	auto c = static_cast<llvm::GlobalVariable*>(constant);

	if (!c->hasInitializer())
		return create_undef_value(c->getType(), tacs, ctx);

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
	const llvm::ConstantAggregateZero * c = static_cast<const llvm::ConstantAggregateZero*>(constant);

	if (c->getType()->isStructTy()) {
		const llvm::StructType * type = static_cast<const llvm::StructType*>(c->getType());

		std::vector<const variable*> operands;
		for (size_t n = 0; n < type->getNumElements(); n++)
			operands.push_back(convert_constant(c->getElementValue(n), tacs, ctx));

		jive::rcd::group_op op(ctx.lookup_declaration(type));
		auto r = ctx.module().create_variable(op.result_type(0), false);
		tacs.push_back(create_tac(op, operands, {r}));
		return r;
	}

	/* FIXME: */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_constantArray(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantArray*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
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
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ConstantStruct*>(constant));

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
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
	return cmap[c->getValueID()](c, tacs, ctx);
}

std::vector<std::unique_ptr<jlm::tac>>
convert_constant(llvm::Constant * c, context & ctx)
{
	std::vector<std::unique_ptr<jlm::tac>> tacs;
	convert_constant(c, tacs, ctx);
	return tacs;
}

std::unique_ptr<const expr>
convert_constant_expression(llvm::Constant * c, context & ctx)
{
	auto tacs = convert_constant(c, ctx);
	return tacs2expr(tacs);
}

}
