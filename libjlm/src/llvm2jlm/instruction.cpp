/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/instruction.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/ipgraph.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>

#include <jive/arch/address.hpp>
#include <jive/arch/address-transform.hpp>
#include <jive/arch/load.hpp>
#include <jive/arch/addresstype.hpp>
#include <jive/arch/store.hpp>
#include <jive/types/bitstring.hpp>
#include <jive/types/float.hpp>
#include <jive/types/record.hpp>
#include <jive/rvsdg/control.hpp>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>

#include <typeindex>

static inline std::vector<const jlm::variable*>
create_result_variables(jlm::ipgraph_module & im, const jive::simple_op & op)
{
	std::vector<const jlm::variable*> variables;
	for (size_t n = 0; n < op.nresults(); n++)
		variables.push_back(im.create_variable(op.result(n).type()));

	return variables;
}

namespace jlm {

const variable *
convert_value(llvm::Value * v, tacsvector_t & tacs, context & ctx)
{
	auto node = ctx.node();
	if (node && ctx.has_value(v)) {
		if (auto callee = dynamic_cast<const fctvariable*>(ctx.lookup_value(v)))
			node->add_dependency(callee->function());

		if (auto data = dynamic_cast<const gblvalue*>(ctx.lookup_value(v)))
			node->add_dependency(data->node());
	}

	if (ctx.has_value(v))
		return ctx.lookup_value(v);

	if (auto c = llvm::dyn_cast<llvm::Constant>(v))
		return convert_constant(c, tacs, ctx);

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

/* constant */

const variable *
convert_constant(llvm::Constant*, std::vector<std::unique_ptr<jlm::tac>>&, context&);

static jive::bitvalue_repr
convert_apint(const llvm::APInt & value)
{
	llvm::APInt v;
	if (value.isNegative())
		v = -value;

	std::string str = value.toString(2, false);
	std::reverse(str.begin(), str.end());

	jive::bitvalue_repr vr(str.c_str());
	if (value.isNegative())
		vr = vr.sext(value.getBitWidth() - str.size());
	else
		vr = vr.zext(value.getBitWidth() - str.size());

	return vr;
}

static const variable *
convert_int_constant(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::ConstantIntVal);
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bitvalue_repr v = convert_apint(constant->getValue());
	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(tac::create(jive::bitconstant_op(v), {}, {r}));
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
	auto r = ctx.module().create_variable(*t);
	tacs.push_back(create_undef_constant_tac(r));
	return r;
}

static const variable *
convert_constantExpr(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::ConstantExprVal);
	auto c = llvm::cast<llvm::ConstantExpr>(constant);

	/*
		FIXME: convert_instruction currently assumes that a instruction's result variable
					 is already added to the context. This is not the case for constants and we
					 therefore need to do some poilerplate checking in convert_instruction to
					 see whether a variable was already declared or we need to create a new
					 variable.
	*/

	/* FIXME: getAsInstruction is none const, forcing all llvm parameters to be none const */
	/* FIXME: The invocation of getAsInstruction() introduces a memory leak. */
	auto instruction = c->getAsInstruction();
	auto v = convert_instruction(instruction, tacs, ctx);
	instruction->dropAllReferences();
	return v;
}

static const variable *
convert_constantFP(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::ConstantFPVal);
	auto c = llvm::cast<llvm::ConstantFP>(constant);

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(create_fpconstant_tac(c->getValueAPF(), r));
	return r;
}

static const variable *
convert_globalVariable(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::GlobalVariableVal);
	return convert_value(c, tacs, ctx);
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
	auto r = ctx.module().create_variable(*t);
	tacs.push_back(ptr_constant_null_op::create(*t,r));
	return r;
}

static const variable *
convert_blockAddress(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::BlockAddressVal);

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_constantAggregateZero(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::ConstantAggregateZeroVal);

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(create_constant_aggregate_zero_tac(r));
	return r;
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

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
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

	auto r = ctx.module().create_variable(*convert_type(c.getType(), ctx));
	tacs.push_back(create_data_array_constant_tac(elements, r));
	return r;
}

static const variable *
convert_constantDataVector(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::ConstantDataVectorVal);
	auto c = llvm::cast<const llvm::ConstantDataVector>(constant);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumElements(); n++)
		elements.push_back(convert_constant(c->getElementAsConstant(n), tacs, ctx));

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(constant_data_vector_op::create(elements, r));
	return r;
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

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(create_struct_constant_tac(elements, r));
	return r;
}

static const variable *
convert_constantVector(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::ConstantVectorVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++)
		elements.push_back(convert_constant(c->getAggregateElement(n), tacs, ctx));

	auto r = ctx.module().create_variable(*convert_type(c->getType(), ctx));
	tacs.push_back(constantvector_op::create(elements, r));
	return r;
}

static inline const variable *
convert_globalAlias(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(constant->getValueID() == llvm::Value::GlobalAliasVal);

	/* FIXME */
	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static inline const variable *
convert_function(
	llvm::Constant * c,
	tacsvector_t & tacs,
	context & ctx)
{
	JLM_DEBUG_ASSERT(c->getValueID() == llvm::Value::FunctionVal);
	return convert_value(c, tacs, ctx);
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

/* instructions */

static inline const variable *
convert_return_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Ret);
	auto i = llvm::cast<llvm::ReturnInst>(instruction);

	auto bb = ctx.get(i->getParent());
	bb->add_outedge(bb->cfg().exit());
	if (!i->getReturnValue())
		return {};

	auto value = convert_value(i->getReturnValue(), tacs, ctx);
	tacs.push_back(assignment_op::create(value, ctx.result()));

	return ctx.result();
}

static inline const variable *
convert_branch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Br);
	auto i = llvm::cast<llvm::BranchInst>(instruction);
	auto bb = ctx.get(i->getParent());

	if (i->isUnconditional()) {
		bb->add_outedge(ctx.get(i->getSuccessor(0)));
		return {};
	}

	bb->add_outedge(ctx.get(i->getSuccessor(1))); /* false */
	bb->add_outedge(ctx.get(i->getSuccessor(0))); /* true */

	auto c = convert_value(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, {{1, 1}}, 0, 2);
	tacs.push_back(tac::create(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back(branch_op::create(2,  tacs.back()->result(0)));

	return nullptr;
}

static inline const variable *
convert_switch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Switch);
	auto i = llvm::cast<llvm::SwitchInst>(instruction);
	auto bb = ctx.get(i->getParent());

	size_t n = 0;
	std::unordered_map<uint64_t, uint64_t> mapping;
	for (auto it = i->case_begin(); it != i->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != i->case_default());
		mapping[it->getCaseValue()->getZExtValue()] = n++;
		bb->add_outedge(ctx.get(it->getCaseSuccessor()));
	}

	bb->add_outedge(ctx.get(i->case_default()->getCaseSuccessor()));
	JLM_DEBUG_ASSERT(i->getNumSuccessors() == n+1);

	auto c = convert_value(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, mapping, n, n+1);
	tacs.push_back(tac::create(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back((branch_op::create(n+1, tacs.back()->result(0))));

	return nullptr;
}

static inline const variable *
convert_unreachable_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Unreachable);
	auto bb = ctx.get(i->getParent());
	bb->add_outedge(bb->cfg().exit());
	return nullptr;
}

static inline const variable *
convert_icmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::ICmp);
	auto i = llvm::cast<const llvm::ICmpInst>(instruction);
	auto t = i->getOperand(0)->getType();

	static std::unordered_map<
		const llvm::CmpInst::Predicate,
		std::unique_ptr<jive::operation>(*)(size_t)> map({
		  {llvm::CmpInst::ICMP_SLT,	[](size_t nbits){jive::bitslt_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_ULT,	[](size_t nbits){jive::bitult_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SLE,	[](size_t nbits){jive::bitsle_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_ULE,	[](size_t nbits){jive::bitule_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_EQ,	[](size_t nbits){jive::biteq_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_NE,	[](size_t nbits){jive::bitne_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SGE,	[](size_t nbits){jive::bitsge_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_UGE,	[](size_t nbits){jive::bituge_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SGT,	[](size_t nbits){jive::bitsgt_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_UGT,	[](size_t nbits){jive::bitugt_op op(nbits); return op.copy();}}
	});

	static std::unordered_map<const llvm::CmpInst::Predicate, jlm::cmp> ptrmap({
	  {llvm::CmpInst::ICMP_ULT,	cmp::lt}, {llvm::CmpInst::ICMP_ULE,	cmp::le}
	,	{llvm::CmpInst::ICMP_EQ,	cmp::eq}, {llvm::CmpInst::ICMP_NE,	cmp::ne}
	,	{llvm::CmpInst::ICMP_UGE,	cmp::ge}, {llvm::CmpInst::ICMP_UGT,	cmp::gt}
	});

	auto p = i->getPredicate();
	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

	std::unique_ptr<jive::operation> binop;
	if (t->isIntegerTy() || (t->isVectorTy() && t->getVectorElementType()->isIntegerTy())) {
		auto it = t->isVectorTy() ? t->getVectorElementType() : t;
		/* FIXME: This is inefficient. We return a unique ptr and then take copy it. */
		binop = map[p](it->getIntegerBitWidth());
	} else if (t->isPointerTy() || (t->isVectorTy() && t->getVectorElementType()->isPointerTy())) {
		auto pt = llvm::cast<llvm::PointerType>(t->isVectorTy() ? t->getVectorElementType() : t);
		binop = std::make_unique<ptrcmp_op>(*convert_type(pt, ctx), ptrmap[p]);
	} else
		JLM_ASSERT(0);

	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	JLM_DEBUG_ASSERT(is<jive::binary_op>(*binop));
	if (t->isVectorTy()) {
		tacs.push_back(vectorbinary_op::create(*static_cast<jive::binary_op*>(binop.get()),
			op1, op2, result));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(binop.get()),
			{op1, op2}, {result}));
	}

	return tacs.back()->result(0);
}

static const variable *
convert_fcmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::FCmp);
	auto i = llvm::cast<const llvm::FCmpInst>(instruction);
	auto t = i->getOperand(0)->getType();

	static std::unordered_map<llvm::CmpInst::Predicate, jlm::fpcmp> map({
		{llvm::CmpInst::FCMP_TRUE, fpcmp::TRUE}, {llvm::CmpInst::FCMP_FALSE, fpcmp::FALSE}
	,	{llvm::CmpInst::FCMP_OEQ, fpcmp::oeq},	{llvm::CmpInst::FCMP_OGT, fpcmp::ogt}
	,	{llvm::CmpInst::FCMP_OGE, fpcmp::oge},	{llvm::CmpInst::FCMP_OLT, fpcmp::olt}
	,	{llvm::CmpInst::FCMP_OLE, fpcmp::ole},	{llvm::CmpInst::FCMP_ONE, fpcmp::one}
	,	{llvm::CmpInst::FCMP_ORD, fpcmp::ord},	{llvm::CmpInst::FCMP_UNO, fpcmp::uno}
	,	{llvm::CmpInst::FCMP_UEQ, fpcmp::ueq},	{llvm::CmpInst::FCMP_UGT, fpcmp::ugt}
	,	{llvm::CmpInst::FCMP_UGE, fpcmp::uge},	{llvm::CmpInst::FCMP_ULT, fpcmp::ult}
	,	{llvm::CmpInst::FCMP_ULE, fpcmp::ule},	{llvm::CmpInst::FCMP_UNE, fpcmp::une}
	});

	auto r = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

	JLM_DEBUG_ASSERT(map.find(i->getPredicate()) != map.end());
	auto fptype = t->isVectorTy() ? t->getVectorElementType() : t;
	fpcmp_op operation(map[i->getPredicate()], convert_fpsize(fptype));

	if (t->isVectorTy())
		tacs.push_back(vectorbinary_op::create(operation, op1, op2, r));
	else
		tacs.push_back(tac::create(operation, {op1, op2}, {r}));

	return tacs.back()->result(0);
}

static inline const variable *
convert_load_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Load);
	auto instruction = static_cast<llvm::LoadInst*>(i);

	/* FIXME: volatile and alignment */
	auto memstate = ctx.memory_state();
	auto value = ctx.module().create_variable(*convert_type(i->getType(), ctx));
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	tacs.push_back(load_op::create(address, instruction->getAlignment(), value, memstate));

	return tacs.back()->result(0);
}

static inline const variable *
convert_store_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Store);
	auto instruction = static_cast<llvm::StoreInst*>(i);

	/* FIXME: volatile and alignement */
	auto memstate = ctx.memory_state();
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	auto value = convert_value(instruction->getValueOperand(), tacs, ctx);
	tacs.push_back(store_op::create(address, value, instruction->getAlignment(), memstate));

	return tacs.back()->result(0);
}

static inline const variable *
convert_phi_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::PHI);

	auto result = ctx.module().create_tacvariable(*convert_type(i->getType(), ctx));

	tacs.push_back(phi_op::create({}, result));
	result->set_tac(tacs.back().get());

	return tacs.back()->result(0);
}

static inline const variable *
convert_getelementptr_instruction(llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::GetElementPtrInst>(inst));
	auto i = llvm::cast<llvm::GetElementPtrInst>(inst);
	auto & m = ctx.module();

	std::vector<const variable*> indices;
	auto base = convert_value(i->getPointerOperand(), tacs, ctx);
	for (auto it = i->idx_begin(); it != i->idx_end(); it++)
		indices.push_back(convert_value(*it, tacs, ctx));

	auto result = m.create_variable(*convert_type(i->getType(), ctx));

	tacs.push_back(getelementptr_op::create(base, indices, result));
	return tacs.back()->result(0);
}

static const llvm::FunctionType *
function_type(const llvm::CallInst * i)
{
	auto f = i->getCalledValue();
	JLM_DEBUG_ASSERT(f->getType()->isPointerTy());
	JLM_DEBUG_ASSERT(f->getType()->getContainedType(0)->isFunctionTy());
	return llvm::cast<const llvm::FunctionType>(f->getType()->getContainedType(0));
}

static const variable *
convert_malloc_call(const llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));
	auto state = ctx.module().create_variable(jive::memtype::instance());

	auto memstate = ctx.memory_state();
	auto size = convert_value(i->getArgOperand(0), tacs, ctx);

	auto malloc = malloc_op::create(size, state, result);
	auto memmerge = memstatemux_op::create_merge({state, memstate}, memstate);

	tacs.push_back(std::move(malloc));
	tacs.push_back(std::move(memmerge));
	return result;
}

static const variable *
convert_call_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Call);
	auto i = llvm::cast<llvm::CallInst>(instruction);

	auto create_arguments = [](
		const llvm::CallInst * i,
		tacsvector_t & tacs,
		context & ctx)
	{
		auto ftype = function_type(i);
		std::vector<const jlm::variable*> arguments;
		for (size_t n = 0; n < ftype->getNumParams(); n++)
			arguments.push_back(convert_value(i->getArgOperand(n), tacs, ctx));

		return arguments;
	};

	auto create_varargs = [](
		const llvm::CallInst * i,
		tacsvector_t & tacs,
		context & ctx)
	{
		auto ftype = function_type(i);
		std::vector<const jlm::variable*> varargs;
		for (size_t n = ftype->getNumParams(); n < i->getNumOperands()-1; n++)
			varargs.push_back(convert_value(i->getArgOperand(n), tacs, ctx));

		tacs.push_back(create_valist_tac(varargs, ctx.module()));
		return tacs.back()->result(0);
	};

	auto create_results = [](
		const llvm::CallInst * i,
		context & ctx)
	{
		auto ftype = function_type(i);
		std::vector<const jlm::variable*> results;
		if (!ftype->getReturnType()->isVoidTy()) {
			auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));
			results.push_back(result);
		}
		results.push_back(ctx.iostate());
		results.push_back(ctx.memory_state());
		results.push_back(ctx.loop_state());

		return results;
	};

	auto is_malloc_call = [](const llvm::CallInst * i)
	{
		auto f = i->getCalledFunction();
		return f && f->getName() == "malloc";
	};

	if (is_malloc_call(i))
		return convert_malloc_call(i, tacs, ctx);


	auto ftype = function_type(i);

	auto arguments = create_arguments(i, tacs, ctx);
	if (ftype->isVarArg())
		arguments.push_back(create_varargs(i, tacs, ctx));
	arguments.push_back(ctx.iostate());
	arguments.push_back(ctx.memory_state());
	arguments.push_back(ctx.loop_state());

	auto results = create_results(i, ctx);


	auto fctvar = convert_value(i->getCalledValue(), tacs, ctx);
	tacs.push_back(call_op::create(fctvar, arguments, results));
	return tacs.back()->result(0);
}

static inline const variable *
convert_select_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Select);
	auto instruction = static_cast<llvm::SelectInst*>(i);

	auto r = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto p = convert_value(instruction->getCondition(), tacs, ctx);
	auto t = convert_value(instruction->getTrueValue(), tacs, ctx);
	auto f = convert_value(instruction->getFalseValue(), tacs, ctx);

	if (i->getType()->isVectorTy())
		tacs.push_back(vectorselect_op::create(p, t, f, r));
	else
		tacs.push_back(select_op::create(p, t, f, r));

	return tacs.back()->result(0);
}

static inline const variable *
convert_binary_operator(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::BinaryOperator>(instruction));
	auto i = llvm::cast<const llvm::BinaryOperator>(instruction);

	static std::unordered_map<
		const llvm::Instruction::BinaryOps,
		std::unique_ptr<jive::operation>(*)(size_t)> bitmap({
			{llvm::Instruction::Add,	[](size_t nbits){jive::bitadd_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::And,	[](size_t nbits){jive::bitand_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::AShr,	[](size_t nbits){jive::bitashr_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Sub,	[](size_t nbits){jive::bitsub_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::UDiv,	[](size_t nbits){jive::bitudiv_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::SDiv,	[](size_t nbits){jive::bitsdiv_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::URem,	[](size_t nbits){jive::bitumod_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::SRem,	[](size_t nbits){jive::bitsmod_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Shl,	[](size_t nbits){jive::bitshl_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::LShr,	[](size_t nbits){jive::bitshr_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Or,		[](size_t nbits){jive::bitor_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Xor,	[](size_t nbits){jive::bitxor_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Mul,	[](size_t nbits){jive::bitmul_op o(nbits); return o.copy();}}
	});

	static std::unordered_map<const llvm::Instruction::BinaryOps, jlm::fpop> fpmap({
	  {llvm::Instruction::FAdd, fpop::add}, {llvm::Instruction::FSub, fpop::sub}
	, {llvm::Instruction::FMul, fpop::mul}, {llvm::Instruction::FDiv, fpop::div}
	, {llvm::Instruction::FRem, fpop::mod}
	});

	static std::unordered_map<const llvm::Type::TypeID, jlm::fpsize> fpsizemap({
	  {llvm::Type::HalfTyID, fpsize::half}
	, {llvm::Type::FloatTyID, fpsize::flt}
	, {llvm::Type::DoubleTyID, fpsize::dbl}
	, {llvm::Type::X86_FP80TyID, fpsize::x86fp80}
	});

	std::unique_ptr<jive::operation> operation;
	auto t = i->getType()->isVectorTy() ? i->getType()->getVectorElementType() : i->getType();
	if (t->isIntegerTy()) {
		JLM_DEBUG_ASSERT(bitmap.find(i->getOpcode()) != bitmap.end());
		operation = bitmap[i->getOpcode()](t->getIntegerBitWidth());
	} else if (t->isFloatingPointTy()) {
		JLM_DEBUG_ASSERT(fpmap.find(i->getOpcode()) != fpmap.end());
		JLM_DEBUG_ASSERT(fpsizemap.find(t->getTypeID()) != fpsizemap.end());
		operation = std::make_unique<fpbin_op>(fpmap[i->getOpcode()], fpsizemap[t->getTypeID()]);
	} else
		JLM_ASSERT(0);

	auto r = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);
	JLM_DEBUG_ASSERT(is<jive::binary_op>(*operation));

	if (i->getType()->isVectorTy()) {
		auto & binop = *static_cast<jive::binary_op*>(operation.get());
		tacs.push_back(vectorbinary_op::create(binop, op1, op2, r));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(operation.get()), {op1, op2}, {r}));
	}

	return tacs.back()->result(0);
}

static inline const variable *
convert_alloca_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Alloca);
	auto i = static_cast<llvm::AllocaInst*>(instruction);

	auto msvar = ctx.module().create_variable(jive::memtype::instance());
	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto memstate = ctx.memory_state();
	auto size = convert_value(i->getArraySize(), tacs, ctx);
	auto vtype = convert_type(i->getAllocatedType(), ctx);

	auto alloca = alloca_op::create(*vtype, size, i->getAlignment(), msvar, result);
	auto memmerge = memstatemux_op::create_merge({alloca->result(1), memstate}, memstate);

	tacs.push_back(std::move(alloca));
	tacs.push_back(std::move(memmerge));
	return result;
}

static inline const variable *
convert_insertvalue_instruction(llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
	/* FIXME: add support */
	JLM_ASSERT(0);
}

static inline const variable *
convert_extractvalue(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::ExtractValue);
	auto ev = llvm::dyn_cast<llvm::ExtractValueInst>(i);

	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto aggregate = convert_value(ev->getOperand(0), tacs, ctx);
	tacs.push_back(extractvalue_op::create(aggregate, ev->getIndices(), result));
	return tacs.back()->result(0);
}

static inline const variable *
convert_extractelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::ExtractElement);

	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto vector = convert_value(i->getOperand(0), tacs, ctx);
	auto index = convert_value(i->getOperand(1), tacs, ctx);
	tacs.push_back(extractelement_op::create(vector, index, result));
	return tacs.back()->result(0);
}

static inline const variable *
convert_shufflevector_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::ShuffleVector);

	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto v1 = convert_value(i->getOperand(0), tacs, ctx);
	auto v2 = convert_value(i->getOperand(1), tacs, ctx);
	auto mask = convert_value(i->getOperand(2), tacs, ctx);
	tacs.push_back(shufflevector_op::create(v1, v2, mask, result));
	return tacs.back()->result(0);
}

static const variable *
convert_insertelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::InsertElement);

	auto result = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto vector = convert_value(i->getOperand(0), tacs, ctx);
	auto value = convert_value(i->getOperand(1), tacs, ctx);
	auto index = convert_value(i->getOperand(2), tacs, ctx);
	tacs.push_back(insertelement_op::create(vector, value, index, result));
	return tacs.back()->result(0);
}

template<class OP> static std::unique_ptr<jive::operation>
create_unop(std::unique_ptr<jive::type> st, std::unique_ptr<jive::type> dt)
{
	return std::unique_ptr<jive::operation>(new OP(std::move(st), std::move(dt)));
}

static const variable *
convert_cast_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<llvm::CastInst>(i));
	auto st = i->getOperand(0)->getType();
	auto dt = i->getType();

	static std::unordered_map<
		unsigned,
		std::unique_ptr<jive::operation>(*)(std::unique_ptr<jive::type>, std::unique_ptr<jive::type>)
	> map({
	  {llvm::Instruction::Trunc, create_unop<trunc_op>}
	, {llvm::Instruction::ZExt, create_unop<zext_op>}
	, {llvm::Instruction::UIToFP, create_unop<uitofp_op>}
	, {llvm::Instruction::SIToFP, create_unop<sitofp_op>}
	, {llvm::Instruction::SExt, create_unop<sext_op>}
	, {llvm::Instruction::PtrToInt, create_unop<ptr2bits_op>}
	, {llvm::Instruction::IntToPtr, create_unop<bits2ptr_op>}
	, {llvm::Instruction::FPTrunc, create_unop<fptrunc_op>}
	, {llvm::Instruction::FPToUI, create_unop<fp2ui_op>}
	, {llvm::Instruction::FPToSI, create_unop<fp2si_op>}
	, {llvm::Instruction::FPExt, create_unop<fpext_op>}
	, {llvm::Instruction::BitCast, create_unop<bitcast_op>}
	});

	auto r = ctx.module().create_variable(*convert_type(i->getType(), ctx));

	auto op = convert_value(i->getOperand(0), tacs, ctx);
	auto srctype = convert_type(st->isVectorTy() ? st->getVectorElementType() : st, ctx);
	auto dsttype = convert_type(dt->isVectorTy() ? dt->getVectorElementType() : dt, ctx);

	JLM_DEBUG_ASSERT(map.find(i->getOpcode()) != map.end());
	auto unop = map[i->getOpcode()](std::move(srctype), std::move(dsttype));
	JLM_DEBUG_ASSERT(is<jive::unary_op>(*unop));

	if (dt->isVectorTy())
		tacs.push_back(vectorunary_op::create(*static_cast<jive::unary_op*>(unop.get()), op, r));
	else
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(unop.get()), {op}, {r}));

	return tacs.back()->result(0);
}

const variable *
convert_instruction(
	llvm::Instruction * i,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	if (i->isCast())
		return convert_cast_instruction(i, tacs, ctx);

	static std::unordered_map<
		unsigned,
		const variable*(*)(llvm::Instruction*, std::vector<std::unique_ptr<jlm::tac>>&, context&)
	> map({
		{llvm::Instruction::Ret, convert_return_instruction}
	,	{llvm::Instruction::Br, convert_branch_instruction}
	,	{llvm::Instruction::Switch, convert_switch_instruction}
	,	{llvm::Instruction::Unreachable, convert_unreachable_instruction}
	,	{llvm::Instruction::Add, convert_binary_operator}
	,	{llvm::Instruction::And, convert_binary_operator}
	,	{llvm::Instruction::AShr, convert_binary_operator}
	,	{llvm::Instruction::Sub, convert_binary_operator}
	,	{llvm::Instruction::UDiv, convert_binary_operator}
	,	{llvm::Instruction::SDiv, convert_binary_operator}
	,	{llvm::Instruction::URem, convert_binary_operator}
	,	{llvm::Instruction::SRem, convert_binary_operator}
	,	{llvm::Instruction::Shl, convert_binary_operator}
	,	{llvm::Instruction::LShr, convert_binary_operator}
	,	{llvm::Instruction::Or, convert_binary_operator}
	,	{llvm::Instruction::Xor, convert_binary_operator}
	,	{llvm::Instruction::Mul, convert_binary_operator}
	,	{llvm::Instruction::FAdd, convert_binary_operator}
	,	{llvm::Instruction::FSub, convert_binary_operator}
	,	{llvm::Instruction::FMul, convert_binary_operator}
	,	{llvm::Instruction::FDiv, convert_binary_operator}
	,	{llvm::Instruction::FRem, convert_binary_operator}
	,	{llvm::Instruction::ICmp, convert_icmp_instruction}
	,	{llvm::Instruction::FCmp, convert_fcmp_instruction}
	,	{llvm::Instruction::Load, convert_load_instruction}
	,	{llvm::Instruction::Store, convert_store_instruction}
	,	{llvm::Instruction::PHI, convert_phi_instruction}
	,	{llvm::Instruction::GetElementPtr, convert_getelementptr_instruction}
	,	{llvm::Instruction::Call, convert_call_instruction}
	,	{llvm::Instruction::Select, convert_select_instruction}
	,	{llvm::Instruction::Alloca, convert_alloca_instruction}
	,	{llvm::Instruction::InsertValue, convert_insertvalue_instruction}
	,	{llvm::Instruction::ExtractValue, convert_extractvalue}
	,	{llvm::Instruction::ExtractElement, convert_extractelement_instruction}
	,	{llvm::Instruction::ShuffleVector, convert_shufflevector_instruction}
	,	{llvm::Instruction::InsertElement, convert_insertelement_instruction}
	});

	JLM_DEBUG_ASSERT(map.find(i->getOpcode()) != map.end());
	return map[i->getOpcode()](i, tacs, ctx);
}

}
