/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/type.hpp>

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/address-transform.h>
#include <jive/arch/load.h>
#include <jive/arch/memorytype.h>
#include <jive/arch/store.h>
#include <jive/types/bitstring.h>
#include <jive/types/float.h>
#include <jive/types/record.h>
#include <jive/vsdg/controltype.h>
#include <jive/vsdg/operators/match.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>

#include <typeindex>

namespace jlm {

const variable *
convert_value(
	const llvm::Value * v,
	context & ctx)
{
	if (ctx.has_value(v))
		return ctx.lookup_value(v);

	if (auto c = dynamic_cast<const llvm::Constant*>(v)) {
		auto attr = static_cast<basic_block*>(&ctx.entry_block()->attribute());
		return attr->append(ctx.cfg(), *convert_constant(c, ctx));
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

/* instructions */

static const variable *
convert_return_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ReturnInst*>(i));
	const llvm::ReturnInst * instruction = static_cast<const llvm::ReturnInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	if (instruction->getReturnValue()) {
		const variable * value = convert_value(instruction->getReturnValue(), ctx);
		attr->append(create_assignment(ctx.result()->type(), value, ctx.result()));
	}

	return nullptr;
}

static const variable *
convert_branch_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BranchInst*>(i));
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	if (instruction->isConditional()) {
		const variable * c = convert_value(instruction->getCondition(), ctx);
		size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
		auto op = jive::match_op(nbits, {{0, 0}}, 1, 2);
		attr->append(create_tac(op, {c}, create_variables(*ctx.cfg(), op)));
	}

	return nullptr;
}

static const variable *
convert_switch_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SwitchInst*>(i));
	const llvm::SwitchInst * instruction = static_cast<const llvm::SwitchInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	std::map<uint64_t, uint64_t> mapping;
	for (auto it = instruction->case_begin(); it != instruction->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != instruction->case_default());
		mapping[it.getCaseValue()->getZExtValue()] = it.getCaseIndex();
	}

	const jlm::variable * c = convert_value(instruction->getCondition(), ctx);
	size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
	auto op = jive::match_op(nbits, mapping, mapping.size(), mapping.size()+1);
	attr->append(create_tac(op, {c}, create_variables(*ctx.cfg(), op)));
	return nullptr;
}

static const variable *
convert_unreachable_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	/* Nothing needs to be done. */
	return nullptr;
}

static const variable *
convert_icmp_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ICmpInst*>(instruction));
	const llvm::ICmpInst * i = static_cast<const llvm::ICmpInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const jlm::variable * op1 = convert_value(i->getOperand(0), ctx);
	const jlm::variable * op2 = convert_value(i->getOperand(1), ctx);

	/* FIXME: */
	if (i->getOperand(0)->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	static std::map<
		const llvm::CmpInst::Predicate,
		std::unique_ptr<jive::operation>(*)(size_t)> map({
			{llvm::CmpInst::ICMP_SLT,	[](size_t nbits){jive::bits::slt_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_ULT,	[](size_t nbits){jive::bits::ult_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_SLE,	[](size_t nbits){jive::bits::sle_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_ULE,	[](size_t nbits){jive::bits::ule_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_EQ,	[](size_t nbits){jive::bits::eq_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_NE,	[](size_t nbits){jive::bits::ne_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_SGE,	[](size_t nbits){jive::bits::sge_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_UGE,	[](size_t nbits){jive::bits::uge_op op(nbits); return op.copy();}}
		,	{llvm::CmpInst::ICMP_SGT,	[](size_t nbits){jive::bits::sgt_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_UGT,	[](size_t nbits){jive::bits::ugt_op op(nbits); return op.copy();}}
	});

	/* FIXME: we don't have any comparison operators for address type yet */
	size_t nbits;
	if (i->getOperand(0)->getType()->isPointerTy()) {
		nbits = 32;
		jive::address_to_bitstring_operation op(nbits,
			std::unique_ptr<jive::base::type>(new jive::addr::type()));
		const variable * new_op1 = ctx.cfg()->create_variable(jive::bits::type(nbits));
		const variable * new_op2 = ctx.cfg()->create_variable(jive::bits::type(nbits));
		op1 = attr->append(create_tac(op, {op1}, {new_op1}))->output(0);
		op2 = attr->append(create_tac(op, {op2}, {new_op2}))->output(0);
	} else
		nbits = i->getOperand(0)->getType()->getIntegerBitWidth();

	auto tac = create_tac(*map[i->getPredicate()](nbits), {op1, op2}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_fcmp_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FCmpInst*>(instruction));
	const llvm::FCmpInst * i = static_cast<const llvm::FCmpInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: vector type is not yet supported */
	if (i->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	/* FIXME: we currently don't have an operation for FCMP_ORD and FCMP_UNO, just use flt::eq_op */

	static std::map<
		const llvm::CmpInst::Predicate,
		std::unique_ptr<jive::operation>(*)()> map({
			{llvm::CmpInst::FCMP_OEQ, [](){jive::flt::eq_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_OGT, [](){jive::flt::gt_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_OGE, [](){jive::flt::ge_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_OLT, [](){jive::flt::lt_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_OLE, [](){jive::flt::le_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_ONE, [](){jive::flt::ne_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_ORD, [](){jive::flt::eq_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_UNO, [](){jive::flt::eq_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_UEQ, [](){jive::flt::eq_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_UGT, [](){jive::flt::gt_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_UGE, [](){jive::flt::ge_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_ULT, [](){jive::flt::lt_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_ULE, [](){jive::flt::le_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_UNE, [](){jive::flt::ne_op op; return op.copy();}}
		,	{llvm::CmpInst::FCMP_FALSE,
				[](){jive::bits::constant_op op(jive::bits::value_repr(1, 0)); return op.copy();}}
		,	{llvm::CmpInst::FCMP_TRUE,
				[](){jive::bits::constant_op op(jive::bits::value_repr(1, 1)); return op.copy();}}
	});

	std::vector<const variable*> operands;
	if (i->getPredicate() != llvm::CmpInst::FCMP_TRUE
	&& i->getPredicate() != llvm::CmpInst::FCMP_FALSE) {
		operands.push_back(convert_value(i->getOperand(0), ctx));
		operands.push_back(convert_value(i->getOperand(1), ctx));
	}

	auto tac = create_tac(*map[i->getPredicate()](), operands, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_load_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::LoadInst*>(i));
	const llvm::LoadInst * instruction = static_cast<const llvm::LoadInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: handle volatile correctly */

	const variable * value = ctx.lookup_value(i);
	const variable * address = convert_value(instruction->getPointerOperand(), ctx);

	std::vector<std::unique_ptr<jive::state::type>> type;
	type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op op(jive::addr::type(), type, dynamic_cast<const jive::value::type&>(value->type()));
	return attr->append(create_tac(op, {address, ctx.state()}, {value}))->output(0);
}

static const variable *
convert_store_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::StoreInst*>(i));
	const llvm::StoreInst * instruction = static_cast<const llvm::StoreInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const variable * address = convert_value(instruction->getPointerOperand(), ctx);
	const variable * value = convert_value(instruction->getValueOperand(), ctx);

	std::vector<std::unique_ptr<jive::state::type>> t;
	t.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op op(jive::addr::type(), t, dynamic_cast<const jive::value::type&>(value->type()));
	attr->append(create_tac(op, {address, value, ctx.state()}, {ctx.state()}));
	return nullptr;
}

static const variable *
convert_phi_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::PHINode*>(i));
	const llvm::PHINode * phi = static_cast<const llvm::PHINode*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	std::vector<const jlm::variable*> operands;
	for (auto edge : bb->inedges()) {
		auto tmp = edge->source();
		const llvm::BasicBlock * ib = nullptr;
		if (ctx.has_basic_block(tmp)) {
			ib = ctx.lookup_basic_block(tmp);
		} else {
			JLM_DEBUG_ASSERT(tmp->ninedges() == 1);
			ib = ctx.lookup_basic_block(tmp->inedges().front()->source());
		}

		const jlm::variable * v = convert_value(phi->getIncomingValueForBlock(ib), ctx);
		operands.push_back(v);
	}

	JLM_DEBUG_ASSERT(operands.size() != 0);
	phi_op op(operands.size(), operands[0]->type());
	return attr->append(create_tac(op, operands, {ctx.lookup_value(phi)}))->output(0);
}

static const variable *
convert_getelementptr_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GetElementPtrInst*>(i));
	const llvm::GetElementPtrInst * instruction = static_cast<const llvm::GetElementPtrInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const jlm::variable * base = convert_value(instruction->getPointerOperand(), ctx);
	for (auto idx = instruction->idx_begin(); idx != instruction->idx_end(); idx++) {
		const jlm::variable * offset = convert_value(idx->get(), ctx);
		const jive::value::type & basetype = dynamic_cast<const jive::value::type&>(base->type());
		const jive::bits::type & offsettype = dynamic_cast<const jive::bits::type&>(offset->type());
		jive::address::arraysubscript_op op(basetype, offsettype);
		base = attr->append(create_tac(op, {base, offset}, {ctx.lookup_value(i)}))->output(0);
	}

	return base;
}

static const variable *
convert_trunc_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::TruncInst*>(i));
	const llvm::TruncInst * instruction = static_cast<const llvm::TruncInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	size_t high = i->getType()->getIntegerBitWidth();
	const jlm::variable * operand = convert_value(instruction->getOperand(0), ctx);
	jive::bits::slice_op op(dynamic_cast<const jive::bits::type&>(operand->type()), 0, high);
	return attr->append(create_tac(op, {operand}, {ctx.lookup_value(i)}))->output(0);
}

static const variable *
convert_call_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::CallInst*>(i));
	const llvm::CallInst * instruction = static_cast<const llvm::CallInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const llvm::Value * f = instruction->getCalledValue();
	const llvm::FunctionType * ftype = llvm::cast<const llvm::FunctionType>(
		llvm::cast<const llvm::PointerType>(f->getType())->getElementType());
	jive::fct::type type = dynamic_cast<jive::fct::type&>(*convert_type(ftype, ctx));

	const jlm::variable * callee = nullptr;
	jlm::clg_node * caller = nullptr;
	if (instruction->getCalledFunction()) {
		/* direct call */
		callee = convert_value(f, ctx);
		caller->add_call(static_cast<const jlm::clg_node*>(callee));
	} else {
		/* indirect call */
		/* FIXME */
		std::vector<std::unique_ptr<jive::state::type>> stype;
		stype.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
		jive::load_op op(jive::addr::type::instance(), stype, type);
		const jlm::variable * tmp = bb->cfg()->create_variable(type);
		callee = attr->append(create_tac(op, {convert_value(f, ctx), ctx.state()}, {tmp}))->output(0);
	}

	/* handle arguments */
	std::vector<const jlm::variable*> arguments;
	arguments.push_back(callee);
	for (size_t n = 0; n < ftype->getNumParams(); n++)
		arguments.push_back(convert_value(instruction->getArgOperand(n), ctx));
	if (ftype->isVarArg()) {
		/* FIXME: sizeof */
		const variable * alloc = bb->cfg()->create_variable(jive::addr::type::instance());
		auto vararg = attr->append(create_tac(alloca_op(4), {ctx.state()}, {alloc, ctx.state()}));
		arguments.push_back(vararg->output(0));
	}
	arguments.push_back(ctx.state());

	/* handle results */
	std::vector<const jlm::variable*> results;
	if (type.nresults() == 2)
		results.push_back(ctx.lookup_value(i));
	results.push_back(ctx.state());

	attr->append(create_tac(jive::fct::apply_op(type), arguments, results));
	return (results.size() == 2) ? results[0] : nullptr;
}

static const variable *
convert_select_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SelectInst*>(i));
	const llvm::SelectInst * instruction = static_cast<const llvm::SelectInst*>(i);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const jlm::variable * condition = convert_value(instruction->getCondition(), ctx);
	const jlm::variable * tv = convert_value(instruction->getTrueValue(), ctx);
	const jlm::variable * fv = convert_value(instruction->getFalseValue(), ctx);
	auto tac = create_tac(select_op(tv->type()), {condition, tv, fv}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_binary_operator(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BinaryOperator*>(instruction));
	const llvm::BinaryOperator * i = static_cast<const llvm::BinaryOperator*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: vector type is not yet supported */
	if (i->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	const jlm::variable * op1 = convert_value(i->getOperand(0), ctx);
	const jlm::variable * op2 = convert_value(i->getOperand(1), ctx);

	if (i->getType()->isIntegerTy()) {
		static std::map<
			const llvm::Instruction::BinaryOps,
			std::unique_ptr<jive::operation>(*)(size_t)> map({
				{llvm::Instruction::Add,	[](size_t nbits){jive::bits::add_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::And,	[](size_t nbits){jive::bits::and_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::AShr,	[](size_t nbits){jive::bits::ashr_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::Sub,	[](size_t nbits){jive::bits::sub_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::UDiv,	[](size_t nbits){jive::bits::udiv_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::SDiv,	[](size_t nbits){jive::bits::sdiv_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::URem,	[](size_t nbits){jive::bits::umod_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::SRem,	[](size_t nbits){jive::bits::smod_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::Shl,	[](size_t nbits){jive::bits::shl_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::LShr,	[](size_t nbits){jive::bits::shr_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::Or,		[](size_t nbits){jive::bits::or_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::Xor,	[](size_t nbits){jive::bits::xor_op o(nbits); return o.copy();}}
			,	{llvm::Instruction::Mul,	[](size_t nbits){jive::bits::mul_op o(nbits); return o.copy();}}
		});

		size_t nbits = i->getType()->getIntegerBitWidth();
		auto tac = create_tac(*map[i->getOpcode()](nbits), {op1, op2}, {ctx.lookup_value(i)});
		return attr->append(std::move(tac))->output(0);
	}

	if (i->getType()->isFloatingPointTy()) {
		static std::map<
			const llvm::Instruction::BinaryOps,
			std::unique_ptr<jive::operation>(*)()> map({
				{llvm::Instruction::FAdd, [](){jive::flt::add_op op; return op.copy();}}
			, {llvm::Instruction::FSub, [](){jive::flt::sub_op op; return op.copy();}}
			, {llvm::Instruction::FMul, [](){jive::flt::mul_op op; return op.copy();}}
			, {llvm::Instruction::FDiv, [](){jive::flt::div_op op; return op.copy();}}
		});

		/* FIXME: support FRem */
		JLM_DEBUG_ASSERT(i->getOpcode() != llvm::Instruction::FRem);
		auto tac = create_tac(*map[i->getOpcode()](), {op1, op2}, {ctx.lookup_value(i)});
		return attr->append(std::move(tac))->output(0);
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const variable *
convert_alloca_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::AllocaInst*>(instruction));
	const llvm::AllocaInst * i = static_cast<const llvm::AllocaInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: the number of bytes is not correct */

	size_t nbytes = 4;
	auto tac = create_tac(alloca_op(nbytes), {ctx.state()}, {ctx.lookup_value(i), ctx.state()});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_zext_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ZExtInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::constant_op c_op(jive::bits::value_repr(new_length - old_length, 0));
	auto c = attr->append(create_tac(c_op, {}, create_variables(*ctx.cfg(), c_op)))->output(0);

	jive::bits::concat_op op({jive::bits::type(old_length), jive::bits::type(new_length-old_length)});
	auto tac = create_tac(op, {convert_value(operand, ctx), c}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_sext_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SExtInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::slice_op s_op(jive::bits::type(old_length), old_length-1, old_length);
	auto bit = attr->append(create_tac(s_op, {convert_value(operand, ctx)},
		create_variables(*ctx.cfg(), s_op)))->output(0);

	std::vector<const variable*> operands(1, convert_value(operand, ctx));
	std::vector<jive::bits::type> operand_types(1, jive::bits::type(old_length));
	for (size_t n = 0; n < new_length - old_length; n++) {
		operand_types.push_back(jive::bits::type(1));
		operands.push_back(bit);
	}

	jive::bits::concat_op op(operand_types);
	return attr->append(create_tac(op, operands, {ctx.lookup_value(i)}))->output(0);
}

static const variable *
convert_fpext_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPExtInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	jive::flt::type type;
	assignment_op op(type);
	const variable * operand = convert_value(i->getOperand(0), ctx);
	return attr->append(create_tac(op, {operand}, {ctx.lookup_value(i)}))->output(0);
}

static const variable *
convert_fptrunc_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPTruncInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	jive::flt::type type;
	assignment_op op(type);
	const variable * operand = convert_value(i->getOperand(0), ctx);
	return attr->append(create_tac(op, {operand}, {ctx.lookup_value(i)}))->output(0);
}

static const variable *
convert_inttoptr_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::IntToPtrInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	jive::bitstring_to_address_operation op(type->getIntegerBitWidth(),
		std::unique_ptr<jive::base::type>(new jive::addr::type()));
	auto tac = create_tac(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_ptrtoint_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::PtrToIntInst*>(instruction));
	const llvm::PtrToIntInst * i = static_cast<const llvm::PtrToIntInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: support vector type */
	if (i->getPointerOperand()->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	const variable * operand = convert_value(i->getPointerOperand(), ctx);

	size_t nbits = i->getType()->getIntegerBitWidth();
	jive::address_to_bitstring_operation op(nbits, jive::addr::type());
	return attr->append(create_tac(op, {operand}, {ctx.lookup_value(i)}))->output(0);
}

static const variable *
convert_uitofp_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UIToFPInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	bits2flt_op op(type->getIntegerBitWidth());
	auto tac = create_tac(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_sitofp_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SIToFPInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	bits2flt_op op(type->getIntegerBitWidth());
	auto tac = create_tac(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_fptoui_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToUIInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	flt2bits_op op(i->getType()->getIntegerBitWidth());
	auto tac = create_tac(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_fptosi_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToSIInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	flt2bits_op op(i->getType()->getIntegerBitWidth());
	auto tac = create_tac(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)});
	return attr->append(std::move(tac))->output(0);
}

static const variable *
convert_bitcast_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BitCastInst*>(i));
	auto attr = static_cast<basic_block*>(&bb->attribute());

	const variable * operand = convert_value(i->getOperand(0), ctx);
	const variable * result = ctx.lookup_value(i);

	/* FIXME: invoke with the right number of bytes */
	alloca_op aop(4);
	const variable * address = bb->cfg()->create_variable(jive::addr::type::instance());
	attr->append(create_tac(aop, {ctx.state()}, {address, ctx.state()}));

	std::vector<std::unique_ptr<jive::state::type>> t;
	t.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op sop(jive::addr::type(), t,
		dynamic_cast<const jive::value::type&>(operand->type()));
	attr->append(create_tac(sop, {address, operand, ctx.state()}, {ctx.state()}));

	std::vector<std::unique_ptr<jive::state::type>> type;
	type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op lop(jive::addr::type(), type,
		dynamic_cast<const jive::value::type&>(result->type()));
	return attr->append(create_tac(lop, {address, ctx.state()}, {result}))->output(0);
}

static const variable *
convert_insertvalue_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::InsertValueInst*>(instruction));
	const llvm::InsertValueInst * i = static_cast<const llvm::InsertValueInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: array type */
	if (i->getType()->isArrayTy())
		JLM_DEBUG_ASSERT(0);

	std::function<const variable *(llvm::InsertValueInst::idx_iterator, const variable *)> f = [&] (
		const llvm::InsertValueInst::idx_iterator & idx,
		const variable * aggregate)
	{
		if (idx == i->idx_end())
			return convert_value(i->getInsertedValueOperand(), ctx);

		/* FIXME: array type */
		const jive::rcd::type * type = dynamic_cast<const jive::rcd::type*>(&aggregate->type());
		std::shared_ptr<const jive::rcd::declaration> decl = type->declaration();

		std::vector<const variable*> operands;
		for (size_t n = 0; n < decl->nelements(); n++) {
			auto op = jive::rcd::select_operation(*type, n);
			auto tac = attr->append(create_tac(op, {aggregate}, create_variables(*ctx.cfg(), op)));
			if (n == *idx)
				operands.push_back(f(std::next(idx), tac->output(0)));
			else
				operands.push_back(tac->output(0));
		}

		jive::rcd::group_op op(decl);
		return attr->append(create_tac(op, operands, {ctx.lookup_value(i)}))->output(0);
	};

	return f(i->idx_begin(), convert_value(i->getAggregateOperand(), ctx));
}

static const variable *
convert_extractvalue_instruction(
	const llvm::Instruction * instruction,
	cfg_node * bb,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ExtractValueInst*>(instruction));
	const llvm::ExtractValueInst * i = static_cast<const llvm::ExtractValueInst*>(instruction);
	auto attr = static_cast<basic_block*>(&bb->attribute());

	/* FIXME: array type */
	if (i->getType()->isArrayTy())
		JLM_DEBUG_ASSERT(0);

	const variable * aggregate = convert_value(i->getAggregateOperand(), ctx);

	for (auto it = i->idx_begin(); it != i->idx_end(); it++) {
		const jive::rcd::type * type = dynamic_cast<const jive::rcd::type*>(&aggregate->type());

		jive::rcd::select_operation op(*type, *it);
		aggregate = attr->append(create_tac(op, {aggregate}, {ctx.lookup_value(i)}))->output(0);
	}

	return aggregate;
}

const variable *
convert_instruction(
	const llvm::Instruction * i,
	cfg_node * bb,
	context & ctx)
{
	static std::unordered_map<
		std::type_index,
		const variable * (*)(const llvm::Instruction*, jlm::cfg_node*, context&)
	> map({
		{std::type_index(typeid(llvm::ReturnInst)), convert_return_instruction}
	,	{std::type_index(typeid(llvm::BranchInst)), convert_branch_instruction}
	,	{std::type_index(typeid(llvm::SwitchInst)), convert_switch_instruction}
	,	{std::type_index(typeid(llvm::UnreachableInst)), convert_unreachable_instruction}
	,	{std::type_index(typeid(llvm::BinaryOperator)), convert_binary_operator}
	,	{std::type_index(typeid(llvm::ICmpInst)), convert_icmp_instruction}
	,	{std::type_index(typeid(llvm::FCmpInst)), convert_fcmp_instruction}
	,	{std::type_index(typeid(llvm::LoadInst)), convert_load_instruction}
	,	{std::type_index(typeid(llvm::StoreInst)), convert_store_instruction}
	,	{std::type_index(typeid(llvm::PHINode)), convert_phi_instruction}
	,	{std::type_index(typeid(llvm::GetElementPtrInst)), convert_getelementptr_instruction}
	,	{std::type_index(typeid(llvm::TruncInst)), convert_trunc_instruction}
	,	{std::type_index(typeid(llvm::CallInst)), convert_call_instruction}
	,	{std::type_index(typeid(llvm::SelectInst)), convert_select_instruction}
	,	{std::type_index(typeid(llvm::AllocaInst)), convert_alloca_instruction}
	,	{std::type_index(typeid(llvm::ZExtInst)), convert_zext_instruction}
	,	{std::type_index(typeid(llvm::SExtInst)), convert_sext_instruction}
	,	{std::type_index(typeid(llvm::FPExtInst)), convert_fpext_instruction}
	,	{std::type_index(typeid(llvm::FPTruncInst)), convert_fptrunc_instruction}
	,	{std::type_index(typeid(llvm::IntToPtrInst)), convert_inttoptr_instruction}
	,	{std::type_index(typeid(llvm::PtrToIntInst)), convert_ptrtoint_instruction}
	,	{std::type_index(typeid(llvm::UIToFPInst)), convert_uitofp_instruction}
	,	{std::type_index(typeid(llvm::SIToFPInst)), convert_sitofp_instruction}
	,	{std::type_index(typeid(llvm::FPToUIInst)), convert_fptoui_instruction}
	,	{std::type_index(typeid(llvm::FPToSIInst)), convert_fptosi_instruction}
	,	{std::type_index(typeid(llvm::BitCastInst)), convert_bitcast_instruction}
	,	{std::type_index(typeid(llvm::InsertValueInst)), convert_insertvalue_instruction}
	,	{std::type_index(typeid(llvm::ExtractValueInst)), convert_extractvalue_instruction}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(*i))) != map.end());
	return map[std::type_index(typeid(*i))](i, bb, ctx);
}

}
