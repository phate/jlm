/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/jlm.hpp>
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
#include <jive/vsdg/controltype.h>
#include <jive/vsdg/operators/match.h>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>

#include <typeindex>

namespace jlm {

const variable *
convert_value(
	const llvm::Value * v,
	const context & ctx)
{
	if (auto c = dynamic_cast<const llvm::Constant*>(v))
		return convert_constant(c, ctx);

	return ctx.lookup_value(v);
}

/* instructions */

static const variable *
convert_return_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ReturnInst*>(i));
	const llvm::ReturnInst * instruction = static_cast<const llvm::ReturnInst*>(i);

	if (instruction->getReturnValue()) {
		const variable * value = convert_value(instruction->getReturnValue(), ctx);
		bb->append(jlm::assignment_op(ctx.result()->type()), {value}, {ctx.result()});
	}

	return nullptr;
}

static const variable *
convert_branch_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BranchInst*>(i));
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(i);

	if (instruction->isConditional()) {
		const variable * c = convert_value(instruction->getCondition(), ctx);
		size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
		bb->append(jive::match_op(nbits, {{0, 0}}, 1, 2), {c});
	}

	return nullptr;
}

static const variable *
convert_switch_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SwitchInst*>(i));
	const llvm::SwitchInst * instruction = static_cast<const llvm::SwitchInst*>(i);

	JLM_DEBUG_ASSERT(bb->outedges().size() == instruction->getNumCases()+1);
	std::map<uint64_t, uint64_t> mapping;
	for (auto it = instruction->case_begin(); it != instruction->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != instruction->case_default());
		mapping[it.getCaseValue()->getZExtValue()] = it.getCaseIndex();
	}

	const jlm::variable * c = convert_value(instruction->getCondition(), ctx);
	size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
	bb->append(jive::match_op(nbits, mapping, mapping.size(), mapping.size()+1), {c});
	return nullptr;
}

static const variable *
convert_unreachable_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	/* Nothing needs to be done. */
	return nullptr;
}

static const variable *
convert_icmp_instruction(
	const llvm::Instruction * instruction,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ICmpInst*>(instruction));
	const llvm::ICmpInst * i = static_cast<const llvm::ICmpInst*>(instruction);

	/* FIXME: this unconditionally casts to integer type, take also care of other types */

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

	size_t nbits = i->getOperand(0)->getType()->getIntegerBitWidth();
	const jlm::variable * op1 = convert_value(i->getOperand(0), ctx);
	const jlm::variable * op2 = convert_value(i->getOperand(1), ctx);
	return bb->append(*map[i->getPredicate()](nbits), {op1, op2}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_fcmp_instruction(
	const llvm::Instruction * instruction,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FCmpInst*>(instruction));
	const llvm::FCmpInst * i = static_cast<const llvm::FCmpInst*>(instruction);

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

	return bb->append(*map[i->getPredicate()](), operands, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_load_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::LoadInst*>(i));
	const llvm::LoadInst * instruction = static_cast<const llvm::LoadInst*>(i);

	/* FIXME: handle volatile correctly */

	const variable * value = ctx.lookup_value(i);
	const variable * address = convert_value(instruction->getPointerOperand(), ctx);

	std::vector<std::unique_ptr<jive::state::type>> type;
	type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op op(jive::addr::type(), type, dynamic_cast<const jive::value::type&>(value->type()));
	return bb->append(op, {address, ctx.state()}, {value})->output(0);
}

static const variable *
convert_store_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::StoreInst*>(i));
	const llvm::StoreInst * instruction = static_cast<const llvm::StoreInst*>(i);

	const variable * address = convert_value(instruction->getPointerOperand(), ctx);
	const variable * value = convert_value(instruction->getValueOperand(), ctx);

	std::vector<std::unique_ptr<jive::state::type>> t;
	t.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op op(jive::addr::type(), t, dynamic_cast<const jive::value::type&>(value->type()));
	bb->append(op, {address, value, ctx.state()}, {ctx.state()});
	return nullptr;
}

static const variable *
convert_phi_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::PHINode*>(i));
	const llvm::PHINode * phi = static_cast<const llvm::PHINode*>(i);

	std::vector<const jlm::variable*> operands;
	for (auto edge : bb->inedges()) {
		const basic_block * tmp = static_cast<basic_block*>(edge->source());
		const llvm::BasicBlock * ib = ctx.lookup_basic_block(tmp);
		const jlm::variable * v = convert_value(phi->getIncomingValueForBlock(ib), ctx);
		operands.push_back(v);
	}

	JLM_DEBUG_ASSERT(operands.size() != 0);
	phi_op op(operands.size(), operands[0]->type());
	return bb->append(op, operands, {ctx.lookup_value(phi)})->output(0);
}

static const variable *
convert_getelementptr_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::GetElementPtrInst*>(i));
	const llvm::GetElementPtrInst * instruction = static_cast<const llvm::GetElementPtrInst*>(i);

	const jlm::variable * base = convert_value(instruction->getPointerOperand(), ctx);
	for (auto idx = instruction->idx_begin(); idx != instruction->idx_end(); idx++) {
		const jlm::variable * offset = convert_value(idx->get(), ctx);
		const jive::value::type & basetype = dynamic_cast<const jive::value::type&>(base->type());
		const jive::bits::type & offsettype = dynamic_cast<const jive::bits::type&>(offset->type());
		jive::address::arraysubscript_op op(basetype, offsettype);
		base = bb->append(op, {base, offset}, {ctx.lookup_value(i)})->output(0);
	}

	return base;
}

static const variable *
convert_trunc_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::TruncInst*>(i));
	const llvm::TruncInst * instruction = static_cast<const llvm::TruncInst*>(i);

	size_t high = i->getType()->getIntegerBitWidth();
	const jlm::variable * operand = convert_value(instruction->getOperand(0), ctx);
	jive::bits::slice_op op(dynamic_cast<const jive::bits::type&>(operand->type()), 0, high);
	return bb->append(op, {operand}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_call_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::CallInst*>(i));
	const llvm::CallInst * instruction = static_cast<const llvm::CallInst*>(i);

	llvm::Function * f = instruction->getCalledFunction();

	jlm::clg_node * caller = bb->cfg()->function();
	jlm::clg_node * callee = caller->clg().lookup_function(f->getName());
	JLM_DEBUG_ASSERT(callee != nullptr);
	caller->add_call(callee);

	std::vector<const jlm::variable*> arguments;
	for (size_t n = 0; n < instruction->getNumArgOperands(); n++)
		arguments.push_back(convert_value(instruction->getArgOperand(n), ctx));
	arguments.push_back(ctx.state());

	jive::fct::type type = dynamic_cast<jive::fct::type&>(*convert_type(f->getFunctionType()));

	std::vector<const jlm::variable*> results;
	if (type.nreturns() == 2)
		results.push_back(ctx.lookup_value(i));
	results.push_back(ctx.state());

	bb->append(jlm::apply_op(callee), arguments, results);
	return (results.size() == 2) ? results[0] : nullptr;
}

static const variable *
convert_select_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SelectInst*>(i));
	const llvm::SelectInst * instruction = static_cast<const llvm::SelectInst*>(i);

	const jlm::variable * condition = convert_value(instruction->getCondition(), ctx);
	const jlm::variable * tv = convert_value(instruction->getTrueValue(), ctx);
	const jlm::variable * fv = convert_value(instruction->getFalseValue(), ctx);
	return bb->append(select_op(tv->type()), {condition, tv, fv}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_binary_operator(
	const llvm::Instruction * instruction,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BinaryOperator*>(instruction));
	const llvm::BinaryOperator * i = static_cast<const llvm::BinaryOperator*>(instruction);

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
		return bb->append(*map[i->getOpcode()](nbits), {op1, op2}, {ctx.lookup_value(i)})->output(0);
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
		return bb->append(*map[i->getOpcode()](), {op1, op2}, {ctx.lookup_value(i)})->output(0);
	}

	JLM_DEBUG_ASSERT(0);
}

static const variable *
convert_alloca_instruction(
	const llvm::Instruction * instruction,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::AllocaInst*>(instruction));
	const llvm::AllocaInst * i = static_cast<const llvm::AllocaInst*>(instruction);

	/* FIXME: the number of bytes is not correct */

	size_t nbytes = 4;
	alloca_op op(nbytes);
	return bb->append(op, {ctx.state()}, {ctx.lookup_value(i), ctx.state()})->output(0);
}

static const variable *
convert_zext_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ZExtInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::constant_op c_op(jive::bits::value_repr(new_length - old_length, 0));
	const variable * c = bb->append(c_op, {})->output(0);

	jive::bits::concat_op op({jive::bits::type(new_length-old_length), jive::bits::type(old_length)});
	return bb->append(op, {convert_value(operand, ctx), c}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_sext_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SExtInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::slice_op s_op(jive::bits::type(old_length), old_length-1, old_length);
	const variable * bit = bb->append(s_op, {convert_value(operand, ctx)})->output(0);

	std::vector<const variable*> operands(1, convert_value(operand, ctx));
	std::vector<jive::bits::type> operand_types(1, jive::bits::type(old_length));
	for (size_t n = 0; n < new_length - old_length; n++) {
		operand_types.push_back(jive::bits::type(1));
		operands.push_back(bit);
	}

	jive::bits::concat_op op(operand_types);
	return bb->append(op, operands, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_fpext_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPExtInst*>(i));

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	jive::flt::type type;
	assignment_op op(type);
	const variable * operand = convert_value(i->getOperand(0), ctx);
	return bb->append(op, {operand}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_fptrunc_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPTruncInst*>(i));

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	jive::flt::type type;
	assignment_op op(type);
	const variable * operand = convert_value(i->getOperand(0), ctx);
	return bb->append(op, {operand}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_inttoptr_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::IntToPtrInst*>(i));

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	jive::bitstring_to_address_operation op(type->getIntegerBitWidth(),
		std::unique_ptr<jive::base::type>(new jive::addr::type()));
	return bb->append(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_ptrtoint_instruction(
	const llvm::Instruction * instruction,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::PtrToIntInst*>(instruction));
	const llvm::PtrToIntInst * i = static_cast<const llvm::PtrToIntInst*>(instruction);

	/* FIXME: support vector type */
	if (i->getPointerOperand()->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	const variable * operand = convert_value(i->getPointerOperand(), ctx);

	size_t nbits = i->getType()->getIntegerBitWidth();
	jive::address_to_bitstring_operation op(nbits, jive::addr::type());
	return bb->append(op, {operand}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_uitofp_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UIToFPInst*>(i));

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	bits2flt_op op(type->getIntegerBitWidth());
	return bb->append(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_sitofp_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SIToFPInst*>(i));

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	bits2flt_op op(type->getIntegerBitWidth());
	return bb->append(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_fptoui_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToUIInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	flt2bits_op op(i->getType()->getIntegerBitWidth());
	return bb->append(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_fptosi_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToSIInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	flt2bits_op op(i->getType()->getIntegerBitWidth());
	return bb->append(op, {convert_value(operand, ctx)}, {ctx.lookup_value(i)})->output(0);
}

static const variable *
convert_bitcast_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BitCastInst*>(i));

	const variable * operand = convert_value(i->getOperand(0), ctx);
	const variable * result = ctx.lookup_value(i);

	/* FIXME: invoke with the right number of bytes */
	alloca_op aop(4);
	const tac * alloc = bb->append(aop, {ctx.state()});
	const variable * address = alloc->output(0);
	const variable * state = alloc->output(1);

	std::vector<std::unique_ptr<jive::state::type>> t;
	t.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op sop(jive::addr::type(), t,
		dynamic_cast<const jive::value::type&>(operand->type()));
	bb->append(sop, {address, operand, state}, {state});

	std::vector<std::unique_ptr<jive::state::type>> type;
	type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op lop(jive::addr::type(), type,
		dynamic_cast<const jive::value::type&>(result->type()));
	return bb->append(lop, {address, state}, {result})->output(0);
}

const variable *
convert_instruction(
	const llvm::Instruction * i,
	basic_block * bb,
	const context & ctx)
{
	static std::unordered_map<
		std::type_index,
		const variable * (*)(const llvm::Instruction*, jlm::basic_block*, const context&)
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
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(*i))) != map.end());
	return map[std::type_index(typeid(*i))](i, bb, ctx);
}

}
