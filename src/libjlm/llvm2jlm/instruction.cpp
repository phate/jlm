/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/llvm2jlm/constant.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/instruction.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/ipgraph.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/address-transform.h>
#include <jive/arch/load.h>
#include <jive/arch/addresstype.h>
#include <jive/arch/store.h>
#include <jive/types/bitstring.h>
#include <jive/types/float.h>
#include <jive/types/record.h>
#include <jive/rvsdg/control.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>

#include <typeindex>

static inline std::vector<const jlm::variable*>
create_result_variables(jlm::module & m, const jive::simple_op & op)
{
	std::vector<const jlm::variable*> variables;
	for (size_t n = 0; n < op.nresults(); n++)
		variables.push_back(m.create_variable(op.result(n).type(), false));

	return variables;
}

static inline void
insert_before_branch(jlm::cfg_node * node, jlm::tacsvector_t & tacs)
{
	using namespace jlm;

	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());

	auto it = bb.rbegin();
	while (it != bb.rend()) {
		if (*it && !is<branch_op>((*it)->operation()))
			break;

		it = std::next(it);
	}

	bb.insert(it.base(), tacs);
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

/* instructions */

static inline const variable *
convert_return_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Ret);
	auto i = llvm::cast<llvm::ReturnInst>(instruction);

	auto bb = ctx.lookup_basic_block(i->getParent());
	bb->add_outedge(bb->cfg()->exit_node());
	if (!i->getReturnValue())
		return {};

	auto value = convert_value(i->getReturnValue(), tacs, ctx);
	tacs.push_back(create_assignment(ctx.result()->type(), value, ctx.result()));

	return ctx.result();
}

static inline const variable *
convert_branch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Br);
	auto i = llvm::cast<llvm::BranchInst>(instruction);
	auto bb = ctx.lookup_basic_block(i->getParent());

	if (i->isUnconditional()) {
		bb->add_outedge(ctx.lookup_basic_block(i->getSuccessor(0)));
		return {};
	}

	bb->add_outedge(ctx.lookup_basic_block(i->getSuccessor(1))); /* false */
	bb->add_outedge(ctx.lookup_basic_block(i->getSuccessor(0))); /* true */

	auto c = convert_value(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, {{1, 1}}, 0, 2);
	tacs.push_back(create_tac(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back(create_branch_tac(2,  tacs.back()->output(0)));

	return nullptr;
}

static inline const variable *
convert_switch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Switch);
	auto i = llvm::cast<llvm::SwitchInst>(instruction);
	auto bb = ctx.lookup_basic_block(i->getParent());

	size_t n = 0;
	std::unordered_map<uint64_t, uint64_t> mapping;
	for (auto it = i->case_begin(); it != i->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != i->case_default());
		mapping[it.getCaseValue()->getZExtValue()] = n++;
		bb->add_outedge(ctx.lookup_basic_block(it.getCaseSuccessor()));
	}

	bb->add_outedge(ctx.lookup_basic_block(i->case_default().getCaseSuccessor()));
	JLM_DEBUG_ASSERT(i->getNumSuccessors() == n+1);

	auto c = convert_value(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, mapping, n, n+1);
	tacs.push_back(create_tac(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back((create_branch_tac(n+1, tacs.back()->output(0))));

	return nullptr;
}

static inline const variable *
convert_unreachable_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Unreachable);
	auto bb = ctx.lookup_basic_block(i->getParent());
	bb->add_outedge(bb->cfg()->exit_node());
	return nullptr;
}

static inline const variable *
convert_icmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::ICmp);
	auto i = llvm::cast<const llvm::ICmpInst>(instruction);
	auto t = i->getOperand(0)->getType();
	JLM_DEBUG_ASSERT(!t->isVectorTy());

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

	if (t->isIntegerTy()) {
		size_t nbits = t->getIntegerBitWidth();
		/* FIXME: This is inefficient. We return a unique ptr and then take copy it. */
		auto op = map[p](nbits);
		tacs.push_back(create_tac(*static_cast<const jive::simple_op*>(op.get()),
			{op1, op2}, {ctx.lookup_value(i)}));
	} else {
		JLM_DEBUG_ASSERT(t->isPointerTy() && map.find(p) != map.end());
		tacs.push_back(create_ptrcmp_tac(ptrmap[p], op1, op2, ctx.lookup_value(i)));
	}

	return tacs.back()->output(0);
}

static inline const variable *
convert_fcmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::FCmp);
	auto i = llvm::cast<const llvm::FCmpInst>(instruction);
	JLM_DEBUG_ASSERT(!i->getOperand(0)->getType()->isVectorTy());

	if (i->getPredicate() == llvm::CmpInst::FCMP_TRUE) {
		jive::bitconstant_op op(jive::bitvalue_repr(1, 1));
		tacs.push_back(create_tac(op, {}, {ctx.lookup_value(i)}));
		return tacs.back()->output(0);
	}

	if (i->getPredicate() == llvm::CmpInst::FCMP_FALSE) {
		jive::bitconstant_op op(jive::bitvalue_repr(1, 0));
		tacs.push_back(create_tac(op, {}, {ctx.lookup_value(i)}));
		return tacs.back()->output(0);
	}

	static std::unordered_map<llvm::CmpInst::Predicate, jlm::fpcmp> map({
		{llvm::CmpInst::FCMP_OEQ, fpcmp::oeq},	{llvm::CmpInst::FCMP_OGT, fpcmp::ogt}
	,	{llvm::CmpInst::FCMP_OGE, fpcmp::oge},	{llvm::CmpInst::FCMP_OLT, fpcmp::olt}
	,	{llvm::CmpInst::FCMP_OLE, fpcmp::ole},	{llvm::CmpInst::FCMP_ONE, fpcmp::one}
	,	{llvm::CmpInst::FCMP_ORD, fpcmp::ord},	{llvm::CmpInst::FCMP_UNO, fpcmp::uno}
	,	{llvm::CmpInst::FCMP_UEQ, fpcmp::ueq},	{llvm::CmpInst::FCMP_UGT, fpcmp::ugt}
	,	{llvm::CmpInst::FCMP_UGE, fpcmp::uge},	{llvm::CmpInst::FCMP_ULT, fpcmp::ult}
	,	{llvm::CmpInst::FCMP_ULE, fpcmp::ule},	{llvm::CmpInst::FCMP_UNE, fpcmp::une}
	});

	JLM_DEBUG_ASSERT(map.find(i->getPredicate()) != map.end());
	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);
	tacs.push_back(create_fpcmp_tac(map[i->getPredicate()], op1, op2, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_load_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::LoadInst*>(i));
	auto instruction = static_cast<llvm::LoadInst*>(i);

	/* FIXME: volatile and alignment */

	auto value = ctx.lookup_value(i);
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	tacs.push_back(create_load_tac(address, ctx.state(), instruction->getAlignment(), value));

	return tacs.back()->output(0);
}

static inline const variable *
convert_store_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::StoreInst*>(i));
	auto instruction = static_cast<llvm::StoreInst*>(i);

	/* FIXME: volatile and alignement */
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	auto value = convert_value(instruction->getValueOperand(), tacs, ctx);
	tacs.push_back(create_store_tac(address, value, instruction->getAlignment(), ctx.state()));

	return nullptr;
}

static inline const variable *
convert_phi_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::PHI);
	auto i = llvm::cast<llvm::PHINode>(instruction);

	std::vector<std::pair<const variable*, cfg_node*>> arguments;
	for (auto it = i->block_begin(); it != i->block_end(); it++) {
		tacsvector_t tacs;
		auto bb = ctx.lookup_basic_block(*it);
		auto v = convert_value(i->getIncomingValueForBlock(*it), tacs, ctx);
		insert_before_branch(bb, tacs);
		arguments.push_back(std::make_pair(v, bb));
	}

	tacs.push_back(create_phi_tac(arguments, {ctx.lookup_value(i)}));

	return tacs.back()->output(0);
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

	jlm::variable * result = nullptr;
	if (ctx.has_value(i))
		result = ctx.lookup_value(i);
	else
		result = m.create_variable(*convert_type(i->getType(), ctx), false);

	tacs.push_back(create_getelementptr_tac(base, indices, result));
	return tacs.back()->output(0);
}

static inline const variable *
convert_trunc_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::Trunc);

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_trunc_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_call_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::Call);
	auto i = llvm::cast<llvm::CallInst>(instruction);

	auto f = i->getCalledValue();
	/* FIXME: currently needed to insert edge in call graph */
	convert_value(f, tacs, ctx);
	JLM_DEBUG_ASSERT(f->getType()->isPointerTy());
	JLM_DEBUG_ASSERT(f->getType()->getContainedType(0)->isFunctionTy());
	auto ftype = llvm::cast<const llvm::FunctionType>(f->getType()->getContainedType(0));
	auto type = convert_type(ftype, ctx);

	/* arguments */
	std::vector<const jlm::variable*> vargs;
	std::vector<const jlm::variable*> arguments;
	for (size_t n = 0; n < ftype->getNumParams(); n++)
		arguments.push_back(convert_value(i->getArgOperand(n), tacs, ctx));
	for (size_t n = ftype->getNumParams(); n < i->getNumOperands()-1; n++)
		vargs.push_back(convert_value(i->getArgOperand(n), tacs, ctx));

	if (ftype->isVarArg()) {
		tacs.push_back(create_valist_tac(vargs, ctx.module()));
		arguments.push_back(tacs.back()->output(0));
	}
	arguments.push_back(ctx.state());

	/* results */
	std::vector<const jlm::variable*> results;
	if (!ftype->getReturnType()->isVoidTy())
		results.push_back(ctx.lookup_value(i));
	results.push_back(ctx.state());

	auto fctvar = convert_value(f, tacs, ctx);
	tacs.push_back(create_call_tac(fctvar, arguments, results));
	return tacs.back()->output(0);
}

static inline const variable *
convert_select_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SelectInst*>(i));
	auto instruction = static_cast<llvm::SelectInst*>(i);

	auto condition = convert_value(instruction->getCondition(), tacs, ctx);
	auto tv = convert_value(instruction->getTrueValue(), tacs, ctx);
	auto fv = convert_value(instruction->getFalseValue(), tacs, ctx);
	tacs.push_back(create_tac(select_op(tv->type()), {condition, tv, fv}, {ctx.lookup_value(i)}));

	return tacs.back()->output(0);
}

static inline const variable *
convert_binary_operator(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::BinaryOperator>(instruction));
	auto i = llvm::cast<const llvm::BinaryOperator>(instruction);
	JLM_DEBUG_ASSERT(!i->getType()->isVectorTy());

	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

	if (i->getType()->isIntegerTy()) {
		static std::unordered_map<
			const llvm::Instruction::BinaryOps,
			std::unique_ptr<jive::operation>(*)(size_t)> map({
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

		size_t nbits = i->getType()->getIntegerBitWidth();
		JLM_DEBUG_ASSERT(map.find(i->getOpcode()) != map.end());
		/* FIXME: This is inefficient. We produce a unique ptr and then copy it. */
		auto op = map[i->getOpcode()](nbits);
		tacs.push_back(create_tac(*static_cast<const jive::simple_op*>(op.get()),
			{op1, op2}, {ctx.lookup_value(i)}));
		return tacs.back()->output(0);
	}

	static std::unordered_map<const llvm::Instruction::BinaryOps, jlm::fpop> map({
	  {llvm::Instruction::FAdd, fpop::add}, {llvm::Instruction::FSub, fpop::sub}
	, {llvm::Instruction::FMul, fpop::mul}, {llvm::Instruction::FDiv, fpop::div}
	, {llvm::Instruction::FRem, fpop::mod}
	});

	JLM_DEBUG_ASSERT(map.find(i->getOpcode()) != map.end());
	tacs.push_back(create_fpbin_tac(map[i->getOpcode()], op1, op2, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_alloca_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::AllocaInst*>(instruction));
	auto i = static_cast<llvm::AllocaInst*>(instruction);

	auto result = ctx.lookup_value(i);
	auto size = convert_value(i->getArraySize(), tacs, ctx);
	auto vtype = convert_type(i->getAllocatedType(), ctx);
	tacs.push_back(create_alloca_tac(*vtype, size, i->getAlignment(), ctx.state(), result));

	return tacs.back()->output(0);
}

static inline const variable *
convert_zext_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::ZExt);
	auto i = llvm::cast<llvm::ZExtInst>(instruction);
	JLM_DEBUG_ASSERT(!i->getSrcTy()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_zext_tac(operand, ctx.lookup_value(i)));

	return tacs.back()->output(0);
}

static inline const variable *
convert_sext_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::SExt);

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_sext_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_fpext_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::FPExt);
	auto i = llvm::cast<llvm::FPExtInst>(instruction);
	JLM_DEBUG_ASSERT(!i->getSrcTy()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_fpext_tac(operand, ctx.lookup_value(i)));

	return tacs.back()->output(0);
}

static inline const variable *
convert_fptrunc_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(instruction->getOpcode() == llvm::Instruction::FPTrunc);
	auto i = llvm::cast<llvm::FPTruncInst>(instruction);
	JLM_DEBUG_ASSERT(!i->getSrcTy()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_fptrunc_tac(operand, ctx.lookup_value(i)));

	return tacs.back()->output(0);
}

static inline const variable *
convert_inttoptr_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::IntToPtrInst*>(i));

	auto argument = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_bits2ptr_tac(argument, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_ptrtoint_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::PtrToIntInst>(i));

	auto argument = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_ptr2bits_tac(argument, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_uitofp_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::UIToFP);
	JLM_DEBUG_ASSERT(!i->getOperand(0)->getType()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_uitofp_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_sitofp_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::SIToFP);
	JLM_DEBUG_ASSERT(!i->getOperand(0)->getType()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_sitofp_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_fptoui_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::FPToUI);
	JLM_DEBUG_ASSERT(!i->getOperand(0)->getType()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_fp2ui_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_fptosi_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::FPToSI);
	JLM_DEBUG_ASSERT(!i->getOperand(0)->getType()->isVectorTy());

	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_fp2si_tac(operand, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_bitcast_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::BitCast);
	auto & m = ctx.module();

	auto operand = convert_value(i->getOperand(0), tacs, ctx);

	jlm::variable * result = nullptr;
	if (ctx.has_value(i))
		result = ctx.lookup_value(i);
	else
		result = m.create_variable(*convert_type(i->getType(), ctx), false);

	tacs.push_back(create_bitcast_tac(operand, result));
	return tacs.back()->output(0);
}

static inline const variable *
convert_insertvalue_instruction(llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
	/* FIXME: add support */
	JLM_ASSERT(0);
}

static inline const variable *
convert_extractvalue_instruction(llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
	/* FIXME: add support */
	JLM_ASSERT(0);
}

const variable *
convert_instruction(
	llvm::Instruction * i,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	static std::unordered_map<
		std::type_index,
		const variable*(*)(llvm::Instruction*, std::vector<std::unique_ptr<jlm::tac>>&, context&)
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
	return map[std::type_index(typeid(*i))](i, tacs, ctx);
}

}
