/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/jlm/llvm2jlm/constant.hpp>
#include <jlm/jlm/llvm2jlm/context.hpp>
#include <jlm/jlm/llvm2jlm/instruction.hpp>
#include <jlm/jlm/llvm2jlm/type.hpp>

#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/ipgraph.hpp>
#include <jlm/jlm/ir/operators.hpp>
#include <jlm/jlm/ir/tac.hpp>

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
		variables.push_back(m.create_variable(op.result(n).type()));

	return variables;
}

static inline void
insert_before_branch(jlm::cfg_node * node, jlm::tacsvector_t & tv)
{
	using namespace jlm;

	JLM_DEBUG_ASSERT(is<basic_block>(node));
	auto & tacs = static_cast<basic_block*>(node)->tacs();

	auto it = tacs.rbegin();
	while (it != tacs.rend()) {
		if (*it && !is<branch_op>((*it)->operation()))
			break;

		it = std::next(it);
	}

	tacs.insert(it.base(), tv);
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

	auto bb = ctx.get(i->getParent());
	bb->add_outedge(bb->cfg().exit());
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
	tacs.push_back(create_branch_tac(2,  tacs.back()->output(0)));

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
		mapping[it.getCaseValue()->getZExtValue()] = n++;
		bb->add_outedge(ctx.get(it.getCaseSuccessor()));
	}

	bb->add_outedge(ctx.get(i->case_default().getCaseSuccessor()));
	JLM_DEBUG_ASSERT(i->getNumSuccessors() == n+1);

	auto c = convert_value(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, mapping, n, n+1);
	tacs.push_back(tac::create(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back((create_branch_tac(n+1, tacs.back()->output(0))));

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

	JLM_DEBUG_ASSERT(is<jive::binary_op>(*binop));
	if (t->isVectorTy()) {
		tacs.push_back(vectorbinary_op::create(*static_cast<jive::binary_op*>(binop.get()),
			op1, op2, ctx.lookup_value(i)));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(binop.get()),
			{op1, op2}, {ctx.lookup_value(i)}));
	}

	return tacs.back()->output(0);
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

	auto r = ctx.lookup_value(i);
	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

	JLM_DEBUG_ASSERT(map.find(i->getPredicate()) != map.end());
	auto fptype = t->isVectorTy() ? t->getVectorElementType() : t;
	fpcmp_op operation(map[i->getPredicate()], convert_fpsize(fptype));

	if (t->isVectorTy())
		tacs.push_back(vectorbinary_op::create(operation, op1, op2, r));
	else
		tacs.push_back(tac::create(operation, {op1, op2}, {r}));

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
		auto bb = ctx.get(*it);
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
		result = m.create_variable(*convert_type(i->getType(), ctx));

	tacs.push_back(create_getelementptr_tac(base, indices, result));
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
	tacs.push_back(tac::create(select_op(tv->type()), {condition, tv, fv}, {ctx.lookup_value(i)}));

	return tacs.back()->output(0);
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

	auto r = ctx.lookup_value(i);
	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);
	JLM_DEBUG_ASSERT(is<jive::binary_op>(*operation));

	if (i->getType()->isVectorTy()) {
		auto & binop = *static_cast<jive::binary_op*>(operation.get());
		tacs.push_back(vectorbinary_op::create(binop, op1, op2, r));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(operation.get()), {op1, op2}, {r}));
	}

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

	auto result = ctx.lookup_value(ev);
	auto aggregate = convert_value(ev->getOperand(0), tacs, ctx);
	tacs.push_back(extractvalue_op::create(aggregate, ev->getIndices(), result));
	return tacs.back()->output(0);
}

static inline const variable *
convert_extractelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::ExtractElement);

	auto vector = convert_value(i->getOperand(0), tacs, ctx);
	auto index = convert_value(i->getOperand(1), tacs, ctx);
	tacs.push_back(extractelement_op::create(vector, index, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static inline const variable *
convert_shufflevector_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::ShuffleVector);

	auto v1 = convert_value(i->getOperand(0), tacs, ctx);
	auto v2 = convert_value(i->getOperand(1), tacs, ctx);
	auto mask = convert_value(i->getOperand(2), tacs, ctx);
	tacs.push_back(shufflevector_op::create(v1, v2, mask, ctx.lookup_value(i)));
	return tacs.back()->output(0);
}

static const variable *
convert_insertelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_DEBUG_ASSERT(i->getOpcode() == llvm::Instruction::InsertElement);

	auto vector = convert_value(i->getOperand(0), tacs, ctx);
	auto value = convert_value(i->getOperand(1), tacs, ctx);
	auto index = convert_value(i->getOperand(2), tacs, ctx);
	tacs.push_back(insertelement_op::create(vector, value, index, ctx.lookup_value(i)));
	return tacs.back()->output(0);
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

	jlm::variable * r;
	if (ctx.has_value(i))
		r = ctx.lookup_value(i);
	else
		r = ctx.module().create_variable(*convert_type(i->getType(), ctx));

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

	return tacs.back()->output(0);
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
	,	{std::type_index(typeid(llvm::CallInst)), convert_call_instruction}
	,	{std::type_index(typeid(llvm::SelectInst)), convert_select_instruction}
	,	{std::type_index(typeid(llvm::AllocaInst)), convert_alloca_instruction}
	,	{std::type_index(typeid(llvm::InsertValueInst)), convert_insertvalue_instruction}
	,	{typeid(llvm::ExtractValueInst), convert_extractvalue}
	,	{typeid(llvm::ExtractElementInst), convert_extractelement_instruction}
	,	{typeid(llvm::ShuffleVectorInst), convert_shufflevector_instruction}
	,	{typeid(llvm::InsertElementInst), convert_insertelement_instruction}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(*i))) != map.end());
	return map[std::type_index(typeid(*i))](i, tacs, ctx);
}

}
