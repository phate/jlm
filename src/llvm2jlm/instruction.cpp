/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/llvm2jlm/constant.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/instruction.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/clg.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>

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

static inline std::vector<const jlm::variable*>
create_result_variables(jlm::module & m, const jive::operation & op)
{
	std::vector<const jlm::variable*> variables;
	for (size_t n = 0; n < op.nresults(); n++)
		variables.push_back(m.create_variable(op.result_type(n), false));

	return variables;
}

namespace jlm {

typedef std::vector<std::unique_ptr<jlm::tac>> tacsvector_t;

static inline const variable *
convert_value(llvm::Value * v, tacsvector_t & tacs, context & ctx)
{
	if (ctx.has_value(v))
		return ctx.lookup_value(v);

	if (auto c = llvm::dyn_cast<llvm::Constant>(v))
		return convert_constant(c, tacs, ctx);

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

/* instructions */

static inline tacsvector_t
convert_return_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ReturnInst*>(i));
	const llvm::ReturnInst * instruction = static_cast<const llvm::ReturnInst*>(i);

	tacsvector_t tacs;
	if (instruction->getReturnValue()) {
		auto value = convert_value(instruction->getReturnValue(), tacs, ctx);
		tacs.push_back(create_assignment(ctx.result()->type(), value, ctx.result()));
	}

	return tacs;
}

static inline tacsvector_t
convert_branch_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BranchInst*>(i));
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(i);

	tacsvector_t tacs;
	if (instruction->isConditional()) {
		auto c = convert_value(instruction->getCondition(), tacs, ctx);
		size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
		auto op = jive::match_op(nbits, {{0, 0}}, 1, 2);
		tacs.push_back(create_tac(op, {c}, create_result_variables(ctx.module(), op)));
		tacs.push_back(create_branch_tac(2,  tacs.back()->output(0)));
	}

	return tacs;
}

static inline tacsvector_t
convert_switch_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SwitchInst*>(i));
	const llvm::SwitchInst * instruction = static_cast<const llvm::SwitchInst*>(i);

	std::map<uint64_t, uint64_t> mapping;
	for (auto it = instruction->case_begin(); it != instruction->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != instruction->case_default());
		mapping[it.getCaseValue()->getZExtValue()] = it.getCaseIndex();
	}

	tacsvector_t tacs;
	auto c = convert_value(instruction->getCondition(), tacs, ctx);
	size_t nbits = dynamic_cast<const jive::bits::type&>(c->type()).nbits();
	auto op = jive::match_op(nbits, mapping, mapping.size(), mapping.size()+1);
	tacs.push_back(create_tac(op, {c}, create_result_variables(ctx.module(), op)));
	tacs.push_back((create_branch_tac(mapping.size()+1, tacs.back()->output(0))));

	return tacs;
}

static inline tacsvector_t
convert_unreachable_instruction(llvm::Instruction * i, context & ctx)
{
	/* Nothing needs to be done. */
	return {};
}

static inline tacsvector_t
convert_icmp_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ICmpInst*>(instruction));
	const llvm::ICmpInst * i = static_cast<const llvm::ICmpInst*>(instruction);

	tacsvector_t tacs;
	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

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
		auto new_op1 = ctx.module().create_variable(jive::bits::type(nbits), false);
		auto new_op2 = ctx.module().create_variable(jive::bits::type(nbits), false);
		tacs.push_back((create_tac(op, {op1}, {new_op1})));
		op1 = tacs.back()->output(0);
		tacs.push_back((create_tac(op, {op2}, {new_op2})));
		op2 = tacs.back()->output(0);
	} else
		nbits = i->getOperand(0)->getType()->getIntegerBitWidth();

	tacs.push_back(create_tac(*map[i->getPredicate()](nbits), {op1, op2}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_fcmp_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FCmpInst*>(instruction));
	const llvm::FCmpInst * i = static_cast<const llvm::FCmpInst*>(instruction);

	/* FIXME: vector type is not yet supported */
	tacsvector_t tacs;
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
		operands.push_back(convert_value(i->getOperand(0), tacs, ctx));
		operands.push_back(convert_value(i->getOperand(1), tacs, ctx));
	}

	tacs.push_back(create_tac(*map[i->getPredicate()](), operands, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_load_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::LoadInst*>(i));
	auto instruction = static_cast<llvm::LoadInst*>(i);

	/* FIXME: volatile and alignment */

	tacsvector_t tacs;
	auto value = ctx.lookup_value(i);
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	tacs.push_back(create_load_tac(address, ctx.state(), value));

	return tacs;
}

static inline tacsvector_t
convert_store_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::StoreInst*>(i));
	auto instruction = static_cast<llvm::StoreInst*>(i);

	/* FIXME: volatile and alignement */
	tacsvector_t tacs;
	auto address = convert_value(instruction->getPointerOperand(), tacs, ctx);
	auto value = convert_value(instruction->getValueOperand(), tacs, ctx);
	tacs.push_back(create_store_tac(address, value, ctx.state()));

	return tacs;
}

static inline tacsvector_t
convert_phi_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::PHINode*>(i));
	const llvm::PHINode * phi = static_cast<const llvm::PHINode*>(i);

	tacsvector_t tacs;
	std::vector<const variable*> operands;
	for (auto edge : ctx.lookup_basic_block(i->getParent())->inedges()) {
		auto tmp = edge->source();
		const llvm::BasicBlock * ib = nullptr;
		if (ctx.has_basic_block(tmp)) {
			ib = ctx.lookup_basic_block(tmp);
		} else {
			JLM_DEBUG_ASSERT(tmp->ninedges() == 1);
			ib = ctx.lookup_basic_block(tmp->inedges().front()->source());
		}

		operands.push_back(convert_value(phi->getIncomingValueForBlock(ib), tacs, ctx));
	}

	JLM_DEBUG_ASSERT(operands.size() != 0);
	phi_op op(operands.size(), operands[0]->type());
	tacs.push_back(create_tac(op, operands, {ctx.lookup_value(phi)}));

	return tacs;
}

static inline tacsvector_t
convert_getelementptr_instruction(llvm::Instruction * inst, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::GetElementPtrInst>(inst));
	auto i = llvm::cast<llvm::GetElementPtrInst>(inst);
	auto & m = ctx.module();

	tacsvector_t tacs;
	std::vector<const variable*> indices;
	auto base = convert_value(i->getPointerOperand(), tacs, ctx);
	for (auto it = i->idx_begin(); it != i->idx_end(); it++)
		indices.push_back(convert_value(*it, tacs, ctx));

	auto r = m.create_variable(*convert_type(i->getType(), ctx), false);
	tacs.push_back(create_ptroffset_tac(base, indices, r));
	return tacs;
}

static inline tacsvector_t
convert_trunc_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::TruncInst*>(i));
	const llvm::TruncInst * instruction = static_cast<const llvm::TruncInst*>(i);

	tacsvector_t tacs;
	size_t high = i->getType()->getIntegerBitWidth();
	auto operand = convert_value(instruction->getOperand(0), tacs, ctx);
	jive::bits::slice_op op(dynamic_cast<const jive::bits::type&>(operand->type()), 0, high);
	tacs.push_back(create_tac(op, {operand}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_call_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::CallInst*>(i));
	auto instruction = static_cast<llvm::CallInst*>(i);

	auto f = instruction->getCalledValue();
	const llvm::FunctionType * ftype = llvm::cast<const llvm::FunctionType>(
		llvm::cast<const llvm::PointerType>(f->getType())->getElementType());
	jive::fct::type type = dynamic_cast<jive::fct::type&>(*convert_type(ftype, ctx));

	tacsvector_t tacs;
	const variable * callee = nullptr;
	auto vcaller = ctx.lookup_value(i->getParent()->getParent());
	auto caller = static_cast<const function_variable*>(vcaller)->function();
	if (instruction->getCalledFunction()) {
		/* direct call */
		callee = convert_value(f, tacs, ctx);
		caller->add_call(static_cast<const function_variable*>(callee)->function());
	} else {
		/* indirect call */
		/* FIXME */
		auto a = convert_value(f, tacs, ctx);
		auto r = ctx.module().create_variable(type, false);
		tacs.push_back(create_load_tac(a, ctx.state(), r));
		callee = tacs.back()->output(0);
	}

	/* handle arguments */
	std::vector<const jlm::variable *> arguments;
	arguments.push_back(callee);
	for (size_t n = 0; n < ftype->getNumParams(); n++)
		arguments.push_back(convert_value(instruction->getArgOperand(n), tacs, ctx));
	if (ftype->isVarArg()) {
		/* FIXME: sizeof */
		//auto alloc = ctx.module().create_variable(jive::addr::type::instance(), false);
		//auto vararg = attr->append(create_tac(alloca_op(4), {ctx.state()}, {alloc, ctx.state()}));
		//arguments.push_back(vararg->output(0));
	}
	arguments.push_back(ctx.state());

	/* handle results */
	std::vector<const jlm::variable*> results;
	if (type.nresults() == 2)
		results.push_back(ctx.lookup_value(i));
	results.push_back(ctx.state());

	tacs.push_back(create_tac(jive::fct::apply_op(type), arguments, results));
	return tacs;
}

static inline tacsvector_t
convert_select_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SelectInst*>(i));
	auto instruction = static_cast<llvm::SelectInst*>(i);

	tacsvector_t tacs;
	auto condition = convert_value(instruction->getCondition(), tacs, ctx);
	auto tv = convert_value(instruction->getTrueValue(), tacs, ctx);
	auto fv = convert_value(instruction->getFalseValue(), tacs, ctx);
	tacs.push_back(create_tac(select_op(tv->type()), {condition, tv, fv}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_binary_operator(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BinaryOperator*>(instruction));
	const llvm::BinaryOperator * i = static_cast<const llvm::BinaryOperator*>(instruction);

	/* FIXME: vector type is not yet supported */
	tacsvector_t tacs;
	if (i->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	auto op1 = convert_value(i->getOperand(0), tacs, ctx);
	auto op2 = convert_value(i->getOperand(1), tacs, ctx);

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
		tacs.push_back(create_tac(*map[i->getOpcode()](nbits), {op1, op2}, {ctx.lookup_value(i)}));
		return tacs;
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
		tacs.push_back(create_tac(*map[i->getOpcode()](), {op1, op2}, {ctx.lookup_value(i)}));
		return tacs;
	}

	JLM_DEBUG_ASSERT(0);
}

static inline tacsvector_t
convert_alloca_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::AllocaInst*>(instruction));
	auto i = static_cast<llvm::AllocaInst*>(instruction);

	/* FIXME: alignment */
	tacsvector_t tacs;
	auto result = ctx.lookup_value(i);
	auto size = convert_value(i->getArraySize(), tacs, ctx);
	auto vtype = convert_type(i->getAllocatedType(), ctx);
	tacs.push_back(create_alloca_tac(*vtype, size, ctx.state(), result));

	return tacs;
}

static inline tacsvector_t
convert_zext_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ZExtInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::constant_op c_op(jive::bits::value_repr(new_length - old_length, 0));
	auto vs = create_result_variables(ctx.module(), c_op);
	tacs.push_back(create_tac(c_op, {}, vs));
	auto c = tacs.back()->output(0);

	jive::bits::concat_op op({jive::bits::type(old_length), jive::bits::type(new_length-old_length)});
	auto v = convert_value(operand, tacs, ctx);
	tacs.push_back(create_tac(op, {v, c}, {ctx.lookup_value(i)}));
	return tacs;
}

static inline tacsvector_t
convert_sext_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SExtInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	size_t new_length = i->getType()->getIntegerBitWidth();
	size_t old_length = operand->getType()->getIntegerBitWidth();
	jive::bits::slice_op s_op(jive::bits::type(old_length), old_length-1, old_length);
	auto vs = create_result_variables(ctx.module(), s_op);
	auto v = convert_value(operand, tacs, ctx);
	tacs.push_back(create_tac(s_op, {v}, vs));
	auto bit = tacs.back()->output(0);

	std::vector<const variable*> operands(1, convert_value(operand, tacs, ctx));
	std::vector<jive::bits::type> operand_types(1, jive::bits::type(old_length));
	for (size_t n = 0; n < new_length - old_length; n++) {
		operand_types.push_back(jive::bits::type(1));
		operands.push_back(bit);
	}

	jive::bits::concat_op op(operand_types);
	tacs.push_back(create_tac(op, operands, {ctx.lookup_value(i)}));
	return tacs;
}

static inline tacsvector_t
convert_fpext_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPExtInst*>(i));

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	tacsvector_t tacs;
	jive::flt::type type;
	assignment_op op(type);
	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_tac(op, {operand}, {ctx.lookup_value(i)}));
	return tacs;
}

static inline tacsvector_t
convert_fptrunc_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPTruncInst*>(i));

	/* FIXME: use assignment operator as long as we don't support floating point types properly */

	tacsvector_t tacs;
	jive::flt::type type;
	assignment_op op(type);
	auto operand = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_tac(op, {operand}, {ctx.lookup_value(i)}));
	return tacs;
}

static inline tacsvector_t
convert_inttoptr_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::IntToPtrInst*>(i));

	tacsvector_t tacs;
	auto argument = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_bits2ptr_tac(argument, ctx.lookup_value(i)));

	return tacs;
}

static inline tacsvector_t
convert_ptrtoint_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(llvm::dyn_cast<const llvm::PtrToIntInst>(i));

	tacsvector_t tacs;
	auto argument = convert_value(i->getOperand(0), tacs, ctx);
	tacs.push_back(create_ptr2bits_tac(argument, ctx.lookup_value(i)));

	return tacs;
}

static inline tacsvector_t
convert_uitofp_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::UIToFPInst*>(i));

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	bits2flt_op op(type->getIntegerBitWidth());
	tacs.push_back(create_tac(op, {convert_value(operand, tacs, ctx)}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_sitofp_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SIToFPInst*>(i));

	llvm::Value * operand = i->getOperand(0);
	llvm::Type * type = operand->getType();

	/* FIXME: support vector type */
	if (type->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	bits2flt_op op(type->getIntegerBitWidth());
	tacs.push_back(create_tac(op, {convert_value(operand, tacs, ctx)}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_fptoui_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToUIInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	flt2bits_op op(i->getType()->getIntegerBitWidth());
	tacs.push_back(create_tac(op, {convert_value(operand, tacs, ctx)}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_fptosi_instruction(llvm::Instruction * i, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::FPToSIInst*>(i));

	llvm::Value * operand = i->getOperand(0);

	/* FIXME: support vector type */
	if (operand->getType()->isVectorTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	flt2bits_op op(i->getType()->getIntegerBitWidth());
	tacs.push_back(create_tac(op, {convert_value(operand, tacs, ctx)}, {ctx.lookup_value(i)}));

	return tacs;
}

static inline tacsvector_t
convert_bitcast_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::BitCastInst*>(instruction));
	auto & i = *llvm::cast<const llvm::BitCastInst>(instruction);
	auto & m = ctx.module();

	/* FIXME: avoid usage of memory to do cast */

	tacsvector_t tacs;
	auto o = convert_value(i.getOperand(0), tacs, ctx);

	jive::bits::type bits32(32);
	auto ptype = create_ptrtype(*convert_type(i.getSrcTy(), ctx));

	auto c = m.create_variable(bits32, false);
	tacs.push_back(create_tac(jive::bits::uint_constant_op(32, 1), {}, {c}));

	auto a = m.create_variable(*ptype, false);
	tacs.push_back(create_alloca_tac(o->type(), c, ctx.state(), a));
	tacs.push_back(create_store_tac(a, o, ctx.state()));
	tacs.push_back(create_load_tac(a, ctx.state(), ctx.lookup_value(&i)));

	return tacs;
}

static inline tacsvector_t
convert_insertvalue_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::InsertValueInst*>(instruction));
	auto i = static_cast<llvm::InsertValueInst*>(instruction);

	/* FIXME: array type */
	if (i->getType()->isArrayTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	std::function<const variable*(llvm::InsertValueInst::idx_iterator, const variable*)> f = [&] (
		const llvm::InsertValueInst::idx_iterator & idx,
		const variable * aggregate)
	{
		if (idx == i->idx_end())
			return convert_value(i->getInsertedValueOperand(), tacs, ctx);

		/* FIXME: array type */
		const jive::rcd::type * type = dynamic_cast<const jive::rcd::type*>(&aggregate->type());
		std::shared_ptr<const jive::rcd::declaration> decl = type->declaration();

		std::vector<const variable*> operands;
		for (size_t n = 0; n < decl->nelements(); n++) {
			auto op = jive::rcd::select_operation(*type, n);
			auto vs = create_result_variables(ctx.module(), op);
			tacs.push_back(create_tac(op, {aggregate}, vs));
			if (n == *idx)
				operands.push_back(f(std::next(idx), tacs.back()->output(0)));
			else
				operands.push_back(tacs.back()->output(0));
		}

		jive::rcd::group_op op(decl);
		tacs.push_back(create_tac(op, operands, {ctx.lookup_value(i)}));
		return tacs.back()->output(0);
	};

	f(i->idx_begin(), convert_value(i->getAggregateOperand(), tacs, ctx));

	return tacs;
}

static inline tacsvector_t
convert_extractvalue_instruction(llvm::Instruction * instruction, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::ExtractValueInst*>(instruction));
	auto i = static_cast<llvm::ExtractValueInst*>(instruction);

	/* FIXME: array type */
	if (i->getType()->isArrayTy())
		JLM_DEBUG_ASSERT(0);

	tacsvector_t tacs;
	auto aggregate = convert_value(i->getAggregateOperand(), tacs, ctx);

	for (auto it = i->idx_begin(); it != i->idx_end(); it++) {
		const jive::rcd::type * type = dynamic_cast<const jive::rcd::type*>(&aggregate->type());

		jive::rcd::select_operation op(*type, *it);
		tacs.push_back(create_tac(op, {aggregate}, {ctx.lookup_value(i)}));
		aggregate = tacs.back()->output(0);
	}

	return tacs;
}

std::vector<std::unique_ptr<jlm::tac>>
convert_instruction(llvm::Instruction * i, context & ctx)
{
	static std::unordered_map<
		std::type_index,
		tacsvector_t(*)(llvm::Instruction*, context&)
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
	return map[std::type_index(typeid(*i))](i, ctx);
}

}
