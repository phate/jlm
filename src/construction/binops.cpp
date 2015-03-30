/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/binops.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/instruction.hpp>

#include <jlm/IR/bitstring.hpp>

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

#include <map>

namespace jlm {

/* integer arithmetic instructions */

typedef std::map<
	llvm::Instruction::BinaryOps,
	const jlm::variable *(*)(
		jlm::basic_block *,
		size_t,
		const jlm::variable*,
		const jlm::variable*,
		const jlm::variable*)
	> int_arithmetic_operators_map;

static int_arithmetic_operators_map int_arthm_ops_map({
		{llvm::Instruction::Add,	jlm::bitadd_tac}
	,	{llvm::Instruction::And,	jlm::bitand_tac}
	, {llvm::Instruction::AShr, jlm::bitashr_tac}
	, {llvm::Instruction::Sub, 	jlm::bitsub_tac}
	, {llvm::Instruction::UDiv, jlm::bitudiv_tac}
	, {llvm::Instruction::SDiv, jlm::bitsdiv_tac}
	, {llvm::Instruction::URem, jlm::bitumod_tac}
	, {llvm::Instruction::SRem, jlm::bitsmod_tac}
	, {llvm::Instruction::Shl, 	jlm::bitshl_tac}
	, {llvm::Instruction::LShr,	jlm::bitshr_tac}
	, {llvm::Instruction::Or,		jlm::bitor_tac}
	, {llvm::Instruction::Xor,	jlm::bitxor_tac}
	, {llvm::Instruction::Mul,	jlm::bitmul_tac}
});

static inline const variable *
convert_int_binary_operator(
	const llvm::BinaryOperator & i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(i.getType()->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getType());

	const jlm::variable * op1 = convert_value(i.getOperand(0), ctx);
	const jlm::variable * op2 = convert_value(i.getOperand(1), ctx);
	return int_arthm_ops_map[i.getOpcode()](bb, type->getBitWidth(), op1, op2, ctx.lookup_value(&i));
}

void
convert_binary_operator(
	const llvm::BinaryOperator & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jlm::variable * result = nullptr;
	switch (instruction->getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_binary_operator(*instruction, bb, ctx);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
}

/* integer comparison instructions */

typedef std::map<
	llvm::CmpInst::Predicate,
	const jlm::variable *(*)(
		jlm::basic_block *,
		size_t,
		const jlm::variable*,
		const jlm::variable*,
		const jlm::variable*)
	> int_comparison_operators_map;

static int_comparison_operators_map int_cmp_ops_map({
		{llvm::CmpInst::ICMP_SLT,	jlm::bitslt_tac}
	,	{llvm::CmpInst::ICMP_ULT,	jlm::bitult_tac}
	,	{llvm::CmpInst::ICMP_SLE,	jlm::bitsle_tac}
	,	{llvm::CmpInst::ICMP_ULE,	jlm::bitule_tac}
	,	{llvm::CmpInst::ICMP_EQ,	jlm::biteq_tac}
	,	{llvm::CmpInst::ICMP_NE,	jlm::bitne_tac}
	,	{llvm::CmpInst::ICMP_SGE,	jlm::bitsge_tac}
	,	{llvm::CmpInst::ICMP_UGE,	jlm::bituge_tac}
	,	{llvm::CmpInst::ICMP_SGT,	jlm::bitsgt_tac}
	,	{llvm::CmpInst::ICMP_UGT,	jlm::bitugt_tac}
});

const jlm::variable *
convert_int_comparison_instruction(
	const llvm::ICmpInst & i,
	basic_block * bb,
	const context & ctx)
{
	const jlm::variable * op1 = convert_value(i.getOperand(0), ctx);
	const jlm::variable * op2 = convert_value(i.getOperand(1), ctx);

	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getOperand(0)->getType());

	return int_cmp_ops_map[i.getPredicate()](bb, type->getBitWidth(), op1, op2, ctx.lookup_value(&i));
}

void
convert_comparison_instruction(
	const llvm::CmpInst & i,
	basic_block * bb,
	const context & ctx)
{
	const jlm::variable * result = nullptr;
	switch(i.getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_comparison_instruction(*static_cast<const llvm::ICmpInst*>(&i), bb, ctx);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
}

}
