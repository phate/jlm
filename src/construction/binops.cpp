/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/binops.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/instruction.hpp>

#include <jlm/IR/tac/bitstring.hpp>

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

#include <map>

namespace jlm {

/* integer arithmetic instructions */

typedef std::map<llvm::Instruction::BinaryOps,
	const jlm::frontend::variable *(*)(jlm::frontend::basic_block *, size_t,
		const jlm::frontend::variable*, const jlm::frontend::variable*)> int_arithmetic_operators_map;

static int_arithmetic_operators_map int_arthm_ops_map({
		{llvm::Instruction::Add,	jlm::frontend::bitadd_tac}
	,	{llvm::Instruction::And,	jlm::frontend::bitand_tac}
	, {llvm::Instruction::AShr, jlm::frontend::bitashr_tac}
	, {llvm::Instruction::Sub, 	jlm::frontend::bitsub_tac}
	, {llvm::Instruction::UDiv, jlm::frontend::bitudiv_tac}
	, {llvm::Instruction::SDiv, jlm::frontend::bitsdiv_tac}
	, {llvm::Instruction::URem, jlm::frontend::bitumod_tac}
	, {llvm::Instruction::SRem, jlm::frontend::bitsmod_tac}
	, {llvm::Instruction::Shl, 	jlm::frontend::bitshl_tac}
	, {llvm::Instruction::LShr,	jlm::frontend::bitshr_tac}
	, {llvm::Instruction::Or,		jlm::frontend::bitor_tac}
	, {llvm::Instruction::Xor,	jlm::frontend::bitxor_tac}
	, {llvm::Instruction::Mul,	jlm::frontend::bitmul_tac}
});

static inline const jlm::frontend::variable *
convert_int_binary_operator(const llvm::BinaryOperator & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx)
{
	JLM_DEBUG_ASSERT(i.getType()->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getType());

	const jlm::frontend::variable * op1 = convert_value(i.getOperand(0), bb, ctx);
	const jlm::frontend::variable * op2 = convert_value(i.getOperand(1), bb, ctx);
	return int_arthm_ops_map[i.getOpcode()](bb, type->getBitWidth(), op1, op2);
}

void
convert_binary_operator(const llvm::BinaryOperator & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jlm::frontend::variable * result = nullptr;
	switch (instruction->getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_binary_operator(*instruction, bb, ctx);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
	ctx.insert_value(instruction, result);
}

/* integer comparison instructions */

typedef std::map<llvm::CmpInst::Predicate,
	const jlm::frontend::variable *(*)(jlm::frontend::basic_block *, size_t,
		const jlm::frontend::variable*, const jlm::frontend::variable*)> int_comparison_operators_map;

static int_comparison_operators_map int_cmp_ops_map({
		{llvm::CmpInst::ICMP_SLT,	jlm::frontend::bitslt_tac}
	,	{llvm::CmpInst::ICMP_ULT,	jlm::frontend::bitult_tac}
	,	{llvm::CmpInst::ICMP_SLE,	jlm::frontend::bitsle_tac}
	,	{llvm::CmpInst::ICMP_ULE,	jlm::frontend::bitule_tac}
	,	{llvm::CmpInst::ICMP_EQ,	jlm::frontend::biteq_tac}
	,	{llvm::CmpInst::ICMP_NE,	jlm::frontend::bitne_tac}
	,	{llvm::CmpInst::ICMP_SGE,	jlm::frontend::bitsge_tac}
	,	{llvm::CmpInst::ICMP_UGE,	jlm::frontend::bituge_tac}
	,	{llvm::CmpInst::ICMP_SGT,	jlm::frontend::bitsgt_tac}
	,	{llvm::CmpInst::ICMP_UGT,	jlm::frontend::bitugt_tac}
});

const jlm::frontend::variable *
convert_int_comparison_instruction(const llvm::ICmpInst & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx)
{
	const jlm::frontend::variable * op1 = convert_value(i.getOperand(0), bb, ctx);
	const jlm::frontend::variable * op2 = convert_value(i.getOperand(1), bb, ctx);

	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getOperand(0)->getType());

	return int_cmp_ops_map[i.getPredicate()](bb, type->getBitWidth(), op1, op2);
}

void
convert_comparison_instruction(const llvm::CmpInst & i, jlm::frontend::basic_block * bb,
	jlm::context & ctx)
{
	const jlm::frontend::variable * result = nullptr;
	switch(i.getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_comparison_instruction(*static_cast<const llvm::ICmpInst*>(&i), bb, ctx);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
	ctx.insert_value(&i, result);
}

}
