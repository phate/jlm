/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/binops.hpp>
#include <jlm/common.hpp>
#include <jlm/instruction.hpp>

#include <jive/frontend/tac/bitstring.h>

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

#include <map>

namespace jlm {

/* integer arithmetic instructions */

typedef std::map<llvm::Instruction::BinaryOps,
	const jive::frontend::output *(*)(jive::frontend::basic_block *, size_t,
		const jive::frontend::output*, const jive::frontend::output*)> int_arithmetic_operators_map;

static int_arithmetic_operators_map int_arthm_ops_map({
		{llvm::Instruction::Add,	jive::frontend::bitadd_tac}
	,	{llvm::Instruction::And,	jive::frontend::bitand_tac}
	, {llvm::Instruction::AShr, jive::frontend::bitashr_tac}
	, {llvm::Instruction::Sub, 	jive::frontend::bitsub_tac}
	, {llvm::Instruction::UDiv, jive::frontend::bitudiv_tac}
	, {llvm::Instruction::SDiv, jive::frontend::bitsdiv_tac}
	, {llvm::Instruction::URem, jive::frontend::bitumod_tac}
	, {llvm::Instruction::SRem, jive::frontend::bitsmod_tac}
	, {llvm::Instruction::Shl, 	jive::frontend::bitshl_tac}
	, {llvm::Instruction::LShr,	jive::frontend::bitshr_tac}
	, {llvm::Instruction::Or,		jive::frontend::bitor_tac}
	, {llvm::Instruction::Xor,	jive::frontend::bitxor_tac}
	, {llvm::Instruction::Mul,	jive::frontend::bitmul_tac}
});

static inline const jive::frontend::output *
convert_int_binary_operator(const llvm::BinaryOperator & i, jive::frontend::basic_block * bb,
	value_map & vmap)
{
	JLM_DEBUG_ASSERT(i.getType()->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getType());

	const jive::frontend::output * op1 = convert_value(i.getOperand(0), bb, vmap);
	const jive::frontend::output * op2 = convert_value(i.getOperand(1), bb, vmap);
	return int_arthm_ops_map[i.getOpcode()](bb, type->getBitWidth(), op1, op2);
}

void
convert_binary_operator(const llvm::BinaryOperator & i, jive::frontend::basic_block * bb,
	value_map & vmap)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jive::frontend::output * result = nullptr;
	switch (instruction->getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_binary_operator(*instruction, bb, vmap);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
	vmap[instruction] = result;
}

/* integer comparison instructions */

typedef std::map<llvm::CmpInst::Predicate,
	const jive::frontend::output *(*)(jive::frontend::basic_block *, size_t,
		const jive::frontend::output*, const jive::frontend::output*)> int_comparison_operators_map;

static int_comparison_operators_map int_cmp_ops_map({
		{llvm::CmpInst::ICMP_SLT,	jive::frontend::bitslt_tac}
	,	{llvm::CmpInst::ICMP_ULT,	jive::frontend::bitult_tac}
	,	{llvm::CmpInst::ICMP_SLE,	jive::frontend::bitsle_tac}
	,	{llvm::CmpInst::ICMP_ULE,	jive::frontend::bitule_tac}
	,	{llvm::CmpInst::ICMP_EQ,	jive::frontend::biteq_tac}
	,	{llvm::CmpInst::ICMP_NE,	jive::frontend::bitne_tac}
	,	{llvm::CmpInst::ICMP_SGE,	jive::frontend::bitsge_tac}
	,	{llvm::CmpInst::ICMP_UGE,	jive::frontend::bituge_tac}
	,	{llvm::CmpInst::ICMP_SGT,	jive::frontend::bitsgt_tac}
	,	{llvm::CmpInst::ICMP_UGT,	jive::frontend::bitugt_tac}
});

const jive::frontend::output *
convert_int_comparison_instruction(const llvm::ICmpInst & i, jive::frontend::basic_block * bb,
	value_map & vmap)
{
	const jive::frontend::output * op1 = convert_value(i.getOperand(0), bb, vmap);
	const jive::frontend::output * op2 = convert_value(i.getOperand(1), bb, vmap);

	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getOperand(0)->getType());

	return int_cmp_ops_map[i.getPredicate()](bb, type->getBitWidth(), op1, op2);
}

void
convert_comparison_instruction(const llvm::CmpInst & i, jive::frontend::basic_block * bb,
	value_map & vmap)
{
	const jive::frontend::output * result = nullptr;
	switch(i.getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_comparison_instruction(*static_cast<const llvm::ICmpInst*>(&i), bb, vmap);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
	vmap[&i] = result;
}

}
