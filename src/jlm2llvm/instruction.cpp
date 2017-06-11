/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/types/bitstring.h>
#include <jive/types/function.h>
#include <jive/vsdg/operators/match.h>

/* FIXME: remove */
#include <jlm/IR/cfg.hpp>

#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

#include <jlm/jlm2llvm/context.hpp>
#include <jlm/jlm2llvm/instruction.hpp>
#include <jlm/jlm2llvm/type.hpp>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace jlm {
namespace jlm2llvm {

static inline llvm::Value *
create_add(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	/* FIXME: nuw and nsw flags */
	return builder.CreateAdd(lhs, rhs);
}

static inline llvm::Value *
create_and(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateAnd(lhs, rhs);
}

static inline llvm::Value *
create_ashr(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateAShr(lhs, rhs);
}

static inline llvm::Value *
create_sub(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	/* FIXME: nuw and nsw flags */
	return builder.CreateSub(lhs, rhs);
}

static inline llvm::Value *
create_udiv(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateUDiv(lhs, rhs);
}

static inline llvm::Value *
create_sdiv(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateSDiv(lhs, rhs);
}

static inline llvm::Value *
create_urem(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateURem(lhs, rhs);
}

static inline llvm::Value *
create_srem(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateSRem(lhs, rhs);
}

static inline llvm::Value *
create_shl(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateShl(lhs, rhs);
}

static inline llvm::Value *
create_shr(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateLShr(lhs, rhs);
}

static inline llvm::Value *
create_or(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateOr(lhs, rhs);
}

static inline llvm::Value *
create_xor(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateXor(lhs, rhs);
}

static inline llvm::Value *
create_mul(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	/* FIXME: nuw and nsw flags */
	return builder.CreateMul(lhs, rhs);
}

static inline llvm::Value *
create_cmpeq(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpEQ(lhs, rhs);
}

static inline llvm::Value *
create_cmpne(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpNE(lhs, rhs);
}

static inline llvm::Value *
create_cmpugt(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpUGT(lhs, rhs);
}

static inline llvm::Value *
create_cmpuge(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpUGE(lhs, rhs);
}

static inline llvm::Value *
create_cmpult(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpULT(lhs, rhs);
}

static inline llvm::Value *
create_cmpule(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpULE(lhs, rhs);
}

static inline llvm::Value *
create_cmpsgt(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpSGT(lhs, rhs);
}

static inline llvm::Value *
create_cmpsge(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpSGE(lhs, rhs);
}

static inline llvm::Value *
create_cmpslt(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpSLT(lhs, rhs);
}

static inline llvm::Value *
create_cmpsle(llvm::IRBuilder<> & builder, llvm::Value * lhs, llvm::Value * rhs)
{
	return builder.CreateICmpSLE(lhs, rhs);
}

static inline llvm::Instruction *
convert_assignment(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(is_assignment_op(tac.operation()));

	ctx.insert(tac.output(0), ctx.value(tac.input(0)));
	return nullptr;
}

static inline llvm::Instruction*
convert_binary(
	const jlm::tac & tac,
	const jlm::cfg_node * node,
	const std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)> & create,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::base::binary_op*>(&tac.operation()));

	JLM_DEBUG_ASSERT(tac.ninputs() == 2);
	auto lhs = ctx.value(tac.input(0));
	auto rhs = ctx.value(tac.input(1));

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto v = create(builder, lhs, rhs);
	ctx.insert(tac.output(0), v);

	return llvm::cast<llvm::Instruction>(v);
}

static inline llvm::Instruction *
convert_bitconstant(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::constant_op*>(&tac.operation()));
	auto & op = *static_cast<const jive::bits::constant_op*>(&tac.operation());

	llvm::Value * v = nullptr;
	auto type = llvm::IntegerType::get(ctx.llvm_module().getContext(), op.value().nbits());
	if (op.value().is_defined()) {
		v = llvm::ConstantInt::get(type, op.value().to_uint());
	} else {
		v = llvm::UndefValue::get(type);
	}

	ctx.insert(tac.output(0), v);

	return nullptr;
}

static inline llvm::Instruction *
convert_apply(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::apply_op*>(&tac.operation()));
	auto & ftype = *static_cast<const jive::fct::type*>(&tac.input(0)->type());
	auto & lctx = ctx.llvm_module().getContext();

	std::vector<llvm::Value*> arguments;
	auto callee = ctx.value(tac.input(0));
	for (size_t n = 1; n < tac.ninputs()-1; n++)
		arguments.push_back(ctx.value(tac.input(n)));

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto v = builder.CreateCall(convert_type(ftype, lctx), callee, arguments);

	JLM_DEBUG_ASSERT(tac.noutputs() == 1 || tac.noutputs() == 2);
	if (tac.noutputs() == 2)
		ctx.insert(tac.output(0), v);

	return v;
}

static inline llvm::Instruction *
convert_match(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&tac.operation()));

	ctx.insert(tac.output(0), ctx.value(tac.input(0)));
	return nullptr;
}

static inline llvm::Instruction *
convert_branch(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(is_branch_op(tac.operation()));
	return nullptr;
}

static inline llvm::Instruction *
convert_phi(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(is_phi_op(tac.operation()));
	auto & op = *static_cast<const jlm::phi_op*>(&tac.operation());
	JLM_DEBUG_ASSERT(node->ninedges() == op.narguments());

	if (dynamic_cast<const jive::state::type*>(&op.type()))
		return nullptr;

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto t = convert_type(op.type(), ctx.llvm_module().getContext());
	auto phi = builder.CreatePHI(t, op.narguments());

	ctx.insert(tac.output(0), phi);
	return phi;
}

llvm::Instruction *
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	using namespace std::placeholders;

	static std::unordered_map<
	  std::type_index
	, std::function<llvm::Instruction*(const jlm::tac & tac, const jlm::cfg_node*, context & ctx)>
	> map({
	  {std::type_index(typeid(jive::bits::add_op)), std::bind(convert_binary,_1,_2,create_add,_3)}
	, {std::type_index(typeid(jive::bits::and_op)), std::bind(convert_binary,_1,_2,create_and,_3)}
	, {std::type_index(typeid(jive::bits::ashr_op)), std::bind(convert_binary,_1,_2,create_ashr,_3)}
	, {std::type_index(typeid(jive::bits::sub_op)), std::bind(convert_binary,_1,_2,create_sub,_3)}
	, {std::type_index(typeid(jive::bits::udiv_op)), std::bind(convert_binary,_1,_2,create_udiv,_3)}
	, {std::type_index(typeid(jive::bits::sdiv_op)), std::bind(convert_binary,_1,_2,create_sdiv,_3)}
	, {std::type_index(typeid(jive::bits::umod_op)), std::bind(convert_binary,_1,_2,create_urem,_3)}
	, {std::type_index(typeid(jive::bits::smod_op)), std::bind(convert_binary,_1,_2,create_srem,_3)}
	, {std::type_index(typeid(jive::bits::shl_op)), std::bind(convert_binary,_1,_2,create_shl,_3)}
	, {std::type_index(typeid(jive::bits::shr_op)), std::bind(convert_binary,_1,_2,create_shr,_3)}
	, {std::type_index(typeid(jive::bits::or_op)), std::bind(convert_binary,_1,_2,create_or,_3)}
	, {std::type_index(typeid(jive::bits::xor_op)), std::bind(convert_binary,_1,_2,create_xor,_3)}
	, {std::type_index(typeid(jive::bits::mul_op)), std::bind(convert_binary,_1,_2,create_mul,_3)}
	, {std::type_index(typeid(jive::bits::eq_op)), std::bind(convert_binary,_1,_2,create_cmpeq,_3)}
	, {std::type_index(typeid(jive::bits::ne_op)), std::bind(convert_binary,_1,_2,create_cmpne,_3)}
	, {std::type_index(typeid(jive::bits::ugt_op)), std::bind(convert_binary,_1,_2,create_cmpugt,_3)}
	, {std::type_index(typeid(jive::bits::uge_op)), std::bind(convert_binary,_1,_2,create_cmpuge,_3)}
	, {std::type_index(typeid(jive::bits::ult_op)), std::bind(convert_binary,_1,_2,create_cmpult,_3)}
	, {std::type_index(typeid(jive::bits::ule_op)), std::bind(convert_binary,_1,_2,create_cmpule,_3)}
	, {std::type_index(typeid(jive::bits::sgt_op)), std::bind(convert_binary,_1,_2,create_cmpsgt,_3)}
	, {std::type_index(typeid(jive::bits::sge_op)), std::bind(convert_binary,_1,_2,create_cmpsge,_3)}
	, {std::type_index(typeid(jive::bits::slt_op)), std::bind(convert_binary,_1,_2,create_cmpslt,_3)}
	, {std::type_index(typeid(jive::bits::sle_op)), std::bind(convert_binary,_1,_2,create_cmpsle,_3)}
	, {std::type_index(typeid(jive::bits::constant_op)), jlm::jlm2llvm::convert_bitconstant}
	, {std::type_index(typeid(jive::match_op)), jlm::jlm2llvm::convert_match}
	, {std::type_index(typeid(jive::fct::apply_op)), jlm::jlm2llvm::convert_apply}
	, {std::type_index(typeid(jlm::assignment_op)), jlm::jlm2llvm::convert_assignment}
	, {std::type_index(typeid(jlm::branch_op)), jlm::jlm2llvm::convert_branch}
	, {std::type_index(typeid(jlm::phi_op)), jlm::jlm2llvm::convert_phi}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(tac.operation()))) != map.end());
	return map[std::type_index(typeid(tac.operation()))](tac, node, ctx);
}

}}
