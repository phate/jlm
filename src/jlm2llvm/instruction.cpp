/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/types/bitstring.h>
#include <jive/types/function.h>
#include <jive/vsdg/control.h>
#include <jive/vsdg/operators/match.h>

#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>

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

static inline llvm::Value *
convert_assignment(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_assignment_op(op));
	return args[0];
}

static inline llvm::Value *
convert_binary(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args,
	const std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)> & create)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::base::binary_op*>(&op) && args.size() == 2);
	return create(builder, args[0], args[1]);
}

static inline llvm::Value *
convert_bitconstant(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::constant_op*>(&op));
	auto & cop = *static_cast<const jive::bits::constant_op*>(&op);
	auto type = llvm::IntegerType::get(builder.getContext(), cop.value().nbits());

	if (cop.value().is_defined())
		return llvm::ConstantInt::get(type, cop.value().to_uint());

	return llvm::UndefValue::get(type);
}

static inline llvm::Value *
convert_ctlconstant(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::ctl::constant_op*>(&op));
	auto & cop = *static_cast<const jive::ctl::constant_op*>(&op);
	JLM_DEBUG_ASSERT(cop.value().nalternatives() == 2);

	auto type = llvm::IntegerType::get(builder.getContext(), 1);
	return llvm::ConstantInt::get(type, cop.value().alternative() == 0);
}

static inline llvm::Value *
convert_apply(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::apply_op*>(&op));
	std::vector<llvm::Value*> arguments({std::next(args.begin()), args.end()});
	JLM_DEBUG_ASSERT(arguments.size() == op.narguments()-2);
	return builder.CreateCall(args[0], arguments);
}

static inline llvm::Value *
convert_match(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::match_op*>(&op));
	return args[0];
}

static inline llvm::Value *
convert_branch(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_branch_op(op));
	return nullptr;
}

static inline llvm::Value *
convert_phi(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_phi_op(op));
	auto & pop = *static_cast<const jlm::phi_op*>(&op);

	if (dynamic_cast<const jive::state::type*>(&pop.type()))
		return nullptr;

	auto t = convert_type(pop.type(), builder.getContext());
	return builder.CreatePHI(t, op.narguments());
}

static inline llvm::Value *
convert_load(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_load_op(op));
	return builder.CreateLoad(args[0]);
}

static inline llvm::Value *
convert_store(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_store_op(op) && args.size() >= 2);
	builder.CreateStore(args[1], args[0]);
	return nullptr;
}

static inline llvm::Value *
convert_alloca(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_alloca_op(op) && args.size() == 1);
	auto & aop = *static_cast<const jlm::alloca_op*>(&op);

	auto t = convert_type(aop.value_type(), builder.getContext());
	return builder.CreateAlloca(t, args[0]);
}

static inline llvm::Value *
convert_ptroffset(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_ptroffset_op(op) && args.size() >= 2);
	auto & pop = *static_cast<const ptroffset_op*>(&op);

	auto t = convert_type(pop.pointee_type(), builder.getContext());
	std::vector<llvm::Value*> indices(std::next(args.begin()), args.end());
	return builder.CreateGEP(t, args[0], indices);
}

static inline llvm::Value *
convert_data_array_constant(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_data_array_constant_op(op));
	auto & cop = *static_cast<const data_array_constant_op*>(&op);

	/* FIXME: support other types */
	auto bt = dynamic_cast<const jive::bits::type*>(&cop.type());
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::type*>(&cop.type()));
	JLM_DEBUG_ASSERT(bt->nbits() == 8);

	std::vector<uint8_t> data;
	for (size_t n = 0; n < args.size(); n++) {
		auto c = llvm::dyn_cast<const llvm::ConstantInt>(args[n]);
		JLM_DEBUG_ASSERT(c);
		data.push_back(c->getZExtValue());
	}

	return llvm::ConstantDataArray::get(builder.getContext(), data);
}

llvm::Value *
convert_operation(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & arguments)
{
	using namespace std::placeholders;
	static auto convert_add		= std::bind(convert_binary, _1, _2, _3, create_add);
	static auto convert_and 	= std::bind(convert_binary, _1, _2, _3, create_and);
	static auto convert_ashr	= std::bind(convert_binary, _1, _2, _3, create_ashr);
	static auto convert_sub		= std::bind(convert_binary, _1, _2, _3, create_sub);
	static auto convert_udiv	= std::bind(convert_binary, _1, _2, _3, create_udiv);
	static auto convert_sdiv	= std::bind(convert_binary, _1, _2, _3, create_sdiv);
	static auto convert_umod	= std::bind(convert_binary, _1, _2, _3, create_urem);
	static auto convert_smod	= std::bind(convert_binary, _1, _2, _3, create_srem);
	static auto convert_shl		= std::bind(convert_binary, _1, _2, _3, create_shl);
	static auto convert_shr		= std::bind(convert_binary, _1, _2, _3, create_shr);
	static auto convert_or		= std::bind(convert_binary, _1, _2, _3, create_or);
	static auto convert_xor		= std::bind(convert_binary, _1, _2, _3, create_xor);
	static auto convert_mul		= std::bind(convert_binary, _1, _2, _3, create_mul);
	static auto convert_eq		= std::bind(convert_binary, _1, _2, _3, create_cmpeq);
	static auto convert_ne		= std::bind(convert_binary, _1, _2, _3, create_cmpne);
	static auto convert_ugt		= std::bind(convert_binary, _1, _2, _3, create_cmpugt);
	static auto convert_uge		= std::bind(convert_binary, _1, _2, _3, create_cmpuge);
	static auto convert_ult		= std::bind(convert_binary, _1, _2, _3, create_cmpult);
	static auto convert_ule		= std::bind(convert_binary, _1, _2, _3, create_cmpule);
	static auto convert_sgt		= std::bind(convert_binary, _1, _2, _3, create_cmpsgt);
	static auto convert_sge		= std::bind(convert_binary, _1, _2, _3, create_cmpsge);
	static auto convert_slt		= std::bind(convert_binary, _1, _2, _3, create_cmpslt);
	static auto convert_sle		= std::bind(convert_binary, _1, _2, _3, create_cmpsle);

	using namespace llvm;

	static std::unordered_map<
		std::type_index
	, std::function<Value*(const jive::operation & op, IRBuilder<>&, const std::vector<Value*>&)>
	> map({
	  {std::type_index(typeid(jive::bits::add_op)), convert_add}
	, {std::type_index(typeid(jive::bits::and_op)), convert_and}
	, {std::type_index(typeid(jive::bits::ashr_op)), convert_ashr}
	, {std::type_index(typeid(jive::bits::sub_op)), convert_sub}
	, {std::type_index(typeid(jive::bits::udiv_op)), convert_udiv}
	, {std::type_index(typeid(jive::bits::sdiv_op)), convert_sdiv}
	, {std::type_index(typeid(jive::bits::umod_op)), convert_umod}
	, {std::type_index(typeid(jive::bits::smod_op)), convert_smod}
	, {std::type_index(typeid(jive::bits::shl_op)), convert_shl}
	, {std::type_index(typeid(jive::bits::shr_op)), convert_shr}
	, {std::type_index(typeid(jive::bits::or_op)), convert_or}
	, {std::type_index(typeid(jive::bits::xor_op)), convert_xor}
	, {std::type_index(typeid(jive::bits::mul_op)), convert_mul}
	, {std::type_index(typeid(jive::bits::eq_op)), convert_eq}
	, {std::type_index(typeid(jive::bits::ne_op)), convert_ne}
	, {std::type_index(typeid(jive::bits::ugt_op)), convert_ugt}
	, {std::type_index(typeid(jive::bits::uge_op)), convert_uge}
	, {std::type_index(typeid(jive::bits::ult_op)), convert_ult}
	, {std::type_index(typeid(jive::bits::ule_op)), convert_ule}
	, {std::type_index(typeid(jive::bits::sgt_op)), convert_sgt}
	, {std::type_index(typeid(jive::bits::sge_op)), convert_sge}
	, {std::type_index(typeid(jive::bits::slt_op)), convert_slt}
	, {std::type_index(typeid(jive::bits::sle_op)), convert_sle}
	, {std::type_index(typeid(jive::bits::constant_op)), jlm::jlm2llvm::convert_bitconstant}
	, {std::type_index(typeid(jive::ctl::constant_op)), convert_ctlconstant}
	, {std::type_index(typeid(jive::match_op)), jlm::jlm2llvm::convert_match}
	, {std::type_index(typeid(jive::fct::apply_op)), jlm::jlm2llvm::convert_apply}
	, {std::type_index(typeid(jlm::assignment_op)), jlm::jlm2llvm::convert_assignment}
	, {std::type_index(typeid(jlm::branch_op)), jlm::jlm2llvm::convert_branch}
	, {std::type_index(typeid(jlm::phi_op)), jlm::jlm2llvm::convert_phi}
	, {std::type_index(typeid(jlm::load_op)), jlm::jlm2llvm::convert_load}
	, {std::type_index(typeid(jlm::store_op)), jlm::jlm2llvm::convert_store}
	, {std::type_index(typeid(jlm::alloca_op)), jlm::jlm2llvm::convert_alloca}
	, {std::type_index(typeid(jlm::ptroffset_op)), jlm::jlm2llvm::convert_ptroffset}
	, {std::type_index(typeid(jlm::data_array_constant_op)), convert_data_array_constant}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return map[std::type_index(typeid(op))](op, builder, arguments);
}

void
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	/*
		Collect all operation arguments except for phi operations,
		since their arguments might not be defined yet, i.e. loops,
		and are patched later.
	*/
	std::vector<llvm::Value*> arguments;
	if (!is_phi_op(tac.operation())) {
		for (size_t n = 0; n < tac.ninputs(); n++) {
			if (!dynamic_cast<const jive::state::type*>(&tac.input(n)->type()))
				arguments.push_back(ctx.value(tac.input(n)));
		}
	}

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto r = convert_operation(tac.operation(), builder, arguments);
	if (r != nullptr) ctx.insert(tac.output(0), r);
}

llvm::Constant *
convert_expression(const jlm::expr & e, context & ctx)
{
	std::vector<llvm::Value*> operands;
	for (size_t n = 0; n < e.noperands(); n++)
		operands.push_back(convert_expression(e.operand(n), ctx));

	llvm::IRBuilder<> builder(ctx.llvm_module().getContext());
	return llvm::dyn_cast<llvm::Constant>(convert_operation(e.operation(), builder, operands));
}

}}
