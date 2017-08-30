/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/memorytype.h>
#include <jive/types/bitstring.h>
#include <jive/types/function.h>
#include <jive/vsdg/control.h>

#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/module.hpp>
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
convert_assignment(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_assignment_op(op));
	return ctx.value(args[0]);
}

static inline llvm::Value *
convert_bitsbinary(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::binary_op*>(&op));

	static std::unordered_map<std::type_index, llvm::Instruction::BinaryOps> map({
	  {std::type_index(typeid(jive::bits::add_op)), llvm::Instruction::Add}
	, {std::type_index(typeid(jive::bits::and_op)), llvm::Instruction::And}
	, {std::type_index(typeid(jive::bits::ashr_op)), llvm::Instruction::AShr}
	, {std::type_index(typeid(jive::bits::sub_op)), llvm::Instruction::Sub}
	, {std::type_index(typeid(jive::bits::udiv_op)), llvm::Instruction::UDiv}
	, {std::type_index(typeid(jive::bits::sdiv_op)), llvm::Instruction::SDiv}
	, {std::type_index(typeid(jive::bits::umod_op)), llvm::Instruction::URem}
	, {std::type_index(typeid(jive::bits::smod_op)), llvm::Instruction::SRem}
	, {std::type_index(typeid(jive::bits::shl_op)), llvm::Instruction::Shl}
	, {std::type_index(typeid(jive::bits::shr_op)), llvm::Instruction::LShr}
	, {std::type_index(typeid(jive::bits::or_op)), llvm::Instruction::Or}
	, {std::type_index(typeid(jive::bits::xor_op)), llvm::Instruction::Xor}
	, {std::type_index(typeid(jive::bits::mul_op)), llvm::Instruction::Mul}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateBinOp(map[std::type_index(typeid(op))], op1, op2);
}

static inline llvm::Value *
convert_bitscompare(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::compare_op*>(&op));

	static std::unordered_map<std::type_index, llvm::CmpInst::Predicate> map({
	  {std::type_index(typeid(jive::bits::eq_op)), llvm::CmpInst::ICMP_EQ}
	, {std::type_index(typeid(jive::bits::ne_op)), llvm::CmpInst::ICMP_NE}
	, {std::type_index(typeid(jive::bits::ugt_op)), llvm::CmpInst::ICMP_UGT}
	, {std::type_index(typeid(jive::bits::uge_op)), llvm::CmpInst::ICMP_UGE}
	, {std::type_index(typeid(jive::bits::ult_op)), llvm::CmpInst::ICMP_ULT}
	, {std::type_index(typeid(jive::bits::ule_op)), llvm::CmpInst::ICMP_ULE}
	, {std::type_index(typeid(jive::bits::sgt_op)), llvm::CmpInst::ICMP_SGT}
	, {std::type_index(typeid(jive::bits::sge_op)), llvm::CmpInst::ICMP_SGE}
	, {std::type_index(typeid(jive::bits::slt_op)), llvm::CmpInst::ICMP_SLT}
	, {std::type_index(typeid(jive::bits::sle_op)), llvm::CmpInst::ICMP_SLE}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateICmp(map[std::type_index(typeid(op))], op1, op2);
}

static inline llvm::Value *
convert_bitconstant(
	const jive::operation & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
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
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::ctl::constant_op*>(&op));
	auto & cop = *static_cast<const jive::ctl::constant_op*>(&op);
	JLM_DEBUG_ASSERT(cop.value().nalternatives() == 2);

	auto type = llvm::IntegerType::get(builder.getContext(), 1);
	return llvm::ConstantInt::get(type, cop.value().alternative() == 0);
}

static inline llvm::Value *
convert_fpconstant(
	const jive::operation & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_fpconstant_op(op));
	auto & cop = *static_cast<const jlm::fpconstant_op*>(&op);

	auto type = convert_type(cop.result(0).type(), ctx);
	return llvm::ConstantFP::get(type, cop.constant());
}

static inline llvm::Value *
convert_undef(
	const jive::operation & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_undef_constant_op(op));
	return llvm::UndefValue::get(convert_type(op.result(0).type(), ctx));
}

static inline llvm::Value *
convert_apply(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::apply_op*>(&op));
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::mem::type*>(&args[args.size()-1]->type()));

	auto function = ctx.value(args[0]);
	std::vector<llvm::Value*> operands;
	for (size_t n = 1; n < args.size()-1; n++) {
		auto argument = args[n];
		if (is_varargtype(argument->type())) {
			JLM_DEBUG_ASSERT(is_tacvariable(argument));
			auto valist = dynamic_cast<const jlm::tacvariable*>(argument)->tac();
			JLM_DEBUG_ASSERT(is_valist_op(valist->operation()));
			for (size_t n = 0; n < valist->ninputs(); n++)
				operands.push_back(ctx.value(valist->input(n)));
			continue;
		}

		operands.push_back(ctx.value(argument));
	}

	return builder.CreateCall(function, operands);
}

static inline llvm::Value *
convert_match(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::ctl::match_op*>(&op));
	return ctx.value(args[0]);
}

static inline llvm::Value *
convert_branch(
	const jive::operation & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_branch_op(op));
	return nullptr;
}

static inline llvm::Value *
convert_phi(
	const jive::operation & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_phi_op(op));
	auto & pop = *static_cast<const jlm::phi_op*>(&op);

	if (dynamic_cast<const jive::mem::type*>(&pop.type()))
		return nullptr;

	auto t = convert_type(pop.type(), ctx);
	return builder.CreatePHI(t, op.narguments());
}

static inline llvm::Value *
convert_load(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_load_op(op));
	return builder.CreateLoad(ctx.value(args[0]));
}

static inline llvm::Value *
convert_store(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_store_op(op) && args.size() >= 2);
	builder.CreateStore(ctx.value(args[1]), ctx.value(args[0]));
	return nullptr;
}

static inline llvm::Value *
convert_alloca(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_alloca_op(op) && args.size() == 2);
	auto & aop = *static_cast<const jlm::alloca_op*>(&op);

	auto t = convert_type(aop.value_type(), ctx);
	return builder.CreateAlloca(t, ctx.value(args[0]));
}

static inline llvm::Value *
convert_ptroffset(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_ptroffset_op(op) && args.size() >= 2);
	auto & pop = *static_cast<const ptroffset_op*>(&op);

	std::vector<llvm::Value*> indices;
	auto t = convert_type(pop.pointee_type(), ctx);
	for (size_t n = 1; n < args.size(); n++)
		indices.push_back(ctx.value(args[n]));

	return builder.CreateGEP(t, ctx.value(args[0]), indices);
}

static inline llvm::Value *
convert_data_array_constant(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_data_array_constant_op(op));
	auto & cop = *static_cast<const data_array_constant_op*>(&op);

	/* FIXME: support other types */
	auto bt = dynamic_cast<const jive::bits::type*>(&cop.type());
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bits::type*>(&cop.type()));
	JLM_DEBUG_ASSERT(bt->nbits() == 8);

	std::vector<uint8_t> data;
	for (size_t n = 0; n < args.size(); n++) {
		auto c = llvm::dyn_cast<const llvm::ConstantInt>(ctx.value(args[n]));
		JLM_DEBUG_ASSERT(c);
		data.push_back(c->getZExtValue());
	}

	return llvm::ConstantDataArray::get(builder.getContext(), data);
}

static inline llvm::Value *
convert_ptrcmp(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_ptrcmp_op(op));
	auto & pop = *static_cast<const ptrcmp_op*>(&op);

	static std::unordered_map<jlm::cmp, llvm::CmpInst::Predicate> map({
	  {cmp::le, llvm::CmpInst::ICMP_ULE}, {cmp::lt, llvm::CmpInst::ICMP_ULT}
	, {cmp::eq, llvm::CmpInst::ICMP_EQ}, {cmp::ne, llvm::CmpInst::ICMP_NE}
	, {cmp::ge, llvm::CmpInst::ICMP_UGE}, {cmp::gt, llvm::CmpInst::ICMP_UGT}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(pop.cmp()) != map.end());
	return builder.CreateICmp(map[pop.cmp()], op1, op2);
}

static inline llvm::Value *
convert_zext(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_zext_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateZExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_fpcmp(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_fpcmp_op(op));
	auto & fpcmp = *static_cast<const jlm::fpcmp_op*>(&op);

	static std::unordered_map<jlm::fpcmp, llvm::CmpInst::Predicate> map({
	  {fpcmp::oeq, llvm::CmpInst::FCMP_OEQ}, {fpcmp::ogt, llvm::CmpInst::FCMP_OGT}
	, {fpcmp::oge, llvm::CmpInst::FCMP_OGE}, {fpcmp::olt, llvm::CmpInst::FCMP_OLT}
	, {fpcmp::ole, llvm::CmpInst::FCMP_OLE}, {fpcmp::one, llvm::CmpInst::FCMP_ONE}
	, {fpcmp::ord, llvm::CmpInst::FCMP_ORD}, {fpcmp::uno, llvm::CmpInst::FCMP_UNO}
	, {fpcmp::ueq, llvm::CmpInst::FCMP_UEQ}, {fpcmp::ugt, llvm::CmpInst::FCMP_UGT}
	, {fpcmp::uge, llvm::CmpInst::FCMP_UGE}, {fpcmp::ult, llvm::CmpInst::FCMP_ULT}
	, {fpcmp::ule, llvm::CmpInst::FCMP_ULE}, {fpcmp::une, llvm::CmpInst::FCMP_UNE}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(fpcmp.cmp()) != map.end());
	return builder.CreateFCmp(map[fpcmp.cmp()], op1, op2);
}

static inline llvm::Value *
convert_fpbin(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_fpbin_op(op));
	auto & fpbin = *static_cast<const jlm::fpbin_op*>(&op);

	static std::unordered_map<jlm::fpop, llvm::Instruction::BinaryOps> map({
	  {fpop::add, llvm::Instruction::FAdd}, {fpop::sub, llvm::Instruction::FSub}
	, {fpop::mul, llvm::Instruction::FMul}, {fpop::div, llvm::Instruction::FDiv}
	, {fpop::mod, llvm::Instruction::FRem}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(fpbin.fpop()) != map.end());
	return builder.CreateBinOp(map[fpbin.fpop()], op1, op2);
}

static inline llvm::Value *
convert_fpext(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_fpext_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateFPExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_valist(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_valist_op(op));
	return nullptr;
}

static inline llvm::Value *
convert_bitcast(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_bitcast_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateBitCast(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_struct_constant(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_struct_constant_op(op));
	auto & cop = *static_cast<const jlm::struct_constant_op*>(&op);

	std::vector<llvm::Constant*> operands;
	for (const auto & arg : args)
		operands.push_back(llvm::cast<llvm::Constant>(ctx.value(arg)));

	auto t = convert_type(cop.type(), ctx);
	return llvm::ConstantStruct::get(t, operands);
}

static inline llvm::Value *
convert_trunc(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_trunc_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateTrunc(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_sext(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_sext_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateSExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_sitofp(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_sitofp_op(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateSIToFP(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_ptr_constant_null(
	const jive::operation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_ptr_constant_null_op(op));

	auto type = static_cast<const jlm::ptrtype*>(&op.result(0).type());
	return llvm::ConstantPointerNull::get(convert_type(*type, ctx));
}

llvm::Value *
convert_operation(
	const jive::operation & op,
	const std::vector<const variable*> & arguments,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	if (dynamic_cast<const jive::bits::binary_op*>(&op))
		return convert_bitsbinary(op, arguments, builder, ctx);

	if (dynamic_cast<const jive::bits::compare_op*>(&op))
		return convert_bitscompare(op, arguments, builder, ctx);

	static std::unordered_map<
		std::type_index
	, llvm::Value*(*)(
			const jive::operation &,
			const std::vector<const variable*> &,
			llvm::IRBuilder<> &,
			context & ctx)
	> map({
	  {std::type_index(typeid(jive::bits::constant_op)), convert_bitconstant}
	, {std::type_index(typeid(jive::ctl::constant_op)), convert_ctlconstant}
	, {std::type_index(typeid(jlm::fpconstant_op)), convert_fpconstant}
	, {std::type_index(typeid(jlm::undef_constant_op)), convert_undef}
	, {std::type_index(typeid(jive::ctl::match_op)), convert_match}
	, {std::type_index(typeid(jive::fct::apply_op)), convert_apply}
	, {std::type_index(typeid(jlm::assignment_op)), convert_assignment}
	, {std::type_index(typeid(jlm::branch_op)), convert_branch}
	, {std::type_index(typeid(jlm::phi_op)), convert_phi}
	, {std::type_index(typeid(jlm::load_op)), convert_load}
	, {std::type_index(typeid(jlm::store_op)), convert_store}
	, {std::type_index(typeid(jlm::alloca_op)), convert_alloca}
	, {std::type_index(typeid(jlm::ptroffset_op)), convert_ptroffset}
	, {std::type_index(typeid(jlm::data_array_constant_op)), convert_data_array_constant}
	, {std::type_index(typeid(jlm::ptrcmp_op)), convert_ptrcmp}
	, {std::type_index(typeid(jlm::zext_op)), convert_zext}
	, {std::type_index(typeid(jlm::fpcmp_op)), convert_fpcmp}
	, {std::type_index(typeid(jlm::fpbin_op)), convert_fpbin}
	, {std::type_index(typeid(jlm::fpext_op)), convert_fpext}
	, {std::type_index(typeid(jlm::valist_op)), convert_valist}
	, {std::type_index(typeid(jlm::bitcast_op)), convert_bitcast}
	, {std::type_index(typeid(jlm::struct_constant_op)), convert_struct_constant}
	, {std::type_index(typeid(jlm::trunc_op)), convert_trunc}
	, {std::type_index(typeid(jlm::sext_op)), convert_sext}
	, {std::type_index(typeid(jlm::sitofp_op)), convert_sitofp}
	, {std::type_index(typeid(jlm::ptr_constant_null_op)), convert_ptr_constant_null}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return map[std::type_index(typeid(op))](op, arguments, builder, ctx);
}

void
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	std::vector<const variable*> operands;
	for (size_t n = 0; n < tac.ninputs(); n++)
		operands.push_back(tac.input(n));

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto r = convert_operation(tac.operation(), operands, builder, ctx);
	if (r != nullptr) ctx.insert(tac.output(0), r);
}

static inline std::vector<std::unique_ptr<jlm::tac>>
expr2tacs(const jlm::expr & e, const context & ctx)
{
	std::function<const jlm::variable *(
		const jlm::expr &,
		const jlm::variable *,
		std::vector<std::unique_ptr<jlm::tac>>&)
	> append = [&](
		const jlm::expr & e,
		const jlm::variable * result,
		std::vector<std::unique_ptr<jlm::tac>> & tacs)
	{
		std::vector<const jlm::variable *> operands;
		for (size_t n = 0; n < e.noperands(); n++) {
			auto v = ctx.jlm_module().create_variable(e.operand(n).type(), false);
			operands.push_back(append(e.operand(n), v, tacs));
		}

		tacs.emplace_back(create_tac(e.operation(), operands, {result}));
		return tacs.back()->output(0);
	};

	std::vector<std::unique_ptr<jlm::tac>> tacs;
	append(e, ctx.jlm_module().create_variable(e.type(), false), tacs);
	return tacs;
}

llvm::Constant *
convert_expression(const jlm::expr & e, context & ctx)
{
	auto tacs = expr2tacs(e, ctx);

	llvm::Value * r = nullptr;
	llvm::IRBuilder<> builder(ctx.llvm_module().getContext());
	for (const auto & tac : tacs) {
		std::vector<const variable*> operands;
		for (size_t n = 0; n < tac->ninputs(); n++)
			operands.push_back(tac->input(n));

		JLM_DEBUG_ASSERT(tac->noutputs() == 1);
		r = convert_operation(tac->operation(), operands, builder, ctx);
		ctx.insert(tac->output(0), r);
	}

	return llvm::dyn_cast<llvm::Constant>(r);
}

}}
