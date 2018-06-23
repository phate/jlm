/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/jlm/ir/cfg-node.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators.hpp>
#include <jlm/jlm/ir/tac.hpp>

#include <jlm/jlm/jlm2llvm/context.hpp>
#include <jlm/jlm/jlm2llvm/instruction.hpp>
#include <jlm/jlm/jlm2llvm/type.hpp>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace jlm {
namespace jlm2llvm {

static inline llvm::Value *
convert_assignment(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<assignment_op>(op));
	return ctx.value(args[0]);
}

static inline llvm::Value *
convert_bitsbinary(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bitbinary_op*>(&op));

	static std::unordered_map<std::type_index, llvm::Instruction::BinaryOps> map({
	  {typeid(jive::bitadd_op), llvm::Instruction::Add}
	, {typeid(jive::bitand_op), llvm::Instruction::And}
	, {typeid(jive::bitashr_op), llvm::Instruction::AShr}
	, {typeid(jive::bitsub_op), llvm::Instruction::Sub}
	, {typeid(jive::bitudiv_op), llvm::Instruction::UDiv}
	, {typeid(jive::bitsdiv_op), llvm::Instruction::SDiv}
	, {typeid(jive::bitumod_op), llvm::Instruction::URem}
	, {typeid(jive::bitsmod_op), llvm::Instruction::SRem}
	, {typeid(jive::bitshl_op), llvm::Instruction::Shl}
	, {typeid(jive::bitshr_op), llvm::Instruction::LShr}
	, {typeid(jive::bitor_op), llvm::Instruction::Or}
	, {typeid(jive::bitxor_op), llvm::Instruction::Xor}
	, {typeid(jive::bitmul_op), llvm::Instruction::Mul}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateBinOp(map[std::type_index(typeid(op))], op1, op2);
}

static inline llvm::Value *
convert_bitscompare(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bitcompare_op*>(&op));

	static std::unordered_map<std::type_index, llvm::CmpInst::Predicate> map({
	  {typeid(jive::biteq_op), llvm::CmpInst::ICMP_EQ}
	, {typeid(jive::bitne_op), llvm::CmpInst::ICMP_NE}
	, {typeid(jive::bitugt_op), llvm::CmpInst::ICMP_UGT}
	, {typeid(jive::bituge_op), llvm::CmpInst::ICMP_UGE}
	, {typeid(jive::bitult_op), llvm::CmpInst::ICMP_ULT}
	, {typeid(jive::bitule_op), llvm::CmpInst::ICMP_ULE}
	, {typeid(jive::bitsgt_op), llvm::CmpInst::ICMP_SGT}
	, {typeid(jive::bitsge_op), llvm::CmpInst::ICMP_SGE}
	, {typeid(jive::bitslt_op), llvm::CmpInst::ICMP_SLT}
	, {typeid(jive::bitsle_op), llvm::CmpInst::ICMP_SLE}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateICmp(map[std::type_index(typeid(op))], op1, op2);
}

static inline llvm::Value *
convert_bitconstant(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::bitconstant_op*>(&op));
	auto & cop = *static_cast<const jive::bitconstant_op*>(&op);
	auto type = llvm::IntegerType::get(builder.getContext(), cop.value().nbits());

	if (cop.value().is_defined())
		return llvm::ConstantInt::get(type, cop.value().to_uint());

	return llvm::UndefValue::get(type);
}

static inline llvm::Value *
convert_ctlconstant(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is_ctlconstant_op(op));
	auto & cop = *static_cast<const jive::ctlconstant_op*>(&op);

	size_t nbits = cop.value().nalternatives() == 2 ? 1 : 32;
	auto type = llvm::IntegerType::get(builder.getContext(), nbits);
	return llvm::ConstantInt::get(type, cop.value().alternative());
}

static inline llvm::Value *
convert_fpconstant(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fpconstant_op>(op));
	auto & cop = *static_cast<const jlm::fpconstant_op*>(&op);

	return llvm::ConstantFP::get(builder.getContext(), cop.constant());
}

static inline llvm::Value *
convert_undef(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<undef_constant_op>(op));
	return llvm::UndefValue::get(convert_type(op.result(0).type(), ctx));
}

static inline llvm::Value *
convert_call(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<call_op>(op));
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::memtype*>(&args[args.size()-1]->type()));

	auto function = ctx.value(args[0]);
	std::vector<llvm::Value*> operands;
	for (size_t n = 1; n < args.size()-1; n++) {
		auto argument = args[n];
		if (is_varargtype(argument->type())) {
			JLM_DEBUG_ASSERT(is_tacvariable(argument));
			auto valist = dynamic_cast<const jlm::tacvariable*>(argument)->tac();
			JLM_DEBUG_ASSERT(is<valist_op>(valist->operation()));
			for (size_t n = 0; n < valist->ninputs(); n++)
				operands.push_back(ctx.value(valist->input(n)));
			continue;
		}

		operands.push_back(ctx.value(argument));
	}

	return builder.CreateCall(function, operands);
}

static inline bool
is_identity_mapping(const jive::match_op & op)
{
	for (const auto & pair : op) {
		if (pair.first != pair.second)
			return false;
	}

	return true;
}

static inline llvm::Value *
convert_match(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<jive::match_op>(op));
	auto mop = static_cast<const jive::match_op*>(&op);

	if (is_identity_mapping(*mop))
		return ctx.value(args[0]);

	if (mop->nalternatives() == 2) {
		JLM_DEBUG_ASSERT(mop->nbits() == 1);
		auto i2 = llvm::IntegerType::get(builder.getContext(), 2);
		auto t = llvm::ConstantInt::getFalse(i2);
		auto f = llvm::ConstantInt::getTrue(i2);
		return builder.CreateSelect(ctx.value(args[0]), t, f);
	}

	/* FIXME: This is not working if the match is not directly connected to a gamma node. */
	return ctx.value(args[0]);
}

static inline llvm::Value *
convert_branch(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<branch_op>(op));
	return nullptr;
}

static inline llvm::Value *
convert_phi(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<phi_op>(op));
	auto & pop = *static_cast<const jlm::phi_op*>(&op);

	if (dynamic_cast<const jive::memtype*>(&pop.type()))
		return nullptr;

	auto t = convert_type(pop.type(), ctx);
	return builder.CreatePHI(t, op.narguments());
}

static inline llvm::Value *
convert_load(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<load_op>(op));
	auto load = static_cast<const load_op*>(&op);

	auto i = builder.CreateLoad(ctx.value(args[0]));
	i->setAlignment(load->alignment());
	return i;
}

static inline llvm::Value *
convert_store(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<store_op>(op) && args.size() >= 2);
	auto store = static_cast<const store_op*>(&op);

	auto i = builder.CreateStore(ctx.value(args[1]), ctx.value(args[0]));
	i->setAlignment(store->alignment());
	return nullptr;
}

static inline llvm::Value *
convert_alloca(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<alloca_op>(op) && args.size() == 2);
	auto & aop = *static_cast<const jlm::alloca_op*>(&op);

	auto t = convert_type(aop.value_type(), ctx);
	auto i = builder.CreateAlloca(t, ctx.value(args[0]));
	i->setAlignment(aop.alignment());
	return i;
}

static inline llvm::Value *
convert_getelementptr(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<getelementptr_op>(op) && args.size() >= 2);
	auto & pop = *static_cast<const getelementptr_op*>(&op);

	std::vector<llvm::Value*> indices;
	auto t = convert_type(pop.pointee_type(), ctx);
	for (size_t n = 1; n < args.size(); n++)
		indices.push_back(ctx.value(args[n]));

	return builder.CreateGEP(t, ctx.value(args[0]), indices);
}

template<typename T> static std::vector<T>
get_bitdata(
	const std::vector<const variable*> & args,
	context & ctx)
{
	std::vector<T> data;
	for (size_t n = 0; n < args.size(); n++) {
		auto c = llvm::dyn_cast<const llvm::ConstantInt>(ctx.value(args[n]));
		JLM_DEBUG_ASSERT(c);
		data.push_back(c->getZExtValue());
	}

	return data;
}

template<typename T> static std::vector<T>
get_fpdata(
	const std::vector<const variable*> & args,
	context & ctx)
{
	std::vector<T> data;
	for (size_t n = 0; n < args.size(); n++) {
		auto c = llvm::dyn_cast<const llvm::ConstantFP>(ctx.value(args[n]));
		JLM_DEBUG_ASSERT(c);
		data.push_back(c->getValueAPF().bitcastToAPInt().getZExtValue());
	}

	return data;
}

static inline llvm::Value *
convert_data_array_constant(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<data_array_constant_op>(op));
	auto & cop = *static_cast<const data_array_constant_op*>(&op);

	if (auto bt = dynamic_cast<const jive::bittype*>(&cop.type())) {
		if (bt->nbits() == 8) {
			auto data = get_bitdata<uint8_t>(args, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 16) {
			auto data = get_bitdata<uint16_t>(args, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 32) {
			auto data = get_bitdata<uint32_t>(args, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 64) {
			auto data = get_bitdata<uint64_t>(args, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else
			JLM_ASSERT(0);
	}

	if (auto ft = dynamic_cast<const fptype*>(&cop.type())) {
		if (ft->size() == fpsize::half) {
			auto data = get_fpdata<uint16_t>(args, ctx);
			return llvm::ConstantDataArray::getFP(builder.getContext(), data);
		} else if (ft->size() == fpsize::flt) {
			auto data = get_fpdata<uint32_t>(args, ctx);
			return llvm::ConstantDataArray::getFP(builder.getContext(), data);
		} else if (ft->size() == fpsize::dbl) {
			auto data = get_fpdata<uint64_t>(args, ctx);
			return llvm::ConstantDataArray::getFP(builder.getContext(), data);
		} else
			JLM_ASSERT(0);
	}

	JLM_ASSERT(0);
}

static inline llvm::Value *
convert_constant_array(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<constant_array_op>(op));

	std::vector<llvm::Constant*> data;
	for (size_t n = 0; n < args.size(); n++) {
		auto c = llvm::dyn_cast<llvm::Constant>(ctx.value(args[n]));
		JLM_DEBUG_ASSERT(c);
		data.push_back(c);
	}

	auto at = dynamic_cast<const arraytype*>(&op.result(0).type());
	auto type = convert_type(*at, ctx);
	return llvm::ConstantArray::get(type, data);
}

static llvm::Value *
convert_constant_aggregate_zero(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<constant_aggregate_zero_op>(op));
	auto type = convert_type(op.result(0).type(), ctx);
	return llvm::ConstantAggregateZero::get(type);
}

static inline llvm::Value *
convert_ptrcmp(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<ptrcmp_op>(op));
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
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<zext_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateZExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_fpcmp(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fpcmp_op>(op));
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
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fpbin_op>(op));
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
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fpext_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateFPExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_fptrunc(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fptrunc_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateFPTrunc(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_fp2ui(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fp2ui_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateFPToUI(ctx.value(args[0]), type);
}

static llvm::Value *
convert_fp2si(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<fp2si_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateFPToSI(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_valist(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<valist_op>(op));
	return nullptr;
}

static inline llvm::Value *
convert_bitcast(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<bitcast_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateBitCast(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_struct_constant(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<struct_constant_op>(op));
	auto & cop = *static_cast<const jlm::struct_constant_op*>(&op);

	std::vector<llvm::Constant*> operands;
	for (const auto & arg : args)
		operands.push_back(llvm::cast<llvm::Constant>(ctx.value(arg)));

	auto t = convert_type(cop.type(), ctx);
	return llvm::ConstantStruct::get(t, operands);
}

static inline llvm::Value *
convert_trunc(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<trunc_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateTrunc(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_sext(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<sext_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateSExt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_sitofp(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<sitofp_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateSIToFP(ctx.value(args[0]), type);
}

static llvm::Value *
convert_uitofp(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<uitofp_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreateUIToFP(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_ptr_constant_null(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<ptr_constant_null_op>(op));

	auto type = static_cast<const jlm::ptrtype*>(&op.result(0).type());
	return llvm::ConstantPointerNull::get(convert_type(*type, ctx));
}

static inline llvm::Value *
convert_select(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<select_op>(op));

	auto c = ctx.value(operands[0]);
	auto t = ctx.value(operands[1]);
	auto f = ctx.value(operands[2]);
	return builder.CreateSelect(c, t, f);
}

static inline llvm::Value *
convert_mux(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<jive::mux_op>(op));
	return nullptr;
}

static inline llvm::Value *
convert_ptr2bits(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<ptr2bits_op>(op));

	auto type = convert_type(op.result(0).type(), ctx);
	return builder.CreatePtrToInt(ctx.value(args[0]), type);
}

static inline llvm::Value *
convert_ctl2bits(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<ctl2bits_op>(op));
	return ctx.value(args[0]);
}

static llvm::Value *
convert_constantvector(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<constantvector_op>(op));

	std::vector<llvm::Constant*> ops;
	for (const auto & operand: operands)
		ops.push_back(llvm::cast<llvm::Constant>(ctx.value(operand)));

	return llvm::ConstantVector::get(ops);
}

static llvm::Value *
convert_constantdatavector(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_DEBUG_ASSERT(is<constant_data_vector_op>(op));
	auto & cop = *static_cast<const constant_data_vector_op*>(&op);

	if (auto bt = dynamic_cast<const jive::bittype*>(&cop.type())) {
		if (bt->nbits() == 8) {
			auto data = get_bitdata<uint8_t>(operands, ctx);
			return llvm::ConstantDataVector::get(builder.getContext(), data);
		} else if (bt->nbits() == 16) {
			auto data = get_bitdata<uint16_t>(operands, ctx);
			return llvm::ConstantDataVector::get(builder.getContext(), data);
		} else if (bt->nbits() == 32) {
			auto data = get_bitdata<uint32_t>(operands, ctx);
			return llvm::ConstantDataVector::get(builder.getContext(), data);
		} else if (bt->nbits() == 64) {
			auto data = get_bitdata<uint64_t>(operands, ctx);
			return llvm::ConstantDataVector::get(builder.getContext(), data);
		} else
			JLM_ASSERT(0);
	}

	if (auto ft = dynamic_cast<const fptype*>(&cop.type())) {
		if (ft->size() == fpsize::half) {
			auto data = get_fpdata<uint16_t>(operands, ctx);
			return llvm::ConstantDataVector::getFP(builder.getContext(), data);
		} else if (ft->size() == fpsize::flt) {
			auto data = get_fpdata<uint32_t>(operands, ctx);
			return llvm::ConstantDataVector::getFP(builder.getContext(), data);
		} else if (ft->size() == fpsize::dbl) {
			auto data = get_fpdata<uint64_t>(operands, ctx);
			return llvm::ConstantDataVector::getFP(builder.getContext(), data);
		} else
		JLM_ASSERT(0);
	}

	JLM_ASSERT(0);
}

llvm::Value *
convert_operation(
	const jive::simple_op & op,
	const std::vector<const variable*> & arguments,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	if (dynamic_cast<const jive::bitbinary_op*>(&op))
		return convert_bitsbinary(op, arguments, builder, ctx);

	if (dynamic_cast<const jive::bitcompare_op*>(&op))
		return convert_bitscompare(op, arguments, builder, ctx);

	static std::unordered_map<
		std::type_index
	, llvm::Value*(*)(
			const jive::simple_op &,
			const std::vector<const variable*> &,
			llvm::IRBuilder<> &,
			context & ctx)
	> map({
	  {typeid(jive::bitconstant_op), convert_bitconstant}
	, {typeid(jive::ctlconstant_op), convert_ctlconstant}
	, {std::type_index(typeid(jlm::fpconstant_op)), convert_fpconstant}
	, {std::type_index(typeid(jlm::undef_constant_op)), convert_undef}
	, {typeid(jive::match_op), convert_match}
	, {typeid(call_op), convert_call}
	, {std::type_index(typeid(jlm::assignment_op)), convert_assignment}
	, {std::type_index(typeid(jlm::branch_op)), convert_branch}
	, {std::type_index(typeid(jlm::phi_op)), convert_phi}
	, {std::type_index(typeid(jlm::load_op)), convert_load}
	, {std::type_index(typeid(jlm::store_op)), convert_store}
	, {std::type_index(typeid(jlm::alloca_op)), convert_alloca}
	, {typeid(jlm::getelementptr_op), convert_getelementptr}
	, {std::type_index(typeid(jlm::data_array_constant_op)), convert_data_array_constant}
	, {std::type_index(typeid(jlm::ptrcmp_op)), convert_ptrcmp}
	, {std::type_index(typeid(jlm::zext_op)), convert_zext}
	, {std::type_index(typeid(jlm::fpcmp_op)), convert_fpcmp}
	, {std::type_index(typeid(jlm::fpbin_op)), convert_fpbin}
	, {std::type_index(typeid(jlm::fpext_op)), convert_fpext}
	, {typeid(fptrunc_op), convert_fptrunc}
	, {typeid(fp2ui_op), convert_fp2ui}
	, {typeid(fp2si_op), convert_fp2si}
	, {std::type_index(typeid(jlm::valist_op)), convert_valist}
	, {std::type_index(typeid(jlm::bitcast_op)), convert_bitcast}
	, {std::type_index(typeid(jlm::struct_constant_op)), convert_struct_constant}
	, {std::type_index(typeid(jlm::trunc_op)), convert_trunc}
	, {std::type_index(typeid(jlm::sext_op)), convert_sext}
	, {std::type_index(typeid(jlm::sitofp_op)), convert_sitofp}
	, {typeid(jlm::uitofp_op), convert_uitofp}
	, {std::type_index(typeid(jlm::ptr_constant_null_op)), convert_ptr_constant_null}
	, {std::type_index(typeid(jlm::select_op)), convert_select}
	, {std::type_index(typeid(jive::mux_op)), convert_mux}
	, {typeid(jlm::constant_array_op), convert_constant_array}
	, {typeid(constant_aggregate_zero_op), convert_constant_aggregate_zero}
	, {typeid(ptr2bits_op), convert_ptr2bits}
	, {typeid(ctl2bits_op), convert_ctl2bits}
	, {typeid(constantvector_op), convert_constantvector}
	, {typeid(constant_data_vector_op), convert_constantdatavector}
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

llvm::Constant *
convert_tacs(const tacsvector_t & tacs, context & ctx)
{
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
