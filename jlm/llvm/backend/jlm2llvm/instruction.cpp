/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/control.hpp>

#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/alias-analyses/Operators.hpp>

#include <jlm/llvm/backend/jlm2llvm/context.hpp>
#include <jlm/llvm/backend/jlm2llvm/instruction.hpp>
#include <jlm/llvm/backend/jlm2llvm/type.hpp>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace jlm {
namespace jlm2llvm {

llvm::Value *
convert_operation(
	const jive::simple_op & op,
	const std::vector<const variable*> & arguments,
	llvm::IRBuilder<> & builder,
	context & ctx);

static inline llvm::Value *
convert_assignment(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<assignment_op>(op));
	return ctx.value(args[0]);
}

static inline llvm::Value *
convert_bitsbinary(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(dynamic_cast<const jive::bitbinary_op*>(&op));

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
	JLM_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateBinOp(map[std::type_index(typeid(op))], op1, op2);
}

static inline llvm::Value *
convert_bitscompare(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(dynamic_cast<const jive::bitcompare_op*>(&op));

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
	JLM_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateICmp(map[std::type_index(typeid(op))], op1, op2);
}

static llvm::APInt
convert_bitvalue_repr(const jive::bitvalue_repr & vr)
{
	JLM_ASSERT(vr.is_defined());

	std::string str = vr.str();
	std::reverse(str.begin(), str.end());

	return llvm::APInt(vr.nbits(), str, 2);
}

static inline llvm::Value *
convert_bitconstant(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(dynamic_cast<const jive::bitconstant_op*>(&op));
	auto value = static_cast<const jive::bitconstant_op*>(&op)->value();

	auto type = llvm::IntegerType::get(builder.getContext(), value.nbits());

	if (value.is_defined())
		return llvm::ConstantInt::get(type, convert_bitvalue_repr(value));

	return llvm::UndefValue::get(type);
}

static inline llvm::Value *
convert_ctlconstant(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is_ctlconstant_op(op));
	auto & cop = *static_cast<const jive::ctlconstant_op*>(&op);

	size_t nbits = cop.value().nalternatives() == 2 ? 1 : 32;
	auto type = llvm::IntegerType::get(builder.getContext(), nbits);
	return llvm::ConstantInt::get(type, cop.value().alternative());
}

static llvm::Value *
convert(
	const ConstantFP & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	return llvm::ConstantFP::get(builder.getContext(), op.constant());
}

static inline llvm::Value *
convert_undef(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<UndefValueOperation>(op));
	return llvm::UndefValue::get(convert_type(op.result(0).type(), ctx));
}

static llvm::Value *
convert(
  const PoisonValueOperation & operation,
  const std::vector<const variable*>&,
  llvm::IRBuilder<>&,
  context & ctx)
{
  return llvm::PoisonValue::get(convert_type(operation.GetType(), ctx));
}

static llvm::Value *
convert(
	const CallOperation & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto function = ctx.value(args[0]);

	std::vector<llvm::Value*> operands;
	for (size_t n = 1; n < args.size(); n++) {
		auto argument = args[n];

		if (jive::is<iostatetype>(argument->type()))
			continue;
		if (jive::is<MemoryStateType>(argument->type()))
			continue;
		if (jive::is<loopstatetype>(argument->type()))
			continue;

		if (jive::is<varargtype>(argument->type())) {
			JLM_ASSERT(is<tacvariable>(argument));
			auto valist = dynamic_cast<const jlm::tacvariable*>(argument)->tac();
			JLM_ASSERT(is<valist_op>(valist->operation()));
			for (size_t n = 0; n < valist->noperands(); n++)
				operands.push_back(ctx.value(valist->operand(n)));
			continue;
		}

		operands.push_back(ctx.value(argument));
	}

	auto ftype = convert_type(op.GetFunctionType(), ctx);
	return builder.CreateCall(ftype, function, operands);
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
	JLM_ASSERT(is<jive::match_op>(op));
	auto mop = static_cast<const jive::match_op*>(&op);

	if (is_identity_mapping(*mop))
		return ctx.value(args[0]);

	if (mop->nalternatives() == 2 && mop->nbits() == 1) {
		auto i1 = llvm::IntegerType::get(builder.getContext(), 1);
		auto t = llvm::ConstantInt::getFalse(i1);
		auto f = llvm::ConstantInt::getTrue(i1);
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
	JLM_ASSERT(is<branch_op>(op));
	return nullptr;
}

static inline llvm::Value *
convert_phi(
	const jive::simple_op & op,
	const std::vector<const variable*> &,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<phi_op>(op));
	auto & phi = *static_cast<const jlm::phi_op*>(&op);

	if (jive::is<iostatetype>(phi.type()))
		return nullptr;
	if (jive::is<MemoryStateType>(phi.type()))
		return nullptr;
	if (jive::is<loopstatetype>(phi.type()))
		return nullptr;

	auto t = convert_type(phi.type(), ctx);
	return builder.CreatePHI(t, op.narguments());
}

static llvm::Value *
convert(
	const LoadOperation & operation,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
    auto type = convert_type(operation.GetLoadedType(), ctx);
    auto loadInstruction = builder.CreateLoad(type, ctx.value(args[0]));
    loadInstruction->setAlignment(llvm::Align(operation.GetAlignment()));
    return loadInstruction;
}

static inline llvm::Value *
convert_store(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<StoreOperation>(op) && args.size() >= 2);
	auto store = static_cast<const StoreOperation*>(&op);

	auto i = builder.CreateStore(ctx.value(args[1]), ctx.value(args[0]));
	i->setAlignment(llvm::Align(store->GetAlignment()));
	return nullptr;
}

static inline llvm::Value *
convert_alloca(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<alloca_op>(op));
	auto & aop = *static_cast<const jlm::alloca_op*>(&op);

	auto t = convert_type(aop.value_type(), ctx);
	auto i = builder.CreateAlloca(t, ctx.value(args[0]));
	i->setAlignment(llvm::Align(aop.alignment()));
	return i;
}

static inline llvm::Value *
convert_getelementptr(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<GetElementPtrOperation>(op) && args.size() >= 2);
	auto & pop = *static_cast<const GetElementPtrOperation*>(&op);

	std::vector<llvm::Value*> indices;
	auto t = convert_type(pop.GetPointeeType(), ctx);
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
		JLM_ASSERT(c);
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
		JLM_ASSERT(c);
		data.push_back(c->getValueAPF().bitcastToAPInt().getZExtValue());
	}

	return data;
}

static llvm::Value *
convert(
	const ConstantDataArray & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<ConstantDataArray>(op));

	if (auto bt = dynamic_cast<const jive::bittype*>(&op.type())) {
		if (bt->nbits() == 8) {
			auto data = get_bitdata<uint8_t>(operands, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 16) {
			auto data = get_bitdata<uint16_t>(operands, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 32) {
			auto data = get_bitdata<uint32_t>(operands, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		} else if (bt->nbits() == 64) {
			auto data = get_bitdata<uint64_t>(operands, ctx);
			return llvm::ConstantDataArray::get(builder.getContext(), data);
		}
	}

	if (auto ft = dynamic_cast<const fptype*>(&op.type())) {
		if (ft->size() == fpsize::half) {
			auto data = get_fpdata<uint16_t>(operands, ctx);
			auto type = llvm::Type::getBFloatTy(builder.getContext());
			return llvm::ConstantDataArray::getFP(type, data);
		} else if (ft->size() == fpsize::flt) {
			auto data = get_fpdata<uint32_t>(operands, ctx);
			auto type = llvm::Type::getFloatTy(builder.getContext());
			return llvm::ConstantDataArray::getFP(type, data);
		} else if (ft->size() == fpsize::dbl) {
			auto data = get_fpdata<uint64_t>(operands, ctx);
			auto type = llvm::Type::getDoubleTy(builder.getContext());
			return llvm::ConstantDataArray::getFP(type, data);
		}
	}

	JLM_UNREACHABLE("This should not have happened!");
}

static llvm::Value *
convert(
	const ConstantArray & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<ConstantArray>(op));

	std::vector<llvm::Constant*> data;
	for (size_t n = 0; n < operands.size(); n++) {
		auto c = llvm::dyn_cast<llvm::Constant>(ctx.value(operands[n]));
		JLM_ASSERT(c);
		data.push_back(c);
	}

	auto at = dynamic_cast<const arraytype*>(&op.result(0).type());
	auto type = convert_type(*at, ctx);
	return llvm::ConstantArray::get(type, data);
}

static llvm::Value *
convert(
	const ConstantAggregateZero & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
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
	JLM_ASSERT(is<ptrcmp_op>(op));
	auto & pop = *static_cast<const ptrcmp_op*>(&op);

	static std::unordered_map<jlm::cmp, llvm::CmpInst::Predicate> map({
	  {cmp::le, llvm::CmpInst::ICMP_ULE}, {cmp::lt, llvm::CmpInst::ICMP_ULT}
	, {cmp::eq, llvm::CmpInst::ICMP_EQ}, {cmp::ne, llvm::CmpInst::ICMP_NE}
	, {cmp::ge, llvm::CmpInst::ICMP_UGE}, {cmp::gt, llvm::CmpInst::ICMP_UGT}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_ASSERT(map.find(pop.cmp()) != map.end());
	return builder.CreateICmp(map[pop.cmp()], op1, op2);
}

static inline llvm::Value *
convert_fpcmp(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<fpcmp_op>(op));
	auto & fpcmp = *static_cast<const jlm::fpcmp_op*>(&op);

	static std::unordered_map<jlm::fpcmp, llvm::CmpInst::Predicate> map({
	  {fpcmp::oeq, llvm::CmpInst::FCMP_OEQ}, {fpcmp::ogt, llvm::CmpInst::FCMP_OGT}
	, {fpcmp::oge, llvm::CmpInst::FCMP_OGE}, {fpcmp::olt, llvm::CmpInst::FCMP_OLT}
	, {fpcmp::ole, llvm::CmpInst::FCMP_OLE}, {fpcmp::one, llvm::CmpInst::FCMP_ONE}
	, {fpcmp::ord, llvm::CmpInst::FCMP_ORD}, {fpcmp::uno, llvm::CmpInst::FCMP_UNO}
	, {fpcmp::ueq, llvm::CmpInst::FCMP_UEQ}, {fpcmp::ugt, llvm::CmpInst::FCMP_UGT}
	, {fpcmp::uge, llvm::CmpInst::FCMP_UGE}, {fpcmp::ult, llvm::CmpInst::FCMP_ULT}
	, {fpcmp::ule, llvm::CmpInst::FCMP_ULE}, {fpcmp::une, llvm::CmpInst::FCMP_UNE}
	, {fpcmp::TRUE, llvm::CmpInst::FCMP_TRUE}, {fpcmp::FALSE, llvm::CmpInst::FCMP_FALSE}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_ASSERT(map.find(fpcmp.cmp()) != map.end());
	return builder.CreateFCmp(map[fpcmp.cmp()], op1, op2);
}

static inline llvm::Value *
convert_fpbin(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<fpbin_op>(op));
	auto & fpbin = *static_cast<const jlm::fpbin_op*>(&op);

	static std::unordered_map<jlm::fpop, llvm::Instruction::BinaryOps> map({
	  {fpop::add, llvm::Instruction::FAdd}, {fpop::sub, llvm::Instruction::FSub}
	, {fpop::mul, llvm::Instruction::FMul}, {fpop::div, llvm::Instruction::FDiv}
	, {fpop::mod, llvm::Instruction::FRem}
	});

	auto op1 = ctx.value(args[0]);
	auto op2 = ctx.value(args[1]);
	JLM_ASSERT(map.find(fpbin.fpop()) != map.end());
	return builder.CreateBinOp(map[fpbin.fpop()], op1, op2);
}

static llvm::Value *
convert_fpneg(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<fpneg_op>(op));
	auto operand = ctx.value(args[0]);
	return builder.CreateUnOp(llvm::Instruction::FNeg, operand);
}

static inline llvm::Value *
convert_valist(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<valist_op>(op));
	return nullptr;
}

static inline llvm::Value *
convert(
	const ConstantStruct & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	std::vector<llvm::Constant*> operands;
	for (const auto & arg : args)
		operands.push_back(llvm::cast<llvm::Constant>(ctx.value(arg)));

	auto t = convert_type(op.type(), ctx);
	return llvm::ConstantStruct::get(t, operands);
}

static inline llvm::Value *
convert(
	const ConstantPointerNullOperation & operation,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto pointerType = convert_type(operation.GetPointerType(), ctx);
	return llvm::ConstantPointerNull::get(pointerType);
}

static inline llvm::Value *
convert_select(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<select_op>(op));
	auto & select = *static_cast<const jlm::select_op*>(&op);

	if (jive::is<jive::statetype>(select.type()))
		return nullptr;

	auto c = ctx.value(operands[0]);
	auto t = ctx.value(operands[1]);
	auto f = ctx.value(operands[2]);
	return builder.CreateSelect(c, t, f);
}

static inline llvm::Value *
convert_ctl2bits(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<ctl2bits_op>(op));
	return ctx.value(args[0]);
}

static llvm::Value *
convert_constantvector(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<constantvector_op>(op));

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
	JLM_ASSERT(is<constant_data_vector_op>(op));
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
		}
	}

	if (auto ft = dynamic_cast<const fptype*>(&cop.type())) {
		if (ft->size() == fpsize::half) {
			auto data = get_fpdata<uint16_t>(operands, ctx);
			auto type = llvm::Type::getBFloatTy(builder.getContext());
			return llvm::ConstantDataVector::getFP(type, data);
		} else if (ft->size() == fpsize::flt) {
			auto data = get_fpdata<uint32_t>(operands, ctx);
			auto type = llvm::Type::getFloatTy(builder.getContext());
			return llvm::ConstantDataVector::getFP(type, data);
		} else if (ft->size() == fpsize::dbl) {
			auto data = get_fpdata<uint64_t>(operands, ctx);
			auto type = llvm::Type::getDoubleTy(builder.getContext());
			return llvm::ConstantDataVector::getFP(type, data);
		}
	}

	JLM_UNREACHABLE("This should not have happened!");
}

static llvm::Value *
convert_extractelement(
	const jive::simple_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<extractelement_op>(op));
	return builder.CreateExtractElement(ctx.value(args[0]), ctx.value(args[1]));
}

static llvm::Value *
convert(
	const shufflevector_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto v1 = ctx.value(operands[0]);
	auto v2 = ctx.value(operands[1]);
	return builder.CreateShuffleVector(v1, v2, op.Mask());
}

static llvm::Value *
convert_insertelement(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<insertelement_op>(op));

	auto vector = ctx.value(operands[0]);
	auto value = ctx.value(operands[1]);
	auto index = ctx.value(operands[2]);
	return builder.CreateInsertElement(vector, value, index);
}

static llvm::Value *
convert_vectorunary(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<vectorunary_op>(op));
	auto vop = static_cast<const vectorunary_op*>(&op);
	return convert_operation(vop->operation(), operands, builder, ctx);
}

static llvm::Value *
convert_vectorbinary(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<vectorbinary_op>(op));
	auto vop = static_cast<const vectorbinary_op*>(&op);
	return convert_operation(vop->operation(), operands, builder, ctx);
}

static llvm::Value *
convert(
	const vectorselect_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto c = ctx.value(operands[0]);
	auto t = ctx.value(operands[1]);
	auto f = ctx.value(operands[2]);
	return builder.CreateSelect(c, t, f);
}

template<llvm::Instruction::CastOps OPCODE> static llvm::Value *
convert_cast(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(llvm::Instruction::isCast(OPCODE));
	auto & dsttype = *static_cast<const jive::valuetype*>(&op.result(0).type());
	auto operand = operands[0];

    if (auto vt = dynamic_cast<const fixedvectortype*>(&operand->type()))
    {
        auto type = convert_type(fixedvectortype(dsttype, vt->size()), ctx);
        return builder.CreateCast(OPCODE, ctx.value(operand), type);
    }

    if (auto vt = dynamic_cast<const scalablevectortype*>(&operand->type()))
    {
        auto type = convert_type(scalablevectortype(dsttype, vt->size()), ctx);
        return builder.CreateCast(OPCODE, ctx.value(operand), type);
    }

	auto type = convert_type(dsttype, ctx);
	return builder.CreateCast(OPCODE, ctx.value(operand), type);
}

static llvm::Value *
convert(
	const ExtractValue & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	std::vector<unsigned> indices(op.begin(), op.end());
	return builder.CreateExtractValue(ctx.value(operands[0]), indices);
}

static llvm::Value *
convert(
	const malloc_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(args.size() == 1);
	auto & lm = ctx.llvm_module();

	auto fcttype = convert_type(op.fcttype(), ctx);
	auto function = lm.getOrInsertFunction("malloc", fcttype);
	auto operands = std::vector<llvm::Value*>(1, ctx.value(args[0]));
	return builder.CreateCall(function, operands);
}

static llvm::Value *
convert(
	const free_op & op,
	const std::vector<const variable*> & args,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto & llvmmod = ctx.llvm_module();

	auto fcttype = convert_type(FunctionType({&op.argument(0).type()}, {}), ctx);
	auto function = llvmmod.getOrInsertFunction("free", fcttype);
	auto operands = std::vector<llvm::Value*>(1, ctx.value(args[0]));
	return builder.CreateCall(function, operands);
}

static llvm::Value *
convert(
	const Memcpy & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	auto destination = ctx.value(operands[0]);
	auto source = ctx.value(operands[1]);
	auto length = ctx.value(operands[2]);
	auto isVolatile = ctx.value(operands[3]);

	return builder.CreateMemCpy(
		destination,
		llvm::MaybeAlign(),
		source,
		llvm::MaybeAlign(),
		length,
		isVolatile);
}

static llvm::Value *
convert(
	const MemStateMergeOperator&,
	const std::vector<const variable*>&,
	llvm::IRBuilder<>&,
	context&)
{
	return nullptr;
}

static llvm::Value *
convert(
	const MemStateSplitOperator&,
	const std::vector<const variable*>&,
	llvm::IRBuilder<>&,
	context&)
{
	return nullptr;
}

static llvm::Value *
convert(
  const aa::LambdaEntryMemStateOperator&,
  const std::vector<const variable*>&,
  llvm::IRBuilder<>&,
  context&)
{
  return nullptr;
}

static llvm::Value *
convert(
    const aa::LambdaExitMemStateOperator&,
    const std::vector<const variable*>&,
    llvm::IRBuilder<>&,
    context&)
{
    return nullptr;
}

static llvm::Value *
convert(
  const aa::CallEntryMemStateOperator&,
  const std::vector<const variable*>&,
  llvm::IRBuilder<>&,
  context&)
{
  return nullptr;
}

static llvm::Value *
convert(
  const aa::CallExitMemStateOperator&,
  const std::vector<const variable*>&,
  llvm::IRBuilder<>&,
  context&)
{
  return nullptr;
}

template<class OP> static llvm::Value *
convert(
	const jive::simple_op & op,
	const std::vector<const variable*> & operands,
	llvm::IRBuilder<> & builder,
	context & ctx)
{
	JLM_ASSERT(is<OP>(op));
	return convert(*static_cast<const OP*>(&op), operands, builder, ctx);
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
    , llvm::Value*(*)(const jive::simple_op &, const std::vector<const variable*> &, llvm::IRBuilder<> &, context & ctx)
  > map({
          {typeid(jive::bitconstant_op),            convert_bitconstant},
          {typeid(jive::ctlconstant_op),            convert_ctlconstant},
          {typeid(ConstantFP),                      convert<ConstantFP>},
          {typeid(UndefValueOperation),             convert_undef},
          {typeid(PoisonValueOperation),            convert<PoisonValueOperation>},
          {typeid(jive::match_op),                  convert_match},
          {typeid(assignment_op),                   convert_assignment},
          {typeid(branch_op),                       convert_branch},
          {typeid(phi_op),                          convert_phi},
          {typeid(LoadOperation),                   convert<LoadOperation>},
          {typeid(StoreOperation),                  convert_store},
          {typeid(alloca_op),                       convert_alloca},
          {typeid(GetElementPtrOperation),          convert_getelementptr},
          {typeid(ConstantDataArray),               convert<ConstantDataArray>},
          {typeid(ptrcmp_op),                       convert_ptrcmp},
          {typeid(fpcmp_op),                        convert_fpcmp},
          {typeid(fpbin_op),                        convert_fpbin},
          {typeid(valist_op),                       convert_valist},
          {typeid(ConstantStruct),                  convert<ConstantStruct>},
          {typeid(ConstantPointerNullOperation),    convert<ConstantPointerNullOperation>},
          {typeid(select_op),                       convert_select},
          {typeid(ConstantArray),                   convert<ConstantArray>},
          {typeid(ConstantAggregateZero),           convert<ConstantAggregateZero>},
          {typeid(ctl2bits_op),                     convert_ctl2bits},
          {typeid(constantvector_op),               convert_constantvector},
          {typeid(constant_data_vector_op),         convert_constantdatavector},
          {typeid(extractelement_op),               convert_extractelement},
          {typeid(shufflevector_op),                convert<shufflevector_op>},
          {typeid(insertelement_op),                convert_insertelement},
          {typeid(vectorunary_op),                  convert_vectorunary},
          {typeid(vectorbinary_op),                 convert_vectorbinary},
          {typeid(vectorselect_op),                 convert<vectorselect_op>},
          {typeid(ExtractValue),                    convert<ExtractValue>},
          {typeid(CallOperation),                   convert<CallOperation>},
          {typeid(malloc_op),                       convert<malloc_op>},
          {typeid(free_op),                         convert<free_op>},
          {typeid(Memcpy),                          convert<Memcpy>},
          {typeid(fpneg_op),                        convert_fpneg},
          {typeid(bitcast_op),                      convert_cast<llvm::Instruction::BitCast>},
          {typeid(fpext_op),                        convert_cast<llvm::Instruction::FPExt>},
          {typeid(fp2si_op),                        convert_cast<llvm::Instruction::FPToSI>},
          {typeid(fp2ui_op),                        convert_cast<llvm::Instruction::FPToUI>},
          {typeid(fptrunc_op),                      convert_cast<llvm::Instruction::FPTrunc>},
          {typeid(bits2ptr_op),                     convert_cast<llvm::Instruction::IntToPtr>},
          {typeid(ptr2bits_op),                     convert_cast<llvm::Instruction::PtrToInt>},
          {typeid(sext_op),                         convert_cast<llvm::Instruction::SExt>},
          {typeid(sitofp_op),                       convert_cast<llvm::Instruction::SIToFP>},
          {typeid(trunc_op),                        convert_cast<llvm::Instruction::Trunc>},
          {typeid(uitofp_op),                       convert_cast<llvm::Instruction::UIToFP>},
          {typeid(zext_op),                         convert_cast<llvm::Instruction::ZExt>},
          {typeid(MemStateMergeOperator),           convert<MemStateMergeOperator>},
          {typeid(MemStateSplitOperator),           convert<MemStateSplitOperator>},
          {typeid(aa::LambdaEntryMemStateOperator), convert<aa::LambdaEntryMemStateOperator>},
          {typeid(aa::LambdaExitMemStateOperator),  convert<aa::LambdaExitMemStateOperator>},
          {typeid(aa::CallEntryMemStateOperator),   convert<aa::CallEntryMemStateOperator>},
          {typeid(aa::CallExitMemStateOperator),    convert<aa::CallExitMemStateOperator>}
        });
  /* FIXME: AddrSpaceCast instruction is not supported */

  JLM_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
  return map[std::type_index(typeid(op))](op, arguments, builder, ctx);
}

void
convert_instruction(const jlm::tac & tac, const jlm::cfg_node * node, context & ctx)
{
	std::vector<const variable*> operands;
	for (size_t n = 0; n < tac.noperands(); n++)
		operands.push_back(tac.operand(n));

	llvm::IRBuilder<> builder(ctx.basic_block(node));
	auto r = convert_operation(tac.operation(), operands, builder, ctx);
	if (r != nullptr) ctx.insert(tac.result(0), r);
}

void
convert_tacs(const tacsvector_t & tacs, context & ctx)
{
	llvm::IRBuilder<> builder(ctx.llvm_module().getContext());
	for (const auto & tac : tacs) {
		std::vector<const variable*> operands;
		for (size_t n = 0; n < tac->noperands(); n++)
			operands.push_back(tac->operand(n));

		JLM_ASSERT(tac->nresults() == 1);
		auto r = convert_operation(tac->operation(), operands, builder, ctx);
		ctx.insert(tac->result(0), r);
	}
}

}}
