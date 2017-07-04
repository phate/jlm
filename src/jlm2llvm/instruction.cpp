/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/memorytype.h>
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
convert_bitsbinary(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
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

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateBinOp(map[std::type_index(typeid(op))], args[0], args[1]);
}

static inline llvm::Value *
convert_bitscompare(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
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

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
	return builder.CreateICmp(map[std::type_index(typeid(op))], args[0], args[1]);
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
convert_fpconstant(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_fpconstant_op(op));
	auto & cop = *static_cast<const jlm::fpconstant_op*>(&op);

	auto type = convert_type(cop.result_type(0), builder.getContext());
	return llvm::ConstantFP::get(type, cop.constant());
}

static inline llvm::Value *
convert_undef(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_undef_constant_op(op));
	return llvm::UndefValue::get(convert_type(op.result_type(0), builder.getContext()));
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

	if (dynamic_cast<const jive::mem::type*>(&pop.type()))
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

static inline llvm::Value *
convert_ptrcmp(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_ptrcmp_op(op));
	auto & pop = *static_cast<const ptrcmp_op*>(&op);

	static std::unordered_map<jlm::cmp, llvm::CmpInst::Predicate> map({
	  {cmp::le, llvm::CmpInst::ICMP_ULE}, {cmp::lt, llvm::CmpInst::ICMP_ULT}
	, {cmp::eq, llvm::CmpInst::ICMP_EQ}, {cmp::ne, llvm::CmpInst::ICMP_NE}
	, {cmp::ge, llvm::CmpInst::ICMP_UGE}, {cmp::gt, llvm::CmpInst::ICMP_UGT}
	});

	JLM_DEBUG_ASSERT(map.find(pop.cmp()) != map.end());
	return builder.CreateICmp(map[pop.cmp()], args[0], args[1]);
}

static inline llvm::Value *
convert_zext(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_zext_op(op));
	return builder.CreateZExt(args[0], convert_type(op.result_type(0), builder.getContext()));
}

static inline llvm::Value *
convert_fpcmp(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
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

	JLM_DEBUG_ASSERT(map.find(fpcmp.cmp()) != map.end());
	return builder.CreateFCmp(map[fpcmp.cmp()], args[0], args[1]);
}

static inline llvm::Value *
convert_fpbin(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_fpbin_op(op));
	auto & fpbin = *static_cast<const jlm::fpbin_op*>(&op);

	static std::unordered_map<jlm::fpop, llvm::Instruction::BinaryOps> map({
	  {fpop::add, llvm::Instruction::FAdd}, {fpop::sub, llvm::Instruction::FSub}
	, {fpop::mul, llvm::Instruction::FMul}, {fpop::div, llvm::Instruction::FDiv}
	, {fpop::mod, llvm::Instruction::FRem}
	});

	JLM_DEBUG_ASSERT(map.find(fpbin.fpop()) != map.end());
	return builder.CreateBinOp(map[fpbin.fpop()], args[0], args[1]);
}

static inline llvm::Value *
convert_fpext(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & args)
{
	JLM_DEBUG_ASSERT(is_fpext_op(op));
	return builder.CreateFPExt(args[0], convert_type(op.result_type(0), builder.getContext()));
}

llvm::Value *
convert_operation(
	const jive::operation & op,
	llvm::IRBuilder<> & builder,
	const std::vector<llvm::Value*> & arguments)
{
	if (dynamic_cast<const jive::bits::binary_op*>(&op))
		return convert_bitsbinary(op, builder, arguments);

	if (dynamic_cast<const jive::bits::compare_op*>(&op))
		return convert_bitscompare(op, builder, arguments);

	static std::unordered_map<
		std::type_index
	, llvm::Value*(*)(const jive::operation &, llvm::IRBuilder<>&, const std::vector<llvm::Value*>&)
	> map({
	  {std::type_index(typeid(jive::bits::constant_op)), convert_bitconstant}
	, {std::type_index(typeid(jive::ctl::constant_op)), convert_ctlconstant}
	, {std::type_index(typeid(jlm::fpconstant_op)), convert_fpconstant}
	, {std::type_index(typeid(jlm::undef_constant_op)), convert_undef}
	, {std::type_index(typeid(jive::match_op)), convert_match}
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
