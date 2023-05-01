/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/LlvmConversionContext.hpp>
#include <jlm/llvm/frontend/LlvmInstructionConversion.hpp>
#include <jlm/llvm/ir/operators.hpp>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Function.h>

namespace jlm {

const variable *
ConvertValue(llvm::Value * v, tacsvector_t & tacs, context & ctx)
{
	auto node = ctx.node();
	if (node && ctx.has_value(v)) {
		if (auto callee = dynamic_cast<const fctvariable*>(ctx.lookup_value(v)))
			node->add_dependency(callee->function());

		if (auto data = dynamic_cast<const gblvalue*>(ctx.lookup_value(v)))
			node->add_dependency(data->node());
	}

	if (ctx.has_value(v))
		return ctx.lookup_value(v);

	if (auto c = llvm::dyn_cast<llvm::Constant>(v))
		return ConvertConstant(c, tacs, ctx);

	JLM_UNREACHABLE("This should not have happened!");
}

/* constant */

const variable *
ConvertConstant(llvm::Constant*, std::vector<std::unique_ptr<jlm::tac>>&, context&);

static jive::bitvalue_repr
convert_apint(const llvm::APInt & value)
{
  llvm::APInt v;
  if (value.isNegative())
    v = -value;

  auto str = toString(value, 2, false);
  std::reverse(str.begin(), str.end());

  jive::bitvalue_repr vr(str.c_str());
  if (value.isNegative())
    vr = vr.sext(value.getBitWidth() - str.size());
  else
    vr = vr.zext(value.getBitWidth() - str.size());

  return vr;
}

static const variable *
convert_int_constant(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::ConstantIntVal);
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(c);

	jive::bitvalue_repr v = convert_apint(constant->getValue());
	tacs.push_back(tac::create(jive::bitconstant_op(v), {}));

	return tacs.back()->result(0);
}

static inline const variable *
convert_undefvalue(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::UndefValueVal);

	auto t = ConvertType(c->getType(), ctx);
	tacs.push_back(UndefValueOperation::Create(*t));

	return tacs.back()->result(0);
}

static const variable *
convert_constantExpr(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::ConstantExprVal);
	auto c = llvm::cast<llvm::ConstantExpr>(constant);

	/*
		FIXME: ConvertInstruction currently assumes that a instruction's result variable
					 is already added to the context. This is not the case for constants and we
					 therefore need to do some poilerplate checking in ConvertInstruction to
					 see whether a variable was already declared or we need to create a new
					 variable.
	*/

	/* FIXME: getAsInstruction is none const, forcing all llvm parameters to be none const */
	/* FIXME: The invocation of getAsInstruction() introduces a memory leak. */
	auto instruction = c->getAsInstruction();
	auto v = ConvertInstruction(instruction, tacs, ctx);
	instruction->dropAllReferences();
	return v;
}

static const variable *
convert_constantFP(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::ConstantFPVal);
	auto c = llvm::cast<llvm::ConstantFP>(constant);

	auto type = ConvertType(c->getType(), ctx);
	tacs.push_back(ConstantFP::create(c->getValueAPF(), *type));

	return tacs.back()->result(0);
}

static const variable *
convert_globalVariable(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::GlobalVariableVal);
	return ConvertValue(c, tacs, ctx);
}

static const variable *
convert_constantPointerNull(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(llvm::dyn_cast<const llvm::ConstantPointerNull>(constant));
	auto & c = *llvm::cast<const llvm::ConstantPointerNull>(constant);

	auto t = ConvertPointerType(c.getType(), ctx);
	tacs.push_back(ConstantPointerNullOperation::Create(*t));

	return tacs.back()->result(0);
}

static const variable *
convert_blockAddress(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::BlockAddressVal);

	JLM_UNREACHABLE("Blockaddress constants are not supported.");
}

static const variable *
convert_constantAggregateZero(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::ConstantAggregateZeroVal);

	auto type = ConvertType(c->getType(), ctx);
	tacs.push_back(ConstantAggregateZero::create(*type));

	return tacs.back()->result(0);
}

static const variable *
convert_constantArray(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::ConstantArrayVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++) {
		auto operand = c->getOperand(n);
		JLM_ASSERT(llvm::dyn_cast<const llvm::Constant>(operand));
		auto constant = llvm::cast<llvm::Constant>(operand);
		elements.push_back(ConvertConstant(constant, tacs, ctx));
	}

	tacs.push_back(ConstantArray::create(elements));

	return tacs.back()->result(0);
}

static const variable *
convert_constantDataArray(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::ConstantDataArrayVal);
	const auto & c = *llvm::cast<const llvm::ConstantDataArray>(constant);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c.getNumElements(); n++)
		elements.push_back(ConvertConstant(c.getElementAsConstant(n), tacs, ctx));

	tacs.push_back(ConstantDataArray::create(elements));

	return tacs.back()->result(0);
}

static const variable *
convert_constantDataVector(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::ConstantDataVectorVal);
	auto c = llvm::cast<const llvm::ConstantDataVector>(constant);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumElements(); n++)
		elements.push_back(ConvertConstant(c->getElementAsConstant(n), tacs, ctx));

	tacs.push_back(constant_data_vector_op::Create(elements));

	return tacs.back()->result(0);
}

static const variable *
ConvertConstantStruct(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::ConstantStructVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++)
		elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

	auto type = ConvertType(c->getType(), ctx);
	tacs.push_back(ConstantStruct::create(elements, *type));

	return tacs.back()->result(0);
}

static const variable *
convert_constantVector(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::ConstantVectorVal);

	std::vector<const variable*> elements;
	for (size_t n = 0; n < c->getNumOperands(); n++)
		elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

	auto type = ConvertType(c->getType(), ctx);
	tacs.push_back(constantvector_op::create(elements, *type));

	return tacs.back()->result(0);
}

static inline const variable *
convert_globalAlias(
	llvm::Constant * constant,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	JLM_ASSERT(constant->getValueID() == llvm::Value::GlobalAliasVal);

	JLM_UNREACHABLE("GlobalAlias constants are not supported.");
}

static inline const variable *
convert_function(
	llvm::Constant * c,
	tacsvector_t & tacs,
	context & ctx)
{
	JLM_ASSERT(c->getValueID() == llvm::Value::FunctionVal);
	return ConvertValue(c, tacs, ctx);
}

static const variable *
ConvertConstant(
  llvm::PoisonValue * poisonValue,
  tacsvector_t & threeAddressCodeVector,
  jlm::context & context)
{
  auto type = ConvertType(poisonValue->getType(), context);
  threeAddressCodeVector.push_back(PoisonValueOperation::Create(*type));

  return threeAddressCodeVector.back()->result(0);
}

template <class T> static const variable *
ConvertConstant(
  llvm::Constant * constant,
  tacsvector_t & threeAddressCodeVector,
  jlm::context & context)
{
  JLM_ASSERT(llvm::dyn_cast<T>(constant));
  return ConvertConstant(llvm::cast<T>(constant), threeAddressCodeVector, context);
}

const variable *
ConvertConstant(
	llvm::Constant * c,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
  static std::unordered_map<
    unsigned,
    const variable*(*)(llvm::Constant*, std::vector<std::unique_ptr<jlm::tac>>&, context & ctx)
    > constantMap({
      {llvm::Value::BlockAddressVal,          convert_blockAddress},
      {llvm::Value::ConstantAggregateZeroVal, convert_constantAggregateZero},
      {llvm::Value::ConstantArrayVal,         convert_constantArray},
      {llvm::Value::ConstantDataArrayVal,     convert_constantDataArray},
      {llvm::Value::ConstantDataVectorVal,    convert_constantDataVector},
      {llvm::Value::ConstantExprVal,          convert_constantExpr},
      {llvm::Value::ConstantFPVal,            convert_constantFP},
      {llvm::Value::ConstantIntVal,           convert_int_constant},
      {llvm::Value::ConstantPointerNullVal,   convert_constantPointerNull},
      {llvm::Value::ConstantStructVal,        ConvertConstantStruct},
      {llvm::Value::ConstantVectorVal,        convert_constantVector},
      {llvm::Value::FunctionVal,              convert_function},
      {llvm::Value::GlobalAliasVal,           convert_globalAlias},
      {llvm::Value::GlobalVariableVal,        convert_globalVariable},
      {llvm::Value::PoisonValueVal,           ConvertConstant<llvm::PoisonValue>},
      {llvm::Value::UndefValueVal,            convert_undefvalue}
    });

  if (constantMap.find(c->getValueID()) != constantMap.end())
    return constantMap[c->getValueID()](c, tacs, ctx);

  JLM_UNREACHABLE("Unsupported LLVM Constant.");
}

std::vector<std::unique_ptr<jlm::tac>>
ConvertConstant(llvm::Constant * c, context & ctx)
{
	std::vector<std::unique_ptr<jlm::tac>> tacs;
  ConvertConstant(c, tacs, ctx);
	return tacs;
}

/* instructions */

static inline const variable *
convert_return_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::Ret);
	auto i = llvm::cast<llvm::ReturnInst>(instruction);

	auto bb = ctx.get(i->getParent());
	bb->add_outedge(bb->cfg().exit());
	if (!i->getReturnValue())
		return {};

	auto value = ConvertValue(i->getReturnValue(), tacs, ctx);
	tacs.push_back(assignment_op::create(value, ctx.result()));

	return ctx.result();
}

static inline const variable *
convert_branch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::Br);
	auto i = llvm::cast<llvm::BranchInst>(instruction);
	auto bb = ctx.get(i->getParent());

	if (i->isUnconditional()) {
		bb->add_outedge(ctx.get(i->getSuccessor(0)));
		return {};
	}

	bb->add_outedge(ctx.get(i->getSuccessor(1))); /* false */
	bb->add_outedge(ctx.get(i->getSuccessor(0))); /* true */

	auto c = ConvertValue(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, {{1, 1}}, 0, 2);
	tacs.push_back(tac::create(op, {c}));
	tacs.push_back(branch_op::create(2,  tacs.back()->result(0)));

	return nullptr;
}

static inline const variable *
convert_switch_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::Switch);
	auto i = llvm::cast<llvm::SwitchInst>(instruction);
	auto bb = ctx.get(i->getParent());

	size_t n = 0;
	std::unordered_map<uint64_t, uint64_t> mapping;
	for (auto it = i->case_begin(); it != i->case_end(); it++) {
		JLM_ASSERT(it != i->case_default());
		mapping[it->getCaseValue()->getZExtValue()] = n++;
		bb->add_outedge(ctx.get(it->getCaseSuccessor()));
	}

	bb->add_outedge(ctx.get(i->case_default()->getCaseSuccessor()));
	JLM_ASSERT(i->getNumSuccessors() == n+1);

	auto c = ConvertValue(i->getCondition(), tacs, ctx);
	auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
	auto op = jive::match_op(nbits, mapping, n, n+1);
	tacs.push_back(tac::create(op, {c}));
	tacs.push_back((branch_op::create(n+1, tacs.back()->result(0))));

	return nullptr;
}

static inline const variable *
convert_unreachable_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::Unreachable);
	auto bb = ctx.get(i->getParent());
	bb->add_outedge(bb->cfg().exit());
	return nullptr;
}

static inline const variable *
convert_icmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::ICmp);
	auto i = llvm::cast<const llvm::ICmpInst>(instruction);
	auto t = i->getOperand(0)->getType();

	static std::unordered_map<
		const llvm::CmpInst::Predicate,
		std::unique_ptr<jive::operation>(*)(size_t)> map({
		  {llvm::CmpInst::ICMP_SLT,	[](size_t nbits){jive::bitslt_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_ULT,	[](size_t nbits){jive::bitult_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SLE,	[](size_t nbits){jive::bitsle_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_ULE,	[](size_t nbits){jive::bitule_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_EQ,	[](size_t nbits){jive::biteq_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_NE,	[](size_t nbits){jive::bitne_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SGE,	[](size_t nbits){jive::bitsge_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_UGE,	[](size_t nbits){jive::bituge_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_SGT,	[](size_t nbits){jive::bitsgt_op op(nbits); return op.copy();}}
		, {llvm::CmpInst::ICMP_UGT,	[](size_t nbits){jive::bitugt_op op(nbits); return op.copy();}}
	});

	static std::unordered_map<const llvm::CmpInst::Predicate, jlm::cmp> ptrmap({
	  {llvm::CmpInst::ICMP_ULT,	cmp::lt}, {llvm::CmpInst::ICMP_ULE,	cmp::le}
	,	{llvm::CmpInst::ICMP_EQ,	cmp::eq}, {llvm::CmpInst::ICMP_NE,	cmp::ne}
	,	{llvm::CmpInst::ICMP_UGE,	cmp::ge}, {llvm::CmpInst::ICMP_UGT,	cmp::gt}
	});

	auto p = i->getPredicate();
	auto op1 = ConvertValue(i->getOperand(0), tacs, ctx);
	auto op2 = ConvertValue(i->getOperand(1), tacs, ctx);

	std::unique_ptr<jive::operation> binop;

	if (t->isIntegerTy() || (t->isVectorTy() && t->getScalarType()->isIntegerTy())) {
		auto it = t->isVectorTy() ? t->getScalarType() : t;
		binop = map[p](it->getIntegerBitWidth());
	} else if (t->isPointerTy() || (t->isVectorTy() && t->getScalarType()->isPointerTy())) {
		auto pt = llvm::cast<llvm::PointerType>(t->isVectorTy() ? t->getScalarType() : t);
		binop = std::make_unique<ptrcmp_op>(*ConvertPointerType(pt, ctx), ptrmap[p]);
	} else
		JLM_UNREACHABLE("This should have never happend.");

	auto type = ConvertType(i->getType(), ctx);

	JLM_ASSERT(is<jive::binary_op>(*binop));
	if (t->isVectorTy()) {
		tacs.push_back(vectorbinary_op::create(*static_cast<jive::binary_op*>(binop.get()),
			op1, op2, *type));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(binop.get()),
			{op1, op2}));
	}

	return tacs.back()->result(0);
}

static const variable *
convert_fcmp_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::FCmp);
	auto i = llvm::cast<const llvm::FCmpInst>(instruction);
	auto t = i->getOperand(0)->getType();

	static std::unordered_map<llvm::CmpInst::Predicate, jlm::fpcmp> map({
		{llvm::CmpInst::FCMP_TRUE, fpcmp::TRUE}, {llvm::CmpInst::FCMP_FALSE, fpcmp::FALSE}
	,	{llvm::CmpInst::FCMP_OEQ, fpcmp::oeq},	{llvm::CmpInst::FCMP_OGT, fpcmp::ogt}
	,	{llvm::CmpInst::FCMP_OGE, fpcmp::oge},	{llvm::CmpInst::FCMP_OLT, fpcmp::olt}
	,	{llvm::CmpInst::FCMP_OLE, fpcmp::ole},	{llvm::CmpInst::FCMP_ONE, fpcmp::one}
	,	{llvm::CmpInst::FCMP_ORD, fpcmp::ord},	{llvm::CmpInst::FCMP_UNO, fpcmp::uno}
	,	{llvm::CmpInst::FCMP_UEQ, fpcmp::ueq},	{llvm::CmpInst::FCMP_UGT, fpcmp::ugt}
	,	{llvm::CmpInst::FCMP_UGE, fpcmp::uge},	{llvm::CmpInst::FCMP_ULT, fpcmp::ult}
	,	{llvm::CmpInst::FCMP_ULE, fpcmp::ule},	{llvm::CmpInst::FCMP_UNE, fpcmp::une}
	});

	auto type = ConvertType(i->getType(), ctx);

	auto op1 = ConvertValue(i->getOperand(0), tacs, ctx);
	auto op2 = ConvertValue(i->getOperand(1), tacs, ctx);

	JLM_ASSERT(map.find(i->getPredicate()) != map.end());
	auto fptype = t->isVectorTy() ? t->getScalarType() : t;
	fpcmp_op operation(map[i->getPredicate()], ExtractFloatingPointSize(fptype));

	if (t->isVectorTy())
		tacs.push_back(vectorbinary_op::create(operation, op1, op2, *type));
	else
		tacs.push_back(tac::create(operation, {op1, op2}));

	return tacs.back()->result(0);
}

static inline const variable *
convert_load_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::Load);
	auto instruction = static_cast<llvm::LoadInst*>(i);

	/* FIXME: volatile */
	auto alignment = instruction->getAlign().value();
	auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
  auto loadedType = ConvertType(instruction->getType(), ctx);

	tacs.push_back(LoadOperation::Create(address, ctx.memory_state(), *loadedType, alignment));
	auto value = tacs.back()->result(0);
	auto state = tacs.back()->result(1);

	tacs.push_back(assignment_op::create(state, ctx.memory_state()));

	return value;
}

static inline const variable *
convert_store_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::Store);
	auto instruction = static_cast<llvm::StoreInst*>(i);

	/* FIXME: volatile */
	auto alignment = instruction->getAlign().value();
	auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
	auto value = ConvertValue(instruction->getValueOperand(), tacs, ctx);

	tacs.push_back(StoreOperation::Create(address, value, ctx.memory_state(), alignment));
	tacs.push_back(assignment_op::create(tacs.back()->result(0), ctx.memory_state()));

	return nullptr;
}

static inline const variable *
convert_phi_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::PHI);

	auto type = ConvertType(i->getType(), ctx);
	tacs.push_back(phi_op::create({}, *type));
	return tacs.back()->result(0);
}

static inline const variable *
convert_getelementptr_instruction(llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(llvm::dyn_cast<const llvm::GetElementPtrInst>(inst));
	auto i = llvm::cast<llvm::GetElementPtrInst>(inst);

	std::vector<const variable*> indices;
	auto base = ConvertValue(i->getPointerOperand(), tacs, ctx);
	for (auto it = i->idx_begin(); it != i->idx_end(); it++)
		indices.push_back(ConvertValue(*it, tacs, ctx));

  auto pointeeType = ConvertType(i->getSourceElementType(), ctx);
  auto resultType = ConvertType(i->getType(), ctx);

	tacs.push_back(GetElementPtrOperation::Create(base, indices, *pointeeType, *resultType));

	return tacs.back()->result(0);
}

static const variable *
convert_malloc_call(const llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
	auto memstate = ctx.memory_state();

	auto size = ConvertValue(i->getArgOperand(0), tacs, ctx);

	tacs.push_back(malloc_op::create(size));
	auto result = tacs.back()->result(0);
	auto mstate = tacs.back()->result(1);

	tacs.push_back(MemStateMergeOperator::Create({mstate, memstate}));
	tacs.push_back(assignment_op::create(tacs.back()->result(0), memstate));

	return result;
}

static const variable *
convert_free_call(const llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
	auto iostate = ctx.iostate();
	auto memstate = ctx.memory_state();

	auto pointer = ConvertValue(i->getArgOperand(0), tacs, ctx);

	tacs.push_back(free_op::create(pointer, {memstate}, iostate));
	tacs.push_back(assignment_op::create(tacs.back()->result(0), memstate));

	return nullptr;
}

static const variable*
convert_memcpy_call(
	const llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
	auto memstate = ctx.memory_state();

	auto destination = ConvertValue(i->getArgOperand(0), tacs, ctx);
	auto source = ConvertValue(i->getArgOperand(1), tacs, ctx);
	auto length = ConvertValue(i->getArgOperand(2), tacs, ctx);
	auto isVolatile = ConvertValue(i->getArgOperand(3), tacs, ctx);

	tacs.push_back(Memcpy::create(destination, source, length, isVolatile, {memstate}));
	tacs.push_back(assignment_op::create(tacs.back()->result(0), memstate));

	return nullptr;
}

static const variable *
convert_call_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::Call);
	auto i = llvm::cast<llvm::CallInst>(instruction);

	auto create_arguments = [](
		const llvm::CallInst * i,
		tacsvector_t & tacs,
		context & ctx)
	{
    auto functionType = i->getFunctionType();
		std::vector<const jlm::variable*> arguments;
		for (size_t n = 0; n < functionType->getNumParams(); n++)
			arguments.push_back(ConvertValue(i->getArgOperand(n), tacs, ctx));

		return arguments;
	};

	auto create_varargs = [](
		const llvm::CallInst * i,
		tacsvector_t & tacs,
		context & ctx)
	{
    auto functionType = i->getFunctionType();
		std::vector<const jlm::variable*> varargs;
		for (size_t n = functionType->getNumParams(); n < i->getNumOperands() - 1; n++)
			varargs.push_back(ConvertValue(i->getArgOperand(n), tacs, ctx));

		tacs.push_back(valist_op::create(varargs));
		return tacs.back()->result(0);
	};

	auto is_malloc_call = [](const llvm::CallInst * i)
	{
		auto f = i->getCalledFunction();
		return f && f->getName() == "malloc";
	};

	auto is_free_call = [](const llvm::CallInst * i)
	{
		auto f = i->getCalledFunction();
		return f && f->getName() == "free";
	};

	auto IsMemcpyCall = [](const llvm::CallInst  * i)
	{
		return llvm::dyn_cast<llvm::MemCpyInst>(i) != nullptr;
	};

	if (is_malloc_call(i))
		return convert_malloc_call(i, tacs, ctx);
	if (is_free_call(i))
		return convert_free_call(i, tacs, ctx);
	if (IsMemcpyCall(i))
		return convert_memcpy_call(i, tacs, ctx);

	auto ftype = i->getFunctionType();

	auto arguments = create_arguments(i, tacs, ctx);
	if (ftype->isVarArg())
		arguments.push_back(create_varargs(i, tacs, ctx));
	arguments.push_back(ctx.iostate());
	arguments.push_back(ctx.memory_state());
	arguments.push_back(ctx.loop_state());

	auto fctvar = ConvertValue(i->getCalledOperand(), tacs, ctx);
	auto call = CallOperation::create(
    fctvar,
    *ConvertFunctionType(ftype, ctx),
    arguments);

	auto result = call->result(0);
	auto iostate = call->result(call->nresults() - 3);
	auto memstate = call->result(call->nresults() - 2);
	auto loopstate = call->result(call->nresults() - 1);

	tacs.push_back(std::move(call));
	tacs.push_back(assignment_op::create(iostate, ctx.iostate()));
	tacs.push_back(assignment_op::create(memstate, ctx.memory_state()));
	tacs.push_back(assignment_op::create(loopstate, ctx.loop_state()));

	return result;
}

static inline const variable *
convert_select_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::Select);
	auto instruction = static_cast<llvm::SelectInst*>(i);

	auto p = ConvertValue(instruction->getCondition(), tacs, ctx);
	auto t = ConvertValue(instruction->getTrueValue(), tacs, ctx);
	auto f = ConvertValue(instruction->getFalseValue(), tacs, ctx);

	if (i->getType()->isVectorTy())
		tacs.push_back(vectorselect_op::create(p, t, f));
	else
		tacs.push_back(select_op::create(p, t, f));

	return tacs.back()->result(0);
}

static inline const variable *
convert_binary_operator(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(llvm::dyn_cast<const llvm::BinaryOperator>(instruction));
	auto i = llvm::cast<const llvm::BinaryOperator>(instruction);

	static std::unordered_map<
		const llvm::Instruction::BinaryOps,
		std::unique_ptr<jive::operation>(*)(size_t)> bitmap({
			{llvm::Instruction::Add,	[](size_t nbits){jive::bitadd_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::And,	[](size_t nbits){jive::bitand_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::AShr,[](size_t nbits){jive::bitashr_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Sub,	[](size_t nbits){jive::bitsub_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::UDiv,[](size_t nbits){jive::bitudiv_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::SDiv,[](size_t nbits){jive::bitsdiv_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::URem,[](size_t nbits){jive::bitumod_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::SRem,[](size_t nbits){jive::bitsmod_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Shl,	[](size_t nbits){jive::bitshl_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::LShr,[](size_t nbits){jive::bitshr_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Or,	[](size_t nbits){jive::bitor_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Xor,	[](size_t nbits){jive::bitxor_op o(nbits); return o.copy();}}
		,	{llvm::Instruction::Mul,	[](size_t nbits){jive::bitmul_op o(nbits); return o.copy();}}
	});

	static std::unordered_map<const llvm::Instruction::BinaryOps, jlm::fpop> fpmap({
	  {llvm::Instruction::FAdd, fpop::add}, {llvm::Instruction::FSub, fpop::sub}
	, {llvm::Instruction::FMul, fpop::mul}, {llvm::Instruction::FDiv, fpop::div}
	, {llvm::Instruction::FRem, fpop::mod}
	});

	static std::unordered_map<const llvm::Type::TypeID, jlm::fpsize> fpsizemap({
	  {llvm::Type::HalfTyID, fpsize::half}
	, {llvm::Type::FloatTyID, fpsize::flt}
	, {llvm::Type::DoubleTyID, fpsize::dbl}
	, {llvm::Type::X86_FP80TyID, fpsize::x86fp80}
	});

	std::unique_ptr<jive::operation> operation;
	auto t = i->getType()->isVectorTy() ? i->getType()->getScalarType() : i->getType();
	if (t->isIntegerTy()) {
		JLM_ASSERT(bitmap.find(i->getOpcode()) != bitmap.end());
		operation = bitmap[i->getOpcode()](t->getIntegerBitWidth());
	} else if (t->isFloatingPointTy()) {
		JLM_ASSERT(fpmap.find(i->getOpcode()) != fpmap.end());
		JLM_ASSERT(fpsizemap.find(t->getTypeID()) != fpsizemap.end());
		operation = std::make_unique<fpbin_op>(fpmap[i->getOpcode()], fpsizemap[t->getTypeID()]);
	} else
		JLM_ASSERT(0);

	auto type = ConvertType(i->getType(), ctx);

	auto op1 = ConvertValue(i->getOperand(0), tacs, ctx);
	auto op2 = ConvertValue(i->getOperand(1), tacs, ctx);
	JLM_ASSERT(is<jive::binary_op>(*operation));

	if (i->getType()->isVectorTy()) {
		auto & binop = *static_cast<jive::binary_op*>(operation.get());
		tacs.push_back(vectorbinary_op::create(binop, op1, op2, *type));
	} else {
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(operation.get()), {op1, op2}));
	}

	return tacs.back()->result(0);
}

static inline const variable *
convert_alloca_instruction(llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(instruction->getOpcode() == llvm::Instruction::Alloca);
	auto i = static_cast<llvm::AllocaInst*>(instruction);

	auto memstate = ctx.memory_state();
	auto size = ConvertValue(i->getArraySize(), tacs, ctx);
	auto vtype = ConvertType(i->getAllocatedType(), ctx);
  auto alignment = i->getAlign().value();

	tacs.push_back(alloca_op::create(*vtype, size, alignment));
	auto result = tacs.back()->result(0);
	auto astate = tacs.back()->result(1);

	tacs.push_back(MemStateMergeOperator::Create({astate, memstate}));
	tacs.push_back(assignment_op::create(tacs.back()->result(0), memstate));

	return result;
}

static const variable *
convert_extractvalue(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::ExtractValue);
	auto ev = llvm::dyn_cast<llvm::ExtractValueInst>(i);

	auto aggregate = ConvertValue(ev->getOperand(0), tacs, ctx);
	tacs.push_back(ExtractValue::create(aggregate, ev->getIndices()));

	return tacs.back()->result(0);
}

static inline const variable *
convert_extractelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::ExtractElement);

	auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
	auto index = ConvertValue(i->getOperand(1), tacs, ctx);
	tacs.push_back(extractelement_op::create(vector, index));

	return tacs.back()->result(0);
}

static const variable *
convert(
        llvm::ShuffleVectorInst * i,
        tacsvector_t & tacs,
        context & ctx)
{
    auto v1 = ConvertValue(i->getOperand(0), tacs, ctx);
    auto v2 = ConvertValue(i->getOperand(1), tacs, ctx);

    std::vector<int> mask;
    for (auto & element : i->getShuffleMask())
        mask.push_back(element);

    tacs.push_back(shufflevector_op::create(v1, v2, mask));

    return tacs.back()->result(0);
}

static const variable *
convert_insertelement_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(i->getOpcode() == llvm::Instruction::InsertElement);

	auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
	auto value = ConvertValue(i->getOperand(1), tacs, ctx);
	auto index = ConvertValue(i->getOperand(2), tacs, ctx);
	tacs.push_back(insertelement_op::create(vector, value, index));

	return tacs.back()->result(0);
}

static const variable *
convert(
  llvm::UnaryOperator * unaryOperator,
  tacsvector_t & threeAddressCodeVector,
  context & ctx)
{
  JLM_ASSERT(unaryOperator->getOpcode() == llvm::Instruction::FNeg);

  auto type = unaryOperator->getType();
  auto scalarType = ConvertType(type->getScalarType(), ctx);
  auto operand = ConvertValue(unaryOperator->getOperand(0), threeAddressCodeVector, ctx);

  if (type->isVectorTy()) {
    auto vectorType = ConvertType(type, ctx);
    threeAddressCodeVector.push_back(vectorunary_op::create(fpneg_op(*scalarType), operand, *vectorType));
  } else {
    threeAddressCodeVector.push_back(fpneg_op::create(operand));
  }

  return threeAddressCodeVector.back()->result(0);
}

template<class OP> static std::unique_ptr<jive::operation>
create_unop(std::unique_ptr<jive::type> st, std::unique_ptr<jive::type> dt)
{
	return std::unique_ptr<jive::operation>(new OP(std::move(st), std::move(dt)));
}

static const variable *
convert_cast_instruction(llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
	JLM_ASSERT(llvm::dyn_cast<llvm::CastInst>(i));
	auto st = i->getOperand(0)->getType();
	auto dt = i->getType();

	static std::unordered_map<
		unsigned,
		std::unique_ptr<jive::operation>(*)(std::unique_ptr<jive::type>, std::unique_ptr<jive::type>)
	> map({
	  {llvm::Instruction::Trunc, create_unop<trunc_op>}
	, {llvm::Instruction::ZExt, create_unop<zext_op>}
	, {llvm::Instruction::UIToFP, create_unop<uitofp_op>}
	, {llvm::Instruction::SIToFP, create_unop<sitofp_op>}
	, {llvm::Instruction::SExt, create_unop<sext_op>}
	, {llvm::Instruction::PtrToInt, create_unop<ptr2bits_op>}
	, {llvm::Instruction::IntToPtr, create_unop<bits2ptr_op>}
	, {llvm::Instruction::FPTrunc, create_unop<fptrunc_op>}
	, {llvm::Instruction::FPToUI, create_unop<fp2ui_op>}
	, {llvm::Instruction::FPToSI, create_unop<fp2si_op>}
	, {llvm::Instruction::FPExt, create_unop<fpext_op>}
	, {llvm::Instruction::BitCast, create_unop<bitcast_op>}
	});

	auto type = ConvertType(i->getType(), ctx);

	auto op = ConvertValue(i->getOperand(0), tacs, ctx);
	auto srctype = ConvertType(st->isVectorTy() ? st->getScalarType() : st, ctx);
	auto dsttype = ConvertType(dt->isVectorTy() ? dt->getScalarType() : dt, ctx);

	JLM_ASSERT(map.find(i->getOpcode()) != map.end());
	auto unop = map[i->getOpcode()](std::move(srctype), std::move(dsttype));
	JLM_ASSERT(is<jive::unary_op>(*unop));

	if (dt->isVectorTy())
		tacs.push_back(vectorunary_op::create(*static_cast<jive::unary_op*>(unop.get()), op, *type));
	else
		tacs.push_back(tac::create(*static_cast<jive::simple_op*>(unop.get()), {op}));

	return tacs.back()->result(0);
}

template<class INSTRUCTIONTYPE> static const variable*
convert(
    llvm::Instruction * instruction,
    tacsvector_t & tacs,
    context & ctx)
{
    JLM_ASSERT(llvm::isa<INSTRUCTIONTYPE>(instruction));
    return convert(llvm::cast<INSTRUCTIONTYPE>(instruction), tacs, ctx);
}

const variable *
ConvertInstruction(
	llvm::Instruction * i,
	std::vector<std::unique_ptr<jlm::tac>> & tacs,
	context & ctx)
{
	if (i->isCast())
		return convert_cast_instruction(i, tacs, ctx);

	static std::unordered_map<
		unsigned,
		const variable*(*)(llvm::Instruction*, std::vector<std::unique_ptr<jlm::tac>>&, context&)
	> map({
		{llvm::Instruction::Ret, convert_return_instruction}
	,	{llvm::Instruction::Br, convert_branch_instruction}
	,	{llvm::Instruction::Switch, convert_switch_instruction}
	,	{llvm::Instruction::Unreachable, convert_unreachable_instruction}
	,	{llvm::Instruction::Add, convert_binary_operator}
	,	{llvm::Instruction::And, convert_binary_operator}
	,	{llvm::Instruction::AShr, convert_binary_operator}
	,	{llvm::Instruction::Sub, convert_binary_operator}
	,	{llvm::Instruction::UDiv, convert_binary_operator}
	,	{llvm::Instruction::SDiv, convert_binary_operator}
	,	{llvm::Instruction::URem, convert_binary_operator}
	,	{llvm::Instruction::SRem, convert_binary_operator}
	,	{llvm::Instruction::Shl, convert_binary_operator}
	,	{llvm::Instruction::LShr, convert_binary_operator}
	,	{llvm::Instruction::Or, convert_binary_operator}
	,	{llvm::Instruction::Xor, convert_binary_operator}
	,	{llvm::Instruction::Mul, convert_binary_operator}
	,	{llvm::Instruction::FAdd, convert_binary_operator}
	,	{llvm::Instruction::FSub, convert_binary_operator}
	,	{llvm::Instruction::FMul, convert_binary_operator}
	,	{llvm::Instruction::FDiv, convert_binary_operator}
	,	{llvm::Instruction::FRem, convert_binary_operator}
	,	{llvm::Instruction::FNeg, convert<llvm::UnaryOperator>}
	,	{llvm::Instruction::ICmp, convert_icmp_instruction}
	,	{llvm::Instruction::FCmp, convert_fcmp_instruction}
	,	{llvm::Instruction::Load, convert_load_instruction}
	,	{llvm::Instruction::Store, convert_store_instruction}
	,	{llvm::Instruction::PHI, convert_phi_instruction}
	,	{llvm::Instruction::GetElementPtr, convert_getelementptr_instruction}
	,	{llvm::Instruction::Call, convert_call_instruction}
	,	{llvm::Instruction::Select, convert_select_instruction}
	,	{llvm::Instruction::Alloca, convert_alloca_instruction}
	,	{llvm::Instruction::ExtractValue, convert_extractvalue}
	,	{llvm::Instruction::ExtractElement, convert_extractelement_instruction}
	,	{llvm::Instruction::ShuffleVector, convert<llvm::ShuffleVectorInst>}
	,	{llvm::Instruction::InsertElement, convert_insertelement_instruction}
	});

	if (map.find(i->getOpcode()) == map.end())
		JLM_UNREACHABLE(strfmt(i->getOpcodeName(), " is not supported.").c_str());

	return map[i->getOpcode()](i, tacs, ctx);
}

}
