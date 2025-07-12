/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/LlvmConversionContext.hpp>
#include <jlm/llvm/frontend/LlvmInstructionConversion.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>

#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>

namespace jlm::llvm
{

// Converts a value into a variable either representing the llvm
// value or representing a function object.
// The distinction stems from the fact that llvm treats functions simply
// as pointers to the function code while we distinguish between the two.
// This function can return either and caller needs to check / adapt.
const Variable *
ConvertValueOrFunction(::llvm::Value * v, tacsvector_t & tacs, context & ctx)
{
  auto node = ctx.node();
  if (node && ctx.has_value(v))
  {
    if (auto callee = dynamic_cast<const fctvariable *>(ctx.lookup_value(v)))
      node->add_dependency(callee->function());

    if (auto data = dynamic_cast<const GlobalValue *>(ctx.lookup_value(v)))
      node->add_dependency(data->node());
  }

  if (ctx.has_value(v))
    return ctx.lookup_value(v);

  if (auto c = ::llvm::dyn_cast<::llvm::Constant>(v))
    return ConvertConstant(c, tacs, ctx);

  JLM_UNREACHABLE("This should not have happened!");
}

// Converts a value into a variable representing the llvm value.
const Variable *
ConvertValue(::llvm::Value * v, tacsvector_t & tacs, context & ctx)
{
  const Variable * var = ConvertValueOrFunction(v, tacs, ctx);
  if (auto fntype = std::dynamic_pointer_cast<const rvsdg::FunctionType>(var->Type()))
  {
    std::unique_ptr<ThreeAddressCode> ptr_cast =
        ThreeAddressCode::create(FunctionToPointerOperation(fntype), { var });
    var = ptr_cast->result(0);
    tacs.push_back(std::move(ptr_cast));
  }
  return var;
}

/* constant */

const Variable *
ConvertConstant(
    ::llvm::Constant *,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    context &);

static rvsdg::bitvalue_repr
convert_apint(const ::llvm::APInt & value)
{
  ::llvm::APInt v;
  if (value.isNegative())
    v = -value;

  auto str = toString(value, 2, false);
  std::reverse(str.begin(), str.end());

  rvsdg::bitvalue_repr vr(str.c_str());
  if (value.isNegative())
    vr = vr.sext(value.getBitWidth() - str.size());
  else
    vr = vr.zext(value.getBitWidth() - str.size());

  return vr;
}

static const Variable *
convert_int_constant(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<ThreeAddressCode>> & tacs,
    context &)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantIntVal);
  const auto constant = ::llvm::cast<const ::llvm::ConstantInt>(c);

  const auto v = convert_apint(constant->getValue());
  tacs.push_back(ThreeAddressCode::create(IntegerConstantOperation(std::move(v)), {}));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_undefvalue(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::UndefValueVal);

  auto t = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(UndefValueOperation::Create(t));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantExpr(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantExprVal);
  auto c = ::llvm::cast<::llvm::ConstantExpr>(constant);

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

static const Variable *
convert_constantFP(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantFPVal);
  auto c = ::llvm::cast<::llvm::ConstantFP>(constant);

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantFP::create(c->getValueAPF(), type));

  return tacs.back()->result(0);
}

static const Variable *
convert_globalVariable(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::GlobalVariableVal);
  return ConvertValue(c, tacs, ctx);
}

static const Variable *
convert_constantPointerNull(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<const ::llvm::ConstantPointerNull>(constant));
  auto & c = *::llvm::cast<const ::llvm::ConstantPointerNull>(constant);

  auto t = ctx.GetTypeConverter().ConvertPointerType(*c.getType());
  tacs.push_back(ConstantPointerNullOperation::Create(t));

  return tacs.back()->result(0);
}

static const Variable *
convert_blockAddress(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    context &)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::BlockAddressVal);

  JLM_UNREACHABLE("Blockaddress constants are not supported.");
}

static const Variable *
convert_constantAggregateZero(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantAggregateZeroVal);

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantAggregateZeroOperation::create(type));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantArray(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantArrayVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
  {
    auto operand = c->getOperand(n);
    JLM_ASSERT(::llvm::dyn_cast<const ::llvm::Constant>(operand));
    auto constant = ::llvm::cast<::llvm::Constant>(operand);
    elements.push_back(ConvertConstant(constant, tacs, ctx));
  }

  tacs.push_back(ConstantArrayOperation::create(elements));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantDataArray(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantDataArrayVal);
  const auto & c = *::llvm::cast<const ::llvm::ConstantDataArray>(constant);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c.getNumElements(); n++)
    elements.push_back(ConvertConstant(c.getElementAsConstant(n), tacs, ctx));

  tacs.push_back(ConstantDataArray::create(elements));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantDataVector(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantDataVectorVal);
  auto c = ::llvm::cast<const ::llvm::ConstantDataVector>(constant);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumElements(); n++)
    elements.push_back(ConvertConstant(c->getElementAsConstant(n), tacs, ctx));

  tacs.push_back(ConstantDataVectorOperation::Create(elements));

  return tacs.back()->result(0);
}

static const Variable *
ConvertConstantStruct(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantStructVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
    elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantStruct::create(elements, type));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantVector(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantVectorVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
    elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantVectorOperation::create(elements, type));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_globalAlias(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    context &)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::GlobalAliasVal);

  JLM_UNREACHABLE("GlobalAlias constants are not supported.");
}

static inline const Variable *
convert_function(::llvm::Constant * c, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::FunctionVal);
  return ConvertValue(c, tacs, ctx);
}

static const Variable *
ConvertConstant(
    ::llvm::PoisonValue * poisonValue,
    tacsvector_t & threeAddressCodeVector,
    llvm::context & context)
{
  auto type = context.GetTypeConverter().ConvertLlvmType(*poisonValue->getType());
  threeAddressCodeVector.push_back(PoisonValueOperation::Create(type));

  return threeAddressCodeVector.back()->result(0);
}

template<class T>
static const Variable *
ConvertConstant(
    ::llvm::Constant * constant,
    tacsvector_t & threeAddressCodeVector,
    llvm::context & context)
{
  JLM_ASSERT(::llvm::dyn_cast<T>(constant));
  return ConvertConstant(::llvm::cast<T>(constant), threeAddressCodeVector, context);
}

const Variable *
ConvertConstant(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  static std::unordered_map<
      unsigned,
      const Variable * (*)(::llvm::Constant *,
                           std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
                           context & ctx)>
      constantMap({ { ::llvm::Value::BlockAddressVal, convert_blockAddress },
                    { ::llvm::Value::ConstantAggregateZeroVal, convert_constantAggregateZero },
                    { ::llvm::Value::ConstantArrayVal, convert_constantArray },
                    { ::llvm::Value::ConstantDataArrayVal, convert_constantDataArray },
                    { ::llvm::Value::ConstantDataVectorVal, convert_constantDataVector },
                    { ::llvm::Value::ConstantExprVal, convert_constantExpr },
                    { ::llvm::Value::ConstantFPVal, convert_constantFP },
                    { ::llvm::Value::ConstantIntVal, convert_int_constant },
                    { ::llvm::Value::ConstantPointerNullVal, convert_constantPointerNull },
                    { ::llvm::Value::ConstantStructVal, ConvertConstantStruct },
                    { ::llvm::Value::ConstantVectorVal, convert_constantVector },
                    { ::llvm::Value::FunctionVal, convert_function },
                    { ::llvm::Value::GlobalAliasVal, convert_globalAlias },
                    { ::llvm::Value::GlobalVariableVal, convert_globalVariable },
                    { ::llvm::Value::PoisonValueVal, ConvertConstant<::llvm::PoisonValue> },
                    { ::llvm::Value::UndefValueVal, convert_undefvalue } });

  if (constantMap.find(c->getValueID()) != constantMap.end())
    return constantMap[c->getValueID()](c, tacs, ctx);

  JLM_UNREACHABLE("Unsupported LLVM Constant.");
}

std::vector<std::unique_ptr<llvm::ThreeAddressCode>>
ConvertConstant(::llvm::Constant * c, context & ctx)
{
  std::vector<std::unique_ptr<llvm::ThreeAddressCode>> tacs;
  ConvertConstant(c, tacs, ctx);
  return tacs;
}

/* instructions */

static inline const Variable *
convert_return_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Ret);
  auto i = ::llvm::cast<::llvm::ReturnInst>(instruction);

  auto bb = ctx.get(i->getParent());
  bb->add_outedge(bb->cfg().exit());
  if (!i->getReturnValue())
    return {};

  auto value = ConvertValue(i->getReturnValue(), tacs, ctx);
  tacs.push_back(AssignmentOperation::create(value, ctx.result()));

  return ctx.result();
}

static const Variable *
ConvertBranchInstruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Br);
  auto i = ::llvm::cast<::llvm::BranchInst>(instruction);
  auto bb = ctx.get(i->getParent());

  JLM_ASSERT(bb->NumOutEdges() == 0);

  if (i->isUnconditional())
  {
    bb->add_outedge(ctx.get(i->getSuccessor(0)));
    return nullptr;
  }

  bb->add_outedge(ctx.get(i->getSuccessor(1))); // False out-edge
  bb->add_outedge(ctx.get(i->getSuccessor(0))); // True out-edge

  auto c = ConvertValue(i->getCondition(), tacs, ctx);
  auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
  auto op = rvsdg::match_op(nbits, { { 1, 1 } }, 0, 2);
  tacs.push_back(ThreeAddressCode::create(op, { c }));
  tacs.push_back(BranchOperation::create(2, tacs.back()->result(0)));

  return nullptr;
}

static const Variable *
ConvertSwitchInstruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Switch);
  auto i = ::llvm::cast<::llvm::SwitchInst>(instruction);
  auto bb = ctx.get(i->getParent());

  JLM_ASSERT(bb->NumOutEdges() == 0);
  std::unordered_map<uint64_t, uint64_t> mapping;
  for (auto it = i->case_begin(); it != i->case_end(); it++)
  {
    JLM_ASSERT(it != i->case_default());
    auto edge = bb->add_outedge(ctx.get(it->getCaseSuccessor()));
    mapping[it->getCaseValue()->getZExtValue()] = edge->index();
  }

  auto defaultEdge = bb->add_outedge(ctx.get(i->case_default()->getCaseSuccessor()));

  auto c = ConvertValue(i->getCondition(), tacs, ctx);
  auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
  auto op = rvsdg::match_op(nbits, mapping, defaultEdge->index(), bb->NumOutEdges());
  tacs.push_back(ThreeAddressCode::create(op, { c }));
  tacs.push_back(BranchOperation::create(bb->NumOutEdges(), tacs.back()->result(0)));

  return nullptr;
}

static inline const Variable *
convert_unreachable_instruction(::llvm::Instruction * i, tacsvector_t &, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Unreachable);
  auto bb = ctx.get(i->getParent());
  bb->add_outedge(bb->cfg().exit());
  return nullptr;
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertIntegerIcmpPredicate(const ::llvm::CmpInst::Predicate predicate, const std::size_t numBits)
{
  switch (predicate)
  {
  case ::llvm::CmpInst::ICMP_SLT:
    return std::make_unique<IntegerSltOperation>(numBits);
  case ::llvm::CmpInst::ICMP_ULT:
    return std::make_unique<IntegerUltOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SLE:
    return std::make_unique<IntegerSleOperation>(numBits);
  case ::llvm::CmpInst::ICMP_ULE:
    return std::make_unique<IntegerUleOperation>(numBits);
  case ::llvm::CmpInst::ICMP_EQ:
    return std::make_unique<IntegerEqOperation>(numBits);
  case ::llvm::CmpInst::ICMP_NE:
    return std::make_unique<IntegerNeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SGE:
    return std::make_unique<IntegerSgeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_UGE:
    return std::make_unique<IntegerUgeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SGT:
    return std::make_unique<IntegerSgtOperation>(numBits);
  case ::llvm::CmpInst::ICMP_UGT:
    return std::make_unique<IntegerUgtOperation>(numBits);
  default:
    JLM_UNREACHABLE("ConvertIntegerIcmpPredicate: Unsupported icmp predicate.");
  }
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertPointerIcmpPredicate(const ::llvm::CmpInst::Predicate predicate)
{
  switch (predicate)
  {
  case ::llvm::CmpInst::ICMP_ULT:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::lt);
  case ::llvm::CmpInst::ICMP_ULE:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::le);
  case ::llvm::CmpInst::ICMP_EQ:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::eq);
  case ::llvm::CmpInst::ICMP_NE:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::ne);
  case ::llvm::CmpInst::ICMP_UGE:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::ge);
  case ::llvm::CmpInst::ICMP_UGT:
    return std::make_unique<PtrCmpOperation>(PointerType::Create(), cmp::gt);
  default:
    JLM_UNREACHABLE("ConvertPointerIcmpPredicate: Unsupported icmp predicate.");
  }
}

static const Variable *
convert(const ::llvm::ICmpInst * instruction, tacsvector_t & tacs, context & ctx)
{
  const auto predicate = instruction->getPredicate();
  const auto operandType = instruction->getOperand(0)->getType();
  auto op1 = ConvertValue(instruction->getOperand(0), tacs, ctx);
  auto op2 = ConvertValue(instruction->getOperand(1), tacs, ctx);

  std::unique_ptr<rvsdg::BinaryOperation> operation;
  if (operandType->isVectorTy() && operandType->getScalarType()->isIntegerTy())
  {
    operation =
        ConvertIntegerIcmpPredicate(predicate, operandType->getScalarType()->getIntegerBitWidth());
  }
  else if (operandType->isVectorTy() && operandType->getScalarType()->isPointerTy())
  {
    operation = ConvertPointerIcmpPredicate(predicate);
  }
  else if (operandType->isIntegerTy())
  {
    operation = ConvertIntegerIcmpPredicate(predicate, operandType->getIntegerBitWidth());
  }
  else if (operandType->isPointerTy())
  {
    operation = ConvertPointerIcmpPredicate(predicate);
  }
  else
  {
    JLM_UNREACHABLE("convert: Unhandled icmp type.");
  }

  if (operandType->isVectorTy())
  {
    const auto instructionType = ctx.GetTypeConverter().ConvertLlvmType(*instruction->getType());
    tacs.push_back(VectorBinaryOperation::create(*operation, op1, op2, instructionType));
  }
  else
  {
    tacs.push_back(ThreeAddressCode::create(*operation, { op1, op2 }));
  }

  return tacs.back()->result(0);
}

static const Variable *
convert_fcmp_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::FCmp);
  auto & typeConverter = ctx.GetTypeConverter();
  auto i = ::llvm::cast<const ::llvm::FCmpInst>(instruction);
  auto t = i->getOperand(0)->getType();

  static std::unordered_map<::llvm::CmpInst::Predicate, llvm::fpcmp> map(
      { { ::llvm::CmpInst::FCMP_TRUE, fpcmp::TRUE },
        { ::llvm::CmpInst::FCMP_FALSE, fpcmp::FALSE },
        { ::llvm::CmpInst::FCMP_OEQ, fpcmp::oeq },
        { ::llvm::CmpInst::FCMP_OGT, fpcmp::ogt },
        { ::llvm::CmpInst::FCMP_OGE, fpcmp::oge },
        { ::llvm::CmpInst::FCMP_OLT, fpcmp::olt },
        { ::llvm::CmpInst::FCMP_OLE, fpcmp::ole },
        { ::llvm::CmpInst::FCMP_ONE, fpcmp::one },
        { ::llvm::CmpInst::FCMP_ORD, fpcmp::ord },
        { ::llvm::CmpInst::FCMP_UNO, fpcmp::uno },
        { ::llvm::CmpInst::FCMP_UEQ, fpcmp::ueq },
        { ::llvm::CmpInst::FCMP_UGT, fpcmp::ugt },
        { ::llvm::CmpInst::FCMP_UGE, fpcmp::uge },
        { ::llvm::CmpInst::FCMP_ULT, fpcmp::ult },
        { ::llvm::CmpInst::FCMP_ULE, fpcmp::ule },
        { ::llvm::CmpInst::FCMP_UNE, fpcmp::une } });

  auto type = typeConverter.ConvertLlvmType(*i->getType());

  auto op1 = ConvertValue(i->getOperand(0), tacs, ctx);
  auto op2 = ConvertValue(i->getOperand(1), tacs, ctx);

  JLM_ASSERT(map.find(i->getPredicate()) != map.end());
  auto fptype = t->isVectorTy() ? t->getScalarType() : t;
  FCmpOperation operation(map[i->getPredicate()], typeConverter.ExtractFloatingPointSize(*fptype));

  if (t->isVectorTy())
    tacs.push_back(VectorBinaryOperation::create(operation, op1, op2, type));
  else
    tacs.push_back(ThreeAddressCode::create(operation, { op1, op2 }));

  return tacs.back()->result(0);
}

static const Variable *
AddIOBarrier(tacsvector_t & tacs, const Variable * operand, const context & ctx)
{
  const auto ioBarrierOperation = std::make_unique<IOBarrierOperation>(operand->Type());
  tacs.push_back(ThreeAddressCode::create(*ioBarrierOperation, { operand, ctx.iostate() }));
  return tacs.back()->result(0);
}

static inline const Variable *
convert_load_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Load);
  auto instruction = static_cast<::llvm::LoadInst *>(i);

  auto alignment = instruction->getAlign().value();
  auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
  auto loadedType = ctx.GetTypeConverter().ConvertLlvmType(*instruction->getType());

  const ThreeAddressCodeVariable * loadedValue = nullptr;
  const ThreeAddressCodeVariable * memoryState = nullptr;
  const ThreeAddressCodeVariable * ioState = nullptr;
  if (instruction->isVolatile())
  {
    auto loadVolatileTac = LoadVolatileOperation::Create(
        address,
        ctx.iostate(),
        ctx.memory_state(),
        loadedType,
        alignment);
    tacs.push_back(std::move(loadVolatileTac));

    loadedValue = tacs.back()->result(0);
    ioState = tacs.back()->result(1);
    memoryState = tacs.back()->result(2);
  }
  else
  {
    address = AddIOBarrier(tacs, address, ctx);
    auto loadTac =
        LoadNonVolatileOperation::Create(address, ctx.memory_state(), loadedType, alignment);
    tacs.push_back(std::move(loadTac));
    loadedValue = tacs.back()->result(0);
    memoryState = tacs.back()->result(1);
  }

  if (ioState)
  {
    tacs.push_back(AssignmentOperation::create(ioState, ctx.iostate()));
  }
  tacs.push_back(AssignmentOperation::create(memoryState, ctx.memory_state()));

  return loadedValue;
}

static inline const Variable *
convert_store_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Store);
  auto instruction = static_cast<::llvm::StoreInst *>(i);

  auto alignment = instruction->getAlign().value();
  auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
  auto value = ConvertValue(instruction->getValueOperand(), tacs, ctx);

  const ThreeAddressCodeVariable * memoryState = nullptr;
  const ThreeAddressCodeVariable * ioState = nullptr;
  if (instruction->isVolatile())
  {
    auto storeVolatileTac = StoreVolatileOperation::Create(
        address,
        value,
        ctx.iostate(),
        ctx.memory_state(),
        alignment);
    tacs.push_back(std::move(storeVolatileTac));
    ioState = tacs.back()->result(0);
    memoryState = tacs.back()->result(1);
  }
  else
  {
    address = AddIOBarrier(tacs, address, ctx);
    auto storeTac =
        StoreNonVolatileOperation::Create(address, value, ctx.memory_state(), alignment);
    tacs.push_back(std::move(storeTac));
    memoryState = tacs.back()->result(0);
  }

  if (ioState)
  {
    tacs.push_back(AssignmentOperation::create(ioState, ctx.iostate()));
  }
  tacs.push_back(AssignmentOperation::create(memoryState, ctx.memory_state()));

  return nullptr;
}

static const Variable *
ConvertPhiInstruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  // Some of the blocks reaching this phi instruction might not be converted yet,
  // so some of the phi's operands may reference instructions that have not yet been converted.
  // For now, a SsaPhiOperation with no operands is created.
  // Once all basic blocks are converted, all SsaPhiOperations are revisited and given operands.
  auto type = ctx.GetTypeConverter().ConvertLlvmType(*i->getType());
  tacs.push_back(SsaPhiOperation::create({}, std::move(type)));
  return tacs.back()->result(0);
}

static const Variable *
convert_getelementptr_instruction(::llvm::Instruction * inst, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<const ::llvm::GetElementPtrInst>(inst));
  auto & typeConverter = ctx.GetTypeConverter();
  auto i = ::llvm::cast<::llvm::GetElementPtrInst>(inst);

  std::vector<const Variable *> indices;
  auto base = ConvertValue(i->getPointerOperand(), tacs, ctx);
  for (auto it = i->idx_begin(); it != i->idx_end(); it++)
    indices.push_back(ConvertValue(*it, tacs, ctx));

  auto pointeeType = typeConverter.ConvertLlvmType(*i->getSourceElementType());
  auto resultType = typeConverter.ConvertLlvmType(*i->getType());

  tacs.push_back(GetElementPtrOperation::Create(base, indices, pointeeType, resultType));

  return tacs.back()->result(0);
}

static const Variable *
convert_malloc_call(const ::llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
  auto memstate = ctx.memory_state();

  auto size = ConvertValue(i->getArgOperand(0), tacs, ctx);

  tacs.push_back(MallocOperation::create(size));
  auto result = tacs.back()->result(0);
  auto mstate = tacs.back()->result(1);

  tacs.push_back(MemoryStateMergeOperation::Create({ mstate, memstate }));
  tacs.push_back(AssignmentOperation::create(tacs.back()->result(0), memstate));

  return result;
}

static const Variable *
convert_free_call(const ::llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
{
  auto iostate = ctx.iostate();
  auto memstate = ctx.memory_state();

  auto pointer = ConvertValue(i->getArgOperand(0), tacs, ctx);

  tacs.push_back(FreeOperation::Create(pointer, { memstate }, iostate));
  auto & freeThreeAddressCode = *tacs.back().get();

  tacs.push_back(AssignmentOperation::create(freeThreeAddressCode.result(0), memstate));
  tacs.push_back(AssignmentOperation::create(freeThreeAddressCode.result(1), iostate));

  return nullptr;
}

/**
 * In LLVM, the memcpy intrinsic is modeled as a call instruction. It expects four arguments, with
 * the fourth argument being a ConstantInt of bit width 1 to encode the volatile flag for the memcpy
 * instruction. This function takes this argument and converts it to a boolean flag.
 *
 * @param value The volatile argument of the memcpy intrinsic.
 * @return Boolean flag indicating whether the memcpy is volatile.
 */
static bool
IsVolatile(const ::llvm::Value & value)
{
  auto constant = ::llvm::dyn_cast<const ::llvm::ConstantInt>(&value);
  JLM_ASSERT(constant != nullptr && constant->getType()->getIntegerBitWidth() == 1);

  auto apInt = constant->getValue();
  JLM_ASSERT(apInt.isZero() || apInt.isOne());

  return apInt.isOne();
}

static const Variable *
convert_memcpy_call(const ::llvm::CallInst * instruction, tacsvector_t & tacs, context & ctx)
{
  auto ioState = ctx.iostate();
  auto memoryState = ctx.memory_state();

  auto destination = ConvertValue(instruction->getArgOperand(0), tacs, ctx);
  auto source = ConvertValue(instruction->getArgOperand(1), tacs, ctx);
  auto length = ConvertValue(instruction->getArgOperand(2), tacs, ctx);

  if (IsVolatile(*instruction->getArgOperand(3)))
  {
    tacs.push_back(MemCpyVolatileOperation::CreateThreeAddressCode(
        *destination,
        *source,
        *length,
        *ioState,
        { memoryState }));
    auto & memCpyVolatileTac = *tacs.back();
    tacs.push_back(AssignmentOperation::create(memCpyVolatileTac.result(0), ioState));
    tacs.push_back(AssignmentOperation::create(memCpyVolatileTac.result(1), memoryState));
  }
  else
  {
    tacs.push_back(
        MemCpyNonVolatileOperation::create(destination, source, length, { memoryState }));
    tacs.push_back(AssignmentOperation::create(tacs.back()->result(0), memoryState));
  }

  return nullptr;
}

static const Variable *
convert_call_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Call);
  auto i = ::llvm::cast<::llvm::CallInst>(instruction);

  auto create_arguments = [](const ::llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
  {
    auto functionType = i->getFunctionType();
    std::vector<const llvm::Variable *> arguments;
    for (size_t n = 0; n < functionType->getNumParams(); n++)
      arguments.push_back(ConvertValue(i->getArgOperand(n), tacs, ctx));

    return arguments;
  };

  auto create_varargs = [](const ::llvm::CallInst * i, tacsvector_t & tacs, context & ctx)
  {
    auto functionType = i->getFunctionType();
    std::vector<const llvm::Variable *> varargs;
    for (size_t n = functionType->getNumParams(); n < i->getNumOperands() - 1; n++)
      varargs.push_back(ConvertValue(i->getArgOperand(n), tacs, ctx));

    tacs.push_back(VariadicArgumentListOperation::create(varargs));
    return tacs.back()->result(0);
  };

  auto is_malloc_call = [](const ::llvm::CallInst * i)
  {
    auto f = i->getCalledFunction();
    return f && f->getName() == "malloc";
  };

  auto is_free_call = [](const ::llvm::CallInst * i)
  {
    auto f = i->getCalledFunction();
    return f && f->getName() == "free";
  };

  auto IsMemcpyCall = [](const ::llvm::CallInst * i)
  {
    return ::llvm::dyn_cast<::llvm::MemCpyInst>(i) != nullptr;
  };

  if (is_malloc_call(i))
    return convert_malloc_call(i, tacs, ctx);
  if (is_free_call(i))
    return convert_free_call(i, tacs, ctx);
  if (IsMemcpyCall(i))
    return convert_memcpy_call(i, tacs, ctx);

  auto ftype = i->getFunctionType();
  auto convertedFType = ctx.GetTypeConverter().ConvertFunctionType(*ftype);

  auto arguments = create_arguments(i, tacs, ctx);
  if (ftype->isVarArg())
    arguments.push_back(create_varargs(i, tacs, ctx));
  arguments.push_back(ctx.iostate());
  arguments.push_back(ctx.memory_state());

  const Variable * callee = ConvertValueOrFunction(i->getCalledOperand(), tacs, ctx);
  // Llvm does not distinguish between "function objects" and
  // "pointers to functions" while we need to be precise in modelling.
  // If the called object is a function object, then we can just
  // feed it to the call operator directly, otherwise we have
  // to cast it into a function object.
  if (is<PointerType>(*callee->Type()))
  {
    std::unique_ptr<ThreeAddressCode> callee_cast =
        ThreeAddressCode::create(PointerToFunctionOperation(convertedFType), { callee });
    callee = callee_cast->result(0);
    tacs.push_back(std::move(callee_cast));
  }
  else if (auto fntype = std::dynamic_pointer_cast<const rvsdg::FunctionType>(callee->Type()))
  {
    // Llvm also allows argument type mismatches if the function
    // features varargs. The code here could be made more precise by
    // validating and accepting only vararg-related mismatches.
    if (*convertedFType != *fntype)
    {
      // Since vararg passing is not modelled explicitly, simply hide the
      // argument mismtach via pointer casts.
      std::unique_ptr<ThreeAddressCode> ptrCast =
          ThreeAddressCode::create(FunctionToPointerOperation(fntype), { callee });
      std::unique_ptr<ThreeAddressCode> fnCast = ThreeAddressCode::create(
          PointerToFunctionOperation(convertedFType),
          { ptrCast->result(0) });
      callee = fnCast->result(0);
      tacs.push_back(std::move(ptrCast));
      tacs.push_back(std::move(fnCast));
    }
  }
  else
  {
    throw std::runtime_error("Unexpected callee type: " + callee->Type()->debug_string());
  }

  auto call = CallOperation::create(callee, convertedFType, arguments);

  auto result = call->result(0);
  auto iostate = call->result(call->nresults() - 2);
  auto memstate = call->result(call->nresults() - 1);

  tacs.push_back(std::move(call));
  tacs.push_back(AssignmentOperation::create(iostate, ctx.iostate()));
  tacs.push_back(AssignmentOperation::create(memstate, ctx.memory_state()));

  return result;
}

static inline const Variable *
convert_select_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Select);
  auto instruction = static_cast<::llvm::SelectInst *>(i);

  auto p = ConvertValue(instruction->getCondition(), tacs, ctx);
  auto t = ConvertValue(instruction->getTrueValue(), tacs, ctx);
  auto f = ConvertValue(instruction->getFalseValue(), tacs, ctx);

  if (i->getType()->isVectorTy())
    tacs.push_back(VectorSelectOperation::create(p, t, f));
  else
    tacs.push_back(SelectOperation::create(p, t, f));

  return tacs.back()->result(0);
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertIntegerBinaryOperation(
    const ::llvm::Instruction::BinaryOps binaryOperation,
    std::size_t numBits)
{
  switch (binaryOperation)
  {
  case ::llvm::Instruction::Add:
    return std::make_unique<IntegerAddOperation>(numBits);
  case ::llvm::Instruction::And:
    return std::make_unique<IntegerAndOperation>(numBits);
  case ::llvm::Instruction::AShr:
    return std::make_unique<IntegerAShrOperation>(numBits);
  case ::llvm::Instruction::LShr:
    return std::make_unique<IntegerLShrOperation>(numBits);
  case ::llvm::Instruction::Mul:
    return std::make_unique<IntegerMulOperation>(numBits);
  case ::llvm::Instruction::Or:
    return std::make_unique<IntegerOrOperation>(numBits);
  case ::llvm::Instruction::SDiv:
    return std::make_unique<IntegerSDivOperation>(numBits);
  case ::llvm::Instruction::Shl:
    return std::make_unique<IntegerShlOperation>(numBits);
  case ::llvm::Instruction::SRem:
    return std::make_unique<IntegerSRemOperation>(numBits);
  case ::llvm::Instruction::Sub:
    return std::make_unique<IntegerSubOperation>(numBits);
  case ::llvm::Instruction::UDiv:
    return std::make_unique<IntegerUDivOperation>(numBits);
  case ::llvm::Instruction::URem:
    return std::make_unique<IntegerURemOperation>(numBits);
  case ::llvm::Instruction::Xor:
    return std::make_unique<IntegerXorOperation>(numBits);
  default:
    JLM_UNREACHABLE("ConvertIntegerBinaryOperation: Unsupported integer binary operation");
  }
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertFloatingPointBinaryOperation(
    const ::llvm::Instruction::BinaryOps binaryOperation,
    fpsize floatingPointSize)
{
  switch (binaryOperation)
  {
  case ::llvm::Instruction::FAdd:
    return std::make_unique<fpbin_op>(fpop::add, floatingPointSize);
  case ::llvm::Instruction::FSub:
    return std::make_unique<fpbin_op>(fpop::sub, floatingPointSize);
  case ::llvm::Instruction::FMul:
    return std::make_unique<fpbin_op>(fpop::mul, floatingPointSize);
  case ::llvm::Instruction::FDiv:
    return std::make_unique<fpbin_op>(fpop::div, floatingPointSize);
  case ::llvm::Instruction::FRem:
    return std::make_unique<fpbin_op>(fpop::mod, floatingPointSize);
  default:
    JLM_UNREACHABLE("ConvertFloatingPointBinaryOperation: Unsupported binary operation");
  }
}

static const Variable *
convert(const ::llvm::BinaryOperator * instruction, tacsvector_t & tacs, context & ctx)
{
  const auto llvmType = instruction->getType();
  auto & typeConverter = ctx.GetTypeConverter();
  const auto opcode = instruction->getOpcode();

  std::unique_ptr<rvsdg::BinaryOperation> operation;
  if (llvmType->isVectorTy() && llvmType->getScalarType()->isIntegerTy())
  {
    const auto numBits = llvmType->getScalarType()->getIntegerBitWidth();
    operation = ConvertIntegerBinaryOperation(opcode, numBits);
  }
  else if (llvmType->isVectorTy() && llvmType->getScalarType()->isFloatingPointTy())
  {
    const auto size = typeConverter.ExtractFloatingPointSize(*llvmType->getScalarType());
    operation = ConvertFloatingPointBinaryOperation(opcode, size);
  }
  else if (llvmType->isIntegerTy())
  {
    operation = ConvertIntegerBinaryOperation(opcode, llvmType->getIntegerBitWidth());
  }
  else if (llvmType->isFloatingPointTy())
  {
    const auto size = typeConverter.ExtractFloatingPointSize(*llvmType);
    operation = ConvertFloatingPointBinaryOperation(opcode, size);
  }
  else
  {
    JLM_ASSERT("convert: Unhandled binary operation type.");
  }

  const auto jlmType = typeConverter.ConvertLlvmType(*llvmType);
  auto operand1 = ConvertValue(instruction->getOperand(0), tacs, ctx);
  auto operand2 = ConvertValue(instruction->getOperand(1), tacs, ctx);

  if (instruction->getOpcode() == ::llvm::Instruction::SDiv
      || instruction->getOpcode() == ::llvm::Instruction::UDiv
      || instruction->getOpcode() == ::llvm::Instruction::SRem
      || instruction->getOpcode() == ::llvm::Instruction::URem)
  {
    operand1 = AddIOBarrier(tacs, operand1, ctx);
  }

  if (llvmType->isVectorTy())
  {
    tacs.push_back(VectorBinaryOperation::create(*operation, operand1, operand2, jlmType));
  }
  else
  {
    tacs.push_back(ThreeAddressCode::create(*operation, { operand1, operand2 }));
  }

  return tacs.back()->result(0);
}

static inline const Variable *
convert_alloca_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Alloca);
  auto i = static_cast<::llvm::AllocaInst *>(instruction);

  auto memstate = ctx.memory_state();
  auto size = ConvertValue(i->getArraySize(), tacs, ctx);
  auto vtype = ctx.GetTypeConverter().ConvertLlvmType(*i->getAllocatedType());
  auto alignment = i->getAlign().value();

  tacs.push_back(AllocaOperation::create(vtype, size, alignment));
  auto result = tacs.back()->result(0);
  auto astate = tacs.back()->result(1);

  tacs.push_back(MemoryStateMergeOperation::Create({ astate, memstate }));
  tacs.push_back(AssignmentOperation::create(tacs.back()->result(0), memstate));

  return result;
}

static const Variable *
convert_extractvalue(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::ExtractValue);
  auto ev = ::llvm::dyn_cast<::llvm::ExtractValueInst>(i);

  auto aggregate = ConvertValue(ev->getOperand(0), tacs, ctx);
  tacs.push_back(ExtractValueOperation::create(aggregate, ev->getIndices()));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_extractelement_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::ExtractElement);

  auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
  auto index = ConvertValue(i->getOperand(1), tacs, ctx);
  tacs.push_back(ExtractElementOperation::create(vector, index));

  return tacs.back()->result(0);
}

static const Variable *
convert(::llvm::ShuffleVectorInst * i, tacsvector_t & tacs, context & ctx)
{
  auto v1 = ConvertValue(i->getOperand(0), tacs, ctx);
  auto v2 = ConvertValue(i->getOperand(1), tacs, ctx);

  std::vector<int> mask;
  for (auto & element : i->getShuffleMask())
    mask.push_back(element);

  tacs.push_back(ShuffleVectorOperation::create(v1, v2, mask));

  return tacs.back()->result(0);
}

static const Variable *
convert_insertelement_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::InsertElement);

  auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
  auto value = ConvertValue(i->getOperand(1), tacs, ctx);
  auto index = ConvertValue(i->getOperand(2), tacs, ctx);
  tacs.push_back(InsertElementOperation::create(vector, value, index));

  return tacs.back()->result(0);
}

static const Variable *
convert(::llvm::UnaryOperator * unaryOperator, tacsvector_t & threeAddressCodeVector, context & ctx)
{
  JLM_ASSERT(unaryOperator->getOpcode() == ::llvm::Instruction::FNeg);
  auto & typeConverter = ctx.GetTypeConverter();

  auto type = unaryOperator->getType();
  auto scalarType = typeConverter.ConvertLlvmType(*type->getScalarType());
  auto operand = ConvertValue(unaryOperator->getOperand(0), threeAddressCodeVector, ctx);

  if (type->isVectorTy())
  {
    auto vectorType = typeConverter.ConvertLlvmType(*type);
    threeAddressCodeVector.push_back(VectorUnaryOperation::create(
        FNegOperation(std::static_pointer_cast<const FloatingPointType>(scalarType)),
        operand,
        vectorType));
  }
  else
  {
    threeAddressCodeVector.push_back(FNegOperation::create(operand));
  }

  return threeAddressCodeVector.back()->result(0);
}

template<class OP>
static std::unique_ptr<rvsdg::Operation>
create_unop(std::shared_ptr<const rvsdg::Type> st, std::shared_ptr<const rvsdg::Type> dt)
{
  return std::unique_ptr<rvsdg::Operation>(new OP(std::move(st), std::move(dt)));
}

static const Variable *
convert_cast_instruction(::llvm::Instruction * i, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<::llvm::CastInst>(i));
  auto & typeConverter = ctx.GetTypeConverter();
  auto st = i->getOperand(0)->getType();
  auto dt = i->getType();

  static std::unordered_map<
      unsigned,
      std::unique_ptr<rvsdg::Operation> (*)(
          std::shared_ptr<const rvsdg::Type>,
          std::shared_ptr<const rvsdg::Type>)>
      map({ { ::llvm::Instruction::Trunc, create_unop<TruncOperation> },
            { ::llvm::Instruction::ZExt, create_unop<ZExtOperation> },
            { ::llvm::Instruction::UIToFP, create_unop<UIToFPOperation> },
            { ::llvm::Instruction::SIToFP, create_unop<SIToFPOperation> },
            { ::llvm::Instruction::SExt, create_unop<SExtOperation> },
            { ::llvm::Instruction::PtrToInt, create_unop<PtrToIntOperation> },
            { ::llvm::Instruction::IntToPtr, create_unop<IntegerToPointerOperation> },
            { ::llvm::Instruction::FPTrunc, create_unop<FPTruncOperation> },
            { ::llvm::Instruction::FPToSI, create_unop<FloatingPointToSignedIntegerOperation> },
            { ::llvm::Instruction::FPToUI, create_unop<FloatingPointToUnsignedIntegerOperation> },
            { ::llvm::Instruction::FPExt, create_unop<FPExtOperation> },
            { ::llvm::Instruction::BitCast, create_unop<BitCastOperation> } });

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*i->getType());

  auto op = ConvertValue(i->getOperand(0), tacs, ctx);
  auto srctype = typeConverter.ConvertLlvmType(*(st->isVectorTy() ? st->getScalarType() : st));
  auto dsttype = typeConverter.ConvertLlvmType(*(dt->isVectorTy() ? dt->getScalarType() : dt));

  JLM_ASSERT(map.find(i->getOpcode()) != map.end());
  auto unop = map[i->getOpcode()](std::move(srctype), std::move(dsttype));
  JLM_ASSERT(is<rvsdg::UnaryOperation>(*unop));

  if (dt->isVectorTy())
    tacs.push_back(
        VectorUnaryOperation::create(*static_cast<rvsdg::UnaryOperation *>(unop.get()), op, type));
  else
    tacs.push_back(
        ThreeAddressCode::create(*static_cast<rvsdg::SimpleOperation *>(unop.get()), { op }));

  return tacs.back()->result(0);
}

template<class INSTRUCTIONTYPE>
static const Variable *
convert(::llvm::Instruction * instruction, tacsvector_t & tacs, context & ctx)
{
  JLM_ASSERT(::llvm::isa<INSTRUCTIONTYPE>(instruction));
  return convert(::llvm::cast<INSTRUCTIONTYPE>(instruction), tacs, ctx);
}

const Variable *
ConvertInstruction(
    ::llvm::Instruction * i,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    context & ctx)
{
  if (i->isCast())
    return convert_cast_instruction(i, tacs, ctx);

  static std::unordered_map<
      unsigned,
      const Variable * (*)(::llvm::Instruction *,
                           std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
                           context &)>
      map({ { ::llvm::Instruction::Ret, convert_return_instruction },
            { ::llvm::Instruction::Br, ConvertBranchInstruction },
            { ::llvm::Instruction::Switch, ConvertSwitchInstruction },
            { ::llvm::Instruction::Unreachable, convert_unreachable_instruction },
            { ::llvm::Instruction::Add, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::And, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::AShr, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::Sub, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::UDiv, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::SDiv, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::URem, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::SRem, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::Shl, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::LShr, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::Or, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::Xor, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::Mul, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FAdd, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FSub, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FMul, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FDiv, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FRem, convert<::llvm::BinaryOperator> },
            { ::llvm::Instruction::FNeg, convert<::llvm::UnaryOperator> },
            { ::llvm::Instruction::ICmp, convert<::llvm::ICmpInst> },
            { ::llvm::Instruction::FCmp, convert_fcmp_instruction },
            { ::llvm::Instruction::Load, convert_load_instruction },
            { ::llvm::Instruction::Store, convert_store_instruction },
            { ::llvm::Instruction::PHI, ConvertPhiInstruction },
            { ::llvm::Instruction::GetElementPtr, convert_getelementptr_instruction },
            { ::llvm::Instruction::Call, convert_call_instruction },
            { ::llvm::Instruction::Select, convert_select_instruction },
            { ::llvm::Instruction::Alloca, convert_alloca_instruction },
            { ::llvm::Instruction::ExtractValue, convert_extractvalue },
            { ::llvm::Instruction::ExtractElement, convert_extractelement_instruction },
            { ::llvm::Instruction::ShuffleVector, convert<::llvm::ShuffleVectorInst> },
            { ::llvm::Instruction::InsertElement, convert_insertelement_instruction } });

  if (map.find(i->getOpcode()) == map.end())
    JLM_UNREACHABLE(util::strfmt(i->getOpcodeName(), " is not supported.").c_str());

  return map[i->getOpcode()](i, tacs, ctx);
}

}
