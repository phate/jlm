/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/control.hpp>

#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>

#include <jlm/llvm/backend/jlm2llvm/context.hpp>
#include <jlm/llvm/backend/jlm2llvm/instruction.hpp>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <typeindex>

namespace jlm::llvm
{

namespace jlm2llvm
{

::llvm::Value *
convert_operation(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & arguments,
    ::llvm::IRBuilder<> & builder,
    context & ctx);

static inline ::llvm::Value *
convert_assignment(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  JLM_ASSERT(is<assignment_op>(op));
  return ctx.value(args[0]);
}

static ::llvm::Value *
CreateBinOpInstruction(
    const ::llvm::Instruction::BinaryOps opcode,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    const context & ctx)
{
  const auto operand1 = ctx.value(args[0]);
  const auto operand2 = ctx.value(args[1]);
  return builder.CreateBinOp(opcode, operand1, operand2);
}

static ::llvm::Value *
CreateICmpInstruction(
    const ::llvm::CmpInst::Predicate predicate,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    const context & ctx)
{
  const auto operand1 = ctx.value(args[0]);
  const auto operand2 = ctx.value(args[1]);
  return builder.CreateICmp(predicate, operand1, operand2);
}

static ::llvm::APInt
convert_bitvalue_repr(const rvsdg::bitvalue_repr & vr)
{
  JLM_ASSERT(vr.is_defined());

  std::string str = vr.str();
  std::reverse(str.begin(), str.end());

  return ::llvm::APInt(vr.nbits(), str, 2);
}

static inline ::llvm::Value *
convert_bitconstant(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> & builder,
    context &)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::bitconstant_op *>(&op));
  auto value = static_cast<const rvsdg::bitconstant_op *>(&op)->value();

  auto type = ::llvm::IntegerType::get(builder.getContext(), value.nbits());

  if (value.is_defined())
    return ::llvm::ConstantInt::get(type, convert_bitvalue_repr(value));

  return ::llvm::UndefValue::get(type);
}

static inline ::llvm::Value *
convert_ctlconstant(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> & builder,
    context &)
{
  JLM_ASSERT(is_ctlconstant_op(op));
  auto & cop = *static_cast<const rvsdg::ctlconstant_op *>(&op);

  size_t nbits = cop.value().nalternatives() == 2 ? 1 : 32;
  auto type = ::llvm::IntegerType::get(builder.getContext(), nbits);
  return ::llvm::ConstantInt::get(type, cop.value().alternative());
}

static ::llvm::Value *
convert(
    const ConstantFP & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> & builder,
    context &)
{
  return ::llvm::ConstantFP::get(builder.getContext(), op.constant());
}

static inline ::llvm::Value *
convert_undef(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  JLM_ASSERT(is<UndefValueOperation>(op));
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto & resultType = *op.result(0);

  // MemoryState has no llvm representation.
  if (is<MemoryStateType>(resultType))
    return nullptr;

  auto type = typeConverter.ConvertJlmType(resultType, llvmContext);
  return ::llvm::UndefValue::get(type);
}

static ::llvm::Value *
convert(
    const PoisonValueOperation & operation,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(operation.GetType(), llvmContext);
  return ::llvm::PoisonValue::get(type);
}

static ::llvm::Value *
convert(
    const CallOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto function = ctx.value(args[0]);
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  std::vector<::llvm::Value *> operands;
  for (size_t n = 1; n < args.size(); n++)
  {
    auto argument = args[n];

    if (rvsdg::is<IOStateType>(argument->type()))
      continue;
    if (rvsdg::is<MemoryStateType>(argument->type()))
      continue;

    if (rvsdg::is<VariableArgumentType>(argument->type()))
    {
      JLM_ASSERT(is<tacvariable>(argument));
      auto valist = dynamic_cast<const llvm::tacvariable *>(argument)->tac();
      JLM_ASSERT(is<valist_op>(valist->operation()));
      for (size_t n = 0; n < valist->noperands(); n++)
        operands.push_back(ctx.value(valist->operand(n)));
      continue;
    }

    operands.push_back(ctx.value(argument));
  }

  auto ftype = typeConverter.ConvertFunctionType(*op.GetFunctionType(), llvmContext);
  return builder.CreateCall(ftype, function, operands);
}

static inline bool
is_identity_mapping(const rvsdg::match_op & op)
{
  for (const auto & pair : op)
  {
    if (pair.first != pair.second)
      return false;
  }

  return true;
}

static inline ::llvm::Value *
convert_match(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<rvsdg::match_op>(op));
  auto mop = static_cast<const rvsdg::match_op *>(&op);

  if (is_identity_mapping(*mop))
    return ctx.value(args[0]);

  if (mop->nalternatives() == 2 && mop->nbits() == 1)
  {
    auto i1 = ::llvm::IntegerType::get(builder.getContext(), 1);
    auto t = ::llvm::ConstantInt::getFalse(i1);
    auto f = ::llvm::ConstantInt::getTrue(i1);
    return builder.CreateSelect(ctx.value(args[0]), t, f);
  }

  /* FIXME: This is not working if the match is not directly connected to a gamma node. */
  return ctx.value(args[0]);
}

static inline ::llvm::Value *
convert_branch(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  JLM_ASSERT(is<branch_op>(op));
  return nullptr;
}

static inline ::llvm::Value *
convert_phi(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<phi_op>(op));
  auto & phi = *static_cast<const llvm::phi_op *>(&op);
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  if (rvsdg::is<IOStateType>(phi.type()))
    return nullptr;
  if (rvsdg::is<MemoryStateType>(phi.type()))
    return nullptr;

  auto t = typeConverter.ConvertJlmType(phi.type(), llvmContext);
  return builder.CreatePHI(t, op.narguments());
}

static ::llvm::Value *
CreateLoadInstruction(
    const rvsdg::ValueType & loadedType,
    const variable * address,
    bool isVolatile,
    size_t alignment,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(loadedType, llvmContext);
  auto loadInstruction = builder.CreateLoad(type, ctx.value(address), isVolatile);
  loadInstruction->setAlignment(::llvm::Align(alignment));
  return loadInstruction;
}

static ::llvm::Value *
convert(
    const LoadNonVolatileOperation & operation,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  return CreateLoadInstruction(
      *operation.GetLoadedType(),
      operands[0],
      false,
      operation.GetAlignment(),
      builder,
      ctx);
}

static ::llvm::Value *
convert(
    const LoadVolatileOperation & operation,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  return CreateLoadInstruction(
      *operation.GetLoadedType(),
      operands[0],
      true,
      operation.GetAlignment(),
      builder,
      ctx);
}

static void
CreateStoreInstruction(
    const variable * address,
    const variable * value,
    bool isVolatile,
    size_t alignment,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto storeInstruction = builder.CreateStore(ctx.value(value), ctx.value(address), isVolatile);
  storeInstruction->setAlignment(::llvm::Align(alignment));
}

static inline ::llvm::Value *
convert_store(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto storeOperation = util::AssertedCast<const StoreNonVolatileOperation>(&operation);
  CreateStoreInstruction(
      operands[0],
      operands[1],
      false,
      storeOperation->GetAlignment(),
      builder,
      ctx);
  return nullptr;
}

static ::llvm::Value *
convert(
    const StoreVolatileOperation & operation,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  CreateStoreInstruction(operands[0], operands[1], true, operation.GetAlignment(), builder, ctx);
  return nullptr;
}

static inline ::llvm::Value *
convert_alloca(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<alloca_op>(op));
  auto & aop = *static_cast<const llvm::alloca_op *>(&op);
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto t = typeConverter.ConvertJlmType(aop.value_type(), llvmContext);
  auto i = builder.CreateAlloca(t, ctx.value(args[0]));
  i->setAlignment(::llvm::Align(aop.alignment()));
  return i;
}

static inline ::llvm::Value *
convert_getelementptr(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<GetElementPtrOperation>(op) && args.size() >= 2);
  auto & pop = *static_cast<const GetElementPtrOperation *>(&op);
  auto & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  std::vector<::llvm::Value *> indices;
  auto t = typeConverter.ConvertJlmType(pop.GetPointeeType(), llvmContext);
  for (size_t n = 1; n < args.size(); n++)
    indices.push_back(ctx.value(args[n]));

  return builder.CreateGEP(t, ctx.value(args[0]), indices);
}

template<typename T>
static std::vector<T>
get_bitdata(const std::vector<const variable *> & args, context & ctx)
{
  std::vector<T> data;
  for (size_t n = 0; n < args.size(); n++)
  {
    auto c = ::llvm::dyn_cast<const ::llvm::ConstantInt>(ctx.value(args[n]));
    JLM_ASSERT(c);
    data.push_back(c->getZExtValue());
  }

  return data;
}

template<typename T>
static std::vector<T>
get_fpdata(const std::vector<const variable *> & args, context & ctx)
{
  std::vector<T> data;
  for (size_t n = 0; n < args.size(); n++)
  {
    auto c = ::llvm::dyn_cast<const ::llvm::ConstantFP>(ctx.value(args[n]));
    JLM_ASSERT(c);
    data.push_back(c->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  return data;
}

static ::llvm::Value *
convert(
    const ConstantDataArray & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<ConstantDataArray>(op));

  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&op.type()))
  {
    if (bt->nbits() == 8)
    {
      auto data = get_bitdata<uint8_t>(operands, ctx);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 16)
    {
      auto data = get_bitdata<uint16_t>(operands, ctx);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 32)
    {
      auto data = get_bitdata<uint32_t>(operands, ctx);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 64)
    {
      auto data = get_bitdata<uint64_t>(operands, ctx);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
  }

  if (auto ft = dynamic_cast<const FloatingPointType *>(&op.type()))
  {
    if (ft->size() == fpsize::half)
    {
      auto data = get_fpdata<uint16_t>(operands, ctx);
      auto type = ::llvm::Type::getBFloatTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
    else if (ft->size() == fpsize::flt)
    {
      auto data = get_fpdata<uint32_t>(operands, ctx);
      auto type = ::llvm::Type::getFloatTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
    else if (ft->size() == fpsize::dbl)
    {
      auto data = get_fpdata<uint64_t>(operands, ctx);
      auto type = ::llvm::Type::getDoubleTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
  }

  JLM_UNREACHABLE("This should not have happened!");
}

static ::llvm::Value *
convert(
    const ConstantArray & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  JLM_ASSERT(is<ConstantArray>(op));
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  std::vector<::llvm::Constant *> data;
  for (size_t n = 0; n < operands.size(); n++)
  {
    auto c = ::llvm::dyn_cast<::llvm::Constant>(ctx.value(operands[n]));
    JLM_ASSERT(c);
    data.push_back(c);
  }

  auto at = std::dynamic_pointer_cast<const ArrayType>(op.result(0));
  auto type = typeConverter.ConvertArrayType(*at, llvmContext);
  return ::llvm::ConstantArray::get(type, data);
}

static ::llvm::Value *
convert(
    const ConstantAggregateZero & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(*op.result(0), llvmContext);
  return ::llvm::ConstantAggregateZero::get(type);
}

static inline ::llvm::Value *
convert_ptrcmp(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<ptrcmp_op>(op));
  auto & pop = *static_cast<const ptrcmp_op *>(&op);

  static std::unordered_map<llvm::cmp, ::llvm::CmpInst::Predicate> map(
      { { cmp::le, ::llvm::CmpInst::ICMP_ULE },
        { cmp::lt, ::llvm::CmpInst::ICMP_ULT },
        { cmp::eq, ::llvm::CmpInst::ICMP_EQ },
        { cmp::ne, ::llvm::CmpInst::ICMP_NE },
        { cmp::ge, ::llvm::CmpInst::ICMP_UGE },
        { cmp::gt, ::llvm::CmpInst::ICMP_UGT } });

  auto op1 = ctx.value(args[0]);
  auto op2 = ctx.value(args[1]);
  JLM_ASSERT(map.find(pop.cmp()) != map.end());
  return builder.CreateICmp(map[pop.cmp()], op1, op2);
}

static inline ::llvm::Value *
convert_fpcmp(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<fpcmp_op>(op));
  auto & fpcmp = *static_cast<const fpcmp_op *>(&op);

  static std::unordered_map<llvm::fpcmp, ::llvm::CmpInst::Predicate> map(
      { { fpcmp::oeq, ::llvm::CmpInst::FCMP_OEQ },
        { fpcmp::ogt, ::llvm::CmpInst::FCMP_OGT },
        { fpcmp::oge, ::llvm::CmpInst::FCMP_OGE },
        { fpcmp::olt, ::llvm::CmpInst::FCMP_OLT },
        { fpcmp::ole, ::llvm::CmpInst::FCMP_OLE },
        { fpcmp::one, ::llvm::CmpInst::FCMP_ONE },
        { fpcmp::ord, ::llvm::CmpInst::FCMP_ORD },
        { fpcmp::uno, ::llvm::CmpInst::FCMP_UNO },
        { fpcmp::ueq, ::llvm::CmpInst::FCMP_UEQ },
        { fpcmp::ugt, ::llvm::CmpInst::FCMP_UGT },
        { fpcmp::uge, ::llvm::CmpInst::FCMP_UGE },
        { fpcmp::ult, ::llvm::CmpInst::FCMP_ULT },
        { fpcmp::ule, ::llvm::CmpInst::FCMP_ULE },
        { fpcmp::une, ::llvm::CmpInst::FCMP_UNE },
        { fpcmp::TRUE, ::llvm::CmpInst::FCMP_TRUE },
        { fpcmp::FALSE, ::llvm::CmpInst::FCMP_FALSE } });

  auto op1 = ctx.value(args[0]);
  auto op2 = ctx.value(args[1]);
  JLM_ASSERT(map.find(fpcmp.cmp()) != map.end());
  return builder.CreateFCmp(map[fpcmp.cmp()], op1, op2);
}

static inline ::llvm::Value *
convert_fpbin(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<fpbin_op>(op));
  auto & fpbin = *static_cast<const llvm::fpbin_op *>(&op);

  static std::unordered_map<llvm::fpop, ::llvm::Instruction::BinaryOps> map(
      { { fpop::add, ::llvm::Instruction::FAdd },
        { fpop::sub, ::llvm::Instruction::FSub },
        { fpop::mul, ::llvm::Instruction::FMul },
        { fpop::div, ::llvm::Instruction::FDiv },
        { fpop::mod, ::llvm::Instruction::FRem } });

  auto op1 = ctx.value(args[0]);
  auto op2 = ctx.value(args[1]);
  JLM_ASSERT(map.find(fpbin.fpop()) != map.end());
  return builder.CreateBinOp(map[fpbin.fpop()], op1, op2);
}

static ::llvm::Value *
convert_fpneg(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<fpneg_op>(op));
  auto operand = ctx.value(args[0]);
  return builder.CreateUnOp(::llvm::Instruction::FNeg, operand);
}

static inline ::llvm::Value *
convert_valist(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  JLM_ASSERT(is<valist_op>(op));
  return nullptr;
}

static inline ::llvm::Value *
convert(
    const ConstantStruct & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  std::vector<::llvm::Constant *> operands;
  for (const auto & arg : args)
    operands.push_back(::llvm::cast<::llvm::Constant>(ctx.value(arg)));

  auto t = typeConverter.ConvertStructType(op.type(), llvmContext);
  return ::llvm::ConstantStruct::get(t, operands);
}

static inline ::llvm::Value *
convert(
    const ConstantPointerNullOperation & operation,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();

  auto pointerType = typeConverter.ConvertPointerType(operation.GetPointerType(), llvmContext);
  return ::llvm::ConstantPointerNull::get(pointerType);
}

static inline ::llvm::Value *
convert_select(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<select_op>(op));
  auto & select = *static_cast<const select_op *>(&op);

  if (rvsdg::is<rvsdg::StateType>(select.type()))
    return nullptr;

  auto c = ctx.value(operands[0]);
  auto t = ctx.value(operands[1]);
  auto f = ctx.value(operands[2]);
  return builder.CreateSelect(c, t, f);
}

static inline ::llvm::Value *
convert_ctl2bits(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  JLM_ASSERT(is<ctl2bits_op>(op));
  return ctx.value(args[0]);
}

static ::llvm::Value *
convert_constantvector(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  JLM_ASSERT(is<constantvector_op>(op));

  std::vector<::llvm::Constant *> ops;
  for (const auto & operand : operands)
    ops.push_back(::llvm::cast<::llvm::Constant>(ctx.value(operand)));

  return ::llvm::ConstantVector::get(ops);
}

static ::llvm::Value *
convert_constantdatavector(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<constant_data_vector_op>(op));
  auto & cop = *static_cast<const constant_data_vector_op *>(&op);

  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&cop.type()))
  {
    if (bt->nbits() == 8)
    {
      auto data = get_bitdata<uint8_t>(operands, ctx);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 16)
    {
      auto data = get_bitdata<uint16_t>(operands, ctx);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 32)
    {
      auto data = get_bitdata<uint32_t>(operands, ctx);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 64)
    {
      auto data = get_bitdata<uint64_t>(operands, ctx);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
  }

  if (auto ft = dynamic_cast<const FloatingPointType *>(&cop.type()))
  {
    if (ft->size() == fpsize::half)
    {
      auto data = get_fpdata<uint16_t>(operands, ctx);
      auto type = ::llvm::Type::getBFloatTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
    else if (ft->size() == fpsize::flt)
    {
      auto data = get_fpdata<uint32_t>(operands, ctx);
      auto type = ::llvm::Type::getFloatTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
    else if (ft->size() == fpsize::dbl)
    {
      auto data = get_fpdata<uint64_t>(operands, ctx);
      auto type = ::llvm::Type::getDoubleTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
  }

  JLM_UNREACHABLE("This should not have happened!");
}

static ::llvm::Value *
convert_extractelement(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<extractelement_op>(op));
  return builder.CreateExtractElement(ctx.value(args[0]), ctx.value(args[1]));
}

static ::llvm::Value *
convert(
    const shufflevector_op & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto v1 = ctx.value(operands[0]);
  auto v2 = ctx.value(operands[1]);
  return builder.CreateShuffleVector(v1, v2, op.Mask());
}

static ::llvm::Value *
convert_insertelement(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<insertelement_op>(op));

  auto vector = ctx.value(operands[0]);
  auto value = ctx.value(operands[1]);
  auto index = ctx.value(operands[2]);
  return builder.CreateInsertElement(vector, value, index);
}

static ::llvm::Value *
convert_vectorunary(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<vectorunary_op>(op));
  auto vop = static_cast<const vectorunary_op *>(&op);
  return convert_operation(vop->operation(), operands, builder, ctx);
}

static ::llvm::Value *
convert_vectorbinary(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<vectorbinary_op>(op));
  auto vop = static_cast<const vectorbinary_op *>(&op);
  return convert_operation(vop->operation(), operands, builder, ctx);
}

static ::llvm::Value *
convert(
    const vectorselect_op &,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto c = ctx.value(operands[0]);
  auto t = ctx.value(operands[1]);
  auto f = ctx.value(operands[2]);
  return builder.CreateSelect(c, t, f);
}

template<::llvm::Instruction::CastOps OPCODE>
static ::llvm::Value *
convert_cast(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(::llvm::Instruction::isCast(OPCODE));
  auto & typeConverter = ctx.GetTypeConverter();
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto dsttype = std::dynamic_pointer_cast<const rvsdg::ValueType>(op.result(0));
  auto operand = operands[0];

  if (const auto vt = dynamic_cast<const FixedVectorType *>(&operand->type()))
  {
    const auto type =
        typeConverter.ConvertJlmType(FixedVectorType(dsttype, vt->size()), llvmContext);
    return builder.CreateCast(OPCODE, ctx.value(operand), type);
  }

  if (const auto vt = dynamic_cast<const ScalableVectorType *>(&operand->type()))
  {
    const auto type =
        typeConverter.ConvertJlmType(ScalableVectorType(dsttype, vt->size()), llvmContext);
    return builder.CreateCast(OPCODE, ctx.value(operand), type);
  }

  auto type = typeConverter.ConvertJlmType(*dsttype, llvmContext);
  return builder.CreateCast(OPCODE, ctx.value(operand), type);
}

static ::llvm::Value *
convert(
    const ExtractValue & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  std::vector<unsigned> indices(op.begin(), op.end());
  return builder.CreateExtractValue(ctx.value(operands[0]), indices);
}

static ::llvm::Value *
convert(
    const malloc_op & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(args.size() == 1);
  auto & typeConverter = ctx.GetTypeConverter();
  auto & lm = ctx.llvm_module();

  auto fcttype = typeConverter.ConvertFunctionType(op.fcttype(), lm.getContext());
  auto function = lm.getOrInsertFunction("malloc", fcttype);
  auto operands = std::vector<::llvm::Value *>(1, ctx.value(args[0]));
  return builder.CreateCall(function, operands);
}

static ::llvm::Value *
convert(
    const FreeOperation & op,
    const std::vector<const variable *> & args,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto & typeConverter = ctx.GetTypeConverter();
  auto & llvmmod = ctx.llvm_module();

  auto fcttype = typeConverter.ConvertFunctionType(
      rvsdg::FunctionType({ op.argument(0) }, {}),
      llvmmod.getContext());
  auto function = llvmmod.getOrInsertFunction("free", fcttype);
  auto operands = std::vector<::llvm::Value *>(1, ctx.value(args[0]));
  return builder.CreateCall(function, operands);
}

static ::llvm::Value *
convert(
    const MemCpyNonVolatileOperation &,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto & destination = *ctx.value(operands[0]);
  auto & source = *ctx.value(operands[1]);
  auto & length = *ctx.value(operands[2]);

  return builder.CreateMemCpy(
      &destination,
      ::llvm::MaybeAlign(),
      &source,
      ::llvm::MaybeAlign(),
      &length,
      false);
}

static ::llvm::Value *
convert(
    const MemCpyVolatileOperation &,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  auto & destination = *ctx.value(operands[0]);
  auto & source = *ctx.value(operands[1]);
  auto & length = *ctx.value(operands[2]);

  return builder.CreateMemCpy(
      &destination,
      ::llvm::MaybeAlign(),
      &source,
      ::llvm::MaybeAlign(),
      &length,
      true);
}

static ::llvm::Value *
convert(
    const MemoryStateMergeOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const MemoryStateSplitOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const LambdaEntryMemoryStateSplitOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const LambdaExitMemoryStateMergeOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const CallEntryMemoryStateMergeOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const CallExitMemoryStateSplitOperation &,
    const std::vector<const variable *> &,
    ::llvm::IRBuilder<> &,
    context &)
{
  return nullptr;
}

static ::llvm::Value *
convert(
    const PointerToFunctionOperation &,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  return ctx.value(operands[0]);
}

static ::llvm::Value *
convert(
    const FunctionToPointerOperation &,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> &,
    context & ctx)
{
  return ctx.value(operands[0]);
}

template<class OP>
static ::llvm::Value *
convert(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & operands,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  JLM_ASSERT(is<OP>(op));
  return convert(*static_cast<const OP *>(&op), operands, builder, ctx);
}

::llvm::Value *
convert_operation(
    const rvsdg::SimpleOperation & op,
    const std::vector<const variable *> & arguments,
    ::llvm::IRBuilder<> & builder,
    context & ctx)
{
  if (is<IntegerAddOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Add, arguments, builder, ctx);
  }
  if (is<IntegerAndOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::And, arguments, builder, ctx);
  }
  if (is<IntegerAShrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::AShr, arguments, builder, ctx);
  }
  if (is<IntegerSubOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Sub, arguments, builder, ctx);
  }
  if (is<IntegerUDivOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::UDiv, arguments, builder, ctx);
  }
  if (is<IntegerSDivOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::SDiv, arguments, builder, ctx);
  }
  if (is<IntegerURemOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::URem, arguments, builder, ctx);
  }
  if (is<IntegerSRemOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::SRem, arguments, builder, ctx);
  }
  if (is<IntegerShlOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Shl, arguments, builder, ctx);
  }
  if (is<IntegerLShrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::LShr, arguments, builder, ctx);
  }
  if (is<IntegerOrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Or, arguments, builder, ctx);
  }
  if (is<IntegerXorOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Xor, arguments, builder, ctx);
  }
  if (is<IntegerMulOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Mul, arguments, builder, ctx);
  }
  if (is<rvsdg::biteq_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_EQ, arguments, builder, ctx);
  }
  if (is<rvsdg::bitne_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_NE, arguments, builder, ctx);
  }
  if (is<rvsdg::bitugt_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_UGT, arguments, builder, ctx);
  }
  if (is<rvsdg::bituge_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_UGE, arguments, builder, ctx);
  }
  if (is<rvsdg::bitult_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_ULT, arguments, builder, ctx);
  }
  if (is<rvsdg::bitule_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_ULE, arguments, builder, ctx);
  }
  if (is<rvsdg::bitsgt_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SGT, arguments, builder, ctx);
  }
  if (is<rvsdg::bitsge_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SGE, arguments, builder, ctx);
  }
  if (is<rvsdg::bitslt_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SLT, arguments, builder, ctx);
  }
  if (is<rvsdg::bitsle_op>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SLE, arguments, builder, ctx);
  }

  static std::unordered_map<
      std::type_index,
      ::llvm::Value * (*)(const rvsdg::SimpleOperation &,
                          const std::vector<const variable *> &,
                          ::llvm::IRBuilder<> &,
                          context & ctx)>
      map({ { typeid(rvsdg::bitconstant_op), convert_bitconstant },
            { typeid(rvsdg::ctlconstant_op), convert_ctlconstant },
            { typeid(ConstantFP), convert<ConstantFP> },
            { typeid(UndefValueOperation), convert_undef },
            { typeid(PoisonValueOperation), convert<PoisonValueOperation> },
            { typeid(rvsdg::match_op), convert_match },
            { typeid(assignment_op), convert_assignment },
            { typeid(branch_op), convert_branch },
            { typeid(phi_op), convert_phi },
            { typeid(LoadNonVolatileOperation), convert<LoadNonVolatileOperation> },
            { typeid(LoadVolatileOperation), convert<LoadVolatileOperation> },
            { typeid(StoreNonVolatileOperation), convert_store },
            { typeid(StoreVolatileOperation), convert<StoreVolatileOperation> },
            { typeid(alloca_op), convert_alloca },
            { typeid(GetElementPtrOperation), convert_getelementptr },
            { typeid(ConstantDataArray), convert<ConstantDataArray> },
            { typeid(ptrcmp_op), convert_ptrcmp },
            { typeid(fpcmp_op), convert_fpcmp },
            { typeid(fpbin_op), convert_fpbin },
            { typeid(valist_op), convert_valist },
            { typeid(ConstantStruct), convert<ConstantStruct> },
            { typeid(ConstantPointerNullOperation), convert<ConstantPointerNullOperation> },
            { typeid(select_op), convert_select },
            { typeid(ConstantArray), convert<ConstantArray> },
            { typeid(ConstantAggregateZero), convert<ConstantAggregateZero> },
            { typeid(ctl2bits_op), convert_ctl2bits },
            { typeid(constantvector_op), convert_constantvector },
            { typeid(constant_data_vector_op), convert_constantdatavector },
            { typeid(extractelement_op), convert_extractelement },
            { typeid(shufflevector_op), convert<shufflevector_op> },
            { typeid(insertelement_op), convert_insertelement },
            { typeid(vectorunary_op), convert_vectorunary },
            { typeid(vectorbinary_op), convert_vectorbinary },
            { typeid(vectorselect_op), convert<vectorselect_op> },
            { typeid(ExtractValue), convert<ExtractValue> },
            { typeid(CallOperation), convert<CallOperation> },
            { typeid(malloc_op), convert<malloc_op> },
            { typeid(FreeOperation), convert<FreeOperation> },
            { typeid(MemCpyNonVolatileOperation), convert<MemCpyNonVolatileOperation> },
            { typeid(MemCpyVolatileOperation), convert<MemCpyVolatileOperation> },
            { typeid(fpneg_op), convert_fpneg },
            { typeid(bitcast_op), convert_cast<::llvm::Instruction::BitCast> },
            { typeid(fpext_op), convert_cast<::llvm::Instruction::FPExt> },
            { typeid(fp2si_op), convert_cast<::llvm::Instruction::FPToSI> },
            { typeid(fp2ui_op), convert_cast<::llvm::Instruction::FPToUI> },
            { typeid(fptrunc_op), convert_cast<::llvm::Instruction::FPTrunc> },
            { typeid(bits2ptr_op), convert_cast<::llvm::Instruction::IntToPtr> },
            { typeid(ptr2bits_op), convert_cast<::llvm::Instruction::PtrToInt> },
            { typeid(sext_op), convert_cast<::llvm::Instruction::SExt> },
            { typeid(sitofp_op), convert_cast<::llvm::Instruction::SIToFP> },
            { typeid(trunc_op), convert_cast<::llvm::Instruction::Trunc> },
            { typeid(uitofp_op), convert_cast<::llvm::Instruction::UIToFP> },
            { typeid(zext_op), convert_cast<::llvm::Instruction::ZExt> },
            { typeid(MemoryStateMergeOperation), convert<MemoryStateMergeOperation> },
            { typeid(MemoryStateSplitOperation), convert<MemoryStateSplitOperation> },
            { typeid(LambdaEntryMemoryStateSplitOperation),
              convert<LambdaEntryMemoryStateSplitOperation> },
            { typeid(LambdaExitMemoryStateMergeOperation),
              convert<LambdaExitMemoryStateMergeOperation> },
            { typeid(CallEntryMemoryStateMergeOperation),
              convert<CallEntryMemoryStateMergeOperation> },
            { typeid(CallExitMemoryStateSplitOperation),
              convert<CallExitMemoryStateSplitOperation> },
            { typeid(PointerToFunctionOperation), convert<PointerToFunctionOperation> },
            { typeid(FunctionToPointerOperation), convert<FunctionToPointerOperation> } });
  /* FIXME: AddrSpaceCast instruction is not supported */

  JLM_ASSERT(map.find(std::type_index(typeid(op))) != map.end());
  return map[std::type_index(typeid(op))](op, arguments, builder, ctx);
}

void
convert_instruction(const llvm::tac & tac, const llvm::cfg_node * node, context & ctx)
{
  std::vector<const variable *> operands;
  for (size_t n = 0; n < tac.noperands(); n++)
    operands.push_back(tac.operand(n));

  ::llvm::IRBuilder<> builder(ctx.basic_block(node));
  auto r = convert_operation(tac.operation(), operands, builder, ctx);
  if (r != nullptr)
    ctx.insert(tac.result(0), r);
}

void
convert_tacs(const tacsvector_t & tacs, context & ctx)
{
  ::llvm::IRBuilder<> builder(ctx.llvm_module().getContext());
  for (const auto & tac : tacs)
  {
    std::vector<const variable *> operands;
    for (size_t n = 0; n < tac->noperands(); n++)
      operands.push_back(tac->operand(n));

    JLM_ASSERT(tac->nresults() == 1);
    auto r = convert_operation(tac->operation(), operands, builder, ctx);
    ctx.insert(tac->result(0), r);
  }
}

}
}
