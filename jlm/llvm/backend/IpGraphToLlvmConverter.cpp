/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>
#include <jlm/rvsdg/control.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <deque>
#include <unordered_map>

#include <typeindex>

namespace jlm::llvm
{

class IpGraphToLlvmConverter::Context final
{
  using const_iterator =
      std::unordered_map<const ControlFlowGraphNode *, ::llvm::BasicBlock *>::const_iterator;

public:
  Context(InterProceduralGraphModule & ipGraphModule, ::llvm::Module & llvmModule)
      : LlvmModule_(llvmModule),
        IpGraphModule_(ipGraphModule)
  {}

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  // FIXME: It should be a const reference, but we still have to create variables to translate
  // expressions.
  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return IpGraphModule_;
  }

  ::llvm::Module &
  llvm_module() const noexcept
  {
    return LlvmModule_;
  }

  const_iterator
  begin() const
  {
    return nodes_.begin();
  }

  const_iterator
  end() const
  {
    return nodes_.end();
  }

  void
  insert(const llvm::ControlFlowGraphNode * node, ::llvm::BasicBlock * bb)
  {
    nodes_[node] = bb;
  }

  void
  insert(const llvm::Variable * variable, ::llvm::Value * value)
  {
    variables_[variable] = value;
  }

  ::llvm::BasicBlock *
  basic_block(const llvm::ControlFlowGraphNode * node) const noexcept
  {
    auto it = nodes_.find(node);
    JLM_ASSERT(it != nodes_.end());
    return it->second;
  }

  ::llvm::Value *
  value(const llvm::Variable * variable) const noexcept
  {
    auto it = variables_.find(variable);
    JLM_ASSERT(it != variables_.end());
    return it->second;
  }

  TypeConverter &
  GetTypeConverter()
  {
    return TypeConverter_;
  }

  static std::unique_ptr<Context>
  Create(InterProceduralGraphModule & ipGraphModule, ::llvm::Module & llvmModule)
  {
    return std::make_unique<Context>(ipGraphModule, llvmModule);
  }

private:
  ::llvm::Module & LlvmModule_;
  InterProceduralGraphModule & IpGraphModule_;
  std::unordered_map<const llvm::Variable *, ::llvm::Value *> variables_;
  std::unordered_map<const llvm::ControlFlowGraphNode *, ::llvm::BasicBlock *> nodes_;
  TypeConverter TypeConverter_;
};

IpGraphToLlvmConverter::~IpGraphToLlvmConverter() noexcept = default;

IpGraphToLlvmConverter::IpGraphToLlvmConverter() = default;

::llvm::Value *
IpGraphToLlvmConverter::convert_assignment(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<AssignmentOperation>(op));
  return Context_->value(args[0]);
}

::llvm::Value *
IpGraphToLlvmConverter::CreateBinOpInstruction(
    const ::llvm::Instruction::BinaryOps opcode,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  const auto operand1 = Context_->value(args[0]);
  const auto operand2 = Context_->value(args[1]);
  return builder.CreateBinOp(opcode, operand1, operand2);
}

::llvm::Value *
IpGraphToLlvmConverter::CreateICmpInstruction(
    const ::llvm::CmpInst::Predicate predicate,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  const auto operand1 = Context_->value(args[0]);
  const auto operand2 = Context_->value(args[1]);
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

::llvm::Value *
IpGraphToLlvmConverter::ConverterIntegerConstant(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> & builder)
{
  const auto & representation =
      util::AssertedCast<const IntegerConstantOperation>(&op)->Representation();
  const auto type = ::llvm::IntegerType::get(builder.getContext(), representation.nbits());

  if (representation.is_defined())
    return ::llvm::ConstantInt::get(type, convert_bitvalue_repr(representation));

  return ::llvm::UndefValue::get(type);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_ctlconstant(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is_ctlconstant_op(op));
  auto & cop = *static_cast<const rvsdg::ctlconstant_op *>(&op);

  size_t nbits = cop.value().nalternatives() == 2 ? 1 : 32;
  auto type = ::llvm::IntegerType::get(builder.getContext(), nbits);
  return ::llvm::ConstantInt::get(type, cop.value().alternative());
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantFP & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> & builder)
{
  return ::llvm::ConstantFP::get(builder.getContext(), op.constant());
}

::llvm::Value *
IpGraphToLlvmConverter::convert_undef(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<UndefValueOperation>(op));
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto & resultType = *op.result(0);

  // MemoryState has no llvm representation.
  if (is<MemoryStateType>(resultType))
    return nullptr;

  auto type = typeConverter.ConvertJlmType(resultType, llvmContext);
  return ::llvm::UndefValue::get(type);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const PoisonValueOperation & operation,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(operation.GetType(), llvmContext);
  return ::llvm::PoisonValue::get(type);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const CallOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  auto function = Context_->value(args[0]);
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

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
      JLM_ASSERT(is<ThreeAddressCodeVariable>(argument));
      auto valist = dynamic_cast<const llvm::ThreeAddressCodeVariable *>(argument)->tac();
      JLM_ASSERT(is<valist_op>(valist->operation()));
      for (size_t n = 0; n < valist->noperands(); n++)
        operands.push_back(Context_->value(valist->operand(n)));
      continue;
    }

    operands.push_back(Context_->value(argument));
  }

  auto ftype = typeConverter.ConvertFunctionType(*op.GetFunctionType(), llvmContext);
  return builder.CreateCall(ftype, function, operands);
}

static bool
is_identity_mapping(const rvsdg::match_op & op)
{
  for (const auto & pair : op)
  {
    if (pair.first != pair.second)
      return false;
  }

  return true;
}

::llvm::Value *
IpGraphToLlvmConverter::convert_match(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<rvsdg::match_op>(op));
  auto mop = static_cast<const rvsdg::match_op *>(&op);

  if (is_identity_mapping(*mop))
    return Context_->value(args[0]);

  if (mop->nalternatives() == 2 && mop->nbits() == 1)
  {
    auto i1 = ::llvm::IntegerType::get(builder.getContext(), 1);
    auto t = ::llvm::ConstantInt::getFalse(i1);
    auto f = ::llvm::ConstantInt::getTrue(i1);
    return builder.CreateSelect(Context_->value(args[0]), t, f);
  }

  /* FIXME: This is not working if the match is not directly connected to a gamma node. */
  return Context_->value(args[0]);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_branch(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<BranchOperation>(op));
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert_phi(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> & builder)
{
  auto & phi = *util::AssertedCast<const SsaPhiOperation>(&op);
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  if (rvsdg::is<IOStateType>(phi.type()))
    return nullptr;
  if (rvsdg::is<MemoryStateType>(phi.type()))
    return nullptr;

  auto t = typeConverter.ConvertJlmType(phi.type(), llvmContext);
  return builder.CreatePHI(t, op.narguments());
}

::llvm::Value *
IpGraphToLlvmConverter::CreateLoadInstruction(
    const rvsdg::ValueType & loadedType,
    const Variable * address,
    bool isVolatile,
    size_t alignment,
    ::llvm::IRBuilder<> & builder)
{
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(loadedType, llvmContext);
  auto loadInstruction = builder.CreateLoad(type, Context_->value(address), isVolatile);
  loadInstruction->setAlignment(::llvm::Align(alignment));
  return loadInstruction;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const LoadNonVolatileOperation & operation,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  return CreateLoadInstruction(
      *operation.GetLoadedType(),
      operands[0],
      false,
      operation.GetAlignment(),
      builder);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const LoadVolatileOperation & operation,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  return CreateLoadInstruction(
      *operation.GetLoadedType(),
      operands[0],
      true,
      operation.GetAlignment(),
      builder);
}

void
IpGraphToLlvmConverter::CreateStoreInstruction(
    const Variable * address,
    const Variable * value,
    bool isVolatile,
    size_t alignment,
    ::llvm::IRBuilder<> & builder)
{
  auto storeInstruction =
      builder.CreateStore(Context_->value(value), Context_->value(address), isVolatile);
  storeInstruction->setAlignment(::llvm::Align(alignment));
}

::llvm::Value *
IpGraphToLlvmConverter::convert_store(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto storeOperation = util::AssertedCast<const StoreNonVolatileOperation>(&operation);
  CreateStoreInstruction(operands[0], operands[1], false, storeOperation->GetAlignment(), builder);
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const StoreVolatileOperation & operation,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  CreateStoreInstruction(operands[0], operands[1], true, operation.GetAlignment(), builder);
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert_alloca(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<AllocaOperation>(op));
  auto & aop = *static_cast<const llvm::AllocaOperation *>(&op);
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto t = typeConverter.ConvertJlmType(aop.value_type(), llvmContext);
  auto i = builder.CreateAlloca(t, Context_->value(args[0]));
  i->setAlignment(::llvm::Align(aop.alignment()));
  return i;
}

::llvm::Value *
IpGraphToLlvmConverter::convert_getelementptr(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<GetElementPtrOperation>(op) && args.size() >= 2);
  auto & pop = *static_cast<const GetElementPtrOperation *>(&op);
  auto & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  std::vector<::llvm::Value *> indices;
  auto t = typeConverter.ConvertJlmType(pop.GetPointeeType(), llvmContext);
  for (size_t n = 1; n < args.size(); n++)
    indices.push_back(Context_->value(args[n]));

  return builder.CreateGEP(t, Context_->value(args[0]), indices);
}

template<typename T>
std::vector<T>
IpGraphToLlvmConverter::get_bitdata(const std::vector<const Variable *> & args)
{
  std::vector<T> data;
  for (size_t n = 0; n < args.size(); n++)
  {
    auto c = ::llvm::dyn_cast<const ::llvm::ConstantInt>(Context_->value(args[n]));
    JLM_ASSERT(c);
    data.push_back(c->getZExtValue());
  }

  return data;
}

template<typename T>
std::vector<T>
IpGraphToLlvmConverter::get_fpdata(const std::vector<const Variable *> & args)
{
  std::vector<T> data;
  for (size_t n = 0; n < args.size(); n++)
  {
    auto c = ::llvm::dyn_cast<const ::llvm::ConstantFP>(Context_->value(args[n]));
    JLM_ASSERT(c);
    data.push_back(c->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  return data;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantDataArray & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<ConstantDataArray>(op));

  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&op.type()))
  {
    if (bt->nbits() == 8)
    {
      auto data = get_bitdata<uint8_t>(operands);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 16)
    {
      auto data = get_bitdata<uint16_t>(operands);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 32)
    {
      auto data = get_bitdata<uint32_t>(operands);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 64)
    {
      auto data = get_bitdata<uint64_t>(operands);
      return ::llvm::ConstantDataArray::get(builder.getContext(), data);
    }
  }

  if (auto ft = dynamic_cast<const FloatingPointType *>(&op.type()))
  {
    if (ft->size() == fpsize::half)
    {
      auto data = get_fpdata<uint16_t>(operands);
      auto type = ::llvm::Type::getBFloatTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
    else if (ft->size() == fpsize::flt)
    {
      auto data = get_fpdata<uint32_t>(operands);
      auto type = ::llvm::Type::getFloatTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
    else if (ft->size() == fpsize::dbl)
    {
      auto data = get_fpdata<uint64_t>(operands);
      auto type = ::llvm::Type::getDoubleTy(builder.getContext());
      return ::llvm::ConstantDataArray::getFP(type, data);
    }
  }

  JLM_UNREACHABLE("This should not have happened!");
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantArrayOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> &)
{
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  std::vector<::llvm::Constant *> data;
  for (size_t n = 0; n < operands.size(); n++)
  {
    auto c = ::llvm::dyn_cast<::llvm::Constant>(Context_->value(operands[n]));
    JLM_ASSERT(c);
    data.push_back(c);
  }

  auto at = std::dynamic_pointer_cast<const ArrayType>(op.result(0));
  auto type = typeConverter.ConvertArrayType(*at, llvmContext);
  return ::llvm::ConstantArray::get(type, data);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantAggregateZeroOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto type = typeConverter.ConvertJlmType(*op.result(0), llvmContext);
  return ::llvm::ConstantAggregateZero::get(type);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_ptrcmp(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
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

  auto op1 = Context_->value(args[0]);
  auto op2 = Context_->value(args[1]);
  JLM_ASSERT(map.find(pop.cmp()) != map.end());
  return builder.CreateICmp(map[pop.cmp()], op1, op2);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_fpcmp(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
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

  auto op1 = Context_->value(args[0]);
  auto op2 = Context_->value(args[1]);
  JLM_ASSERT(map.find(fpcmp.cmp()) != map.end());
  return builder.CreateFCmp(map[fpcmp.cmp()], op1, op2);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_fpbin(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<fpbin_op>(op));
  auto & fpbin = *static_cast<const llvm::fpbin_op *>(&op);

  static std::unordered_map<llvm::fpop, ::llvm::Instruction::BinaryOps> map(
      { { fpop::add, ::llvm::Instruction::FAdd },
        { fpop::sub, ::llvm::Instruction::FSub },
        { fpop::mul, ::llvm::Instruction::FMul },
        { fpop::div, ::llvm::Instruction::FDiv },
        { fpop::mod, ::llvm::Instruction::FRem } });

  auto op1 = Context_->value(args[0]);
  auto op2 = Context_->value(args[1]);
  JLM_ASSERT(map.find(fpbin.fpop()) != map.end());
  return builder.CreateBinOp(map[fpbin.fpop()], op1, op2);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_fpneg(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<FNegOperation>(op));
  auto operand = Context_->value(args[0]);
  return builder.CreateUnOp(::llvm::Instruction::FNeg, operand);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_valist(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<valist_op>(op));
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantStruct & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> &)
{
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  std::vector<::llvm::Constant *> operands;
  for (const auto & arg : args)
    operands.push_back(::llvm::cast<::llvm::Constant>(Context_->value(arg)));

  auto t = typeConverter.ConvertStructType(op.type(), llvmContext);
  return ::llvm::ConstantStruct::get(t, operands);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ConstantPointerNullOperation & operation,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();

  auto pointerType = typeConverter.ConvertPointerType(operation.GetPointerType(), llvmContext);
  return ::llvm::ConstantPointerNull::get(pointerType);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_select(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto & select = *util::AssertedCast<const SelectOperation>(&op);

  if (rvsdg::is<rvsdg::StateType>(select.type()))
    return nullptr;

  auto c = Context_->value(operands[0]);
  auto t = Context_->value(operands[1]);
  auto f = Context_->value(operands[2]);
  return builder.CreateSelect(c, t, f);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_ctl2bits(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<ctl2bits_op>(op));
  return Context_->value(args[0]);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_constantvector(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> &)
{
  JLM_ASSERT(is<constantvector_op>(op));

  std::vector<::llvm::Constant *> ops;
  for (const auto & operand : operands)
    ops.push_back(::llvm::cast<::llvm::Constant>(Context_->value(operand)));

  return ::llvm::ConstantVector::get(ops);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_constantdatavector(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<constant_data_vector_op>(op));
  auto & cop = *static_cast<const constant_data_vector_op *>(&op);

  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&cop.type()))
  {
    if (bt->nbits() == 8)
    {
      auto data = get_bitdata<uint8_t>(operands);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 16)
    {
      auto data = get_bitdata<uint16_t>(operands);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 32)
    {
      auto data = get_bitdata<uint32_t>(operands);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
    else if (bt->nbits() == 64)
    {
      auto data = get_bitdata<uint64_t>(operands);
      return ::llvm::ConstantDataVector::get(builder.getContext(), data);
    }
  }

  if (auto ft = dynamic_cast<const FloatingPointType *>(&cop.type()))
  {
    if (ft->size() == fpsize::half)
    {
      auto data = get_fpdata<uint16_t>(operands);
      auto type = ::llvm::Type::getBFloatTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
    else if (ft->size() == fpsize::flt)
    {
      auto data = get_fpdata<uint32_t>(operands);
      auto type = ::llvm::Type::getFloatTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
    else if (ft->size() == fpsize::dbl)
    {
      auto data = get_fpdata<uint64_t>(operands);
      auto type = ::llvm::Type::getDoubleTy(builder.getContext());
      return ::llvm::ConstantDataVector::getFP(type, data);
    }
  }

  JLM_UNREACHABLE("This should not have happened!");
}

::llvm::Value *
IpGraphToLlvmConverter::convert_extractelement(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<extractelement_op>(op));
  return builder.CreateExtractElement(Context_->value(args[0]), Context_->value(args[1]));
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const shufflevector_op & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto v1 = Context_->value(operands[0]);
  auto v2 = Context_->value(operands[1]);
  return builder.CreateShuffleVector(v1, v2, op.Mask());
}

::llvm::Value *
IpGraphToLlvmConverter::convert_insertelement(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<insertelement_op>(op));

  auto vector = Context_->value(operands[0]);
  auto value = Context_->value(operands[1]);
  auto index = Context_->value(operands[2]);
  return builder.CreateInsertElement(vector, value, index);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_vectorunary(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<vectorunary_op>(op));
  auto vop = static_cast<const vectorunary_op *>(&op);
  return convert_operation(vop->operation(), operands, builder);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_vectorbinary(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<vectorbinary_op>(op));
  auto vop = static_cast<const vectorbinary_op *>(&op);
  return convert_operation(vop->operation(), operands, builder);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const VectorSelectOperation &,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto c = Context_->value(operands[0]);
  auto t = Context_->value(operands[1]);
  auto f = Context_->value(operands[2]);
  return builder.CreateSelect(c, t, f);
}

template<::llvm::Instruction::CastOps OPCODE>
::llvm::Value *
IpGraphToLlvmConverter::convert_cast(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(::llvm::Instruction::isCast(OPCODE));
  auto & typeConverter = Context_->GetTypeConverter();
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto dsttype = std::dynamic_pointer_cast<const rvsdg::ValueType>(op.result(0));
  auto operand = operands[0];

  if (const auto vt = dynamic_cast<const FixedVectorType *>(&operand->type()))
  {
    const auto type =
        typeConverter.ConvertJlmType(FixedVectorType(dsttype, vt->size()), llvmContext);
    return builder.CreateCast(OPCODE, Context_->value(operand), type);
  }

  if (const auto vt = dynamic_cast<const ScalableVectorType *>(&operand->type()))
  {
    const auto type =
        typeConverter.ConvertJlmType(ScalableVectorType(dsttype, vt->size()), llvmContext);
    return builder.CreateCast(OPCODE, Context_->value(operand), type);
  }

  auto type = typeConverter.ConvertJlmType(*dsttype, llvmContext);
  return builder.CreateCast(OPCODE, Context_->value(operand), type);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const ExtractValue & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  std::vector<unsigned> indices(op.begin(), op.end());
  return builder.CreateExtractValue(Context_->value(operands[0]), indices);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const malloc_op & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(args.size() == 1);
  auto & typeConverter = Context_->GetTypeConverter();
  auto & lm = Context_->llvm_module();

  auto fcttype = typeConverter.ConvertFunctionType(op.fcttype(), lm.getContext());
  auto function = lm.getOrInsertFunction("malloc", fcttype);
  auto operands = std::vector<::llvm::Value *>(1, Context_->value(args[0]));
  return builder.CreateCall(function, operands);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const FreeOperation & op,
    const std::vector<const Variable *> & args,
    ::llvm::IRBuilder<> & builder)
{
  auto & typeConverter = Context_->GetTypeConverter();
  auto & llvmmod = Context_->llvm_module();

  auto fcttype = typeConverter.ConvertFunctionType(
      rvsdg::FunctionType({ op.argument(0) }, {}),
      llvmmod.getContext());
  auto function = llvmmod.getOrInsertFunction("free", fcttype);
  auto operands = std::vector<::llvm::Value *>(1, Context_->value(args[0]));
  return builder.CreateCall(function, operands);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const MemCpyNonVolatileOperation &,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto & destination = *Context_->value(operands[0]);
  auto & source = *Context_->value(operands[1]);
  auto & length = *Context_->value(operands[2]);

  return builder.CreateMemCpy(
      &destination,
      ::llvm::MaybeAlign(),
      &source,
      ::llvm::MaybeAlign(),
      &length,
      false);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const MemCpyVolatileOperation &,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  auto & destination = *Context_->value(operands[0]);
  auto & source = *Context_->value(operands[1]);
  auto & length = *Context_->value(operands[2]);

  return builder.CreateMemCpy(
      &destination,
      ::llvm::MaybeAlign(),
      &source,
      ::llvm::MaybeAlign(),
      &length,
      true);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const MemoryStateMergeOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const MemoryStateSplitOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const LambdaEntryMemoryStateSplitOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const LambdaExitMemoryStateMergeOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const CallEntryMemoryStateMergeOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const CallExitMemoryStateSplitOperation &,
    const std::vector<const Variable *> &,
    ::llvm::IRBuilder<> &)
{
  return nullptr;
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const PointerToFunctionOperation &,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> &)
{
  return Context_->value(operands[0]);
}

::llvm::Value *
IpGraphToLlvmConverter::convert(
    const FunctionToPointerOperation &,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> &)
{
  return Context_->value(operands[0]);
}

template<class OP>
::llvm::Value *
IpGraphToLlvmConverter::convert(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & operands,
    ::llvm::IRBuilder<> & builder)
{
  JLM_ASSERT(is<OP>(op));
  return convert(*static_cast<const OP *>(&op), operands, builder);
}

::llvm::Value *
IpGraphToLlvmConverter::convert_operation(
    const rvsdg::SimpleOperation & op,
    const std::vector<const Variable *> & arguments,
    ::llvm::IRBuilder<> & builder)
{
  if (is<IntegerAddOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Add, arguments, builder);
  }
  if (is<IntegerAndOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::And, arguments, builder);
  }
  if (is<IntegerAShrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::AShr, arguments, builder);
  }
  if (is<IntegerSubOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Sub, arguments, builder);
  }
  if (is<IntegerUDivOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::UDiv, arguments, builder);
  }
  if (is<IntegerSDivOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::SDiv, arguments, builder);
  }
  if (is<IntegerURemOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::URem, arguments, builder);
  }
  if (is<IntegerSRemOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::SRem, arguments, builder);
  }
  if (is<IntegerShlOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Shl, arguments, builder);
  }
  if (is<IntegerLShrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::LShr, arguments, builder);
  }
  if (is<IntegerOrOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Or, arguments, builder);
  }
  if (is<IntegerXorOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Xor, arguments, builder);
  }
  if (is<IntegerMulOperation>(op))
  {
    return CreateBinOpInstruction(::llvm::Instruction::Mul, arguments, builder);
  }
  if (is<IntegerEqOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_EQ, arguments, builder);
  }
  if (is<IntegerNeOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_NE, arguments, builder);
  }
  if (is<IntegerUgtOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_UGT, arguments, builder);
  }
  if (is<IntegerUgeOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_UGE, arguments, builder);
  }
  if (is<IntegerUltOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_ULT, arguments, builder);
  }
  if (is<IntegerUleOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_ULE, arguments, builder);
  }
  if (is<IntegerSgtOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SGT, arguments, builder);
  }
  if (is<IntegerSgeOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SGE, arguments, builder);
  }
  if (is<IntegerSltOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SLT, arguments, builder);
  }
  if (is<IntegerSleOperation>(op))
  {
    return CreateICmpInstruction(::llvm::CmpInst::ICMP_SLE, arguments, builder);
  }
  if (is<IOBarrierOperation>(op))
  {
    return Context_->value(arguments[0]);
  }
  if (is<IntegerConstantOperation>(op))
  {
    return ConverterIntegerConstant(op, arguments, builder);
  }
  if (is<rvsdg::ctlconstant_op>(op))
  {
    return convert_ctlconstant(op, arguments, builder);
  }
  if (is<ConstantFP>(op))
  {
    return convert<ConstantFP>(op, arguments, builder);
  }
  if (is<UndefValueOperation>(op))
  {
    return convert_undef(op, arguments, builder);
  }
  if (is<PoisonValueOperation>(op))
  {
    return convert<PoisonValueOperation>(op, arguments, builder);
  }
  if (is<rvsdg::match_op>(op))
  {
    return convert_match(op, arguments, builder);
  }
  if (is<AssignmentOperation>(op))
  {
    return convert_assignment(op, arguments, builder);
  }
  if (is<BranchOperation>(op))
  {
    return convert_branch(op, arguments, builder);
  }
  if (is<SsaPhiOperation>(op))
  {
    return convert_phi(op, arguments, builder);
  }
  if (is<LoadNonVolatileOperation>(op))
  {
    return convert<LoadNonVolatileOperation>(op, arguments, builder);
  }
  if (is<LoadVolatileOperation>(op))
  {
    return convert<LoadVolatileOperation>(op, arguments, builder);
  }
  if (is<StoreNonVolatileOperation>(op))
  {
    return convert_store(op, arguments, builder);
  }
  if (is<StoreVolatileOperation>(op))
  {
    return convert<StoreVolatileOperation>(op, arguments, builder);
  }
  if (is<AllocaOperation>(op))
  {
    return convert_alloca(op, arguments, builder);
  }
  if (is<GetElementPtrOperation>(op))
  {
    return convert_getelementptr(op, arguments, builder);
  }
  if (is<ConstantDataArray>(op))
  {
    return convert<ConstantDataArray>(op, arguments, builder);
  }
  if (is<ptrcmp_op>(op))
  {
    return convert_ptrcmp(op, arguments, builder);
  }
  if (is<fpcmp_op>(op))
  {
    return convert_fpcmp(op, arguments, builder);
  }
  if (is<fpbin_op>(op))
  {
    return convert_fpbin(op, arguments, builder);
  }
  if (is<valist_op>(op))
  {
    return convert_valist(op, arguments, builder);
  }
  if (is<ConstantStruct>(op))
  {
    return convert<ConstantStruct>(op, arguments, builder);
  }
  if (is<ConstantPointerNullOperation>(op))
  {
    return convert<ConstantPointerNullOperation>(op, arguments, builder);
  }
  if (is<SelectOperation>(op))
  {
    return convert_select(op, arguments, builder);
  }
  if (is<ConstantArrayOperation>(op))
  {
    return convert<ConstantArrayOperation>(op, arguments, builder);
  }
  if (is<ConstantAggregateZeroOperation>(op))
  {
    return convert<ConstantAggregateZeroOperation>(op, arguments, builder);
  }
  if (is<ctl2bits_op>(op))
  {
    return convert_ctl2bits(op, arguments, builder);
  }
  if (is<constantvector_op>(op))
  {
    return convert_constantvector(op, arguments, builder);
  }
  if (is<constant_data_vector_op>(op))
  {
    return convert_constantdatavector(op, arguments, builder);
  }
  if (is<extractelement_op>(op))
  {
    return convert_extractelement(op, arguments, builder);
  }
  if (is<shufflevector_op>(op))
  {
    return convert<shufflevector_op>(op, arguments, builder);
  }
  if (is<insertelement_op>(op))
  {
    return convert_insertelement(op, arguments, builder);
  }
  if (is<vectorunary_op>(op))
  {
    return convert_vectorunary(op, arguments, builder);
  }
  if (is<vectorbinary_op>(op))
  {
    return convert_vectorbinary(op, arguments, builder);
  }
  if (is<VectorSelectOperation>(op))
  {
    return convert<VectorSelectOperation>(op, arguments, builder);
  }
  if (is<ExtractValue>(op))
  {
    return convert<ExtractValue>(op, arguments, builder);
  }
  if (is<CallOperation>(op))
  {
    return convert<CallOperation>(op, arguments, builder);
  }
  if (is<malloc_op>(op))
  {
    return convert<malloc_op>(op, arguments, builder);
  }
  if (is<FreeOperation>(op))
  {
    return convert<FreeOperation>(op, arguments, builder);
  }
  if (is<MemCpyNonVolatileOperation>(op))
  {
    return convert<MemCpyNonVolatileOperation>(op, arguments, builder);
  }
  if (is<MemCpyVolatileOperation>(op))
  {
    return convert<MemCpyVolatileOperation>(op, arguments, builder);
  }
  if (is<FNegOperation>(op))
  {
    return convert_fpneg(op, arguments, builder);
  }
  if (is<bitcast_op>(op))
  {
    return convert_cast<::llvm::Instruction::BitCast>(op, arguments, builder);
  }
  if (is<FPExtOperation>(op))
  {
    return convert_cast<::llvm::Instruction::FPExt>(op, arguments, builder);
  }
  if (is<FloatingPointToSignedIntegerOperation>(op))
  {
    return convert_cast<::llvm::Instruction::FPToSI>(op, arguments, builder);
  }
  if (is<FloatingPointToUnsignedIntegerOperation>(op))
  {
    return convert_cast<::llvm::Instruction::FPToUI>(op, arguments, builder);
  }
  if (is<FPTruncOperation>(op))
  {
    return convert_cast<::llvm::Instruction::FPTrunc>(op, arguments, builder);
  }
  if (is<IntegerToPointerOperation>(op))
  {
    return convert_cast<::llvm::Instruction::IntToPtr>(op, arguments, builder);
  }
  if (is<PtrToIntOperation>(op))
  {
    return convert_cast<::llvm::Instruction::PtrToInt>(op, arguments, builder);
  }
  if (is<SExtOperation>(op))
  {
    return convert_cast<::llvm::Instruction::SExt>(op, arguments, builder);
  }
  if (is<SIToFPOperation>(op))
  {
    return convert_cast<::llvm::Instruction::SIToFP>(op, arguments, builder);
  }
  if (is<TruncOperation>(op))
  {
    return convert_cast<::llvm::Instruction::Trunc>(op, arguments, builder);
  }
  if (is<UIToFPOperation>(op))
  {
    return convert_cast<::llvm::Instruction::UIToFP>(op, arguments, builder);
  }
  if (is<ZExtOperation>(op))
  {
    return convert_cast<::llvm::Instruction::ZExt>(op, arguments, builder);
  }
  if (is<MemoryStateMergeOperation>(op))
  {
    return convert<MemoryStateMergeOperation>(op, arguments, builder);
  }
  if (is<MemoryStateSplitOperation>(op))
  {
    return convert<MemoryStateSplitOperation>(op, arguments, builder);
  }
  if (is<LambdaEntryMemoryStateSplitOperation>(op))
  {
    return convert<LambdaEntryMemoryStateSplitOperation>(op, arguments, builder);
  }
  if (is<LambdaExitMemoryStateMergeOperation>(op))
  {
    return convert<LambdaExitMemoryStateMergeOperation>(op, arguments, builder);
  }
  if (is<CallEntryMemoryStateMergeOperation>(op))
  {
    return convert<CallEntryMemoryStateMergeOperation>(op, arguments, builder);
  }
  if (is<CallExitMemoryStateSplitOperation>(op))
  {
    return convert<CallExitMemoryStateSplitOperation>(op, arguments, builder);
  }
  if (is<PointerToFunctionOperation>(op))
  {
    return convert<PointerToFunctionOperation>(op, arguments, builder);
  }
  if (is<FunctionToPointerOperation>(op))
  {
    return convert<FunctionToPointerOperation>(op, arguments, builder);
  }

  JLM_UNREACHABLE(util::strfmt("Unhandled operation type: ", op.debug_string()).c_str());
}

void
IpGraphToLlvmConverter::convert_instruction(
    const llvm::ThreeAddressCode & tac,
    const llvm::ControlFlowGraphNode * node)
{
  std::vector<const Variable *> operands;
  for (size_t n = 0; n < tac.noperands(); n++)
    operands.push_back(tac.operand(n));

  ::llvm::IRBuilder<> builder(Context_->basic_block(node));
  auto r = convert_operation(tac.operation(), operands, builder);
  if (r != nullptr)
    Context_->insert(tac.result(0), r);
}

void
IpGraphToLlvmConverter::convert_tacs(const tacsvector_t & tacs)
{
  ::llvm::IRBuilder<> builder(Context_->llvm_module().getContext());
  for (const auto & tac : tacs)
  {
    std::vector<const Variable *> operands;
    for (size_t n = 0; n < tac->noperands(); n++)
      operands.push_back(tac->operand(n));

    JLM_ASSERT(tac->nresults() == 1);
    auto r = convert_operation(tac->operation(), operands, builder);
    Context_->insert(tac->result(0), r);
  }
}

static const llvm::ThreeAddressCode *
get_match(const llvm::ThreeAddressCode * branch)
{
  JLM_ASSERT(is<ThreeAddressCodeVariable>(branch->operand(0)));
  auto tv = static_cast<const ThreeAddressCodeVariable *>(branch->operand(0));
  return tv->tac();
}

static bool
has_return_value(const ControlFlowGraph & cfg)
{
  for (size_t n = 0; n < cfg.exit()->nresults(); n++)
  {
    auto result = cfg.exit()->result(n);
    if (rvsdg::is<rvsdg::ValueType>(result->type()))
      return true;
  }

  return false;
}

void
IpGraphToLlvmConverter::create_return(const ControlFlowGraphNode * node)
{
  JLM_ASSERT(node->NumOutEdges() == 1);
  JLM_ASSERT(node->OutEdge(0)->sink() == node->cfg().exit());
  ::llvm::IRBuilder<> builder(Context_->basic_block(node));
  auto & cfg = node->cfg();

  /* return without result */
  if (!has_return_value(cfg))
  {
    builder.CreateRetVoid();
    return;
  }

  auto result = cfg.exit()->result(0);
  JLM_ASSERT(rvsdg::is<rvsdg::ValueType>(result->type()));
  builder.CreateRet(Context_->value(result));
}

void
IpGraphToLlvmConverter::create_unconditional_branch(const ControlFlowGraphNode * node)
{
  JLM_ASSERT(node->NumOutEdges() == 1);
  JLM_ASSERT(node->OutEdge(0)->sink() != node->cfg().exit());
  ::llvm::IRBuilder<> builder(Context_->basic_block(node));
  auto target = node->OutEdge(0)->sink();

  builder.CreateBr(Context_->basic_block(target));
}

void
IpGraphToLlvmConverter::create_conditional_branch(const ControlFlowGraphNode * node)
{
  JLM_ASSERT(node->NumOutEdges() == 2);
  JLM_ASSERT(node->OutEdge(0)->sink() != node->cfg().exit());
  JLM_ASSERT(node->OutEdge(1)->sink() != node->cfg().exit());
  ::llvm::IRBuilder<> builder(Context_->basic_block(node));

  auto branch = static_cast<const BasicBlock *>(node)->tacs().last();
  JLM_ASSERT(branch && is<BranchOperation>(branch));
  JLM_ASSERT(Context_->value(branch->operand(0))->getType()->isIntegerTy(1));

  auto condition = Context_->value(branch->operand(0));
  auto bbfalse = Context_->basic_block(node->OutEdge(0)->sink());
  auto bbtrue = Context_->basic_block(node->OutEdge(1)->sink());
  builder.CreateCondBr(condition, bbtrue, bbfalse);
}

void
IpGraphToLlvmConverter::create_switch(const ControlFlowGraphNode * node)
{
  JLM_ASSERT(node->NumOutEdges() >= 2);
  ::llvm::LLVMContext & llvmContext = Context_->llvm_module().getContext();
  auto & typeConverter = Context_->GetTypeConverter();
  auto bb = static_cast<const BasicBlock *>(node);
  ::llvm::IRBuilder<> builder(Context_->basic_block(node));

  auto branch = bb->tacs().last();
  JLM_ASSERT(branch && is<BranchOperation>(branch));
  auto condition = Context_->value(branch->operand(0));
  auto match = get_match(branch);

  if (is<rvsdg::match_op>(match))
  {
    JLM_ASSERT(match->result(0) == branch->operand(0));
    auto mop = static_cast<const rvsdg::match_op *>(&match->operation());

    auto defbb = Context_->basic_block(node->OutEdge(mop->default_alternative())->sink());
    auto sw = builder.CreateSwitch(condition, defbb);
    for (const auto & alt : *mop)
    {
      auto & type = *std::static_pointer_cast<const rvsdg::bittype>(mop->argument(0));
      auto value =
          ::llvm::ConstantInt::get(typeConverter.ConvertBitType(type, llvmContext), alt.first);
      sw->addCase(value, Context_->basic_block(node->OutEdge(alt.second)->sink()));
    }
  }
  else
  {
    auto defbb = Context_->basic_block(node->OutEdge(node->NumOutEdges() - 1)->sink());
    auto sw = builder.CreateSwitch(condition, defbb);
    for (size_t n = 0; n < node->NumOutEdges() - 1; n++)
    {
      auto value = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(builder.getContext()), n);
      sw->addCase(value, Context_->basic_block(node->OutEdge(n)->sink()));
    }
  }
}

void
IpGraphToLlvmConverter::create_terminator_instruction(const llvm::ControlFlowGraphNode * node)
{
  JLM_ASSERT(is<BasicBlock>(node));
  auto & tacs = static_cast<const BasicBlock *>(node)->tacs();
  auto & cfg = node->cfg();

  // unconditional branch or return statement
  if (node->NumOutEdges() == 1)
  {
    auto target = node->OutEdge(0)->sink();
    if (target == cfg.exit())
      return create_return(node);

    return create_unconditional_branch(node);
  }

  auto branch = tacs.last();
  JLM_ASSERT(branch && is<BranchOperation>(branch));

  // conditional branch
  if (Context_->value(branch->operand(0))->getType()->isIntegerTy(1))
    return create_conditional_branch(node);

  // switch
  create_switch(node);
}

::llvm::Attribute::AttrKind
IpGraphToLlvmConverter::ConvertAttributeKind(const Attribute::kind & kind)
{
  typedef ::llvm::Attribute::AttrKind ak;

  static std::unordered_map<Attribute::kind, ::llvm::Attribute::AttrKind> map(
      { { Attribute::kind::None, ak::None },

        { Attribute::kind::FirstEnumAttr, ak::FirstEnumAttr },
        { Attribute::kind::AllocAlign, ak::AllocAlign },
        { Attribute::kind::AllocatedPointer, ak::AllocatedPointer },
        { Attribute::kind::AlwaysInline, ak::AlwaysInline },
        { Attribute::kind::Builtin, ak::Builtin },
        { Attribute::kind::Cold, ak::Cold },
        { Attribute::kind::Convergent, ak::Convergent },
        { Attribute::kind::CoroDestroyOnlyWhenComplete, ak::CoroDestroyOnlyWhenComplete },
        { Attribute::kind::DeadOnUnwind, ak::DeadOnUnwind },
        { Attribute::kind::DisableSanitizerInstrumentation, ak::DisableSanitizerInstrumentation },
        { Attribute::kind::FnRetThunkExtern, ak::FnRetThunkExtern },
        { Attribute::kind::Hot, ak::Hot },
        { Attribute::kind::ImmArg, ak::ImmArg },
        { Attribute::kind::InReg, ak::InReg },
        { Attribute::kind::InlineHint, ak::InlineHint },
        { Attribute::kind::JumpTable, ak::JumpTable },
        { Attribute::kind::Memory, ak::Memory },
        { Attribute::kind::MinSize, ak::MinSize },
        { Attribute::kind::MustProgress, ak::MustProgress },
        { Attribute::kind::Naked, ak::Naked },
        { Attribute::kind::Nest, ak::Nest },
        { Attribute::kind::NoAlias, ak::NoAlias },
        { Attribute::kind::NoBuiltin, ak::NoBuiltin },
        { Attribute::kind::NoCallback, ak::NoCallback },
        { Attribute::kind::NoCapture, ak::NoCapture },
        { Attribute::kind::NoCfCheck, ak::NoCfCheck },
        { Attribute::kind::NoDuplicate, ak::NoDuplicate },
        { Attribute::kind::NoFree, ak::NoFree },
        { Attribute::kind::NoImplicitFloat, ak::NoImplicitFloat },
        { Attribute::kind::NoInline, ak::NoInline },
        { Attribute::kind::NoMerge, ak::NoMerge },
        { Attribute::kind::NoProfile, ak::NoProfile },
        { Attribute::kind::NoRecurse, ak::NoRecurse },
        { Attribute::kind::NoRedZone, ak::NoRedZone },
        { Attribute::kind::NoReturn, ak::NoReturn },
        { Attribute::kind::NoSanitizeBounds, ak::NoSanitizeBounds },
        { Attribute::kind::NoSanitizeCoverage, ak::NoSanitizeCoverage },
        { Attribute::kind::NoSync, ak::NoSync },
        { Attribute::kind::NoUndef, ak::NoUndef },
        { Attribute::kind::NoUnwind, ak::NoUnwind },
        { Attribute::kind::NonLazyBind, ak::NonLazyBind },
        { Attribute::kind::NonNull, ak::NonNull },
        { Attribute::kind::NullPointerIsValid, ak::NullPointerIsValid },
        { Attribute::kind::OptForFuzzing, ak::OptForFuzzing },
        { Attribute::kind::OptimizeForDebugging, ak::OptimizeForDebugging },
        { Attribute::kind::OptimizeForSize, ak::OptimizeForSize },
        { Attribute::kind::OptimizeNone, ak::OptimizeNone },
        { Attribute::kind::PresplitCoroutine, ak::PresplitCoroutine },
        { Attribute::kind::ReadNone, ak::ReadNone },
        { Attribute::kind::ReadOnly, ak::ReadOnly },
        { Attribute::kind::Returned, ak::Returned },
        { Attribute::kind::ReturnsTwice, ak::ReturnsTwice },
        { Attribute::kind::SExt, ak::SExt },
        { Attribute::kind::SafeStack, ak::SafeStack },
        { Attribute::kind::SanitizeAddress, ak::SanitizeAddress },
        { Attribute::kind::SanitizeHWAddress, ak::SanitizeHWAddress },
        { Attribute::kind::SanitizeMemTag, ak::SanitizeMemTag },
        { Attribute::kind::SanitizeMemory, ak::SanitizeMemory },
        { Attribute::kind::SanitizeThread, ak::SanitizeThread },
        { Attribute::kind::ShadowCallStack, ak::ShadowCallStack },
        { Attribute::kind::SkipProfile, ak::SkipProfile },
        { Attribute::kind::Speculatable, ak::Speculatable },
        { Attribute::kind::SpeculativeLoadHardening, ak::SpeculativeLoadHardening },
        { Attribute::kind::StackProtect, ak::StackProtect },
        { Attribute::kind::StackProtectReq, ak::StackProtectReq },
        { Attribute::kind::StackProtectStrong, ak::StackProtectStrong },
        { Attribute::kind::StrictFP, ak::StrictFP },
        { Attribute::kind::SwiftAsync, ak::SwiftAsync },
        { Attribute::kind::SwiftError, ak::SwiftError },
        { Attribute::kind::SwiftSelf, ak::SwiftSelf },
        { Attribute::kind::WillReturn, ak::WillReturn },
        { Attribute::kind::Writable, ak::Writable },
        { Attribute::kind::WriteOnly, ak::WriteOnly },
        { Attribute::kind::ZExt, ak::ZExt },
        { Attribute::kind::LastEnumAttr, ak::LastEnumAttr },

        { Attribute::kind::FirstTypeAttr, ak::FirstTypeAttr },
        { Attribute::kind::ByRef, ak::ByRef },
        { Attribute::kind::ByVal, ak::ByVal },
        { Attribute::kind::ElementType, ak::ElementType },
        { Attribute::kind::InAlloca, ak::InAlloca },
        { Attribute::kind::Preallocated, ak::Preallocated },
        { Attribute::kind::StructRet, ak::StructRet },
        { Attribute::kind::LastTypeAttr, ak::LastTypeAttr },

        { Attribute::kind::FirstIntAttr, ak::FirstIntAttr },
        { Attribute::kind::Alignment, ak::Alignment },
        { Attribute::kind::AllocKind, ak::AllocKind },
        { Attribute::kind::AllocSize, ak::AllocSize },
        { Attribute::kind::Dereferenceable, ak::Dereferenceable },
        { Attribute::kind::DereferenceableOrNull, ak::DereferenceableOrNull },
        { Attribute::kind::NoFPClass, ak::NoFPClass },
        { Attribute::kind::StackAlignment, ak::StackAlignment },
        { Attribute::kind::UWTable, ak::UWTable },
        { Attribute::kind::VScaleRange, ak::VScaleRange },
        { Attribute::kind::LastIntAttr, ak::LastIntAttr },

        { Attribute::kind::EndAttrKinds, ak::EndAttrKinds } });

  JLM_ASSERT(map.find(kind) != map.end());
  return map[kind];
}

::llvm::Attribute
IpGraphToLlvmConverter::ConvertEnumAttribute(const llvm::enum_attribute & attribute)
{
  auto & llvmContext = Context_->llvm_module().getContext();
  auto kind = ConvertAttributeKind(attribute.kind());
  return ::llvm::Attribute::get(llvmContext, kind);
}

::llvm::Attribute
IpGraphToLlvmConverter::ConvertIntAttribute(const llvm::int_attribute & attribute)
{
  auto & llvmContext = Context_->llvm_module().getContext();
  auto kind = ConvertAttributeKind(attribute.kind());
  return ::llvm::Attribute::get(llvmContext, kind, attribute.value());
}

::llvm::Attribute
IpGraphToLlvmConverter::ConvertTypeAttribute(const llvm::type_attribute & attribute)
{
  auto & typeConverter = Context_->GetTypeConverter();
  auto & llvmContext = Context_->llvm_module().getContext();

  auto kind = ConvertAttributeKind(attribute.kind());
  auto type = typeConverter.ConvertJlmType(attribute.type(), llvmContext);
  return ::llvm::Attribute::get(llvmContext, kind, type);
}

::llvm::Attribute
IpGraphToLlvmConverter::ConvertStringAttribute(const llvm::string_attribute & attribute)
{
  auto & llvmContext = Context_->llvm_module().getContext();
  return ::llvm::Attribute::get(llvmContext, attribute.kind(), attribute.value());
}

::llvm::AttributeSet
IpGraphToLlvmConverter::convert_attributes(const attributeset & attributeSet)
{
  ::llvm::AttrBuilder builder(Context_->llvm_module().getContext());
  for (auto & attribute : attributeSet.EnumAttributes())
    builder.addAttribute(ConvertEnumAttribute(attribute));

  for (auto & attribute : attributeSet.IntAttributes())
    builder.addAttribute(ConvertIntAttribute(attribute));

  for (auto & attribute : attributeSet.TypeAttributes())
    builder.addAttribute(ConvertTypeAttribute(attribute));

  for (auto & attribute : attributeSet.StringAttributes())
    builder.addAttribute(ConvertStringAttribute(attribute));

  return ::llvm::AttributeSet::get(Context_->llvm_module().getContext(), builder);
}

::llvm::AttributeList
IpGraphToLlvmConverter::convert_attributes(const function_node & f)
{
  JLM_ASSERT(f.cfg());

  auto & llvmctx = Context_->llvm_module().getContext();

  auto fctset = convert_attributes(f.attributes());

  // FIXME: return value attributes are currently not supported
  auto retset = ::llvm::AttributeSet();

  std::vector<::llvm::AttributeSet> argsets;
  for (size_t n = 0; n < f.cfg()->entry()->narguments(); n++)
  {
    auto argument = f.cfg()->entry()->argument(n);

    if (rvsdg::is<rvsdg::StateType>(argument->type()))
      continue;

    argsets.push_back(convert_attributes(argument->attributes()));
  }

  return ::llvm::AttributeList::get(llvmctx, fctset, retset, argsets);
}

std::vector<ControlFlowGraphNode *>
IpGraphToLlvmConverter::ConvertBasicBlocks(
    const ControlFlowGraph & controlFlowGraph,
    ::llvm::Function & function)
{
  auto nodes = breadth_first(controlFlowGraph);

  uint64_t basicBlockCounter = 0;
  for (const auto & node : nodes)
  {
    if (node == controlFlowGraph.entry())
      continue;
    if (node == controlFlowGraph.exit())
      continue;

    auto name = util::strfmt("bb", basicBlockCounter++);
    auto * basicBlock = ::llvm::BasicBlock::Create(function.getContext(), name, &function);
    Context_->insert(node, basicBlock);
  }

  return nodes;
}

void
IpGraphToLlvmConverter::convert_cfg(ControlFlowGraph & cfg, ::llvm::Function & f)
{
  JLM_ASSERT(is_closed(cfg));

  auto add_arguments = [&](const ControlFlowGraph & cfg, ::llvm::Function & f)
  {
    size_t n = 0;
    for (auto & llvmarg : f.args())
    {
      auto jlmarg = cfg.entry()->argument(n++);
      Context_->insert(jlmarg, &llvmarg);
    }
  };

  straighten(cfg);

  auto nodes = ConvertBasicBlocks(cfg, f);

  add_arguments(cfg, f);

  // create non-terminator instructions
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    JLM_ASSERT(is<BasicBlock>(node));
    auto & tacs = static_cast<const BasicBlock *>(node)->tacs();
    for (const auto & tac : tacs)
      convert_instruction(*tac, node);
  }

  // create cfg structure
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    create_terminator_instruction(node);
  }

  // patch phi instructions
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    JLM_ASSERT(is<BasicBlock>(node));
    auto & tacs = static_cast<const BasicBlock *>(node)->tacs();
    for (const auto & tac : tacs)
    {
      if (!is<SsaPhiOperation>(tac->operation()))
        continue;

      if (rvsdg::is<IOStateType>(tac->result(0)->type()))
        continue;
      if (rvsdg::is<MemoryStateType>(tac->result(0)->type()))
        continue;

      JLM_ASSERT(node->NumInEdges() == tac->noperands());
      auto & op = *static_cast<const SsaPhiOperation *>(&tac->operation());
      auto phi = ::llvm::dyn_cast<::llvm::PHINode>(Context_->value(tac->result(0)));
      for (size_t n = 0; n < tac->noperands(); n++)
        phi->addIncoming(
            Context_->value(tac->operand(n)),
            Context_->basic_block(op.GetIncomingNode(n)));
    }
  }
}

void
IpGraphToLlvmConverter::convert_function(const function_node & node)
{
  if (!node.cfg())
    return;

  auto & im = Context_->module();
  auto f = ::llvm::cast<::llvm::Function>(Context_->value(im.variable(&node)));

  auto attributes = convert_attributes(node);
  f->setAttributes(attributes);

  convert_cfg(*node.cfg(), *f);
}

void
IpGraphToLlvmConverter::convert_data_node(const data_node & node)
{
  if (!node.initialization())
    return;

  auto & jm = Context_->module();
  auto init = node.initialization();
  convert_tacs(init->tacs());

  auto gv = ::llvm::dyn_cast<::llvm::GlobalVariable>(Context_->value(jm.variable(&node)));
  gv->setInitializer(::llvm::dyn_cast<::llvm::Constant>(Context_->value(init->value())));
}

const ::llvm::GlobalValue::LinkageTypes &
IpGraphToLlvmConverter::convert_linkage(const llvm::linkage & linkage)
{
  static std::unordered_map<llvm::linkage, ::llvm::GlobalValue::LinkageTypes> map(
      { { llvm::linkage::external_linkage, ::llvm::GlobalValue::ExternalLinkage },
        { llvm::linkage::available_externally_linkage,
          ::llvm::GlobalValue::AvailableExternallyLinkage },
        { llvm::linkage::link_once_any_linkage, ::llvm::GlobalValue::LinkOnceAnyLinkage },
        { llvm::linkage::link_once_odr_linkage, ::llvm::GlobalValue::LinkOnceODRLinkage },
        { llvm::linkage::weak_any_linkage, ::llvm::GlobalValue::WeakAnyLinkage },
        { llvm::linkage::weak_odr_linkage, ::llvm::GlobalValue::WeakODRLinkage },
        { llvm::linkage::appending_linkage, ::llvm::GlobalValue::AppendingLinkage },
        { llvm::linkage::internal_linkage, ::llvm::GlobalValue::InternalLinkage },
        { llvm::linkage::private_linkage, ::llvm::GlobalValue::PrivateLinkage },
        { llvm::linkage::external_weak_linkage, ::llvm::GlobalValue::ExternalWeakLinkage },
        { llvm::linkage::common_linkage, ::llvm::GlobalValue::CommonLinkage } });

  JLM_ASSERT(map.find(linkage) != map.end());
  return map[linkage];
}

void
IpGraphToLlvmConverter::convert_ipgraph()
{
  auto & typeConverter = Context_->GetTypeConverter();
  auto & jm = Context_->module();
  auto & lm = Context_->llvm_module();

  // forward declare all nodes
  for (const auto & node : jm.ipgraph())
  {
    auto v = jm.variable(&node);

    if (auto dataNode = dynamic_cast<const data_node *>(&node))
    {
      auto type = typeConverter.ConvertJlmType(*dataNode->GetValueType(), lm.getContext());
      auto linkage = convert_linkage(dataNode->linkage());

      auto gv = new ::llvm::GlobalVariable(
          lm,
          type,
          dataNode->constant(),
          linkage,
          nullptr,
          dataNode->name());
      gv->setSection(dataNode->Section());
      Context_->insert(v, gv);
    }
    else if (auto n = dynamic_cast<const function_node *>(&node))
    {
      auto type = typeConverter.ConvertFunctionType(n->fcttype(), lm.getContext());
      auto linkage = convert_linkage(n->linkage());
      auto f = ::llvm::Function::Create(type, linkage, n->name(), &lm);
      Context_->insert(v, f);
    }
    else
      JLM_ASSERT(0);
  }

  // convert all nodes
  for (const auto & node : jm.ipgraph())
  {
    if (auto n = dynamic_cast<const data_node *>(&node))
    {
      convert_data_node(*n);
    }
    else if (auto n = dynamic_cast<const function_node *>(&node))
    {
      convert_function(*n);
    }
    else
      JLM_ASSERT(0);
  }
}

std::unique_ptr<::llvm::Module>
IpGraphToLlvmConverter::ConvertModule(
    InterProceduralGraphModule & ipGraphModule,
    ::llvm::LLVMContext & llvmContext)
{
  std::unique_ptr<::llvm::Module> llvmModule(new ::llvm::Module("module", llvmContext));
  llvmModule->setSourceFileName(ipGraphModule.source_filename().to_str());
  llvmModule->setTargetTriple(ipGraphModule.target_triple());
  llvmModule->setDataLayout(ipGraphModule.data_layout());

  Context_ = Context::Create(ipGraphModule, *llvmModule);
  convert_ipgraph();

  return llvmModule;
}

std::unique_ptr<::llvm::Module>
IpGraphToLlvmConverter::CreateAndConvertModule(
    InterProceduralGraphModule & ipGraphModule,
    ::llvm::LLVMContext & ctx)
{
  IpGraphToLlvmConverter converter;
  return converter.ConvertModule(ipGraphModule, ctx);
}

}
