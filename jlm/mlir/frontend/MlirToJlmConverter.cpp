/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/mlir/MLIRConverterCommon.hpp>

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/TopologicalSortUtils.h>

#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>

namespace jlm::mlir
{

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ReadAndConvertMlir(const util::FilePath & filePath)
{
  auto config = ::mlir::ParserConfig(Context_.get());
  std::unique_ptr<::mlir::Block> block = std::make_unique<::mlir::Block>();
  auto result = ::mlir::parseSourceFile(filePath.to_str(), block.get(), config);
  if (result.failed())
  {
    JLM_ASSERT("Parsing MLIR input file failed.");
  }
  return ConvertMlir(block);
}

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ConvertMlir(std::unique_ptr<::mlir::Block> & block)
{
  auto & topNode = block->front();
  auto omegaNode = ::mlir::dyn_cast<::mlir::rvsdg::OmegaNode>(topNode);
  return ConvertOmega(omegaNode);
}

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ConvertOmega(::mlir::rvsdg::OmegaNode & omegaNode)
{
  auto rvsdgModule = llvm::RvsdgModule::Create(util::FilePath(""), std::string(), std::string());
  auto & graph = rvsdgModule->Rvsdg();
  auto & root = graph.GetRootRegion();
  ConvertRegion(omegaNode.getRegion(), root);

  return rvsdgModule;
}

::llvm::SmallVector<jlm::rvsdg::Output *>
MlirToJlmConverter::ConvertRegion(::mlir::Region & region, rvsdg::Region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return ConvertBlock(region.front(), rvsdgRegion);
}

::llvm::SmallVector<jlm::rvsdg::Output *>
MlirToJlmConverter::GetConvertedInputs(
    ::mlir::Operation & mlirOp,
    const std::unordered_map<void *, rvsdg::Output *> & outputMap)
{
  ::llvm::SmallVector<jlm::rvsdg::Output *> inputs;
  for (::mlir::Value operand : mlirOp.getOperands())
  {
    auto key = operand.getAsOpaquePointer();
    JLM_ASSERT(outputMap.find(key) != outputMap.end());
    inputs.push_back(outputMap.at(key));
  }
  return inputs;
}

::llvm::SmallVector<jlm::rvsdg::Output *>
MlirToJlmConverter::ConvertBlock(::mlir::Block & block, rvsdg::Region & rvsdgRegion)
{
  ::mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store the mapping from
  // MLIR values to RVSDG outputs in a hash map for easy lookup
  std::unordered_map<void *, rvsdg::Output *> outputMap;

  for (size_t i = 0; i < block.getNumArguments(); i++)
  {
    auto arg = block.getArgument(i);
    auto key = arg.getAsOpaquePointer();
    outputMap[key] = rvsdgRegion.argument(i);
  }

  for (auto & mlirOp : block.getOperations())
  {
    if (auto argument = ::mlir::dyn_cast<::mlir::rvsdg::OmegaArgument>(mlirOp))
    {
      auto valueType = argument.getValueType();
      auto importedType = argument.getImportedValue().getType();
      std::shared_ptr<rvsdg::Type> jlmValueType = ConvertType(valueType);
      std::shared_ptr<rvsdg::Type> jlmImportedType = ConvertType(importedType);

      jlm::llvm::GraphImport::Create(
          *rvsdgRegion.graph(),
          std::dynamic_pointer_cast<const rvsdg::ValueType>(jlmValueType),
          std::dynamic_pointer_cast<const rvsdg::ValueType>(jlmImportedType),
          argument.getNameAttr().cast<::mlir::StringAttr>().str(),
          llvm::FromString(argument.getLinkageAttr().cast<::mlir::StringAttr>().str()));

      auto key = argument.getResult().getAsOpaquePointer();
      outputMap[key] = rvsdgRegion.argument(rvsdgRegion.narguments() - 1);
    }
    else
    {
      ::llvm::SmallVector<jlm::rvsdg::Output *> inputs = GetConvertedInputs(mlirOp, outputMap);

      if (auto * node = ConvertOperation(mlirOp, rvsdgRegion, inputs))
      {
        for (size_t i = 0; i < mlirOp.getNumResults(); i++)
        {
          auto result = mlirOp.getResult(i);
          auto key = result.getAsOpaquePointer();
          outputMap[key] = node->output(i);
        }
      }
    }
  }

  // The results of the region/block are encoded in the terminator operation
  ::mlir::Operation * terminator = block.getTerminator();

  return GetConvertedInputs(*terminator, outputMap);
}

rvsdg::Node *
MlirToJlmConverter::ConvertCmpIOp(
    ::mlir::arith::CmpIOp & CompOp,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs,
    size_t nbits)
{
  if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::eq)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerEqOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ne)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerNeOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sge)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerSgeOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sgt)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerSgtOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sle)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerSleOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::slt)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerSltOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::uge)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerUgeOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ugt)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerUgtOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ule)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerUleOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ult)
  {
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::IntegerUltOperation(nbits),
        { inputs[0], inputs[1] });
  }
  else
  {
    JLM_UNREACHABLE("frontend : Unknown comparison predicate.");
  }
}

rvsdg::Node *
MlirToJlmConverter::ConvertICmpOp(
    ::mlir::LLVM::ICmpOp & operation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs)
{
  if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::eq)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::eq);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::ne)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::ne);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::sge)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::ge);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::sgt)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::gt);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::sle)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::le);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (operation.getPredicate() == ::mlir::LLVM::ICmpPredicate::slt)
  {
    auto newOp =
        std::make_unique<llvm::PtrCmpOperation>(llvm::PointerType::Create(), llvm::cmp::lt);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        std::move(newOp),
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else
  {
    JLM_UNREACHABLE("MLIR frontend: Unknown pointer compare operation");
  }
}

rvsdg::Node *
MlirToJlmConverter::ConvertFPBinaryNode(
    const ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs)
{
  if (inputs.size() != 2)
    return nullptr;
  auto op = llvm::fpop::add;
  auto size = llvm::fpsize::half;
  if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::AddFOp>(&mlirOperation))
  {
    op = llvm::fpop::add;
    size = ConvertFPSize(castedOp.getType().cast<::mlir::FloatType>().getWidth());
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::SubFOp>(&mlirOperation))
  {
    op = llvm::fpop::sub;
    size = ConvertFPSize(castedOp.getType().cast<::mlir::FloatType>().getWidth());
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::MulFOp>(&mlirOperation))
  {
    op = llvm::fpop::mul;
    size = ConvertFPSize(castedOp.getType().cast<::mlir::FloatType>().getWidth());
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::DivFOp>(&mlirOperation))
  {
    op = llvm::fpop::div;
    size = ConvertFPSize(castedOp.getType().cast<::mlir::FloatType>().getWidth());
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::RemFOp>(&mlirOperation))
  {
    op = llvm::fpop::mod;
    size = ConvertFPSize(castedOp.getType().cast<::mlir::FloatType>().getWidth());
  }
  else
  {
    return nullptr;
  }
  return &rvsdg::SimpleNode::Create(
      rvsdgRegion,
      llvm::FBinaryOperation(op, size),
      { inputs[0], inputs[1] });
}

llvm::fpcmp
MlirToJlmConverter::TryConvertFPCMP(const ::mlir::arith::CmpFPredicate & op)
{
  const auto & map = GetFpCmpPredicateMap();
  return map.LookupKey(op);
}

rvsdg::Node *
MlirToJlmConverter::ConvertBitBinaryNode(
    ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs)
{
  if (inputs.size() != 2 || mlirOperation.getNumResults() != 1)
    return nullptr;

  auto type = mlirOperation.getResult(0).getType();

  if (!type.isa<::mlir::IntegerType>())
    return nullptr;

  auto integerType = type.cast<::mlir::IntegerType>();
  auto width = integerType.getWidth();

  if (::mlir::isa<::mlir::arith::AddIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerAddOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::SubIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerSubOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::MulIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerMulOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::DivSIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerSDivOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::DivUIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerUDivOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::RemSIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerSRemOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::RemUIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerURemOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::LLVM::ShlOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerShlOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::LLVM::AShrOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerAShrOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::LLVM::LShrOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerLShrOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::AndIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerAndOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::OrIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerOrOperation>({ inputs[0], inputs[1] }, width);
  }
  else if (::mlir::isa<::mlir::arith::XOrIOp>(mlirOperation))
  {
    return &rvsdg::CreateOpNode<jlm::llvm::IntegerXorOperation>({ inputs[0], inputs[1] }, width);
  }
  else
  {
    return nullptr;
  }
}

rvsdg::Node *
MlirToJlmConverter::ConvertOperation(
    ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs)
{

  // ** region Arithmetic Integer Operation **
  auto convertedBitBinaryNode = ConvertBitBinaryNode(mlirOperation, rvsdgRegion, inputs);
  // If the operation was converted it means it has been casted to a bit binary operation
  if (convertedBitBinaryNode)
    return convertedBitBinaryNode;
  // ** endregion Arithmetic Integer Operation **

  // ** region Arithmetic Float Operation **
  auto convertedFloatBinaryNode = ConvertFPBinaryNode(mlirOperation, rvsdgRegion, inputs);
  // If the operation was converted it means it has been casted to a fp binary operation
  if (convertedFloatBinaryNode)
    return convertedFloatBinaryNode;
  // ** endregion Arithmetic Float Operation **

  if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ExtUIOp>(&mlirOperation))
  {
    auto st = std::dynamic_pointer_cast<const rvsdg::bittype>(inputs[0]->Type());
    if (!st)
      JLM_UNREACHABLE("Expected bitstring type for ExtUIOp operation.");
    ::mlir::Type type = castedOp.getType();
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(
        *&llvm::ZExtOperation::Create(*(inputs[0]), ConvertType(type)));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ExtSIOp>(&mlirOperation))
  {
    auto outputType = castedOp.getOut().getType();
    auto convertedOutputType = ConvertType(outputType);
    if (!::mlir::isa<::mlir::IntegerType>(castedOp.getType()))
      JLM_UNREACHABLE("Expected IntegerType for ExtSIOp operation output.");
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*llvm::SExtOperation::create(
        castedOp.getType().cast<::mlir::IntegerType>().getWidth(),
        inputs[0]));
  }
  else if (auto sitofpOp = ::mlir::dyn_cast<::mlir::arith::SIToFPOp>(&mlirOperation))
  {
    auto st = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(inputs[0]->Type());
    if (!st)
      JLM_UNREACHABLE("Expected bits type for SIToFPOp operation.");

    auto mlirOutputType = sitofpOp.getType();
    std::shared_ptr<rvsdg::Type> rt = ConvertType(mlirOutputType);

    llvm::SIToFPOperation op(std::move(st), std::move(rt));
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        op,
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }

  else if (::mlir::isa<::mlir::rvsdg::OmegaNode>(&mlirOperation))
  {
    // Omega doesn't have a corresponding RVSDG node so we return nullptr
    return nullptr;
  }
  else if (::mlir::isa<::mlir::rvsdg::LambdaNode>(&mlirOperation))
  {
    return ConvertLambda(mlirOperation, rvsdgRegion, inputs);
  }
  else if (auto callOp = ::mlir::dyn_cast<::mlir::jlm::Call>(&mlirOperation))
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> argumentTypes;
    for (auto arg : callOp.getArgs())
    {
      auto type = arg.getType();
      argumentTypes.push_back(ConvertType(type));
    }
    argumentTypes.push_back(llvm::IOStateType::Create());
    argumentTypes.push_back(llvm::MemoryStateType::Create());
    std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes;
    for (auto res : callOp.getResults())
    {
      auto type = res.getType();
      resultTypes.push_back(ConvertType(type));
    }

    auto callOperation =
        llvm::CallOperation(std::make_shared<rvsdg::FunctionType>(argumentTypes, resultTypes));
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        callOperation,
        std::vector<jlm::rvsdg::Output *>(inputs.begin(), inputs.end()));
  }
  else if (auto constant = ::mlir::dyn_cast<::mlir::arith::ConstantIntOp>(&mlirOperation))
  {
    auto type = constant.getType();
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return &jlm::llvm::IntegerConstantOperation::Create(
        rvsdgRegion,
        integerType.getWidth(),
        constant.value());
  }
  else if (auto constant = ::mlir::dyn_cast<::mlir::arith::ConstantFloatOp>(&mlirOperation))
  {
    auto type = constant.getType();
    if (!::mlir::isa<::mlir::FloatType>(type))
      JLM_UNREACHABLE("Expected FloatType for ConstantFloatOp operation.");
    auto floatType = ::mlir::cast<::mlir::FloatType>(type);

    auto size = ConvertFPSize(floatType.getWidth());
    auto & output =
        rvsdg::SimpleNode::Create(rvsdgRegion, llvm::ConstantFP(size, constant.value()), {});
    return &output;
  }

  else if (auto negOp = ::mlir::dyn_cast<::mlir::arith::NegFOp>(&mlirOperation))
  {
    auto type = negOp.getResult().getType();
    auto floatType = ::mlir::cast<::mlir::FloatType>(type);

    llvm::fpsize size = ConvertFPSize(floatType.getWidth());
    auto & output =
        rvsdg::SimpleNode::Create(rvsdgRegion, jlm::llvm::FNegOperation(size), { inputs[0] });
    return &output;
  }

  else if (auto extOp = ::mlir::dyn_cast<::mlir::arith::ExtFOp>(&mlirOperation))
  {
    auto type = extOp.getResult().getType();
    auto floatType = ::mlir::cast<::mlir::FloatType>(type);

    llvm::fpsize size = ConvertFPSize(floatType.getWidth());
    auto & output = rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::FPExtOperation(inputs[0]->Type(), llvm::FloatingPointType::Create(size)),
        { inputs[0] });
    return &output;
  }

  else if (auto truncOp = ::mlir::dyn_cast<::mlir::arith::TruncIOp>(&mlirOperation))
  {
    auto type = truncOp.getResult().getType();
    auto intType = ::mlir::cast<::mlir::IntegerType>(type);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(
        *llvm::TruncOperation::create(intType.getIntOrFloatBitWidth(), inputs[0]));
  }

  // Binary Integer Comparision operations
  else if (auto ComOp = ::mlir::dyn_cast<::mlir::arith::CmpIOp>(&mlirOperation))
  {
    auto type = ComOp.getOperandTypes()[0];
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return ConvertCmpIOp(ComOp, rvsdgRegion, inputs, integerType.getWidth());
  }

  else if (auto ComOp = ::mlir::dyn_cast<::mlir::arith::CmpFOp>(&mlirOperation))
  {
    auto type = ComOp.getOperandTypes()[0];
    auto floatType = ::mlir::cast<::mlir::FloatType>(type);
    auto op = llvm::FCmpOperation(
        TryConvertFPCMP(ComOp.getPredicate()),
        ConvertFPSize(floatType.getWidth()));
    return &rvsdg::SimpleNode::Create(rvsdgRegion, op, std::vector(inputs.begin(), inputs.end()));
  }

  // Pointer compare is mapped to LLVM::ICmpOp
  else if (auto iComOp = ::mlir::dyn_cast<::mlir::LLVM::ICmpOp>(&mlirOperation))
  {
    return ConvertICmpOp(iComOp, rvsdgRegion, inputs);
  }

  else if (auto UndefOp = ::mlir::dyn_cast<::mlir::jlm::Undef>(&mlirOperation))
  {
    auto type = UndefOp.getResult().getType();
    std::shared_ptr<jlm::rvsdg::Type> jlmType = ConvertType(type);
    auto jlmUndefOutput = jlm::llvm::UndefValueOperation::Create(rvsdgRegion, jlmType);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*jlmUndefOutput);
  }

  else if (auto ArrayOp = ::mlir::dyn_cast<::mlir::jlm::ConstantDataArray>(&mlirOperation))
  {
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(
        *llvm::ConstantDataArray::Create(std::vector(inputs.begin(), inputs.end())));
  }

  else if (auto ZeroOp = ::mlir::dyn_cast<::mlir::LLVM::ZeroOp>(&mlirOperation))
  {
    auto type = ZeroOp.getType();
    // NULL pointers are a special case of ZeroOp
    if (::mlir::isa<::mlir::LLVM::LLVMPointerType>(type))
    {
      return rvsdg::TryGetOwnerNode<rvsdg::Node>(
          *llvm::ConstantPointerNullOperation::Create(&rvsdgRegion, ConvertType(type)));
    }
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(
        *llvm::ConstantAggregateZeroOperation::Create(rvsdgRegion, ConvertType(type)));
  }

  else if (auto VarArgOp = ::mlir::dyn_cast<::mlir::jlm::CreateVarArgList>(&mlirOperation))
  {
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*llvm::VariadicArgumentListOperation::Create(
        rvsdgRegion,
        std::vector(inputs.begin(), inputs.end())));
  }

  // Memory operations

  else if (auto FreeOp = ::mlir::dyn_cast<::mlir::jlm::Free>(&mlirOperation))
  {
    llvm::FreeOperation freeOp(inputs.size() - 2);
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        freeOp,
        std::vector(inputs.begin(), inputs.end()));
  }

  else if (auto AllocaOp = ::mlir::dyn_cast<::mlir::jlm::Alloca>(&mlirOperation))
  {
    auto outputType = AllocaOp.getValueType();

    std::shared_ptr<jlm::rvsdg::Type> jlmType = ConvertType(outputType);
    if (!rvsdg::is<const rvsdg::ValueType>(jlmType))
      JLM_UNREACHABLE("Expected ValueType for AllocaOp operation.");

    auto jlmValueType = std::dynamic_pointer_cast<const rvsdg::ValueType>(jlmType);
    if (!rvsdg::is<const rvsdg::bittype>(inputs[0]->Type()))
      JLM_UNREACHABLE("Expected bittype for AllocaOp operation.");

    auto jlmBitType = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(inputs[0]->Type());
    auto allocaOp = jlm::llvm::AllocaOperation(jlmValueType, jlmBitType, AllocaOp.getAlignment());
    auto operands = std::vector(inputs.begin(), inputs.end());

    return &rvsdg::SimpleNode::Create(rvsdgRegion, allocaOp, operands);
  }
  else if (auto MemstateMergeOp = ::mlir::dyn_cast<::mlir::rvsdg::MemStateMerge>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto memoryStateMergeOutput = jlm::llvm::MemoryStateMergeOperation::Create(operands);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*memoryStateMergeOutput);
  }
  else if (
      auto LambdaEntryMemstateSplitOp =
          ::mlir::dyn_cast<::mlir::rvsdg::LambdaEntryMemoryStateSplitOperation>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto lambdaMemoryStateSplitOutput = jlm::llvm::LambdaEntryMemoryStateSplitOperation::Create(
        *operands.front(),
        LambdaEntryMemstateSplitOp.getNumResults());
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*lambdaMemoryStateSplitOutput.front());
  }
  else if (
      auto LambdaExitMemstateMergeOp =
          ::mlir::dyn_cast<::mlir::rvsdg::LambdaExitMemoryStateMergeOperation>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto & lambdaMemoryStateMergeOutput =
        jlm::llvm::LambdaExitMemoryStateMergeOperation::Create(rvsdgRegion, operands);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(lambdaMemoryStateMergeOutput);
  }
  else if (
      auto CallEntryMemstateMergeOp =
          ::mlir::dyn_cast<::mlir::rvsdg::CallEntryMemoryStateMerge>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto & callMemoryStateMergeOutput =
        jlm::llvm::CallEntryMemoryStateMergeOperation::Create(rvsdgRegion, operands);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(callMemoryStateMergeOutput);
  }
  else if (
      auto CallExitMemstateSplitOp =
          ::mlir::dyn_cast<::mlir::rvsdg::CallExitMemoryStateSplit>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto callMemoryStateSplitOutput = jlm::llvm::CallExitMemoryStateSplitOperation::Create(
        *operands.front(),
        CallExitMemstateSplitOp.getNumResults());
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*callMemoryStateSplitOutput.front());
  }
  else if (auto IOBarrierOp = ::mlir::dyn_cast<::mlir::jlm::IOBarrier>(&mlirOperation))
  {
    // auto operands = std::vector(inputs.begin(), inputs.end());
    auto type = IOBarrierOp.getResult().getType();
    auto ioBarrierOp = jlm::llvm::IOBarrierOperation(ConvertType(type));
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        ioBarrierOp,
        std::vector(inputs.begin(), inputs.end()));
  }
  else if (auto MallocOp = ::mlir::dyn_cast<::mlir::jlm::Malloc>(&mlirOperation))
  {
    auto mallocOutputs = jlm::llvm::MallocOperation::create(inputs[0]);
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*mallocOutputs[0]);
  }
  else if (auto StoreOp = ::mlir::dyn_cast<::mlir::jlm::Store>(&mlirOperation))
  {
    auto address = inputs[0];
    auto value = inputs[1];
    auto memoryStateInputs = std::vector(std::next(inputs.begin(), 2), inputs.end());
    auto & storeNode = jlm::llvm::StoreNonVolatileOperation::CreateNode(
        *address,
        *value,
        memoryStateInputs,
        StoreOp.getAlignment());
    return &storeNode;
  }
  else if (auto LoadOp = ::mlir::dyn_cast<::mlir::jlm::Load>(&mlirOperation))
  {
    auto address = inputs[0];
    auto memoryStateInputs = std::vector(std::next(inputs.begin()), inputs.end());
    auto outputType = LoadOp.getOutput().getType();
    std::shared_ptr<jlm::rvsdg::Type> jlmType = ConvertType(outputType);
    if (!rvsdg::is<const rvsdg::ValueType>(jlmType))
      JLM_UNREACHABLE("Expected ValueType for LoadOp operation output.");
    auto jlmValueType = std::dynamic_pointer_cast<const rvsdg::ValueType>(jlmType);
    auto & loadNode = llvm::LoadNonVolatileOperation::CreateNode(
        *address,
        memoryStateInputs,
        jlmValueType,
        LoadOp.getAlignment());
    return &loadNode;
  }
  else if (auto GepOp = ::mlir::dyn_cast<::mlir::LLVM::GEPOp>(&mlirOperation))
  {
    auto elemType = GepOp.getElemType();
    std::shared_ptr<jlm::rvsdg::Type> pointeeType = ConvertType(elemType);
    if (!rvsdg::is<const rvsdg::ValueType>(pointeeType))
      JLM_UNREACHABLE("Expected ValueType for GepOp operation pointee.");

    auto pointeeValueType = std::dynamic_pointer_cast<const rvsdg::ValueType>(pointeeType);

    auto jlmGepOp = jlm::llvm::GetElementPtrOperation::Create(
        inputs[0],
        std::vector(std::next(inputs.begin()), inputs.end()),
        pointeeValueType,
        llvm::PointerType::Create());
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*jlmGepOp);
  }
  // * region Structural nodes **
  else if (auto MlirCtrlConst = ::mlir::dyn_cast<::mlir::rvsdg::ConstantCtrl>(&mlirOperation))
  {
    JLM_ASSERT(::mlir::isa<::mlir::rvsdg::RVSDG_CTRLType>(MlirCtrlConst.getType()));
    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*rvsdg::control_constant(
        &rvsdgRegion,
        ::mlir::cast<::mlir::rvsdg::RVSDG_CTRLType>(MlirCtrlConst.getType()).getNumOptions(),
        MlirCtrlConst.getValue()));
  }
  else if (auto mlirGammaNode = ::mlir::dyn_cast<::mlir::rvsdg::GammaNode>(&mlirOperation))
  {
    auto rvsdgGammaNode = rvsdg::GammaNode::create(
        inputs[0],                    // predicate
        mlirGammaNode.getNumRegions() // nalternatives
    );

    // Add inputs to the gamma node and to all it's subregions
    for (size_t i = 1; i < inputs.size(); i++)
    {
      rvsdgGammaNode->AddEntryVar(inputs[i]);
    }

    ::llvm::SmallVector<::llvm::SmallVector<jlm::rvsdg::Output *>> regionResults;
    for (size_t i = 0; i < mlirGammaNode.getNumRegions(); i++)
    {
      regionResults.push_back(
          ConvertRegion(mlirGammaNode.getRegion(i), *rvsdgGammaNode->subregion(i)));
    }

    // Connect the outputs
    //! Here we connect all subregion result to output of the gamma node
    for (size_t exitvarIndex = 0; exitvarIndex < regionResults[0].size(); exitvarIndex++)
    {
      std::vector<rvsdg::Output *> exitvars;
      for (size_t regionIndex = 0; regionIndex < mlirGammaNode.getNumRegions(); regionIndex++)
      {
        JLM_ASSERT(regionResults[regionIndex].size() == regionResults[0].size());
        exitvars.push_back(regionResults[regionIndex][exitvarIndex]);
      }
      rvsdgGammaNode->AddExitVar(exitvars);
    }

    return rvsdgGammaNode;
  }
  else if (auto mlirThetaNode = ::mlir::dyn_cast<::mlir::rvsdg::ThetaNode>(&mlirOperation))
  {
    auto rvsdgThetaNode = rvsdg::ThetaNode::create(&rvsdgRegion);

    // Add loop vars to the theta node
    for (size_t i = 0; i < inputs.size(); i++)
    {
      rvsdgThetaNode->AddLoopVar(inputs[i]);
    }

    auto regionResults = ConvertRegion(mlirThetaNode.getRegion(), *rvsdgThetaNode->subregion());

    rvsdgThetaNode->set_predicate(regionResults[0]);

    return rvsdgThetaNode;
  }
  else if (auto mlirDeltaNode = ::mlir::dyn_cast<::mlir::rvsdg::DeltaNode>(&mlirOperation))
  {
    auto & deltaRegion = mlirDeltaNode.getRegion();
    auto & deltaBlock = deltaRegion.front();
    auto terminator = deltaBlock.getTerminator();

    auto mlirOutputType = terminator->getOperand(0).getType();
    std::shared_ptr<rvsdg::Type> outputType = ConvertType(mlirOutputType);
    auto outputValueType = std::dynamic_pointer_cast<const rvsdg::ValueType>(outputType);
    auto linakgeString = mlirDeltaNode.getLinkage().str();
    auto rvsdgDeltaNode = llvm::DeltaNode::Create(
        &rvsdgRegion,
        outputValueType,
        mlirDeltaNode.getName().str(),
        jlm::llvm::FromString(linakgeString),
        mlirDeltaNode.getSection().str(),
        mlirDeltaNode.getConstant());

    auto outputVector = ConvertRegion(mlirDeltaNode.getRegion(), *rvsdgDeltaNode->subregion());

    if (outputVector.size() != 1)
      JLM_UNREACHABLE("Expected 1 output for Delta operation.");

    rvsdgDeltaNode->finalize(outputVector[0]);

    return rvsdgDeltaNode;
  }
  else if (auto mlirMatch = ::mlir::dyn_cast<::mlir::rvsdg::Match>(&mlirOperation))
  {
    std::unordered_map<uint64_t, uint64_t> mapping;
    uint64_t defaultAlternative = 0;
    for (auto & attr : mlirMatch.getMapping())
    {
      JLM_ASSERT(attr.isa<::mlir::rvsdg::MatchRuleAttr>());
      auto matchRuleAttr = attr.cast<::mlir::rvsdg::MatchRuleAttr>();
      if (matchRuleAttr.isDefault())
      {
        defaultAlternative = matchRuleAttr.getIndex();
        continue;
      }
      // In our Mlir implementation, an index is associated with a single value
      mapping[matchRuleAttr.getValues().front()] = matchRuleAttr.getIndex();
    }

    return rvsdg::TryGetOwnerNode<rvsdg::Node>(*rvsdg::match_op::Create(
        *(inputs[0]),                 // predicate
        mapping,                      // mapping
        defaultAlternative,           // defaultAlternative
        mlirMatch.getMapping().size() // numAlternatives
        ));
  }
  // ** endregion Structural nodes **

  else if (
      ::mlir::isa<::mlir::rvsdg::LambdaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::OmegaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::GammaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::ThetaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::DeltaResult>(&mlirOperation)
      // This is a terminating operation that doesn't have a corresponding RVSDG node
      || ::mlir::isa<::mlir::rvsdg::OmegaArgument>(&mlirOperation)) // Handled at the top level
  {
    return nullptr;
  }
  else
  {
    auto message = util::strfmt(
        "Operation not implemented: ",
        mlirOperation.getName().getStringRef().str(),
        "\n");
    JLM_UNREACHABLE(message.c_str());
  }
}

llvm::fpsize
MlirToJlmConverter::ConvertFPSize(unsigned int size)
{
  switch (size)
  {
  case 16:
    return llvm::fpsize::half;
  case 32:
    return llvm::fpsize::flt;
  case 64:
    return llvm::fpsize::dbl;
  case 80:
    return llvm::fpsize::x86fp80;
  case 128:
    return llvm::fpsize::fp128;
  default:
    auto message = util::strfmt("Unsupported floating point size: ", size, "\n");
    JLM_UNREACHABLE(message.c_str());
    break;
  }
}

jlm::rvsdg::Node *
MlirToJlmConverter::ConvertLambda(
    ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::Output *> & inputs)
{
  // Get the name of the function
  auto functionNameAttribute = mlirOperation.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != nullptr);
  auto functionName = ::mlir::cast<::mlir::StringAttr>(functionNameAttribute);

  auto lambdaOp = ::mlir::dyn_cast<::mlir::rvsdg::LambdaNode>(&mlirOperation);
  auto & lambdaRegion = lambdaOp.getRegion();
  auto numNonContextVars = lambdaRegion.getNumArguments() - lambdaOp.getNumOperands();
  auto & lambdaBlock = lambdaRegion.front();
  auto lamdbaTerminator = lambdaBlock.getTerminator();

  // Create the RVSDG function signature
  std::vector<std::shared_ptr<const rvsdg::Type>> argumentTypes;
  for (size_t argumentIndex = 0; argumentIndex < numNonContextVars; argumentIndex++)
  {
    auto type = lambdaRegion.getArgument(argumentIndex).getType();
    argumentTypes.push_back(ConvertType(type));
  }
  std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes;
  for (auto returnType : lamdbaTerminator->getOperandTypes())
  {
    resultTypes.push_back(ConvertType(returnType));
  }
  auto functionType = rvsdg::FunctionType::Create(std::move(argumentTypes), std::move(resultTypes));

  // FIXME
  // The linkage should be part of the MLIR attributes so it can be extracted here
  auto rvsdgLambda = rvsdg::LambdaNode::Create(
      rvsdgRegion,
      llvm::LlvmLambdaOperation::Create(
          functionType,
          functionName.getValue().str(),
          llvm::linkage::external_linkage));

  for (auto input : inputs)
  {
    rvsdgLambda->AddContextVar(*input);
  }

  auto jlmLambdaRegion = rvsdgLambda->subregion();
  auto regionResults = ConvertRegion(lambdaRegion, *jlmLambdaRegion);

  rvsdgLambda->finalize(std::vector<rvsdg::Output *>(regionResults.begin(), regionResults.end()));

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::Type>
MlirToJlmConverter::ConvertType(const ::mlir::Type & type)
{
  if (auto ctrlType = ::mlir::dyn_cast<::mlir::rvsdg::RVSDG_CTRLType>(type))
  {
    return std::make_unique<rvsdg::ControlType>(ctrlType.getNumOptions());
  }
  else if (auto intType = ::mlir::dyn_cast<::mlir::IntegerType>(type))
  {
    return std::make_unique<rvsdg::bittype>(intType.getWidth());
  }
  else if (::mlir::isa<::mlir::Float16Type>(type))
  {
    return std::make_unique<llvm::FloatingPointType>(llvm::fpsize::half);
  }
  else if (::mlir::isa<::mlir::Float32Type>(type))
  {
    return std::make_unique<llvm::FloatingPointType>(llvm::fpsize::flt);
  }
  else if (::mlir::isa<::mlir::Float64Type>(type))
  {
    return std::make_unique<llvm::FloatingPointType>(llvm::fpsize::dbl);
  }
  else if (::mlir::isa<::mlir::Float80Type>(type))
  {
    return std::make_unique<llvm::FloatingPointType>(llvm::fpsize::x86fp80);
  }
  else if (::mlir::isa<::mlir::Float128Type>(type))
  {
    return std::make_unique<llvm::FloatingPointType>(llvm::fpsize::fp128);
  }
  else if (::mlir::isa<::mlir::rvsdg::MemStateEdgeType>(type))
  {
    return std::make_unique<llvm::MemoryStateType>();
  }
  else if (::mlir::isa<::mlir::rvsdg::IOStateEdgeType>(type))
  {
    return std::make_unique<llvm::IOStateType>();
  }
  else if (::mlir::isa<::mlir::LLVM::LLVMPointerType>(type))
  {
    return std::make_unique<llvm::PointerType>();
  }
  else if (::mlir::isa<::mlir::jlm::VarargListType>(type))
  {
    return std::make_unique<llvm::VariableArgumentType>();
  }
  else if (auto arrayType = ::mlir::dyn_cast<::mlir::LLVM::LLVMArrayType>(type))
  {
    auto mlirElementType = arrayType.getElementType();
    std::shared_ptr<rvsdg::Type> elementType = ConvertType(mlirElementType);
    auto elemenValueType = std::dynamic_pointer_cast<const rvsdg::ValueType>(elementType);
    return std::make_unique<llvm::ArrayType>(elemenValueType, arrayType.getNumElements());
  }
  else if (auto functionType = ::mlir::dyn_cast<::mlir::FunctionType>(type))
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> argumentTypes;
    for (auto argumentType : functionType.getInputs())
    {
      argumentTypes.push_back(ConvertType(argumentType));
    }
    std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes;
    for (auto resultType : functionType.getResults())
    {
      resultTypes.push_back(ConvertType(resultType));
    }
    return std::make_unique<rvsdg::FunctionType>(argumentTypes, resultTypes);
  }
  else
  {
    JLM_UNREACHABLE("Type conversion not implemented\n");
  }
}

} // jlm::mlirrvsdg
