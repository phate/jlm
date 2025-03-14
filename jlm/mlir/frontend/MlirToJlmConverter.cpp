/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

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
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>

namespace jlm::mlir
{

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ReadAndConvertMlir(const util::filepath & filePath)
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
  auto rvsdgModule = llvm::RvsdgModule::Create(util::filepath(""), std::string(), std::string());
  ConvertBlock(*block, rvsdgModule->Rvsdg().GetRootRegion());

  return rvsdgModule;
}

::llvm::SmallVector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertRegion(::mlir::Region & region, rvsdg::Region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return ConvertBlock(region.front(), rvsdgRegion);
}

::llvm::SmallVector<jlm::rvsdg::output *>
MlirToJlmConverter::GetConvertedInputs(
    ::mlir::Operation & mlirOp,
    const std::unordered_map<::mlir::Operation *, rvsdg::Node *> & operationsMap,
    const rvsdg::Region & rvsdgRegion)
{
  ::llvm::SmallVector<jlm::rvsdg::output *> inputs;
  for (::mlir::Value operand : mlirOp.getOperands())
  {
    if (::mlir::Operation * producer = operand.getDefiningOp())
    {
      JLM_ASSERT(operationsMap.find(producer) != operationsMap.end());
      JLM_ASSERT(::mlir::isa<::mlir::OpResult>(operand));
      inputs.push_back(
          operationsMap.at(producer)->output(operand.cast<::mlir::OpResult>().getResultNumber()));
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      JLM_ASSERT(::mlir::isa<::mlir::BlockArgument>(operand));
      inputs.push_back(rvsdgRegion.argument(operand.cast<::mlir::BlockArgument>().getArgNumber()));
    }
  }
  return inputs;
}

::llvm::SmallVector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertBlock(::mlir::Block & block, rvsdg::Region & rvsdgRegion)
{
  ::mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<::mlir::Operation *, rvsdg::Node *> operationsMap;
  for (auto & mlirOp : block.getOperations())
  {
    ::llvm::SmallVector<jlm::rvsdg::output *> inputs =
        GetConvertedInputs(mlirOp, operationsMap, rvsdgRegion);

    if (auto * node = ConvertOperation(mlirOp, rvsdgRegion, inputs))
    {
      operationsMap[&mlirOp] = node;
    }
  }

  // The results of the region/block are encoded in the terminator operation
  ::mlir::Operation * terminator = block.getTerminator();

  return GetConvertedInputs(*terminator, operationsMap, rvsdgRegion);
}

rvsdg::Node *
MlirToJlmConverter::ConvertCmpIOp(
    ::mlir::arith::CmpIOp & CompOp,
    const ::llvm::SmallVector<rvsdg::output *> & inputs,
    size_t nbits)
{
  if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::eq)
  {
    return rvsdg::output::GetNode(*rvsdg::biteq_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ne)
  {
    return rvsdg::output::GetNode(*rvsdg::bitne_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sge)
  {
    return rvsdg::output::GetNode(*rvsdg::bitsge_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sgt)
  {
    return rvsdg::output::GetNode(*rvsdg::bitsgt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sle)
  {
    return rvsdg::output::GetNode(*rvsdg::bitsle_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::slt)
  {
    return rvsdg::output::GetNode(*rvsdg::bitslt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::uge)
  {
    return rvsdg::output::GetNode(*rvsdg::bituge_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ugt)
  {
    return rvsdg::output::GetNode(*rvsdg::bitugt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ule)
  {
    return rvsdg::output::GetNode(*rvsdg::bitule_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ult)
  {
    return rvsdg::output::GetNode(*rvsdg::bitult_op::create(nbits, inputs[0], inputs[1]));
  }
  else
  {
    JLM_UNREACHABLE("frontend : Unknown comparison predicate.");
  }
}

rvsdg::Node *
MlirToJlmConverter::ConvertFPBinaryNode(
    const ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::output *> & inputs)
{
  if (inputs.size() != 2)
    return nullptr;
  llvm::fpop op;
  llvm::fpsize size;
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
      llvm::fpbin_op(op, size),
      { inputs[0], inputs[1] });
}

rvsdg::Node *
MlirToJlmConverter::ConvertBitBinaryNode(
    const ::mlir::Operation & mlirOperation,
    const ::llvm::SmallVector<rvsdg::output *> & inputs)
{
  if (inputs.size() != 2)
    return nullptr;
  if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::AddIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitadd_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::AndIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitand_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShRUIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitashr_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::MulIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitmul_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::OrIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitor_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::DivSIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitsdiv_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShLIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitshl_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShRUIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitshr_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::RemSIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitsmod_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::SubIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitsub_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::DivUIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitudiv_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::RemUIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitumod_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::XOrIOp>(&mlirOperation))
  {
    return rvsdg::output::GetNode(*rvsdg::bitxor_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }

  return nullptr;
}

rvsdg::Node *
MlirToJlmConverter::ConvertOperation(
    ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::output *> & inputs)
{

  // ** region Arithmetic Integer Operation **
  auto convertedBitBinaryNode = ConvertBitBinaryNode(mlirOperation, inputs);
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
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(&inputs[0]->type());
    if (!st)
      JLM_UNREACHABLE("Expected bitstring type for ExtUIOp operation.");
    ::mlir::Type type = castedOp.getType();
    return rvsdg::output::GetNode(*&llvm::ZExtOperation::Create(*(inputs[0]), ConvertType(type)));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ExtSIOp>(&mlirOperation))
  {
    auto outputType = castedOp.getOut().getType();
    auto convertedOutputType = ConvertType(outputType);
    if (!::mlir::isa<::mlir::IntegerType>(castedOp.getType()))
      JLM_UNREACHABLE("Expected IntegerType for ExtSIOp operation output.");
    return rvsdg::output::GetNode(*llvm::sext_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0]));
  }
  else if (auto sitofpOp = ::mlir::dyn_cast<::mlir::arith::SIToFPOp>(&mlirOperation))
  {
    auto st = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(inputs[0]->Type());
    if (!st)
      JLM_UNREACHABLE("Expected bits type for SIToFPOp operation.");

    auto mlirOutputType = sitofpOp.getType();
    std::shared_ptr<rvsdg::Type> rt = ConvertType(mlirOutputType);

    llvm::sitofp_op op(std::move(st), std::move(rt));
    return &rvsdg::SimpleNode::Create(
        rvsdgRegion,
        op,
        std::vector<jlm::rvsdg::output *>(inputs.begin(), inputs.end()));
  }

  else if (::mlir::isa<::mlir::rvsdg::OmegaNode>(&mlirOperation))
  {
    ConvertOmega(mlirOperation, rvsdgRegion);
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
        std::vector<jlm::rvsdg::output *>(inputs.begin(), inputs.end()));
  }
  else if (auto constant = ::mlir::dyn_cast<::mlir::arith::ConstantIntOp>(&mlirOperation))
  {
    auto type = constant.getType();
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return rvsdg::output::GetNode(
        *rvsdg::create_bitconstant(&rvsdgRegion, integerType.getWidth(), constant.value()));
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
    auto & output = rvsdg::SimpleNode::Create(
        rvsdgRegion,
        jlm::llvm::FNegOperation(size),
        {inputs[0]});
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
        {inputs[0]});
    return &output;
  }

  else if (auto truncOp = ::mlir::dyn_cast<::mlir::arith::TruncIOp>(&mlirOperation))
  {
    auto type = truncOp.getResult().getType();
    auto intType = ::mlir::cast<::mlir::IntegerType>(type);
    return rvsdg::output::GetNode(*jlm::llvm::TruncOperation::create(intType.getIntOrFloatBitWidth(), inputs[0]));
  }

  // Binary Integer Comparision operations
  else if (auto ComOp = ::mlir::dyn_cast<::mlir::arith::CmpIOp>(&mlirOperation))
  {
    auto type = ComOp.getOperandTypes()[0];
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return ConvertCmpIOp(ComOp, inputs, integerType.getWidth());
  }

  else if (auto UndefOp = ::mlir::dyn_cast<::mlir::jlm::Undef>(&mlirOperation))
  {
    auto type = UndefOp.getResult().getType();
    std::shared_ptr<jlm::rvsdg::Type> jlmType = ConvertType(type);
    auto jlmUndefOutput = jlm::llvm::UndefValueOperation::Create(rvsdgRegion, jlmType);
    return rvsdg::output::GetNode(*jlmUndefOutput);
  }

  else if (auto ArrayOp = ::mlir::dyn_cast<::mlir::jlm::ConstantDataArray>(&mlirOperation))
  {
    return rvsdg::output::GetNode(
        *llvm::ConstantDataArray::Create(std::vector(inputs.begin(), inputs.end())));
  }

  else if (auto ZeroOp = ::mlir::dyn_cast<::mlir::LLVM::ZeroOp>(&mlirOperation))
  {
    auto type = ZeroOp.getType();
    return rvsdg::output::GetNode(
        *llvm::ConstantAggregateZero::Create(rvsdgRegion, ConvertType(type)));
  }

  else if (auto VarArgOp = ::mlir::dyn_cast<::mlir::jlm::CreateVarArgList>(&mlirOperation))
  {
    return rvsdg::output::GetNode(
        *llvm::valist_op::Create(rvsdgRegion, std::vector(inputs.begin(), inputs.end())));
  }

  // Memory operations

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
    auto allocaOp = jlm::llvm::alloca_op(jlmValueType, jlmBitType, AllocaOp.getAlignment());
    auto operands = std::vector(inputs.begin(), inputs.end());

    return &rvsdg::SimpleNode::Create(rvsdgRegion, allocaOp, operands);
  }
  else if (auto MemstateMergeOp = ::mlir::dyn_cast<::mlir::rvsdg::MemStateMerge>(&mlirOperation))
  {
    auto operands = std::vector(inputs.begin(), inputs.end());
    auto memoryStateMergeOutput = jlm::llvm::MemoryStateMergeOperation::Create(operands);
    return rvsdg::output::GetNode(*memoryStateMergeOutput);
  }
  else if (auto StoreOp = ::mlir::dyn_cast<::mlir::jlm::Store>(&mlirOperation))
  {
    auto address = inputs[0];
    auto value = inputs[1];
    auto memoryStateInputs = std::vector(std::next(inputs.begin(), 2), inputs.end());
    auto & storeNode = jlm::llvm::StoreNonVolatileNode::CreateNode(
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
    auto & loadNode = jlm::llvm::LoadNonVolatileNode::CreateNode(
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
    return rvsdg::output::GetNode(*jlmGepOp);
  }
  // * region Structural nodes **
  else if (auto MlirCtrlConst = ::mlir::dyn_cast<::mlir::rvsdg::ConstantCtrl>(&mlirOperation))
  {
    JLM_ASSERT(::mlir::isa<::mlir::rvsdg::RVSDG_CTRLType>(MlirCtrlConst.getType()));
    return rvsdg::output::GetNode(*rvsdg::control_constant(
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

    ::llvm::SmallVector<::llvm::SmallVector<jlm::rvsdg::output *>> regionResults;
    for (size_t i = 0; i < mlirGammaNode.getNumRegions(); i++)
    {
      regionResults.push_back(
          ConvertRegion(mlirGammaNode.getRegion(i), *rvsdgGammaNode->subregion(i)));
    }

    // Connect the outputs
    //! Here we connect all subregion result to output of the gamma node
    for (size_t exitvarIndex = 0; exitvarIndex < regionResults[0].size(); exitvarIndex++)
    {
      std::vector<rvsdg::output *> exitvars;
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
    auto rvsdgDeltaNode = llvm::delta::node::Create(
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

    return rvsdg::output::GetNode(*rvsdg::match_op::Create(
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
      || ::mlir::isa<::mlir::rvsdg::DeltaResult>(&mlirOperation))
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
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

void
MlirToJlmConverter::ConvertOmega(::mlir::Operation & mlirOmega, rvsdg::Region & rvsdgRegion)
{
  JLM_ASSERT(mlirOmega.getRegions().size() == 1);
  ConvertRegion(mlirOmega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::Node *
MlirToJlmConverter::ConvertLambda(
    ::mlir::Operation & mlirOperation,
    rvsdg::Region & rvsdgRegion,
    const ::llvm::SmallVector<rvsdg::output *> & inputs)
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

  rvsdgLambda->finalize(std::vector<rvsdg::output *>(regionResults.begin(), regionResults.end()));

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::Type>
MlirToJlmConverter::ConvertType(::mlir::Type & type)
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
  else
  {
    JLM_UNREACHABLE("Type conversion not implemented\n");
  }
}

} // jlm::mlirrvsdg
