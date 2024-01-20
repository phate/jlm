/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/frontend/MlirToRvsdg.hpp>

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/TopologicalSortUtils.h>

namespace jlm::mlir
{

std::unique_ptr<::mlir::Block>
MlirToJlmConverter::ReadRvsdgMlir(const util::filepath & filePath)
{
  auto config = ::mlir::ParserConfig(Context_.get());
  std::unique_ptr<::mlir::Block> block = std::make_unique<::mlir::Block>();
  auto result = ::mlir::parseSourceFile(filePath.to_str(), block.get(), config);
  if (result.failed())
  {
    throw util::error("Parsing MLIR input file failed.");
  }
  return block;
}

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ConvertMlir(std::unique_ptr<::mlir::Block> & block)
{
  auto rvsdgModule = llvm::RvsdgModule::Create(util::filepath(""), std::string(), std::string());
  ConvertBlock(*block, *rvsdgModule->Rvsdg().root());

  return rvsdgModule;
}

std::vector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertRegion(::mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return ConvertBlock(region.front(), rvsdgRegion);
}

std::vector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertBlock(::mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  ::mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<::mlir::Operation *, rvsdg::node *> operations;
  for (auto & mlirOp : block.getOperations())
  {
    std::vector<const jlm::rvsdg::output *> inputs;
    for (auto operand : mlirOp.getOperands())
    {
      if (auto * producer = operand.getDefiningOp())
      {
        size_t index = GetOperandIndex(producer, operand);
        inputs.push_back(operations[producer]->output(index));
      }
      else
      {
        // If there is no defining op, the Value is necessarily a Block argument.
        auto blockArg = operand.cast<::mlir::BlockArgument>();
        inputs.push_back(rvsdgRegion.argument(blockArg.getArgNumber()));
      }
    }

    if (auto * node = ConvertOperation(mlirOp, rvsdgRegion, inputs))
    {
      operations[&mlirOp] = node;
    }
  }

  // The results of the region/block are encoded in the terminator operation
  auto terminator = block.getTerminator();
  std::vector<jlm::rvsdg::output *> results;
  for (auto operand : terminator->getOperands())
  {
    if (auto * producer = operand.getDefiningOp())
    {
      size_t index = GetOperandIndex(producer, operand);
      results.push_back(operations[producer]->output(index));
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      auto blockArg = operand.cast<::mlir::BlockArgument>();
      results.push_back(rvsdgRegion.argument(blockArg.getArgNumber()));
    }
  }

  return results;
}

rvsdg::node *
MlirToJlmConverter::ConvertOperation(
    ::mlir::Operation & mlirOperation,
    rvsdg::region & rvsdgRegion,
    std::vector<const rvsdg::output *> & inputs)
{
  if (::mlir::isa<::mlir::rvsdg::OmegaNode>(&mlirOperation))
  {
    ConvertOmega(mlirOperation, rvsdgRegion);
    // Omega doesn't have a corresponding RVSDG node so we return nullptr
    return nullptr;
  }
  else if (::mlir::isa<::mlir::rvsdg::LambdaNode>(&mlirOperation))
  {
    return ConvertLambda(mlirOperation, rvsdgRegion);
  }
  else if (auto constant = ::mlir::dyn_cast<::mlir::arith::ConstantIntOp>(&mlirOperation))
  {
    auto type = constant.getType();
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return rvsdg::node_output::node(
        rvsdg::create_bitconstant(&rvsdgRegion, integerType.getWidth(), constant.value()));
  }
  else if (
      ::mlir::isa<::mlir::rvsdg::LambdaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::OmegaResult>(&mlirOperation))
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
    return nullptr;
  }
  else
  {
    JLM_UNREACHABLE("Operation is not implemented.\n");
  }
}

void
MlirToJlmConverter::ConvertOmega(::mlir::Operation & mlirOmega, rvsdg::region & rvsdgRegion)
{
  // The Omega has a single region
  JLM_ASSERT(mlirOmega.getRegions().size() == 1);
  ConvertRegion(mlirOmega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::node *
MlirToJlmConverter::ConvertLambda(::mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion)
{
  // Get the name of the function
  auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != nullptr);
  auto functionName = ::mlir::cast<::mlir::StringAttr>(functionNameAttribute);

  // A lambda node has only the function signature as the result
  JLM_ASSERT(mlirLambda.getNumResults() == 1);
  auto result = mlirLambda.getResult(0).getType();

  if (result.getTypeID() != ::mlir::rvsdg::LambdaRefType::getTypeID())
  {
    JLM_ASSERT("The result from lambda node is not a LambdaRefType\n");
  }

  // Create the RVSDG function signature
  auto lambdaRefType = ::mlir::cast<::mlir::rvsdg::LambdaRefType>(result);
  std::vector<std::unique_ptr<rvsdg::type>> argumentTypes;
  for (auto argumentType : lambdaRefType.getParameterTypes())
  {
    argumentTypes.push_back(ConvertType(argumentType));
  }
  std::vector<std::unique_ptr<rvsdg::type>> resultTypes;
  for (auto returnType : lambdaRefType.getReturnTypes())
  {
    resultTypes.push_back(ConvertType(returnType));
  }
  llvm::FunctionType functionType(std::move(argumentTypes), std::move(resultTypes));

  auto rvsdgLambda = llvm::lambda::node::create(
      &rvsdgRegion,
      functionType,
      functionName.getValue().str(),
      llvm::linkage::external_linkage);

  auto lambdaRegion = rvsdgLambda->subregion();
  auto regionResults = ConvertRegion(mlirLambda.getRegion(0), *lambdaRegion);

  rvsdgLambda->finalize(regionResults);

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::type>
MlirToJlmConverter::ConvertType(::mlir::Type & type)
{
  if (::mlir::isa<::mlir::IntegerType>(type))
  {
    auto * intType = static_cast<::mlir::IntegerType *>(&type);
    return std::make_unique<rvsdg::bittype>(intType->getWidth());
  }
  else if (::mlir::isa<::mlir::rvsdg::LoopStateEdgeType>(type))
  {
    return std::make_unique<llvm::loopstatetype>();
  }
  else if (::mlir::isa<::mlir::rvsdg::MemStateEdgeType>(type))
  {
    return std::make_unique<llvm::MemoryStateType>();
  }
  else if (::mlir::isa<::mlir::rvsdg::IOStateEdgeType>(type))
  {
    return std::make_unique<llvm::iostatetype>();
  }
  else
  {
    JLM_UNREACHABLE("Type conversion not implemented\n");
  }
}

// TODO
// Consider tracking outputs instead of operations to avoid the need for this function
size_t
MlirToJlmConverter::GetOperandIndex(::mlir::Operation * producer, ::mlir::Value & operand)
{
  if (producer->getNumResults() == 1)
  {
    return 0;
  }

  size_t index = 0;
  for (auto tmp : producer->getResults())
  {
    if (tmp == operand)
    {
      break;
    }
    index++;
  }

  return index;
}

} // jlm::mlirrvsdg
