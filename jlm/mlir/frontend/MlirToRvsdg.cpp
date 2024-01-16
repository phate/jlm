/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/mlir/frontend/MlirToRvsdg.hpp"

#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

namespace jlm::mlirrvsdg
{

std::unique_ptr<mlir::Block>
MlirToRvsdg::readRvsdgMlir(const util::filepath & filePath)
{
  mlir::ParserConfig config = mlir::ParserConfig(Context_.get());
  std::unique_ptr<mlir::Block> block = std::make_unique<mlir::Block>();
  auto result = mlir::parseSourceFile(filePath.to_str(), block.get(), config);
  if (result.failed())
  {
    throw util::error("Parsing MLIR input file failed.");
  }
  return block;
}

std::unique_ptr<llvm::RvsdgModule>
MlirToRvsdg::convertMlir(std::unique_ptr<mlir::Block> & block)
{
  std::string dataLayout;
  std::string targetTriple;
  util::filepath sourceFileName("");
  auto rvsdgModule = llvm::RvsdgModule::Create(sourceFileName, targetTriple, dataLayout);
  auto & graph = rvsdgModule->Rvsdg();
  auto root = graph.root();

  convertBlock(*block.get(), *root);

  return rvsdgModule;
}

std::unique_ptr<std::vector<jlm::rvsdg::output *>>
MlirToRvsdg::convertRegion(mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return convertBlock(region.front(), rvsdgRegion);
}

std::unique_ptr<std::vector<jlm::rvsdg::output *>>
MlirToRvsdg::convertBlock(mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<mlir::Operation *, rvsdg::node *> operations;
  for (mlir::Operation & mlirOp : block.getOperations())
  {
    std::vector<const jlm::rvsdg::output *> inputs;
    for (mlir::Value operand : mlirOp.getOperands())
    {
      if (mlir::Operation * producer = operand.getDefiningOp())
      {
        size_t index = getOperandIndex(producer, operand);
        inputs.push_back(operations[producer]->output(index));
      }
      else
      {
        // If there is no defining op, the Value is necessarily a Block argument.
        auto blockArg = operand.cast<mlir::BlockArgument>();
        inputs.push_back(rvsdgRegion.argument(blockArg.getArgNumber()));
      }
    }

    if (rvsdg::node * node = convertOperation(mlirOp, rvsdgRegion, inputs))
    {
      operations[&mlirOp] = node;
    }
  }

  // The results of the region/block are encoded in the terminator operation
  auto terminator = block.getTerminator();
  std::unique_ptr<std::vector<jlm::rvsdg::output *>> results =
      std::make_unique<std::vector<jlm::rvsdg::output *>>();
  for (mlir::Value operand : terminator->getOperands())
  {
    if (mlir::Operation * producer = operand.getDefiningOp())
    {
      size_t index = getOperandIndex(producer, operand);
      results->push_back(operations[producer]->output(index));
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      auto blockArg = operand.cast<mlir::BlockArgument>();
      results->push_back(rvsdgRegion.argument(blockArg.getArgNumber()));
    }
  }

  return results;
}

rvsdg::node *
MlirToRvsdg::convertOperation(
    mlir::Operation & mlirOperation,
    rvsdg::region & rvsdgRegion,
    std::vector<const rvsdg::output *> & inputs)
{
  if (mlirOperation.getName().getStringRef() == mlir::rvsdg::OmegaNode::getOperationName())
  {
    convertOmega(mlirOperation, rvsdgRegion);
    // Omega doesn't have a corresponding RVSDG node so we return nullptr
    return nullptr;
  }
  else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::LambdaNode::getOperationName())
  {
    return convertLambda(mlirOperation, rvsdgRegion);
  }
  else if (mlirOperation.getName().getStringRef() == mlir::arith::ConstantIntOp::getOperationName())
  {
    auto constant = static_cast<mlir::arith::ConstantIntOp>(&mlirOperation);
    auto type = constant.getType();
    JLM_ASSERT(type.getTypeID() == mlir::IntegerType::getTypeID());
    auto * integerType = static_cast<mlir::IntegerType *>(&type);

    return rvsdg::node_output::node(
        rvsdg::create_bitconstant(&rvsdgRegion, integerType->getWidth(), constant.value()));
  }
  else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::LambdaResult::getOperationName())
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
    return nullptr;
  }
  else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::OmegaResult::getOperationName())
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
    return nullptr;
  }
  else
  {
    throw util::error(
        "Operation is not implemented:" + mlirOperation.getName().getStringRef().str() + "\n");
  }
}

void
MlirToRvsdg::convertOmega(mlir::Operation & mlirOmega, rvsdg::region & rvsdgRegion)
{
  // The Omega has a single region
  JLM_ASSERT(mlirOmega.getRegions().size() == 1);
  convertRegion(mlirOmega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::node *
MlirToRvsdg::convertLambda(mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion)
{
  // Get the name of the function
  auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != nullptr);
  auto * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);

  // A lambda node has only the function signature as the result
  JLM_ASSERT(mlirLambda.getNumResults() == 1);
  auto result = mlirLambda.getResult(0).getType();

  if (result.getTypeID() != mlir::rvsdg::LambdaRefType::getTypeID())
  {
    throw util::error("The result from lambda node is not a LambdaRefType\n");
  }

  // Create the RVSDG function signature
  auto * lambdaRefType = static_cast<mlir::rvsdg::LambdaRefType *>(&result);
  std::vector<std::unique_ptr<rvsdg::type>> argumentTypes;
  for (auto argumentType : lambdaRefType->getParameterTypes())
  {
    argumentTypes.push_back(convertType(argumentType));
  }
  std::vector<std::unique_ptr<rvsdg::type>> resultTypes;
  for (auto returnType : lambdaRefType->getReturnTypes())
  {
    resultTypes.push_back(convertType(returnType));
  }
  llvm::FunctionType functionType(std::move(argumentTypes), std::move(resultTypes));

  auto rvsdgLambda = llvm::lambda::node::create(
      &rvsdgRegion,
      functionType,
      functionName->getValue().str(),
      llvm::linkage::external_linkage);

  JLM_ASSERT(mlirLambda.getRegions().size() == 1);
  auto lambdaRegion = rvsdgLambda->subregion();
  auto regionResults = convertRegion(mlirLambda.getRegion(0), *lambdaRegion);

  rvsdgLambda->finalize(*regionResults);

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::type>
MlirToRvsdg::convertType(mlir::Type & type)
{
  if (type.getTypeID() == mlir::IntegerType::getTypeID())
  {
    auto * intType = static_cast<mlir::IntegerType *>(&type);
    return std::make_unique<rvsdg::bittype>(intType->getWidth());
  }
  else if (type.getTypeID() == mlir::rvsdg::LoopStateEdgeType::getTypeID())
  {
    return std::make_unique<llvm::loopstatetype>();
  }
  else if (type.getTypeID() == mlir::rvsdg::MemStateEdgeType::getTypeID())
  {
    return std::make_unique<llvm::MemoryStateType>();
  }
  else if (type.getTypeID() == mlir::rvsdg::IOStateEdgeType::getTypeID())
  {
    return std::make_unique<llvm::iostatetype>();
  }
  else
  {
    throw util::error("Type conversion not implemented\n");
  }
}

size_t
MlirToRvsdg::getOperandIndex(mlir::Operation * producer, mlir::Value & operand)
{
  // TODO
  // Is there a more elegant way of getting the index of the
  // operand of an operation?
  size_t index = 0;
  if (producer->getNumResults() > 1)
  {
    for (mlir::Value tmp : producer->getResults())
    {
      if (tmp == operand)
      {
        break;
      }
      index++;
    }
  }
  return index;
}

} // jlm::mlirrvsdg
