/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifdef MLIR_ENABLED

#include "jlm/mlir/frontend/rvsdggen.hpp"

#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

// #include <jlm/llvm/ir/operators/GetElementPtr.hpp>
// #include <jlm/llvm/ir/operators/load.hpp>
// #include <jlm/llvm/ir/operators/operators.hpp>
// #include <jlm/llvm/ir/operators/sext.hpp>
// #include <jlm/llvm/ir/operators/store.hpp>
// #include <jlm/rvsdg/bitstring/type.hpp>

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

namespace jlm::mlirrvsdg
{

std::unique_ptr<mlir::Block>
RVSDGGen::readRvsdgMlir(const util::filepath & filePath)
{
  // Configer the parser
  mlir::ParserConfig config = mlir::ParserConfig(Context_.get());
  // Variable for storing the result
  std::unique_ptr<mlir::Block> block = std::make_unique<mlir::Block>();
  // Read the input file
  auto result = mlir::parseSourceFile(filePath.to_str(), block.get(), config);
  if (result.failed())
  {
    throw util::error("Parsing MLIR input file failed.");
  }
  return block;
}

std::unique_ptr<llvm::RvsdgModule>
RVSDGGen::convertMlir(std::unique_ptr<mlir::Block> & block)
{
  // Create RVSDG module
  std::string dataLayout;
  std::string targetTriple;
  util::filepath sourceFileName("");
  auto rvsdgModule = llvm::RvsdgModule::Create(sourceFileName, targetTriple, dataLayout);

  // Get the root region
  auto & graph = rvsdgModule->Rvsdg();
  auto root = graph.root();

  // Convert the MLIR into an RVSDG graph
  convertBlock(*block.get(), *root);

  return rvsdgModule;
}

std::unique_ptr<std::vector<jlm::rvsdg::output *>>
RVSDGGen::convertRegion(mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return convertBlock(region.front(), rvsdgRegion);
}

std::unique_ptr<std::vector<jlm::rvsdg::output *>>
RVSDGGen::convertBlock(mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  // Transform the block such that operations are in topological order
  mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<mlir::Operation *, rvsdg::node *> operations;
  for (mlir::Operation & mlirOp : block.getOperations())
  {
    // Get the inputs of the MLIR operation
    std::vector<const jlm::rvsdg::output *> inputs;
    for (mlir::Value operand : mlirOp.getOperands())
    {
      if (mlir::Operation * producer = operand.getDefiningOp())
      {
        // TODO
        // Is there a more elegant way of getting the index of the
        // result that is the operand of the current operation?
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

  // Get all the results of the region
  auto terminator = block.getTerminator();
  std::unique_ptr<std::vector<jlm::rvsdg::output *>> results =
      std::make_unique<std::vector<jlm::rvsdg::output *>>();
  for (mlir::Value operand : terminator->getOperands())
  {
    if (mlir::Operation * producer = operand.getDefiningOp())
    {
      results->push_back(operations[producer]->output(0));
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
RVSDGGen::convertOperation(mlir::Operation & mlirOperation, rvsdg::region & rvsdgRegion, std::vector<const rvsdg::output *> & inputs)
{
    if (mlirOperation.getName().getStringRef() == mlir::rvsdg::OmegaNode::getOperationName())
    {
      convertOmega(mlirOperation, rvsdgRegion);
      // Omega doesn't have a corresponding RVSDG node so we return NULL
      return NULL;
    }
    else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::LambdaNode::getOperationName())
    {
      return convertLambda(mlirOperation, rvsdgRegion);
    }
    else if (mlirOperation.getName().getStringRef() == mlir::arith::ConstantIntOp::getOperationName())
    {
      auto constant = static_cast<mlir::arith::ConstantIntOp>(&mlirOperation);

      // Need the type to know the width of the constant
      auto type = constant.getType();
      JLM_ASSERT(type.getTypeID() == mlir::IntegerType::getTypeID());
      auto * integerType = static_cast<mlir::IntegerType *>(&type);

      return rvsdg::node_output::node(
          rvsdg::create_bitconstant(&rvsdgRegion, integerType->getWidth(), constant.value()));
    }
    else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::LambdaResult::getOperationName())
    {
      // This is a terminating operation, so do nothing
      return NULL;
    }
    else if (mlirOperation.getName().getStringRef() == mlir::rvsdg::OmegaResult::getOperationName())
    {
      // This is a terminating operation, so do nothing
      return NULL;
    }
    else
    {
      throw util::error(
          "Operation is not implemented:" + mlirOperation.getName().getStringRef().str() + "\n");
    } 
}

void
RVSDGGen::convertOmega(mlir::Operation & mlirOmega, rvsdg::region & rvsdgRegion)
{
  // The Omega consists of a single region
  JLM_ASSERT(mlirOmega.getRegions().size() == 1);
  convertRegion(mlirOmega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::node *
RVSDGGen::convertLambda(mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion)
{
  // Get the name of the function
  auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != NULL);
  mlir::StringAttr * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);

  // A lambda node has only the function signature as the result
  JLM_ASSERT(mlirLambda.getNumResults() == 1);

  // Get the MLIR function signature
  auto result = mlirLambda.getResult(0).getType();

  if (result.getTypeID() != mlir::rvsdg::LambdaRefType::getTypeID())
  {
    throw util::error("The result from lambda node is not a LambdaRefType\n");
  }

  // Creat the RVSDG function signature
  mlir::rvsdg::LambdaRefType * lambdaRefType = static_cast<mlir::rvsdg::LambdaRefType *>(&result);
  std::vector<const jlm::rvsdg::type *> arguments;
  for (auto argumentType : lambdaRefType->getParameterTypes())
  {
    auto argument = convertType(argumentType);
    arguments.push_back(argument);
  }
  std::vector<const jlm::rvsdg::type *> results;
  for (auto returnType : lambdaRefType->getReturnTypes())
  {
    auto result = convertType(returnType);
    results.push_back(result);
  }
  llvm::FunctionType functionType(arguments, results);

  auto rvsdgLambda = llvm::lambda::node::create(
      &rvsdgRegion,
      functionType,
      functionName->getValue().str(),
      llvm::linkage::external_linkage);

  // Get the region and convert it
  JLM_ASSERT(mlirLambda.getRegions().size() == 1);
  auto lambdaRegion = rvsdgLambda->subregion();
  auto regionResults = convertRegion(mlirLambda.getRegion(0), *lambdaRegion);

  rvsdgLambda->finalize(*regionResults);

  return rvsdgLambda;
}

rvsdg::type *
RVSDGGen::convertType(mlir::Type & type)
{
  // TODO
  // Fix memory leak
  if (type.getTypeID() == mlir::IntegerType::getTypeID())
  {
    auto * intType = static_cast<mlir::IntegerType *>(&type);
    return new rvsdg::bittype(intType->getWidth());
  }
  else if (type.getTypeID() == mlir::rvsdg::LoopStateEdgeType::getTypeID())
  {
    return new llvm::loopstatetype();
  }
  else if (type.getTypeID() == mlir::rvsdg::MemStateEdgeType::getTypeID())
  {
    return new llvm::MemoryStateType();
  }
  else if (type.getTypeID() == mlir::rvsdg::IOStateEdgeType::getTypeID())
  {
    return new llvm::iostatetype();
  }
  else
  {
    throw util::error("Type conversion not implemented\n");
  }
}

} // jlm::mlirrvsdg

#endif // MLIR_ENABLED
