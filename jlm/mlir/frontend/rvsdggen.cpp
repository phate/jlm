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

void
RVSDGGen::convertBlock(mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  for (mlir::BlockArgument & arg : block.getArguments())
  {
    ::llvm::outs() << "Block argument: " << arg << "\n";
  }

  // Transform the block such that operations are in topological order
  mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<mlir::Operation *, rvsdg::node *> operations;
  for (mlir::Operation & mlirOp : block.getOperations())
  {
    ::llvm::outs() << "- Current operation " << mlirOp.getName() << "\n";

    // Get the inputs of the MLIR operation
    std::vector<const jlm::rvsdg::output *> inputs;
    for (mlir::Value operand : mlirOp.getOperands())
    {
      if (mlir::Operation * producer = operand.getDefiningOp())
      {
        ::llvm::outs() << "  - Operand produced by operation '" << producer->getName() << "'\n";
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
        ::llvm::outs() << "  - Operand produced by Block argument, number "
                       << blockArg.getArgNumber() << "\n";
        inputs.push_back(rvsdgRegion.argument(blockArg.getArgNumber()));
      }
    }

    if (mlirOp.getName().getStringRef() == mlir::rvsdg::OmegaNode::getOperationName())
    {
      ::llvm::outs() << "- Current operation " << mlirOp.getName() << "\n";
      convertOmega(mlirOp, rvsdgRegion);
    }
    else if (mlirOp.getName().getStringRef() == mlir::rvsdg::LambdaNode::getOperationName())
    {
      operations[&mlirOp] = convertLambda(mlirOp, rvsdgRegion);
    }
    else if (mlirOp.getName().getStringRef() == mlir::arith::ConstantIntOp::getOperationName())
    {
      auto constant = static_cast<mlir::arith::ConstantIntOp>(&mlirOp);

      // Need the type to know the width of the constant
      auto type = constant.getType();
      JLM_ASSERT(type.getTypeID() == mlir::IntegerType::getTypeID());
      auto * integerType = static_cast<mlir::IntegerType *>(&type);

      operations[&mlirOp] = rvsdg::node_output::node(
          rvsdg::create_bitconstant(&rvsdgRegion, integerType->getWidth(), constant.value()));
    }
    else if (mlirOp.getName().getStringRef() == mlir::rvsdg::LambdaResult::getOperationName())
    {
      ::llvm::outs() << "- This is a terminating operation, so do nothing: " << mlirOp.getName()
                     << "\n";
    }
    else if (mlirOp.getName().getStringRef() == mlir::rvsdg::OmegaResult::getOperationName())
    {
      ::llvm::outs() << "- This is a terminating operation, so do nothing: " << mlirOp.getName()
                     << "\n";
    }
    else
    {
      throw util::error(
          "Operation is not implemented:" + mlirOp.getName().getStringRef().str() + "\n");
    }
  }

  // All nodes in the region have been created, so all that's left to do is to connect
  // the results of the regions to their producers.

  auto terminator = block.getTerminator();
  ::llvm::outs() << "Terminator: " << terminator->getName();
  ::llvm::outs() << " with " << terminator->getOperands().size() << " operands\n";
  ::llvm::outs() << "  # RVSDG region with " << rvsdgRegion.nresults() << " arguments\n";
  ::llvm::outs() << "  # RVSDG region with " << rvsdgRegion.nresults() << " results\n";
  //  ::llvm::outs() << "  # RVSDG region with " << rvsdgRegion.node()->noutputs() << " outputs\n";
  //  ::llvm::outs() << "  # RVSDG region with " << rvsdgRegion.node()->ninputs() << " inputs\n";
  // Print information about the producer of each of the operands.
  for (mlir::Value operand : terminator->getOperands())
  {
    if (mlir::Operation * producer = operand.getDefiningOp())
    {
      //      rvsdgRegion.append_result(operations[producer]->output(0));
      ::llvm::outs() << "  - Operand produced by operation '" << producer->getName() << "'\n";
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      auto blockArg = operand.cast<mlir::BlockArgument>();
      //    rvsdgRegion.append_result(rvsdgRegion.argument(blockArg.getArgNumber()));
      ::llvm::outs() << "  - Operand produced by Block argument, number " << blockArg.getArgNumber()
                     << "\n";
    }
  }
}

void
RVSDGGen::convertRegion(mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  ::llvm::outs() << "  - Converting region\n";
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  convertBlock(region.front(), rvsdgRegion);
}

void
RVSDGGen::convertOmega(mlir::Operation & omega, rvsdg::region & rvsdgRegion)
{
  ::llvm::outs() << "  ** Converting Omega **\n";
  // The Omega consists of a single region
  JLM_ASSERT(omega.getRegions().size() == 1);
  convertRegion(omega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::node *
RVSDGGen::convertLambda(mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion)
{
  ::llvm::outs() << "  ** Converting Lambda **\n";

  // Get the name of the function
  auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != NULL);
  mlir::StringAttr * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);
  ::llvm::outs() << "Function name: " << functionName->getValue().str() << "\n";

  // A lambda node has only the function signature as the result
  JLM_ASSERT(mlirLambda.getNumResults() == 1);

  // Get the MLIR function signature
  auto result = mlirLambda.getResult(0).getType();
  ::llvm::outs() << "  - Function signature: '" << result << "'\n";

  if (result.getTypeID() != mlir::rvsdg::LambdaRefType::getTypeID())
  {
    throw util::error("The result from lambda node is not a LambdaRefType\n");
  }

  // Creat the RVSDG function signature
  std::vector<const jlm::rvsdg::type *> arguments;
  std::vector<const jlm::rvsdg::type *> results;
  mlir::rvsdg::LambdaRefType * lambdaRefType = static_cast<mlir::rvsdg::LambdaRefType *>(&result);
  for (auto argumentType : lambdaRefType->getParameterTypes())
  {
    auto argument = convertType(argumentType);
    ::llvm::outs() << "  - Argument: '" << argument->debug_string() << "\n";
    arguments.push_back(argument);
  }
  for (auto returnType : lambdaRefType->getReturnTypes())
  {
    auto result = convertType(returnType);
    ::llvm::outs() << "  - Result: '" << result->debug_string() << "\n";
    results.push_back(result);
  }
  llvm::FunctionType functionType(arguments, results);

  auto rvsdgLambda = llvm::lambda::node::create(
      &rvsdgRegion,
      functionType,
      functionName->getValue().str(),
      llvm::linkage::external_linkage);

  // Add all the inputs, i.e., function arguments of the lambda,
  ::llvm::outs() << "  - Function arguments: '" << rvsdgLambda->nfctarguments() << "\n";
  std::vector<const jlm::rvsdg::argument *> functionArguments;
  for (size_t i = 0; i < rvsdgLambda->nfctarguments(); i++)
  {
    functionArguments.push_back(rvsdgLambda->fctargument(i));
  }

  // Add all the outputs of the lambda,
  // which have the same types as the results
  std::vector<const jlm::rvsdg::structural_output *> outputs;
  for (auto outputType : results)
  {
    rvsdg::port port(*outputType);
    outputs.push_back(rvsdg::structural_output::create(rvsdgLambda, port));
  }

  // Get the region and convert it
  JLM_ASSERT(mlirLambda.getRegions().size() == 1);
  auto lambdaRegion = rvsdgLambda->subregion();
  convertRegion(mlirLambda.getRegion(0), *lambdaRegion);

  return rvsdgLambda;

  /*
    FunctionType functionType({&vt}, {&vt});

    auto lambda = lambda::node::create(rvsdgRegion, functionType, "f", linkage::external_linkage);
    lambda->finalize({lambda->fctargument(0)});

    std::vector<jlm::rvsdg::argument*> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);
  */
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
