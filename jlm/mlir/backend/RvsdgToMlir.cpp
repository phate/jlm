/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/mlir/backend/RvsdgToMlir.hpp"

#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/Verifier.h"

namespace jlm::rvsdgmlir
{

void
RvsdgToMlir::print(mlir::rvsdg::OmegaNode & omega, const util::filepath & filePath)
{
  if (failed(mlir::verify(omega)))
  {
    omega.emitError("module verification error");
    throw util::error("Verification of RVSDG-MLIR failed");
  }
  if (filePath == "")
  {
    ::llvm::raw_os_ostream os(std::cout);
    omega.print(os);
  }
  else
  {
    std::error_code ec;
    ::llvm::raw_fd_ostream os(filePath.to_str(), ec);
    omega.print(os);
  }
}

mlir::rvsdg::OmegaNode
RvsdgToMlir::convertModule(const llvm::RvsdgModule & rvsdgModule)
{
  auto & graph = rvsdgModule.Rvsdg();
  return convertOmega(graph);
}

mlir::rvsdg::OmegaNode
RvsdgToMlir::convertOmega(const rvsdg::graph & graph)
{
  auto omega = Builder_->create<mlir::rvsdg::OmegaNode>(Builder_->getUnknownLoc());
  mlir::Region & region = omega.getRegion();
  auto & omegaBlock = region.emplaceBlock();

  auto subregion = graph.root();
  ::llvm::SmallVector<mlir::Value> regionResults = convertRegion(*subregion, omegaBlock);

  auto omegaResult =
      Builder_->create<mlir::rvsdg::OmegaResult>(Builder_->getUnknownLoc(), regionResults);
  omegaBlock.push_back(omegaResult);

  return omega;
}

::llvm::SmallVector<mlir::Value>
RvsdgToMlir::convertRegion(rvsdg::region & region, mlir::Block & block)
{
  for (size_t i = 0; i < region.narguments(); ++i)
  {
    auto type = convertType(region.argument(i)->type());
    block.addArgument(type, Builder_->getUnknownLoc());
  }

  // Create an MLIR operation for each RVSDG node and store each pair in a
  // hash map for easy lookup of corresponding MLIR operation
  std::unordered_map<rvsdg::node *, mlir::Value> nodes;
  for (rvsdg::node * rvsdgNode : rvsdg::topdown_traverser(&region))
  {
    // TODO
    // Get the inputs of the node
    // for (size_t i=0; i < rvsdgNode->ninputs(); i++)
    //{
    //  ::llvm::outs() << rvsdgNode->input(i) << "\n";
    //}
    nodes[rvsdgNode] = convertNode(*rvsdgNode, block);
  }

  ::llvm::SmallVector<mlir::Value> results;
  for (size_t i = 0; i < region.nresults(); ++i)
  {
    auto result = region.result(i);
    auto output = result->origin();
    rvsdg::node * outputNode = rvsdg::node_output::node(output);
    if (outputNode == nullptr)
    {
      // The result is connected directly to an argument
      results.push_back(block.getArgument(output->index()));
    }
    else
    {
      // The identified node should always exist in the hash map of nodes
      JLM_ASSERT(nodes.find(outputNode) != nodes.end());
      results.push_back(nodes[outputNode]);
    }
  }
  return results;
}

mlir::Value
RvsdgToMlir::convertNode(const rvsdg::node & node, mlir::Block & block)
{
  if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(&node))
  {
    return convertSimpleNode(*simpleNode, block);
  }
  else if (auto lambda = dynamic_cast<const llvm::lambda::node *>(&node))
  {
    return convertLambda(*lambda, block);
  }
  else
  {
    throw util::error("Unimplemented structural node: " + node.operation().debug_string());
  }
}

mlir::Value
RvsdgToMlir::convertSimpleNode(const rvsdg::simple_node & node, mlir::Block & block)
{
  if (auto bitsOp = dynamic_cast<const rvsdg::bitconstant_op *>(&(node.operation())))
  {
    auto value = bitsOp->value();
    auto constOp = Builder_->create<mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value.to_uint(),
        value.nbits());
    block.push_back(constOp);

    return constOp;
  }
  else
  {
    throw util::error("Unimplemented simple node: " + node.operation().debug_string());
  }
}

mlir::Value
RvsdgToMlir::convertLambda(const llvm::lambda::node & lambdaNode, mlir::Block & block)
{
  ::llvm::SmallVector<mlir::Type> arguments;
  for (size_t i = 0; i < lambdaNode.nfctarguments(); ++i)
  {
    arguments.push_back(convertType(lambdaNode.fctargument(i)->type()));
  }
  ::llvm::ArrayRef argumentsArray(arguments);

  ::llvm::SmallVector<mlir::Type> results;
  for (size_t i = 0; i < lambdaNode.nfctresults(); ++i)
  {
    results.push_back(convertType(lambdaNode.fctresult(i)->type()));
  }
  ::llvm::ArrayRef resultsArray(results);

  ::llvm::SmallVector<mlir::Type> lambdaRef;
  auto refType = Builder_->getType<::mlir::rvsdg::LambdaRefType>(argumentsArray, resultsArray);
  lambdaRef.push_back(refType);

  ::llvm::SmallVector<mlir::Value> inputs;

  // Add function attributes, e.g., the function name
  ::llvm::SmallVector<mlir::NamedAttribute> attributes;
  auto attributeName = Builder_->getStringAttr("sym_name");
  auto attributeValue = Builder_->getStringAttr(lambdaNode.name());
  auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
  attributes.push_back(symbolName);
  ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

  auto lambda = Builder_->create<mlir::rvsdg::LambdaNode>(
      Builder_->getUnknownLoc(),
      lambdaRef,
      inputs,
      attributesRef);
  block.push_back(lambda);

  mlir::Region & region = lambda.getRegion();
  auto & lambdaBlock = region.emplaceBlock();

  auto regionResults = convertRegion(*lambdaNode.subregion(), lambdaBlock);
  auto lambdaResult =
      Builder_->create<mlir::rvsdg::LambdaResult>(Builder_->getUnknownLoc(), regionResults);
  lambdaBlock.push_back(lambdaResult);

  return lambda;
}

mlir::Type
RvsdgToMlir::convertType(const rvsdg::type & type)
{
  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&type))
  {
    return Builder_->getIntegerType(bt->nbits());
  }
  else if (dynamic_cast<const llvm::loopstatetype *>(&type))
  {
    return Builder_->getType<::mlir::rvsdg::LoopStateEdgeType>();
  }
  else if (dynamic_cast<const llvm::iostatetype *>(&type))
  {
    return Builder_->getType<::mlir::rvsdg::IOStateEdgeType>();
  }
  else if (dynamic_cast<const llvm::MemoryStateType *>(&type))
  {
    return Builder_->getType<::mlir::rvsdg::MemStateEdgeType>();
  }
  else
  {
    throw util::error("Type conversion not implemented: " + type.debug_string());
  }
}

} // namespace jlm::rvsdgmlir
