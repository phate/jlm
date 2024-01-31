/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/backend/JlmToMlirConverter.hpp>

#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/IR/Verifier.h>

namespace jlm::mlir
{

void
JlmToMlirConverter::Print(::mlir::rvsdg::OmegaNode & omega, const util::filepath & filePath)
{
  if (failed(::mlir::verify(omega)))
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

::mlir::rvsdg::OmegaNode
JlmToMlirConverter::ConvertModule(const llvm::RvsdgModule & rvsdgModule)
{
  return ConvertOmega(rvsdgModule.Rvsdg());
}

::mlir::rvsdg::OmegaNode
JlmToMlirConverter::ConvertOmega(const rvsdg::graph & graph)
{
  auto omega = Builder_->create<::mlir::rvsdg::OmegaNode>(Builder_->getUnknownLoc());
  auto & omegaBlock = omega.getRegion().emplaceBlock();

  ::llvm::SmallVector<::mlir::Value> regionResults = ConvertRegion(*graph.root(), omegaBlock);

  auto omegaResult =
      Builder_->create<::mlir::rvsdg::OmegaResult>(Builder_->getUnknownLoc(), regionResults);
  omegaBlock.push_back(omegaResult);

  return omega;
}

::llvm::SmallVector<::mlir::Value>
JlmToMlirConverter::ConvertRegion(rvsdg::region & region, ::mlir::Block & block)
{
  for (size_t i = 0; i < region.narguments(); ++i)
  {
    auto type = ConvertType(region.argument(i)->type());
    block.addArgument(type, Builder_->getUnknownLoc());
  }

  // Create an MLIR operation for each RVSDG node and store each pair in a
  // hash map for easy lookup of corresponding MLIR operation
  std::unordered_map<rvsdg::node *, ::mlir::Value> nodes;
  for (rvsdg::node * rvsdgNode : rvsdg::topdown_traverser(&region))
  {
    // TODO
    // Get the inputs of the node
    // for (size_t i=0; i < rvsdgNode->ninputs(); i++)
    //{
    //  ::llvm::outs() << rvsdgNode->input(i) << "\n";
    //}
    nodes[rvsdgNode] = ConvertNode(*rvsdgNode, block);
  }

  ::llvm::SmallVector<::mlir::Value> results;
  for (size_t i = 0; i < region.nresults(); ++i)
  {
    auto output = region.result(i)->origin();
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

::mlir::Value
JlmToMlirConverter::ConvertNode(const rvsdg::node & node, ::mlir::Block & block)
{
  if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(&node))
  {
    return ConvertSimpleNode(*simpleNode, block);
  }
  else if (auto lambda = dynamic_cast<const llvm::lambda::node *>(&node))
  {
    return ConvertLambda(*lambda, block);
  }
  else
  {
    JLM_UNREACHABLE(
        util::strfmt("Unimplemented structural node: " + node.operation().debug_string()).c_str());
  }
}

::mlir::Value
JlmToMlirConverter::ConvertSimpleNode(const rvsdg::simple_node & node, ::mlir::Block & block)
{
  if (auto bitsOp = dynamic_cast<const rvsdg::bitconstant_op *>(&(node.operation())))
  {
    auto value = bitsOp->value();
    auto constOp = Builder_->create<::mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value.to_uint(),
        value.nbits());
    block.push_back(constOp);

    return constOp;
  }
  else
  {
    JLM_UNREACHABLE(
        util::strfmt("Unimplemented simple node: " + node.operation().debug_string()).c_str());
  }
}

::mlir::Value
JlmToMlirConverter::ConvertLambda(const llvm::lambda::node & lambdaNode, ::mlir::Block & block)
{
  ::llvm::SmallVector<::mlir::Type> arguments;
  for (size_t i = 0; i < lambdaNode.nfctarguments(); ++i)
  {
    arguments.push_back(ConvertType(lambdaNode.fctargument(i)->type()));
  }

  ::llvm::SmallVector<::mlir::Type> results;
  for (size_t i = 0; i < lambdaNode.nfctresults(); ++i)
  {
    results.push_back(ConvertType(lambdaNode.fctresult(i)->type()));
  }

  ::llvm::SmallVector<::mlir::Type> lambdaRef;
  auto refType = Builder_->getType<::mlir::rvsdg::LambdaRefType>(
      ::llvm::ArrayRef(arguments),
      ::llvm::ArrayRef(results));
  lambdaRef.push_back(refType);

  ::llvm::SmallVector<::mlir::Value> inputs;
  // TODO
  // Populate the inputs

  // Add function attributes, e.g., the function name and linkage
  ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
  auto symbolName = Builder_->getNamedAttr(
      Builder_->getStringAttr("sym_name"),
      Builder_->getStringAttr(lambdaNode.name()));
  attributes.push_back(symbolName);
  auto linkage = Builder_->getNamedAttr(
      Builder_->getStringAttr("linkage"),
      Builder_->getStringAttr(llvm::ToString(lambdaNode.linkage())));
  attributes.push_back(linkage);

  auto lambda = Builder_->create<::mlir::rvsdg::LambdaNode>(
      Builder_->getUnknownLoc(),
      lambdaRef,
      inputs,
      ::llvm::ArrayRef<::mlir::NamedAttribute>(attributes));
  block.push_back(lambda);

  auto & lambdaBlock = lambda.getRegion().emplaceBlock();
  auto regionResults = ConvertRegion(*lambdaNode.subregion(), lambdaBlock);
  auto lambdaResult =
      Builder_->create<::mlir::rvsdg::LambdaResult>(Builder_->getUnknownLoc(), regionResults);
  lambdaBlock.push_back(lambdaResult);

  return lambda;
}

::mlir::Type
JlmToMlirConverter::ConvertType(const rvsdg::type & type)
{
  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&type))
  {
    return Builder_->getIntegerType(bt->nbits());
  }
  else if (rvsdg::is<llvm::loopstatetype>(type))
  {
    return Builder_->getType<::mlir::rvsdg::LoopStateEdgeType>();
  }
  else if (rvsdg::is<llvm::iostatetype>(type))
  {
    return Builder_->getType<::mlir::rvsdg::IOStateEdgeType>();
  }
  else if (rvsdg::is<llvm::MemoryStateType>(type))
  {
    return Builder_->getType<::mlir::rvsdg::MemStateEdgeType>();
  }
  else
  {
    JLM_UNREACHABLE(
        util::strfmt("Type conversion not implemented: " + type.debug_string()).c_str());
  }
}

} // namespace jlm::mlir
