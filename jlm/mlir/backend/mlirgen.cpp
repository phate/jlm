/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/mlir/backend/mlirgen.hpp"

#ifdef MLIR_ENABLED

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

namespace jlm::rvsdgmlir
{

void
MLIRGen::print(mlir::rvsdg::OmegaNode & omega, const util::filepath & filePath)
{
  // Verify the module
  if (failed(mlir::verify(omega)))
  {
    omega.emitError("module verification error");
    throw util::error("Verification of RVSDG-MLIR failed");
  }
  // Print the module
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
MLIRGen::convertModule(const llvm::RvsdgModule & rvsdgModule)
{
  auto & graph = rvsdgModule.Rvsdg();

  // TODO
  // Add assert checking that the graph consists of a single omega node

  return convertOmega(graph);
}

mlir::rvsdg::OmegaNode
MLIRGen::convertOmega(const rvsdg::graph & graph)
{
  // Create the MLIR omega node
  mlir::rvsdg::OmegaNode omega =
      Builder_->create<mlir::rvsdg::OmegaNode>(Builder_->getUnknownLoc());

  // Create a block for the region as this is currently not done automatically
  mlir::Region & region = omega.getRegion();
  mlir::Block * omegaBlock = new mlir::Block;
  region.push_back(omegaBlock);

  // Convert the region of the omega
  auto subregion = graph.root();
  ::llvm::SmallVector<mlir::Value> regionResults = convertSubregion(*subregion, *omegaBlock);

  // Handle the result of the omega
  auto omegaResult =
      Builder_->create<mlir::rvsdg::OmegaResult>(Builder_->getUnknownLoc(), regionResults);
  omegaBlock->push_back(omegaResult);

  return omega;
}

::llvm::SmallVector<mlir::Value>
MLIRGen::convertSubregion(rvsdg::region & region, mlir::Block & block)
{
  // Handle arguments of the region
  for (size_t i = 0; i < region.narguments(); ++i)
  {
    auto type = convertType(region.argument(i)->type());
    block.addArgument(type, Builder_->getUnknownLoc());
  }

  // Create an MLIR node for each RVSDG node and store each pair in a
  // hash map for easy lookup of corresponding MLIR nodes
  std::unordered_map<rvsdg::node *, mlir::Value> nodes;
  for (rvsdg::node * rvsdgNode : rvsdg::topdown_traverser(&region))
  {
    nodes[rvsdgNode] = convertNode(*rvsdgNode, block);
  }

  // Handle results of the region
  ::llvm::SmallVector<mlir::Value> results;
  for (size_t i = 0; i < region.nresults(); ++i)
  {
    // Get the result of the RVSDG region
    auto result = region.result(i);
    // Get the output of the RVSDG node driving the result
    auto output = result->origin();
    // Get the RVSDG node that generates the output
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
MLIRGen::convertNode(const rvsdg::node & node, mlir::Block & block)
{
  if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(&node))
  {
    return convertSimpleNode(*simpleNode, block);
  }
  else if (auto lambda = dynamic_cast<const llvm::lambda::node *>(&node))
  {
    return convertLambda(*lambda, block);
    /*
    } else if (auto gamma = dynamic_cast<llvm::gamma_node *>(&node)) {
      convertGamma(*gamma, block);
    }  else if (auto theta = dynamic_cast<llvm::theta_node *>(&node)) {
      convertTheta(*theta, block);
    } else if (auto delta = dynamic_cast<llvm::delta::node *>(&node)) {
      convertDelta(*delta, block);
    } else if (auto phi = dynamic_cast<llvm::phi::node *>(&node)) {
      convertPhi(*phi, block);
    */
  }
  else
  {
    throw util::error("Unimplemented structural node: " + node.operation().debug_string());
  }
}

mlir::Value
MLIRGen::convertSimpleNode(const rvsdg::simple_node & node, mlir::Block & block)
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
MLIRGen::convertLambda(const llvm::lambda::node & lambdaNode, mlir::Block & block)
{

  // Handle function arguments
  ::llvm::SmallVector<mlir::Type> arguments;
  for (size_t i = 0; i < lambdaNode.nfctarguments(); ++i)
  {
    arguments.push_back(convertType(lambdaNode.fctargument(i)->type()));
  }
  ::llvm::ArrayRef argumentsArray(arguments);

  // Handle function results
  ::llvm::SmallVector<mlir::Type> results;
  for (size_t i = 0; i < lambdaNode.nfctresults(); ++i)
  {
    results.push_back(convertType(lambdaNode.fctresult(i)->type()));
  }
  ::llvm::ArrayRef resultsArray(results);

  /*
    // Context arguments
    for (size_t i = 0; i < node->ncvarguments(); ++i) {
      // s << print_input_origin(node.cvargument(i)->input()) << ": " <<
    print_type(&ln.cvargument(i)->type()); throw util::error("Context arguments in convertLambda()
    has not been implemented");
    }
  */
  // TODO
  // Consider replacing the lambda ref creation with
  // mlir::rvsdg::LambdaRefTyp::get();
  // static LambdaRefType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<mlir::Type>
  // parameterTypes, ::llvm::ArrayRef<mlir::Type> returnTypes);

  // LambdaNodes return a LambdaRefType
  ::llvm::SmallVector<mlir::Type> lambdaRef;
  auto refType = Builder_->getType<::mlir::rvsdg::LambdaRefType>(argumentsArray, resultsArray);
  lambdaRef.push_back(refType);

  // TODO
  // Add the inputs to the function
  ::llvm::SmallVector<mlir::Value> inputs;

  // Add function attributes
  ::llvm::SmallVector<mlir::NamedAttribute> attributes;
  auto attributeName = Builder_->getStringAttr("sym_name");
  auto attributeValue = Builder_->getStringAttr(lambdaNode.name());
  auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
  attributes.push_back(symbolName);
  ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

  // Create the lambda node and add it to the region/block it resides in
  auto lambda = Builder_->create<mlir::rvsdg::LambdaNode>(
      Builder_->getUnknownLoc(),
      lambdaRef,
      inputs,
      attributesRef);
  block.push_back(lambda);

  // Create a block for the region as this is not done automatically
  mlir::Region & region = lambda.getRegion();
  mlir::Block * lambdaBlock = new mlir::Block;
  region.push_back(lambdaBlock);

  // Convert the region and get all the results generated by the region
  auto regionResults = convertSubregion(*lambdaNode.subregion(), *lambdaBlock);
  auto lambdaResult =
      Builder_->create<mlir::rvsdg::LambdaResult>(Builder_->getUnknownLoc(), regionResults);
  lambdaBlock->push_back(lambdaResult);

  return lambda;
}

mlir::Type
MLIRGen::convertType(const rvsdg::type & type)
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
    /*
    } else if (auto varargType =dynamic_cast<const jlm::varargtype*>(&type)) {
      s << "!jlm.varargList";
    } else if (auto pointerType = dynamic_cast<const jlm::PointerType*>(&type)){
      s << print_pointer_type(pointer_type);
    } else if (auto arrayType = dynamic_cast<const jlm::arraytype*>(&type)){
      s << print_array_type(array_type);
    } else if (auto structType = dynamic_cast<const jlm::structtype*>(&type)){
      s << print_struct_type(struct_type);
    } else if (auto controlType = dynamic_cast<const jive::ctltype*>(&type)){
      s << "!rvsdg.ctrl<" << control_type->nalternatives() << ">";
    */
  }
  else
  {
    throw util::error("Type conversion not implemented: " + type.debug_string());
  }
}

} // namespace jlm::rvsdgmlir

#endif // MLIR_ENABLED
