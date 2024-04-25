/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/backend/JlmToMlirConverter.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
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
    nodes[rvsdgNode] = ConvertNode(*rvsdgNode, block, nodes);
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
JlmToMlirConverter::ConvertNode(
    const rvsdg::node & node,
    ::mlir::Block & block,
    std::unordered_map<rvsdg::node *, ::mlir::Value> nodes)
{
  // Create a list of inputs to the MLIR operation
  // TODO try to change to pointers
  ::llvm::SmallVector<::mlir::Value> inputs;
  for (size_t i = 0; i < node.ninputs(); i++)
  {
    if (auto output = dynamic_cast<jlm::rvsdg::simple_output *>(node.input(i)->origin()))
    {
      inputs.push_back(nodes[output->node()]);
    }
    else if (auto arg = dynamic_cast<jlm::rvsdg::argument *>(node.input(i)->origin()))
    {
      inputs.push_back(block.getArgument(arg->index()));
    }
    else
    {
      JLM_UNREACHABLE("Unhandled origin type.");
    }
  }

  if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(&node))
  {
    return ConvertSimpleNode(*simpleNode, block, inputs);
  }
  else if (auto lambda = dynamic_cast<const llvm::lambda::node *>(&node))
  {
    return ConvertLambda(*lambda, block);
  }
  else
  {
    auto message = util::strfmt("Unimplemented structural node: ", node.operation().debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

::mlir::Operation *
JlmToMlirConverter::ConvertBitBinaryNode(
    const jlm::rvsdg::simple_op & bitOp,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  ::mlir::Operation * MlirOp;
  if (jlm::rvsdg::is<const rvsdg::bitadd_op>(bitOp))
  {
    MlirOp = Builder_->create<::mlir::LLVM::AddOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitand_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::AndIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitashr_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::ShRUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitmul_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::MulIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitor_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::OrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitsdiv_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::DivSIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitshl_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::ShLIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitshr_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::ShRUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitsmod_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::RemSIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitsmulh_op>(bitOp))
  {
    JLM_UNREACHABLE("Binary bit bitOp smulh not supported");
  }
  else if (jlm::rvsdg::is<const rvsdg::bitsub_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::SubIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitudiv_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::DivUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitumod_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::RemUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitumulh_op>(bitOp))
  {
    JLM_UNREACHABLE("Binary bit bitOp umulh not supported");
  }
  else if (jlm::rvsdg::is<const rvsdg::bitxor_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::XOrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else
  {
    JLM_UNREACHABLE("Unknown binary bitop");
  }

  return MlirOp;
}

::mlir::Operation *
JlmToMlirConverter::BitCompareNode(
    const jlm::rvsdg::simple_op & bitOp,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  ::mlir::arith::CmpIPredicate compPredicate;
  if (jlm::rvsdg::is<const rvsdg::biteq_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::eq;
  else if (jlm::rvsdg::is<const rvsdg::bitne_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::ne;
  else if (jlm::rvsdg::is<const rvsdg::bitsge_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::sge;
  else if (jlm::rvsdg::is<const rvsdg::bitsgt_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::sgt;
  else if (jlm::rvsdg::is<const rvsdg::bitsle_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::sle;
  else if (jlm::rvsdg::is<const rvsdg::bitslt_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::slt;
  else if (jlm::rvsdg::is<const rvsdg::bituge_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::uge;
  else if (jlm::rvsdg::is<const rvsdg::bitugt_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::ugt;
  else if (jlm::rvsdg::is<const rvsdg::bitule_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::ule;
  else if (jlm::rvsdg::is<const rvsdg::bitult_op>(bitOp))
    compPredicate = ::mlir::arith::CmpIPredicate::ult;
  else
    JLM_UNREACHABLE("Unknown bitcompare operation");

  auto MlirOp = Builder_->create<::mlir::arith::CmpIOp>(
      Builder_->getUnknownLoc(),
      compPredicate,
      inputs[0],
      inputs[1]);
  return MlirOp;
}

::mlir::Value
JlmToMlirConverter::ConvertSimpleNode(
    const rvsdg::simple_node & node,
    ::mlir::Block & block,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  ::mlir::Operation * MlirOp;
  if (auto bitOp = dynamic_cast<const rvsdg::bitconstant_op *>(&(node.operation())))
  {
    auto value = bitOp->value();
    MlirOp = Builder_->create<::mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value.to_uint(),
        value.nbits());
  }
  else if (jlm::rvsdg::is<const rvsdg::bitbinary_op>(node.operation()))
  {
    MlirOp = ConvertBitBinaryNode(node.operation(), inputs);
  }
  else if (jlm::rvsdg::is<const rvsdg::bitcompare_op>(node.operation()))
  {
    MlirOp = BitCompareNode(node.operation(), inputs);
  }
  else if (auto bitOp = dynamic_cast<const jlm::llvm::zext_op *>(&(node.operation())))
  {
    MlirOp = Builder_->create<::mlir::arith::ExtUIOp>(
        Builder_->getUnknownLoc(),
        Builder_->getIntegerType(bitOp->ndstbits()),
        inputs[0]);
  }
  else
  {
    auto message = util::strfmt("Unimplemented simple node: ", node.operation().debug_string());
    JLM_UNREACHABLE(message.c_str());
  }

  block.push_back(MlirOp);
  return ::mlir::Value(MlirOp->getResult(0));
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
    auto message = util::strfmt("Type conversion not implemented: ", type.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

} // namespace jlm::mlir
