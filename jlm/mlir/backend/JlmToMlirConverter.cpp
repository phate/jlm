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
  std::unordered_map<rvsdg::node *, ::mlir::Operation *> operationsMap;
  for (rvsdg::node * rvsdgNode : rvsdg::topdown_traverser(&region))
  {
    ::llvm::SmallVector<::mlir::Value> inputs =
        GetConvertedInputs(*rvsdgNode, operationsMap, block);

    operationsMap[rvsdgNode] = ConvertNode(*rvsdgNode, block, inputs);
  }

  // This code is used to get the results of the region
  //! It is similar to the GetConvertedInputs function
  ::llvm::SmallVector<::mlir::Value> results;
  for (size_t i = 0; i < region.nresults(); i++)
  {
    if (jlm::rvsdg::node_output * nodeOuput =
            dynamic_cast<jlm::rvsdg::node_output *>(region.result(i)->origin()))
    {
      results.push_back(operationsMap.at(nodeOuput->node())->getResult(nodeOuput->index()));
    }
    else if (auto arg = dynamic_cast<jlm::rvsdg::argument *>(region.result(i)->origin()))
    {
      results.push_back(block.getArgument(arg->index()));
    }
    else
    {
      auto message = util::strfmt(
          "Unimplemented input type: ",
          region.result(i)->origin()->debug_string(),
          ": ",
          region.result(i)->origin()->type().debug_string(),
          " for region result: ",
          region.result(i)->debug_string(),
          " at index: ",
          i);
      JLM_UNREACHABLE(message.c_str());
    }
  }

  return results;
}

::llvm::SmallVector<::mlir::Value>
JlmToMlirConverter::GetConvertedInputs(
    const rvsdg::node & node,
    const std::unordered_map<rvsdg::node *, ::mlir::Operation *> & operationsMap,
    ::mlir::Block & block)
{
  ::llvm::SmallVector<::mlir::Value> inputs;
  for (size_t i = 0; i < node.ninputs(); i++)
  {
    if (auto nodeOuput = dynamic_cast<jlm::rvsdg::node_output *>(node.input(i)->origin()))
    {
      inputs.push_back(operationsMap.at(nodeOuput->node())->getResult(nodeOuput->index()));
    }
    else if (auto arg = dynamic_cast<jlm::rvsdg::argument *>(node.input(i)->origin()))
    {
      inputs.push_back(block.getArgument(arg->index()));
    }
    else
    {
      auto message = util::strfmt(
          "Unimplemented input type: ",
          node.input(i)->origin()->debug_string(),
          ": ",
          node.input(i)->origin()->type().debug_string(),
          " for node: ",
          node.operation().debug_string(),
          " at index: ",
          i);
      JLM_UNREACHABLE(message.c_str());
    }
  }
  return inputs;
}

::mlir::Operation *
JlmToMlirConverter::ConvertNode(
    const rvsdg::node & node,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(&node))
  {
    return ConvertSimpleNode(*simpleNode, block, inputs);
  }
  else if (auto lambda = dynamic_cast<const llvm::lambda::node *>(&node))
  {
    return ConvertLambda(*lambda, block);
  }
  else if (auto gamma = dynamic_cast<const rvsdg::gamma_node *>(&node))
  {
    return ConvertGamma(*gamma, block, inputs);
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

::mlir::Operation *
JlmToMlirConverter::ConvertSimpleNode(
    const rvsdg::simple_node & node,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
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
  // ** region structural nodes **
  else if (auto ctlOp = dynamic_cast<const jlm::rvsdg::ctlconstant_op *>(&node.operation()))
  {
    MlirOp = Builder_->create<::mlir::rvsdg::ConstantCtrl>(
        Builder_->getUnknownLoc(),
        ConvertType(node.output(0)->type()), // Control, ouput type
        ctlOp->value().alternative());
  }
  else if (auto matchOp = dynamic_cast<const jlm::rvsdg::match_op *>(&(node.operation())))
  {
    // ** region Create the MLIR mapping vector **
    //! MLIR match operation can match multiple values to one index
    //! But jlm implements this with multiple mappings
    //! For easy conversion, we only created one mapping per value
    ::llvm::SmallVector<::mlir::Attribute> mappingVector;
    for (auto mapping : *matchOp)
    {
      ::mlir::rvsdg::MatchRuleAttr matchRule = ::mlir::rvsdg::MatchRuleAttr::get(
          Builder_->getContext(),
          ::llvm::ArrayRef(static_cast<int64_t>(mapping.first)),
          mapping.second);

      mappingVector.push_back(matchRule);
    }
    //! The default alternative has an empty mapping
    mappingVector.push_back(::mlir::rvsdg::MatchRuleAttr::get(
        Builder_->getContext(),
        ::llvm::ArrayRef<int64_t>(),
        matchOp->default_alternative()));
    // ** endregion Create the MLIR mapping vector **

    MlirOp = Builder_->create<::mlir::rvsdg::Match>(
        Builder_->getUnknownLoc(),
        ConvertType(node.output(0)->type()), // Control, ouput type
        inputs[0],                           // input
        ::mlir::ArrayAttr::get(Builder_->getContext(), ::llvm::ArrayRef(mappingVector)));
  }
  // ** endregion structural nodes **
  else
  {
    auto message = util::strfmt("Unimplemented simple node: ", node.operation().debug_string());
    JLM_UNREACHABLE(message.c_str());
  }

  block.push_back(MlirOp);
  return MlirOp;
}

::mlir::Operation *
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

::mlir::Operation *
JlmToMlirConverter::ConvertGamma(
    const rvsdg::gamma_node & gammaNode,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  auto & gammaOp = *util::AssertedCast<const rvsdg::gamma_op>(&gammaNode.operation());

  ::llvm::SmallVector<::mlir::Type> typeRangeOuput;
  for (size_t i = 0; i < gammaNode.noutputs(); ++i)
  {
    typeRangeOuput.push_back(ConvertType(gammaNode.output(i)->type()));
  }

  // The predicate is always the first input
  // Predicate is used to select the region to execute
  ::mlir::Value predicate = inputs[0];

  auto gamma = Builder_->create<::mlir::rvsdg::GammaNode>(
      Builder_->getUnknownLoc(),
      ::mlir::TypeRange(::llvm::ArrayRef(typeRangeOuput)), // Ouputs types
      predicate,
      ::mlir::ValueRange({ std::next(inputs.begin()), inputs.end() }), // Inputs
      gammaOp.nalternatives()                                          // regionsCount
  );
  block.push_back(gamma);

  for (size_t i = 0; i < gammaOp.nalternatives(); ++i)
  {
    auto & gammaBlock = gamma.getRegion(i).emplaceBlock();
    auto regionResults = ConvertRegion(*gammaNode.subregion(i), gammaBlock);
    auto gammaResult =
        Builder_->create<::mlir::rvsdg::GammaResult>(Builder_->getUnknownLoc(), regionResults);
    gammaBlock.push_back(gammaResult);
  }

  return gamma;
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
  else if (auto clt = dynamic_cast<const rvsdg::ctltype *>(&type))
  {
    return Builder_->getType<::mlir::rvsdg::RVSDG_CTRLType>(clt->nalternatives());
  }
  else
  {
    auto message = util::strfmt("Type conversion not implemented: ", type.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

} // namespace jlm::mlir
