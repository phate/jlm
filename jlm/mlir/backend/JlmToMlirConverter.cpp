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

#include "JlmToMlirConverter.hpp"
#include <unistd.h>

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
  // // load llvm dialect
  // ::mlir::DialectRegistry registry;
  // registry.insert<::mlir::LLVM::LLVMDialect>();
  // registry.insert<::mlir::rvsdg::RVSDGDialect>();
  // ::mlir::MLIRContext context(registry);

  // // Create a new MLIR module
  // auto module = ::mlir::ModuleOp::create(Builder_->getUnknownLoc());

  // // Create a new MLIR builder
  // Builder_ = std::make_unique<::mlir::OpBuilder>(&context);

  // // Convert the RVSDG module to MLIR
  // auto omega = ConvertOmega(rvsdgModule.Rvsdg());

  // // Add the omega node to the module
  // module.push_back(omega);

  // // Return the module
  // return omega;

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
  // ::mlir::Value * previous_node = nullptr;
  for (rvsdg::node * rvsdgNode : rvsdg::topdown_traverser(&region))
  {

    // TODO try change to pointers
    ::llvm::SmallVector<::mlir::Value> inputs;
    for (size_t i = 0; i < rvsdgNode->ninputs(); i++)
    {
      if (auto output = dynamic_cast<jlm::rvsdg::simple_output *>(rvsdgNode->input(i)->origin()))
      {
        inputs.push_back(nodes[output->node()]);
      }
      else if (auto arg = dynamic_cast<jlm::rvsdg::argument *>(rvsdgNode->input(i)->origin()))
      {
        inputs.push_back(block.getArgument(arg->index()));
      }

      //  rvsdgNode->input(i)->origin()->debug_string();
    }
    nodes[rvsdgNode] = ConvertNode(*rvsdgNode, block, inputs);
    // previous_node = &(nodes[rvsdgNode]);
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
    ::llvm::SmallVector<::mlir::Value> inputs)
{
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

::mlir::Value
JlmToMlirConverter::ConvertSimpleNode(
    const rvsdg::simple_node & node,
    ::mlir::Block & block,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  if (auto bitOp = dynamic_cast<const rvsdg::bitconstant_op *>(&(node.operation())))
  {
    auto value = bitOp->value();
    auto constOp = Builder_->create<::mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value.to_uint(),
        value.nbits());
    block.push_back(constOp);

    return constOp;
  }
  else if (auto bitOp = dynamic_cast<const rvsdg::bitbinary_op *>(&(node.operation())))
  {
    if (dynamic_cast<const rvsdg::bitadd_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::LLVM::AddOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitand_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::AndIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitashr_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::ShRUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitmul_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::MulIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitor_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::OrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitsdiv_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::DivSIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitshl_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::ShLIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitshr_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::ShRUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitsmod_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::RemSIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitsmulh_op *>(bitOp))
    {
      assert(false && "Binary bit bitOp smulh not supported");
    }
    else if (dynamic_cast<const rvsdg::bitsub_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::SubIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitudiv_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::DivUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitumod_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::RemUIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else if (dynamic_cast<const rvsdg::bitumulh_op *>(bitOp))
    {
      assert(false && "Binary bit bitOp umulh not supported");
    }
    else if (dynamic_cast<const rvsdg::bitxor_op *>(bitOp))
    {
      auto MlirOp =
          Builder_->create<::mlir::arith::XOrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
      block.push_back(MlirOp);
      return MlirOp;
    }
    else
    {
      auto message =
          util::strfmt("Unknown binary bitop on simple node: ", node.operation().debug_string());
      JLM_UNREACHABLE(message.c_str());
    }

    // fprintf(fd, "type : %s\n", bitsOp->type().debug_string().c_str());
  }
  else if (auto bitCompOp = dynamic_cast<const rvsdg::bitcompare_op *>(&(node.operation())))
  {
    ::mlir::arith::CmpIPredicate comp_type;
    if (dynamic_cast<const rvsdg::biteq_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::eq;
    }
    else if (dynamic_cast<const rvsdg::bitne_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::ne;
    }
    else if (dynamic_cast<const rvsdg::bitsge_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::sge;
    }
    else if (dynamic_cast<const rvsdg::bitsgt_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::sgt;
    }
    else if (dynamic_cast<const rvsdg::bitsle_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::sle;
    }
    else if (dynamic_cast<const rvsdg::bitslt_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::slt;
    }
    else if (dynamic_cast<const rvsdg::bituge_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::uge;
    }
    else if (dynamic_cast<const rvsdg::bitugt_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::ugt;
    }
    else if (dynamic_cast<const rvsdg::bitule_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::ule;
    }
    else if (dynamic_cast<const rvsdg::bitult_op *>(bitCompOp))
    {
      comp_type = ::mlir::arith::CmpIPredicate::ult;
    }
    else
    {
      assert(false && "Unknown bitcompare operation");
    }
    auto MlirOp = Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        comp_type,
        inputs[0],
        inputs[1]);
    block.push_back(MlirOp);
    return MlirOp;
  }
  else if (auto bitOp = dynamic_cast<const jlm::llvm::zext_op *>(&(node.operation())))
  {
    auto MlirOp = Builder_->create<::mlir::arith::ExtUIOp>(
        Builder_->getUnknownLoc(),
        Builder_->getIntegerType(bitOp->ndstbits()),
        inputs[0]);
    block.push_back(MlirOp);
    return MlirOp;
  }
  else
  {
    auto message = util::strfmt("Unimplemented simple node: ", node.operation().debug_string());
    JLM_UNREACHABLE(message.c_str());
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
    auto message = util::strfmt("Type conversion not implemented: ", type.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

} // namespace jlm::mlir
