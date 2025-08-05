/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/MLIRConverterCommon.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/UnitType.hpp>

#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

namespace jlm::mlir
{

void
JlmToMlirConverter::Print(::mlir::rvsdg::OmegaNode & omega, const util::FilePath & filePath)
{
  if (failed(::mlir::verify(omega)))
  {
    omega.emitError("module verification error");
    throw util::Error("Verification of RVSDG-MLIR failed");
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
  auto & graph = rvsdgModule.Rvsdg();

  auto omega = Builder_->create<::mlir::rvsdg::OmegaNode>(Builder_->getUnknownLoc());
  auto & omegaBlock = omega.getRegion().emplaceBlock();

  ::llvm::SmallVector<::mlir::Value> regionResults =
      ConvertRegion(graph.GetRootRegion(), omegaBlock, true);

  auto omegaResult =
      Builder_->create<::mlir::rvsdg::OmegaResult>(Builder_->getUnknownLoc(), regionResults);
  omegaBlock.push_back(omegaResult);
  return omega;
}

::llvm::SmallVector<::mlir::Value>
JlmToMlirConverter::ConvertRegion(rvsdg::Region & region, ::mlir::Block & block, bool isRoot)
{
  std::unordered_map<rvsdg::Output *, ::mlir::Value> valueMap;
  size_t argIndex = 0;
  for (size_t i = 0; i < region.narguments(); ++i)
  {
    auto arg = region.argument(i);
    if (isRoot) // Omega arguments are treated separately
    {
      auto imp = util::AssertedCast<llvm::GraphImport>(arg);
      block.push_back(Builder_->create<::mlir::rvsdg::OmegaArgument>(
          Builder_->getUnknownLoc(),
          ConvertType(*imp->ImportedType()),
          ConvertType(*imp->ValueType()),
          Builder_->getStringAttr(llvm::ToString(imp->Linkage())),
          Builder_->getStringAttr(imp->Name())));
      valueMap[arg] = block.back().getResult(0); // Add the output of the omega argument
    }
    else
    {
      block.addArgument(ConvertType(*arg->Type()), Builder_->getUnknownLoc());
      valueMap[arg] = block.getArgument(argIndex);
      ++argIndex;
    }
  }

  // Create an MLIR operation for each RVSDG node and store each pair in a
  // hash map for easy lookup of corresponding MLIR operation
  for (rvsdg::Node * rvsdgNode : rvsdg::TopDownTraverser(&region))
  {
    ::llvm::SmallVector<::mlir::Value> inputs = GetConvertedInputs(*rvsdgNode, valueMap);

    auto convertedNode = ConvertNode(*rvsdgNode, block, inputs);
    for (size_t i = 0; i < rvsdgNode->noutputs(); i++)
    {
      valueMap[rvsdgNode->output(i)] = convertedNode->getResult(i);
    }
  }

  // This code is used to get the results of the region
  //! It is similar to the GetConvertedInputs function
  ::llvm::SmallVector<::mlir::Value> results;
  for (size_t i = 0; i < region.nresults(); i++)
  {
    auto it = valueMap.find(region.result(i)->origin());
    if (it != valueMap.end())
    {
      results.push_back(it->second);
    }
    else
    {
      auto message = util::strfmt(
          "Unimplemented input type: ",
          region.result(i)->origin()->debug_string(),
          ": ",
          region.result(i)->origin()->Type()->debug_string(),
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
    const rvsdg::Node & node,
    const std::unordered_map<rvsdg::Output *, ::mlir::Value> & valueMap)
{
  ::llvm::SmallVector<::mlir::Value> inputs;
  for (size_t i = 0; i < node.ninputs(); i++)
  {
    auto it = valueMap.find(node.input(i)->origin());
    if (it != valueMap.end())
    {
      inputs.push_back(it->second);
    }
    else
    {
      auto message = util::strfmt(
          "Unimplemented input type: ",
          node.input(i)->origin()->debug_string(),
          ": ",
          node.input(i)->origin()->Type()->debug_string(),
          " for node: ",
          node.DebugString(),
          " at index: ",
          i);
      JLM_UNREACHABLE(message.c_str());
    }
  }
  return inputs;
}

::mlir::Operation *
JlmToMlirConverter::ConvertNode(
    const rvsdg::Node & node,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
  {
    return ConvertSimpleNode(*simpleNode, block, inputs);
  }
  else if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(&node))
  {
    return ConvertLambda(*lambda, block, inputs);
  }
  else if (auto gamma = dynamic_cast<const rvsdg::GammaNode *>(&node))
  {
    return ConvertGamma(*gamma, block, inputs);
  }
  else if (auto theta = dynamic_cast<const rvsdg::ThetaNode *>(&node))
  {
    return ConvertTheta(*theta, block, inputs);
  }
  else if (auto delta = dynamic_cast<const llvm::DeltaNode *>(&node))
  {
    return ConvertDelta(*delta, block, inputs);
  }
  else
  {
    auto message = util::strfmt("Unimplemented structural node: ", node.DebugString());
    JLM_UNREACHABLE(message.c_str());
  }
}

::mlir::Operation *
JlmToMlirConverter::ConvertFpBinaryNode(
    const jlm::llvm::FBinaryOperation & op,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  switch (op.fpop())
  {
  case jlm::llvm::fpop::add:
    return Builder_->create<::mlir::arith::AddFOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  case jlm::llvm::fpop::sub:
    return Builder_->create<::mlir::arith::SubFOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  case jlm::llvm::fpop::mul:
    return Builder_->create<::mlir::arith::MulFOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  case jlm::llvm::fpop::div:
    return Builder_->create<::mlir::arith::DivFOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  case jlm::llvm::fpop::mod:
    return Builder_->create<::mlir::arith::RemFOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  default:
    JLM_UNREACHABLE("Unknown binary bitop");
  }
}

::mlir::Operation *
JlmToMlirConverter::ConvertFpCompareNode(
    const llvm::FCmpOperation & op,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  const auto & map = GetFpCmpPredicateMap();
  auto predicate = map.LookupValue(op.cmp());
  return Builder_->create<::mlir::arith::CmpFOp>(
      Builder_->getUnknownLoc(),
      Builder_->getAttr<::mlir::arith::CmpFPredicateAttr>(predicate),
      inputs[0],
      inputs[1]);
}

::mlir::Operation *
JlmToMlirConverter::ConvertBitBinaryNode(
    const rvsdg::SimpleOperation & bitOp,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  ::mlir::Operation * MlirOp = nullptr;
  if (jlm::rvsdg::is<const rvsdg::bitadd_op>(bitOp))
  {
    MlirOp =
        Builder_->create<::mlir::arith::AddIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
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
    const rvsdg::SimpleOperation & bitOp,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  auto compPredicate = ::mlir::arith::CmpIPredicate::eq;
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
  {
    auto message = util::strfmt("Unknown compare operation: ", bitOp.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }

  auto MlirOp = Builder_->create<::mlir::arith::CmpIOp>(
      Builder_->getUnknownLoc(),
      compPredicate,
      inputs[0],
      inputs[1]);
  return MlirOp;
}

::mlir::Operation *
JlmToMlirConverter::ConvertPointerCompareNode(
    const llvm::PtrCmpOperation & operation,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  auto compPredicate = ::mlir::LLVM::ICmpPredicate::eq;
  if (operation.cmp() == llvm::cmp::eq)
    compPredicate = ::mlir::LLVM::ICmpPredicate::eq;
  else if (operation.cmp() == llvm::cmp::ne)
    compPredicate = ::mlir::LLVM::ICmpPredicate::ne;
  else if (operation.cmp() == llvm::cmp::gt)
    compPredicate = ::mlir::LLVM::ICmpPredicate::sgt;
  else if (operation.cmp() == llvm::cmp::ge)
    compPredicate = ::mlir::LLVM::ICmpPredicate::sge;
  else if (operation.cmp() == llvm::cmp::lt)
    compPredicate = ::mlir::LLVM::ICmpPredicate::slt;
  else if (operation.cmp() == llvm::cmp::le)
    compPredicate = ::mlir::LLVM::ICmpPredicate::sle;
  else
  {
    auto message = util::strfmt("Unknown pointer compare operation: ", operation.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }

  auto MlirOp = Builder_->create<::mlir::LLVM::ICmpOp>(
      Builder_->getUnknownLoc(),
      compPredicate,
      inputs[0],
      inputs[1]);
  return MlirOp;
}

::mlir::Operation *
JlmToMlirConverter::ConvertSimpleNode(
    const rvsdg::SimpleNode & node,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  ::mlir::Operation * MlirOp = nullptr;
  auto & operation = node.GetOperation();
  if (auto bitOp = dynamic_cast<const rvsdg::bitconstant_op *>(&operation))
  {
    auto value = bitOp->value();
    MlirOp = Builder_->create<::mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value.to_uint(),
        value.nbits());
  }
  else if (
      auto integerConstOp = dynamic_cast<const jlm::llvm::IntegerConstantOperation *>(&operation))
  {
    auto isNegative = integerConstOp->Representation().is_negative();
    auto value = isNegative ? integerConstOp->Representation().to_int()
                            : integerConstOp->Representation().to_uint();
    MlirOp = Builder_->create<::mlir::arith::ConstantIntOp>(
        Builder_->getUnknownLoc(),
        value,
        integerConstOp->Representation().nbits());
  }
  else if (auto fpBinOp = dynamic_cast<const jlm::llvm::FBinaryOperation *>(&operation))
  {
    MlirOp = ConvertFpBinaryNode(*fpBinOp, inputs);
  }

  else if (rvsdg::is<jlm::llvm::IntegerBinaryOperation>(operation))
  {
    MlirOp = ConvertIntegerBinaryOperation(
        *dynamic_cast<const jlm::llvm::IntegerBinaryOperation *>(&operation),
        inputs);
  }
  else if (auto fpOp = dynamic_cast<const llvm::ConstantFP *>(&operation))
  {
    auto size = ConvertFPType(fpOp->size());
    auto value = fpOp->constant();
    MlirOp =
        Builder_->create<::mlir::arith::ConstantFloatOp>(Builder_->getUnknownLoc(), value, size);
  }
  else if (auto zeroOp = dynamic_cast<const llvm::ConstantAggregateZeroOperation *>(&operation))
  {
    auto type = ConvertType(*zeroOp->result(0));
    MlirOp = Builder_->create<::mlir::LLVM::ZeroOp>(Builder_->getUnknownLoc(), type);
  }
  else if (auto arrOp = dynamic_cast<const llvm::ConstantDataArray *>(&operation))
  {
    auto arrayType = ConvertType(*arrOp->result(0));
    MlirOp = Builder_->create<::mlir::jlm::ConstantDataArray>(
        Builder_->getUnknownLoc(),
        arrayType,
        inputs);
  }
  else if (auto zeroOp = dynamic_cast<const llvm::ConstantAggregateZeroOperation *>(&operation))
  {
    auto type = ConvertType(*zeroOp->result(0));
    MlirOp = Builder_->create<::mlir::LLVM::ZeroOp>(Builder_->getUnknownLoc(), type);
  }
  else if (
      auto constantPointerNullOp =
          dynamic_cast<const llvm::ConstantPointerNullOperation *>(&operation))
  {
    // NULL pointers are a special case of ZeroOp
    auto type = ConvertType(*constantPointerNullOp->result(0));
    MlirOp = Builder_->create<::mlir::LLVM::ZeroOp>(Builder_->getUnknownLoc(), type);
  }
  else if (jlm::rvsdg::is<const rvsdg::BitBinaryOperation>(operation))
  {
    MlirOp = ConvertBitBinaryNode(operation, inputs);
  }
  else if (auto fpBinOp = dynamic_cast<const jlm::llvm::FBinaryOperation *>(&operation))
  {
    MlirOp = ConvertFpBinaryNode(*fpBinOp, inputs);
  }
  else if (rvsdg::is<const jlm::llvm::FNegOperation>(operation))
  {
    MlirOp = Builder_->create<::mlir::arith::NegFOp>(Builder_->getUnknownLoc(), inputs[0]);
  }
  else if (auto fpextOp = dynamic_cast<const jlm::llvm::FPExtOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::arith::ExtFOp>(
        Builder_->getUnknownLoc(),
        ConvertType(*fpextOp->result(0)),
        inputs[0]);
  }

  else if (jlm::rvsdg::is<const rvsdg::BitCompareOperation>(operation))
  {
    MlirOp = BitCompareNode(operation, inputs);
  }
  else if (auto fpCmpOp = dynamic_cast<const llvm::FCmpOperation *>(&operation))
  {
    MlirOp = ConvertFpCompareNode(*fpCmpOp, inputs);
  }
  else if (auto pointerCompareOp = dynamic_cast<const llvm::PtrCmpOperation *>(&operation))
  {
    MlirOp = ConvertPointerCompareNode(*pointerCompareOp, inputs);
  }
  else if (const auto zextOperation = dynamic_cast<const llvm::ZExtOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::arith::ExtUIOp>(
        Builder_->getUnknownLoc(),
        Builder_->getIntegerType(zextOperation->ndstbits()),
        inputs[0]);
  }
  else if (auto sextOp = dynamic_cast<const jlm::llvm::SExtOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::arith::ExtSIOp>(
        Builder_->getUnknownLoc(),
        Builder_->getIntegerType(sextOp->ndstbits()),
        inputs[0]);
  }
  else if (auto sitofpOp = dynamic_cast<const llvm::SIToFPOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::arith::SIToFPOp>(
        Builder_->getUnknownLoc(),
        ConvertType(*sitofpOp->result(0)),
        inputs[0]);
  }
  else if (auto truncOp = dynamic_cast<const jlm::llvm::TruncOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::arith::TruncIOp>(
        Builder_->getUnknownLoc(),
        ConvertType(*truncOp->result(0)),
        inputs[0]);
  }
  // ** region structural nodes **
  else if (auto ctlOp = dynamic_cast<const rvsdg::ctlconstant_op *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::rvsdg::ConstantCtrl>(
        Builder_->getUnknownLoc(),
        ConvertType(*node.output(0)->Type()), // Control, ouput type
        ctlOp->value().alternative());
  }
  else if (auto vaOp = dynamic_cast<const llvm::VariadicArgumentListOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::CreateVarArgList>(
        Builder_->getUnknownLoc(),
        ConvertType(*vaOp->result(0)),
        inputs);
  }
  else if (auto undefOp = dynamic_cast<const llvm::UndefValueOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::Undef>(
        Builder_->getUnknownLoc(),
        ConvertType(undefOp->GetType()));
  }
  else if (auto freeOp = dynamic_cast<const jlm::llvm::FreeOperation *>(&operation))
  {
    auto nMemstates = freeOp->narguments() - 2; // Subtract for pointer and io state

    std::vector<::mlir::Type> memoryStates(
        nMemstates,
        Builder_->getType<::mlir::rvsdg::MemStateEdgeType>());
    MlirOp = Builder_->create<::mlir::jlm::Free>(
        Builder_->getUnknownLoc(),
        ::mlir::TypeRange(::llvm::ArrayRef(memoryStates)),
        Builder_->getType<::mlir::rvsdg::IOStateEdgeType>(),
        inputs[0],
        ::mlir::ValueRange({ std::next(inputs.begin()), std::prev(inputs.end()) }),
        inputs[inputs.size() - 1]);
  }
  else if (auto alloca_op = dynamic_cast<const jlm::llvm::AllocaOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::Alloca>(
        Builder_->getUnknownLoc(),
        ConvertType(*alloca_op->result(0)),                               // ptr
        ConvertType(*alloca_op->result(1)),                               // memstate
        ConvertType(alloca_op->value_type()),                             // value type
        inputs[0],                                                        // size
        alloca_op->alignment(),                                           // alignment
        ::mlir::ValueRange({ std::next(inputs.begin()), inputs.end() })); // inputMemStates
  }
  else if (auto malloc_op = dynamic_cast<const jlm::llvm::MallocOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::Malloc>(
        Builder_->getUnknownLoc(),
        ConvertType(*malloc_op->result(0)), // ptr
        ConvertType(*malloc_op->result(1)), // memstate
        inputs[0]                           // size
    );
  }
  else if (auto load_op = dynamic_cast<const jlm::llvm::LoadOperation *>(&operation))
  {
    // Can have more than a single memory state
    ::llvm::SmallVector<::mlir::Type> memStateTypes;
    for (size_t i = 1; i < load_op->nresults(); i++)
    {
      memStateTypes.push_back(ConvertType(*load_op->result(i)));
    }
    MlirOp = Builder_->create<::mlir::jlm::Load>(
        Builder_->getUnknownLoc(),
        ConvertType(*load_op->result(0)),                               // ptr
        GetMemStateRange(load_op->nresults() - 1),                      // memstate(s)
        inputs[0],                                                      // pointer
        Builder_->getUI32IntegerAttr(load_op->GetAlignment()),          // alignment
        ::mlir::ValueRange({ std::next(inputs.begin()), inputs.end() }) // inputMemStates
    );
  }
  else if (auto store_op = dynamic_cast<const jlm::llvm::StoreOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::Store>(
        Builder_->getUnknownLoc(),
        GetMemStateRange(store_op->nresults()),                                    // memstate(s)
        inputs[0],                                                                 // ptr
        inputs[1],                                                                 // value
        Builder_->getUI32IntegerAttr(store_op->GetAlignment()),                    // alignment
        ::mlir::ValueRange({ std::next(std::next(inputs.begin())), inputs.end() }) // inputMemStates
    );
  }
  else if (rvsdg::is<jlm::llvm::MemoryStateMergeOperation>(operation))
  {
    MlirOp = Builder_->create<::mlir::rvsdg::MemStateMerge>(
        Builder_->getUnknownLoc(),
        ConvertType(*node.output(0)->Type()),
        inputs);
  }
  else if (rvsdg::is<jlm::llvm::IOBarrierOperation>(operation))
  {
    MlirOp = Builder_->create<::mlir::jlm::IOBarrier>(
        Builder_->getUnknownLoc(),
        ConvertType(*node.output(0)->Type()),
        inputs[0],
        inputs[1]);
  }
  else if (auto op = dynamic_cast<const llvm::GetElementPtrOperation *>(&operation))
  {
    MlirOp = Builder_->create<::mlir::LLVM::GEPOp>(
        Builder_->getUnknownLoc(),
        ConvertType(*op->result(0)),                                      // resultType
        ConvertType(op->GetPointeeType()),                                // elementType
        inputs[0],                                                        // basePtr
        ::mlir::ValueRange({ std::next(inputs.begin()), inputs.end() })); // indices
  }
  else if (auto selectOp = dynamic_cast<const llvm::SelectOperation *>(&operation))
  {
    assert(selectOp->nresults() == 1);
    assert(inputs.size() == 3);
    MlirOp = Builder_->create<::mlir::arith::SelectOp>(
        Builder_->getUnknownLoc(),
        ConvertType(*selectOp->result(0)),
        inputs[0],
        inputs[1],
        inputs[2]);
  }
  else if (auto matchOp = dynamic_cast<const rvsdg::MatchOperation *>(&operation))
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
        ConvertType(*node.output(0)->Type()), // Control, ouput type
        inputs[0],                            // input
        ::mlir::ArrayAttr::get(Builder_->getContext(), ::llvm::ArrayRef(mappingVector)));
  }
  else if (auto callOp = dynamic_cast<const jlm::llvm::CallOperation *>(&operation))
  {
    auto functionType = *callOp->GetFunctionType();
    ::llvm::SmallVector<::mlir::Type> argumentTypes;
    for (size_t i = 0; i < functionType.NumArguments(); i++)
    {
      argumentTypes.push_back(ConvertType(functionType.ArgumentType(i)));
    }
    ::llvm::SmallVector<::mlir::Type> resultTypes;
    for (size_t i = 0; i < functionType.NumResults(); i++)
    {
      resultTypes.push_back(ConvertType(functionType.ResultType(i)));
    }
    MlirOp = Builder_->create<::mlir::jlm::Call>(
        Builder_->getUnknownLoc(),
        resultTypes,
        inputs[0], // func ptr
        ::mlir::ValueRange(
            { std::next(inputs.begin()), std::prev(std::prev(inputs.end())) }), // args
        inputs[inputs.size() - 2],                                              // io
        inputs[inputs.size() - 1]                                               // mem
    );
  }
  else if (
      auto lambdaStateSplit =
          dynamic_cast<const jlm::llvm::LambdaEntryMemoryStateSplitOperation *>(&operation))
  {
    ::llvm::SmallVector<::mlir::Type> resultTypes;
    for (size_t i = 0; i < lambdaStateSplit->nresults(); i++)
    {
      resultTypes.push_back(ConvertType(*lambdaStateSplit->result(i).get()));
    }
    ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
    MlirOp = Builder_->create<::mlir::rvsdg::LambdaEntryMemoryStateSplitOperation>(
        Builder_->getUnknownLoc(),
        ::llvm::ArrayRef(resultTypes), // output types
        inputs[0],                     // input
        ::llvm::ArrayRef(attributes));
  }
  else if (
      auto lambdaStateMerge =
          dynamic_cast<const jlm::llvm::LambdaExitMemoryStateMergeOperation *>(&operation))
  {
    ::llvm::SmallVector<::mlir::Type> resultTypes;
    for (size_t i = 0; i < lambdaStateMerge->nresults(); i++)
    {
      resultTypes.push_back(ConvertType(*lambdaStateMerge->result(i).get()));
    }
    ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
    MlirOp = Builder_->create<::mlir::rvsdg::LambdaExitMemoryStateMergeOperation>(
        Builder_->getUnknownLoc(),
        ::llvm::ArrayRef(resultTypes), // output type
        ::mlir::ValueRange(inputs),    // inputs
        ::llvm::ArrayRef(attributes));
  }
  else if (
      auto callStateSplit =
          dynamic_cast<const jlm::llvm::CallExitMemoryStateSplitOperation *>(&operation))
  {
    ::llvm::SmallVector<::mlir::Type> resultTypes;
    for (size_t i = 0; i < callStateSplit->nresults(); i++)
    {
      resultTypes.push_back(ConvertType(*callStateSplit->result(i).get()));
    }
    ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
    MlirOp = Builder_->create<::mlir::rvsdg::CallExitMemoryStateSplit>(
        Builder_->getUnknownLoc(),
        ::llvm::ArrayRef(resultTypes), // output types
        inputs[0],                     // input
        ::llvm::ArrayRef(attributes));
  }
  else if (
      auto callStateMerge =
          dynamic_cast<const jlm::llvm::CallEntryMemoryStateMergeOperation *>(&operation))
  {
    ::llvm::SmallVector<::mlir::Type> resultTypes;
    for (size_t i = 0; i < callStateMerge->nresults(); i++)
    {
      resultTypes.push_back(ConvertType(*callStateMerge->result(i).get()));
    }
    ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
    MlirOp = Builder_->create<::mlir::rvsdg::CallEntryMemoryStateMerge>(
        Builder_->getUnknownLoc(),
        ::llvm::ArrayRef(resultTypes), // output type
        ::mlir::ValueRange(inputs),    // inputs
        ::llvm::ArrayRef(attributes));
  }
  // ** endregion structural nodes **
  else
  {
    auto message = util::strfmt("Unimplemented simple node: ", operation.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }

  block.push_back(MlirOp);
  return MlirOp;
}

::llvm::SmallVector<::mlir::Type>
JlmToMlirConverter::GetMemStateRange(size_t nresults)
{
  ::llvm::SmallVector<::mlir::Type> typeRange;
  for (size_t i = 0; i < nresults; ++i)
  {
    typeRange.push_back(Builder_->getType<::mlir::rvsdg::MemStateEdgeType>());
  }
  return typeRange;
}

::mlir::Operation *
JlmToMlirConverter::ConvertLambda(
    const rvsdg::LambdaNode & lambdaNode,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  // Add function attributes, e.g., the function name and linkage
  ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
  auto symbolName = Builder_->getNamedAttr(
      Builder_->getStringAttr("sym_name"),
      Builder_->getStringAttr(
          dynamic_cast<llvm::LlvmLambdaOperation &>(lambdaNode.GetOperation()).name()));
  attributes.push_back(symbolName);
  auto linkage = Builder_->getNamedAttr(
      Builder_->getStringAttr("linkage"),
      Builder_->getStringAttr(llvm::ToString(
          dynamic_cast<llvm::LlvmLambdaOperation &>(lambdaNode.GetOperation()).linkage())));
  attributes.push_back(linkage);

  auto lambda = Builder_->create<::mlir::rvsdg::LambdaNode>(
      Builder_->getUnknownLoc(),
      ConvertType(*lambdaNode.output()->Type()),
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
    const rvsdg::GammaNode & gammaNode,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  auto & gammaOp = *util::AssertedCast<const rvsdg::GammaOperation>(&gammaNode.GetOperation());

  ::llvm::SmallVector<::mlir::Type> typeRangeOuput;
  for (size_t i = 0; i < gammaNode.noutputs(); ++i)
  {
    typeRangeOuput.push_back(ConvertType(*gammaNode.output(i)->Type()));
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

::mlir::Operation *
JlmToMlirConverter::ConvertTheta(
    const rvsdg::ThetaNode & thetaNode,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  ::llvm::SmallVector<::mlir::Type> outputTypeRange;
  for (size_t i = 0; i < thetaNode.noutputs(); ++i)
  {
    outputTypeRange.push_back(ConvertType(*thetaNode.output(i)->Type()));
  }

  ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
  auto theta = Builder_->create<::mlir::rvsdg::ThetaNode>(
      Builder_->getUnknownLoc(),
      ::mlir::TypeRange(::llvm::ArrayRef(outputTypeRange)),
      ::mlir::ValueRange(::llvm::ArrayRef(inputs)),
      attributes);

  block.push_back(theta);
  auto & thetaBlock = theta.getRegion().emplaceBlock();
  auto regionResults = ConvertRegion(*thetaNode.subregion(), thetaBlock);
  auto results = ::mlir::ValueRange({ std::next(regionResults.begin()), regionResults.end() });
  auto thetaResult = Builder_->create<::mlir::rvsdg::ThetaResult>(
      Builder_->getUnknownLoc(),
      regionResults[0],
      results);
  thetaBlock.push_back(thetaResult);
  return theta;
}

::mlir::Operation *
JlmToMlirConverter::ConvertDelta(
    const llvm::DeltaNode & deltaNode,
    ::mlir::Block & block,
    const ::llvm::SmallVector<::mlir::Value> & inputs)
{
  auto delta = Builder_->create<::mlir::rvsdg::DeltaNode>(
      Builder_->getUnknownLoc(),
      Builder_->getType<::mlir::LLVM::LLVMPointerType>(),
      inputs,
      ::llvm::StringRef(deltaNode.name()),
      ::llvm::StringRef(llvm::ToString(deltaNode.linkage())),
      ::llvm::StringRef(deltaNode.Section()),
      deltaNode.constant());
  block.push_back(delta);
  auto & deltaBlock = delta.getRegion().emplaceBlock();
  auto regionResults = ConvertRegion(*deltaNode.subregion(), deltaBlock);
  JLM_ASSERT(regionResults.size() == 1); // Delta nodes have 1 output
  auto deltaResult =
      Builder_->create<::mlir::rvsdg::DeltaResult>(Builder_->getUnknownLoc(), regionResults[0]);
  deltaBlock.push_back(deltaResult);
  return delta;
}

::mlir::FloatType
JlmToMlirConverter::ConvertFPType(const llvm::fpsize size)
{
  switch (size)
  {
  case jlm::llvm::fpsize::half:
    return Builder_->getF16Type();
  case jlm::llvm::fpsize::flt:
    return Builder_->getF32Type();
  case jlm::llvm::fpsize::dbl:
    return Builder_->getF64Type();
  case jlm::llvm::fpsize::x86fp80:
    return Builder_->getF80Type();
  case jlm::llvm::fpsize::fp128:
    return Builder_->getF128Type();
  default:
    auto message = util::strfmt(
        "Floating point type conversion not implemented: ",
        llvm::FloatingPointType(size).debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

::mlir::FunctionType
JlmToMlirConverter::ConvertFunctionType(const jlm::rvsdg::FunctionType & functionType)
{
  ::llvm::SmallVector<::mlir::Type> argumentTypes;
  for (size_t i = 0; i < functionType.NumArguments(); i++)
  {
    argumentTypes.push_back(ConvertType(functionType.ArgumentType(i)));
  }
  ::llvm::SmallVector<::mlir::Type> resultTypes;
  for (size_t i = 0; i < functionType.NumResults(); i++)
  {
    resultTypes.push_back(ConvertType(functionType.ResultType(i)));
  }
  return Builder_->getFunctionType(argumentTypes, resultTypes);
}

::mlir::Type
JlmToMlirConverter::ConvertType(const rvsdg::Type & type)
{
  if (auto bt = dynamic_cast<const rvsdg::bittype *>(&type))
  {
    return Builder_->getIntegerType(bt->nbits());
  }
  else if (auto fpt = dynamic_cast<const jlm::llvm::FloatingPointType *>(&type))
  {
    return ConvertFPType(fpt->size());
  }
  else if (rvsdg::is<llvm::IOStateType>(type))
  {
    return Builder_->getType<::mlir::rvsdg::IOStateEdgeType>();
  }
  else if (rvsdg::is<llvm::MemoryStateType>(type))
  {
    return Builder_->getType<::mlir::rvsdg::MemStateEdgeType>();
  }
  else if (auto clt = dynamic_cast<const rvsdg::ControlType *>(&type))
  {
    return Builder_->getType<::mlir::rvsdg::RVSDG_CTRLType>(clt->nalternatives());
  }
  else if (rvsdg::is<llvm::PointerType>(type))
  {
    return Builder_->getType<::mlir::LLVM::LLVMPointerType>();
  }
  else if (auto arrayType = dynamic_cast<const llvm::ArrayType *>(&type))
  {
    return Builder_->getType<::mlir::LLVM::LLVMArrayType>(
        ConvertType(arrayType->element_type()),
        arrayType->nelements());
  }
  else if (auto functionType = dynamic_cast<const jlm::rvsdg::FunctionType *>(&type))
  {
    return ConvertFunctionType(*functionType);
  }
  else if (rvsdg::is<const llvm::VariableArgumentType>(type))
  {
    return Builder_->getType<::mlir::jlm::VarargListType>();
  }
  else if (rvsdg::is<const rvsdg::UnitType>(type))
  {
    return Builder_->getType<::mlir::NoneType>();
  }
  else
  {
    auto message = util::strfmt("Type conversion not implemented: ", type.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

::mlir::Operation *
JlmToMlirConverter::ConvertIntegerBinaryOperation(
    const jlm::llvm::IntegerBinaryOperation & operation,
    ::llvm::SmallVector<::mlir::Value> inputs)
{
  if (rvsdg::is<jlm::llvm::IntegerAddOperation>(operation))
  {
    return Builder_->create<::mlir::arith::AddIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSubOperation>(operation))
  {
    return Builder_->create<::mlir::arith::SubIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerMulOperation>(operation))
  {
    return Builder_->create<::mlir::arith::MulIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSDivOperation>(operation))
  {
    return Builder_->create<::mlir::arith::DivSIOp>(
        Builder_->getUnknownLoc(),
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerUDivOperation>(operation))
  {
    return Builder_->create<::mlir::arith::DivUIOp>(
        Builder_->getUnknownLoc(),
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSRemOperation>(operation))
  {
    return Builder_->create<::mlir::arith::RemSIOp>(
        Builder_->getUnknownLoc(),
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerURemOperation>(operation))
  {
    return Builder_->create<::mlir::arith::RemUIOp>(
        Builder_->getUnknownLoc(),
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerAShrOperation>(operation))
  {
    return Builder_->create<::mlir::LLVM::AShrOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerShlOperation>(operation))
  {
    return Builder_->create<::mlir::LLVM::ShlOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerLShrOperation>(operation))
  {
    return Builder_->create<::mlir::LLVM::LShrOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerAndOperation>(operation))
  {
    return Builder_->create<::mlir::arith::AndIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerOrOperation>(operation))
  {
    return Builder_->create<::mlir::arith::OrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerXorOperation>(operation))
  {
    return Builder_->create<::mlir::arith::XOrIOp>(Builder_->getUnknownLoc(), inputs[0], inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerEqOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::eq,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerNeOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::ne,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSgeOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::sge,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSgtOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::sgt,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSleOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::sle,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerSltOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::slt,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerUgeOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::uge,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerUgtOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::ugt,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerUleOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::ule,
        inputs[0],
        inputs[1]);
  }
  else if (rvsdg::is<jlm::llvm::IntegerUltOperation>(operation))
  {
    return Builder_->create<::mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        ::mlir::arith::CmpIPredicate::ult,
        inputs[0],
        inputs[1]);
  }
  else
  {
    auto message =
        util::strfmt("Unimplemented integer binary operation: ", operation.debug_string());
    JLM_UNREACHABLE(message.c_str());
  }
}

} // namespace jlm::mlir
