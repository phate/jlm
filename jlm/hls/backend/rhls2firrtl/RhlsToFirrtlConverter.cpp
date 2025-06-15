/*
 * Copyright 2021 Magnus Sjalander <work@sjalander.com> and
 * David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/util/strfmt.hpp>

#include <llvm/ADT/SmallPtrSet.h>

namespace jlm::hls
{

// Handles nodes with 2 inputs and 1 output
circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenSimpleNode(const jlm::rvsdg::SimpleNode * node)
{
  // Only handles nodes with a single output
  if (node->noutputs() != 1)
  {
    throw std::logic_error(node->DebugString() + " has more than 1 output");
  }

  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  // Get the body of the module such that we can add contents to the module
  auto body = module.getBodyBlock();

  ::llvm::SmallVector<mlir::Value> inBundles;

  // Get input signals
  for (size_t i = 0; i < node->ninputs(); i++)
  {
    // Get the input bundle
    auto bundle = GetInPort(module, i);
    // Get the data signal from the bundle
    GetSubfield(body, bundle, "data");
    inBundles.push_back(bundle);
  }

  // Get the output bundle
  auto outBundle = GetOutPort(module, 0);
  // Get the data signal from the bundle
  auto outData = GetSubfield(body, outBundle, "data");

  if (rvsdg::is<llvm::IntegerAddOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddAddOp(body, input0, input1);
    // Connect the op to the output data
    // We drop the carry bit
    Connect(body, outData, DropMSBs(body, op, 1));
  }
  else if (rvsdg::is<llvm::IntegerSubOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddSubOp(body, input0, input1);
    // Connect the op to the output data
    // We drop the carry bit
    Connect(body, outData, DropMSBs(body, op, 1));
  }
  else if (rvsdg::is<llvm::IntegerAndOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddAndOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerXorOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddXorOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerOrOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddOrOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (auto bitmulOp = dynamic_cast<const llvm::IntegerMulOperation *>(&(node->GetOperation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddMulOp(body, input0, input1);
    // Connect the op to the output data
    // Multiplication results are double the input width, so we drop the upper half of the result
    Connect(body, outData, DropMSBs(body, op, bitmulOp->Type().nbits()));
  }
  else if (rvsdg::is<llvm::IntegerSDivOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto divOp = AddDivOp(body, sIntOp0, sIntOp1);
    auto uIntOp = AddAsUIntOp(body, divOp);
    // Connect the op to the output data
    Connect(body, outData, DropMSBs(body, uIntOp, 1));
  }
  else if (rvsdg::is<llvm::IntegerLShrOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddDShrOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerAShrOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto shrOp = AddDShrOp(body, sIntOp0, input1);
    auto uIntOp = AddAsUIntOp(body, shrOp);
    // Connect the op to the output data
    Connect(body, outData, uIntOp);
  }
  else if (rvsdg::is<llvm::IntegerShlOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto bitsOp = AddBitsOp(body, input1, 7, 0);
    auto op = AddDShlOp(body, input0, bitsOp);
    int outSize = JlmSize(node->output(0)->Type().get());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    // Connect the op to the output data
    Connect(body, outData, slice);
  }
  else if (rvsdg::is<llvm::IntegerSRemOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto remOp = AddRemOp(body, sIntOp0, sIntOp1);
    auto uIntOp = AddAsUIntOp(body, remOp);
    Connect(body, outData, uIntOp);
  }
  else if (rvsdg::is<llvm::IntegerEqOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddEqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerNeOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddNeqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerSgtOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddGtOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerUltOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddLtOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerUleOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddLeqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerUgtOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddGtOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerSgeOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddGeqOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerUgeOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddGeqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerSleOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddLeqOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (rvsdg::is<llvm::IntegerSltOperation>(node))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sInt0 = AddAsSIntOp(body, input0);
    auto sInt1 = AddAsSIntOp(body, input1);
    auto op = AddLtOp(body, sInt0, sInt1);
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const llvm::ZExtOperation *>(&(node->GetOperation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, input0);
  }
  else if (rvsdg::is<const llvm::TruncOperation>(node->GetOperation()))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    int outSize = JlmSize(node->output(0)->Type().get());
    Connect(body, outData, AddBitsOp(body, inData, outSize - 1, 0));
  }
  else if (dynamic_cast<const llvm::LambdaExitMemoryStateMergeOperation *>(&(node->GetOperation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, inData);
  }
  else if (dynamic_cast<const llvm::MemoryStateMergeOperation *>(&(node->GetOperation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, inData);
  }
  else if (auto op = dynamic_cast<const llvm::SExtOperation *>(&(node->GetOperation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto sintOp = AddAsSIntOp(body, input0);
    auto padOp = AddPadOp(body, sintOp, op->ndstbits());
    auto uintOp = AddAsUIntOp(body, padOp);
    Connect(body, outData, uintOp);
  }
  else if (auto op = dynamic_cast<const llvm::IntegerConstantOperation *>(&(node->GetOperation())))
  {
    auto & value = op->Representation();
    auto size = value.nbits();
    // Create a constant of UInt<size>(value) and connect to output data
    auto constant = GetConstant(body, size, value.to_uint());
    Connect(body, outData, constant);
  }
  else if (auto op = dynamic_cast<const jlm::rvsdg::ctlconstant_op *>(&(node->GetOperation())))
  {
    auto value = op->value().alternative();
    auto size = ceil(log2(op->value().nalternatives()));
    auto constant = GetConstant(body, size, value);
    Connect(body, outData, constant);
  }
  else if (dynamic_cast<const llvm::bitcast_op *>(&(node->GetOperation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, input0);
  }
  else if (dynamic_cast<const llvm::IntegerToPointerOperation *>(&(node->GetOperation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, input0);
  }
  else if (auto op = dynamic_cast<const jlm::rvsdg::match_op *>(&(node->GetOperation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    auto outData = GetSubfield(body, outBundle, "data");
    int inSize = JlmSize(node->input(0)->Type().get());
    int outSize = JlmSize(node->output(0)->Type().get());
    if (IsIdentityMapping(*op))
    {
      if (inSize == outSize)
      {
        Connect(body, outData, inData);
      }
      else
      {
        Connect(body, outData, AddBitsOp(body, inData, outSize - 1, 0));
      }
    }
    else
    {
      auto size = op->nbits();
      mlir::Value result = GetConstant(body, size, op->default_alternative());
      for (auto it = op->begin(); it != op->end(); it++)
      {
        auto comparison = AddEqOp(body, inData, GetConstant(body, size, it->first));
        auto value = GetConstant(body, size, it->second);
        result = AddMuxOp(body, comparison, value, result);
      }
      if ((unsigned long)outSize != size)
      {
        result = AddBitsOp(body, result, outSize - 1, 0);
      }
      Connect(body, outData, result);
    }
  }
  else if (auto op = dynamic_cast<const llvm::GetElementPtrOperation *>(&(node->GetOperation())))
  {
    // Start of with base pointer
    auto input0 = GetSubfield(body, inBundles[0], "data");
    mlir::Value result = AddCvtOp(body, input0);

    // TODO: support structs
    const jlm::rvsdg::Type * pointeeType = &op->GetPointeeType();
    for (size_t i = 1; i < node->ninputs(); i++)
    {
      int bits = JlmSize(pointeeType);
      if (dynamic_cast<const rvsdg::bittype *>(pointeeType)
          || dynamic_cast<const llvm::FloatingPointType *>(pointeeType))
      {
        pointeeType = nullptr;
      }
      else if (auto arrayType = dynamic_cast<const llvm::ArrayType *>(pointeeType))
      {
        pointeeType = &arrayType->element_type();
      }
      else if (auto vectorType = dynamic_cast<const llvm::VectorType *>(pointeeType))
      {
        pointeeType = vectorType->Type().get();
      }
      else
      {
        throw std::logic_error(pointeeType->debug_string() + " pointer not implemented!");
      }
      // GEP inputs are signed
      auto input = GetSubfield(body, inBundles[i], "data");
      auto asSInt = AddAsSIntOp(body, input);
      int bytes = bits / 8;
      auto constantOp = GetConstant(body, GetPointerSizeInBits(), bytes);
      auto cvtOp = AddCvtOp(body, constantOp);
      auto offset = AddMulOp(body, asSInt, cvtOp);
      result = AddAddOp(body, result, offset);
    }
    auto asUInt = AddAsUIntOp(body, result);
    Connect(body, outData, AddBitsOp(body, asUInt, GetPointerSizeInBits() - 1, 0));
  }
  else if (auto op = dynamic_cast<const llvm::extractelement_op *>(&(node->GetOperation())))
  {
    // Start of with base pointer
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto vt = dynamic_cast<const llvm::VectorType *>(op->argument(0).get());
    auto vec = Builder_->create<circt::firrtl::WireOp>(
        Builder_->getUnknownLoc(),
        circt::firrtl::FVectorType::get(GetFirrtlType(vt->Type().get()), vt->size()),
        "vec");
    auto elementBits = JlmSize(vt->Type().get());
    body->push_back(vec);
    for (size_t i = 0; i < vt->size(); ++i)
    {
      auto subindexOp = Builder_->create<circt::firrtl::SubindexOp>(
          Builder_->getUnknownLoc(),
          vec.getResult(),
          i);
      body->push_back(subindexOp);
      Connect(
          body,
          subindexOp,
          AddBitsOp(body, input0, elementBits * (i + 1) - 1, elementBits * i));
    }
    auto subaccessOp = Builder_->create<circt::firrtl::SubaccessOp>(
        Builder_->getUnknownLoc(),
        vec.getResult(),
        input1);
    body->push_back(subaccessOp);
    Connect(body, outData, subaccessOp);
  }
  else if (dynamic_cast<const llvm::UndefValueOperation *>(&(node->GetOperation())))
  {
    ConnectInvalid(body, outData);
  }
  else if (auto op = dynamic_cast<const MuxOperation *>(&(node->GetOperation())))
  {
    JLM_ASSERT(op->discarding);
    auto select = GetSubfield(body, inBundles[0], "data");
    ConnectInvalid(body, outData);
    for (size_t i = 1; i < node->ninputs(); i++)
    {
      auto data = GetSubfield(body, inBundles[i], "data");
      auto constant = GetConstant(body, JlmSize(node->input(0)->Type().get()), i - 1);
      auto eqOp = AddEqOp(body, select, constant);
      auto whenOp = AddWhenOp(body, eqOp, false);
      auto thenBody = whenOp.getThenBodyBuilder().getBlock();
      Connect(thenBody, outData, data);
    }
  }
  else
  {
    throw std::logic_error("Simple node " + node->DebugString() + " not implemented!");
  }

  // Generate the output valid signal
  auto oneBitValue = GetConstant(body, 1, 1);
  mlir::Value prevAnd = oneBitValue;
  for (size_t i = 0; i < node->ninputs(); i++)
  {
    auto bundle = inBundles[i];
    prevAnd = AddAndOp(body, prevAnd, GetSubfield(body, bundle, "valid"));
  }
  // Connect the valide signal to the output bundle
  auto outValid = GetSubfield(body, outBundle, "valid");
  Connect(body, outValid, prevAnd);

  // Generate the ready signal
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto andReady = AddAndOp(body, outReady, prevAnd);
  // Connect it to the ready signal of the two input bundles
  for (size_t i = 0; i < node->ninputs(); i++)
  {
    auto bundle = inBundles[i];
    auto ready = GetSubfield(body, bundle, "ready");
    Connect(body, ready, andReady);
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenSink(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  // Create a constant of UInt<1>(1)
  auto intType = GetIntType(1);
  auto constant = Builder_->create<circt::firrtl::ConstantOp>(
      Builder_->getUnknownLoc(),
      intType,
      ::llvm::APInt(1, 1));
  body->push_back(constant);

  // Get the input bundle
  auto bundle = GetInPort(module, 0);
  // Get the ready signal from the bundle (first signal in the bundle)
  auto ready = GetSubfield(body, bundle, "ready");
  // Connect the constant to the ready signal
  Connect(body, ready, constant);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenLoopConstBuffer(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto clock = GetClockSignal(module);

  // Input signals
  auto predBundle = GetInPort(module, 0);
  auto predReady = GetSubfield(body, predBundle, "ready");
  auto predValid = GetSubfield(body, predBundle, "valid");
  auto predData = GetSubfield(body, predBundle, "data");

  auto inBundle = GetInPort(module, 1);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");

  // Output signals
  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto outData = GetSubfield(body, outBundle, "data");

  auto dataReg = Builder_->create<circt::firrtl::RegOp>(
      Builder_->getUnknownLoc(),
      GetIntType(node->input(1)->Type().get()),
      clock,
      Builder_->getStringAttr("data_reg"));
  body->push_back(dataReg);
  // predicate 0 updates register, passes through and consumes input
  // we always start with predicate 0 due to pred_buf
  // predicate 1 uses data in register
  Connect(body, predReady, AddAndOp(body, outReady, outValid));
  Connect(
      body,
      inReady,
      AddAndOp(body, AddAndOp(body, outReady, AddNotOp(body, predData)), predValid));

  Connect(body, outValid, AddAndOp(body, AddOrOp(body, predData, inValid), predValid));
  Connect(body, outData, dataReg.getResult());
  auto dataPassThrough = AddAndOp(body, inValid, AddNotOp(body, predData));
  auto dataPassThroughBody =
      AddWhenOp(body, dataPassThrough, false).getThenBodyBuilder().getBlock();
  Connect(dataPassThroughBody, outData, inData);

  auto inFire = AddAndOp(body, inReady, inValid);
  auto inFireBody = AddWhenOp(body, inFire, false).getThenBodyBuilder().getBlock();
  Connect(inFireBody, dataReg.getResult(), inData);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenFork(const jlm::rvsdg::SimpleNode * node)
{
  auto op = dynamic_cast<const jlm::hls::ForkOperation *>(&node->GetOperation());
  bool isConstant = op->IsConstant();
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  // Input signals
  auto inBundle = GetInPort(module, 0);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");

  auto oneBitValue = GetConstant(body, 1, 1);
  auto zeroBitValue = GetConstant(body, 1, 0);

  //
  // Output registers
  //
  if (isConstant)
  {
    Connect(body, inReady, oneBitValue);
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      // Get the bundle
      auto port = GetOutPort(module, i);
      auto portValid = GetSubfield(body, port, "valid");
      auto portData = GetSubfield(body, port, "data");
      Connect(body, portValid, inValid);
      Connect(body, portData, inData);
    }
  }
  else
  {
    auto clock = GetClockSignal(module);
    auto reset = GetResetSignal(module);
    ::llvm::SmallVector<circt::firrtl::RegResetOp> firedRegs;
    ::llvm::SmallVector<circt::firrtl::AndPrimOp> whenConditions;
    // outputs can only fire if input is valid. This should not be necessary, unless other
    // components misbehave
    mlir::Value allFired = inValid;
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      std::string validName("out");
      validName.append(std::to_string(i));
      validName.append("_fired_reg");
      auto firedReg = Builder_->create<circt::firrtl::RegResetOp>(
          Builder_->getUnknownLoc(),
          GetIntType(1),
          clock,
          reset,
          zeroBitValue,
          Builder_->getStringAttr(validName));
      body->push_back(firedReg);
      firedRegs.push_back(firedReg);

      // Get the bundle
      auto port = GetOutPort(module, i);
      auto portReady = GetSubfield(body, port, "ready");
      auto portValid = GetSubfield(body, port, "valid");
      auto portData = GetSubfield(body, port, "data");

      auto notFiredReg = AddNotOp(body, firedReg.getResult());
      auto andOp = AddAndOp(body, inValid, notFiredReg.getResult());
      Connect(body, portValid, andOp);
      Connect(body, portData, inData);

      auto orOp = AddOrOp(body, portReady, firedReg.getResult());
      allFired = AddAndOp(body, allFired, orOp);

      // Conditions needed for the when statements
      whenConditions.push_back(AddAndOp(body, portReady, portValid));
    }
    allFired = AddNodeOp(body, allFired, "all_fired").getResult();
    Connect(body, inReady, allFired);

    // When statement
    auto condition = AddNotOp(body, allFired);
    auto whenOp = AddWhenOp(body, condition, true);
    // getThenBlock() cause an error during commpilation
    // So we first get the builder and then its associated body
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    // Then region
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      auto nestedWhen = AddWhenOp(thenBody, whenConditions[i], false);
      auto nestedBody = nestedWhen.getThenBodyBuilder().getBlock();
      Connect(nestedBody, firedRegs[i].getResult(), oneBitValue);
    }
    // Else region
    auto elseBody = whenOp.getElseBodyBuilder().getBlock();
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      Connect(elseBody, firedRegs[i].getResult(), zeroBitValue);
    }
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenStateGate(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  //
  // Output registers
  //
  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  ::llvm::SmallVector<circt::firrtl::RegResetOp> firedRegs;
  ::llvm::SmallVector<circt::firrtl::AndPrimOp> whenConditions;
  auto oneBitValue = GetConstant(body, 1, 1);
  auto zeroBitValue = GetConstant(body, 1, 0);
  mlir::Value allInsValid = oneBitValue;
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    auto inBundle = GetInPort(module, i);
    //        auto inReady = GetSubfield(body, inBundle, "ready");
    auto inValid = GetSubfield(body, inBundle, "valid");
    //        auto inData  = GetSubfield(body, inBundle, "data");
    allInsValid = AddAndOp(body, allInsValid, inValid);
  }
  allInsValid = AddNodeOp(body, allInsValid, "all_ins_valid").getResult();
  mlir::Value allFired = oneBitValue;
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    std::string validName("out");
    validName.append(std::to_string(i));
    validName.append("_fired_reg");
    auto firedReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(firedReg);
    firedRegs.push_back(firedReg);

    // Get the bundle
    auto out = GetOutPort(module, i);
    auto outReady = GetSubfield(body, out, "ready");
    auto outValid = GetSubfield(body, out, "valid");
    auto outData = GetSubfield(body, out, "data");
    auto in = GetInPort(module, i);
    auto inData = GetSubfield(body, in, "data");

    auto notFiredReg = AddNotOp(body, firedReg.getResult());
    auto andOp = AddAndOp(body, allInsValid, notFiredReg);
    Connect(body, outValid, andOp);
    Connect(body, outData, inData);

    auto orOp = AddOrOp(body, AddAndOp(body, outValid, outReady), firedReg.getResult());
    allFired = AddAndOp(body, allFired, orOp);

    // Conditions needed for the when statements
    whenConditions.push_back(AddAndOp(body, outReady, outValid));
  }
  allFired = AddNodeOp(body, allFired, "all_fired").getResult();
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    auto in = GetInPort(module, i);
    auto inReady = GetSubfield(body, in, "ready");
    Connect(body, inReady, allFired);
  }

  // When statement
  auto condition = AddNotOp(body, allFired);
  auto whenOp = AddWhenOp(body, condition, true);
  // getThenBlock() cause an error during commpilation
  // So we first get the builder and then its associated body
  auto thenBody = whenOp.getThenBodyBuilder().getBlock();
  // Then region
  for (size_t i = 0; i < node->noutputs(); i++)
  {
    auto nestedWhen = AddWhenOp(thenBody, whenConditions[i], false);
    auto nestedBody = nestedWhen.getThenBodyBuilder().getBlock();
    Connect(nestedBody, firedRegs[i].getResult(), oneBitValue);
  }
  // Else region
  auto elseBody = whenOp.getElseBodyBuilder().getBlock();
  for (size_t i = 0; i < node->noutputs(); i++)
  {
    Connect(elseBody, firedRegs[i].getResult(), zeroBitValue);
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsMemResp(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, false);
  auto body = module.getBodyBlock();

  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    auto outBundle = GetOutPort(module, i);
    auto outValid = GetSubfield(body, outBundle, "valid");
    auto outData = GetSubfield(body, outBundle, "data");
    Connect(body, outValid, zeroBitValue);
    ConnectInvalid(body, outData);
  }
  for (size_t j = 0; j < node->ninputs(); ++j)
  {
    mlir::BlockArgument memRes = GetInPort(module, j);
    auto memResValid = GetSubfield(body, memRes, "valid");
    auto memResReady = GetSubfield(body, memRes, "ready");
    auto memResBundle = GetSubfield(body, memRes, "data");
    auto memResId = GetSubfield(body, memResBundle, "id");
    auto memResData = GetSubfield(body, memResBundle, "data");
    auto portWidth =
        memResData->getResult(0).getType().cast<circt::firrtl::IntType>().getWidth().value();

    auto elseBody = body;
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      bool isStore = rvsdg::is<rvsdg::StateType>(node->output(i)->Type());
      auto outBundle = GetOutPort(module, i);
      auto outValid = GetSubfield(elseBody, outBundle, "valid");
      auto outReady = GetSubfield(elseBody, outBundle, "ready");
      auto outData = GetSubfield(elseBody, outBundle, "data");
      auto condition =
          AddAndOp(elseBody, memResValid, AddEqOp(elseBody, GetConstant(elseBody, 8, i), memResId));
      auto whenOp = AddWhenOp(elseBody, condition, true);
      auto thenBody = whenOp.getThenBodyBuilder().getBlock();
      Connect(thenBody, outValid, oneBitValue);
      Connect(thenBody, memResReady, outReady);
      // don't connect data for stores
      if (!isStore)
      {
        int nbits = JlmSize(node->output(i)->Type().get());
        if (nbits == portWidth)
        {
          Connect(thenBody, outData, memResData);
        }
        else
        {
          Connect(thenBody, outData, AddBitsOp(thenBody, memResData, nbits - 1, 0));
        }
      }
      elseBody = whenOp.getElseBodyBuilder().getBlock();
    }

    // Connect to ready for other ids - for example stores
    Connect(elseBody, memResReady, oneBitValue);
    // Assert we don't get a response to the same ID on several in ports - if this shows up we need
    // taken logic for outputs
    for (size_t i = 0; i < j; ++i)
    {
      mlir::BlockArgument memRes2 = GetInPort(module, i);
      auto memResValid2 = GetSubfield(body, memRes2, "valid");
      auto memResBundle2 = GetSubfield(body, memRes2, "data");
      auto memResId2 = GetSubfield(body, memResBundle2, "id");
      auto id_assert = Builder_->create<circt::firrtl::AssertOp>(
          Builder_->getUnknownLoc(),
          GetClockSignal(module),
          AddNotOp(
              body,
              AddAndOp(
                  body,
                  AddAndOp(body, memResValid, memResValid2),
                  AddEqOp(body, memResId, memResId2))),
          AddNotOp(body, GetResetSignal(module)),
          "overlapping reponse id",
          mlir::ValueRange(),
          "response_id_assert_" + std::to_string(j) + "_" + std::to_string(i));
      body->push_back(id_assert);
    }
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsMemReq(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, false);
  auto body = module.getBodyBlock();
  auto op = dynamic_cast<const mem_req_op *>(&node->GetOperation());

  auto loadTypes = op->GetLoadTypes();
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrDatas;
  ::llvm::SmallVector<mlir::Value> loadIds;

  auto storeTypes = op->GetStoreTypes();
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrDatas;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataDatas;
  ::llvm::SmallVector<mlir::Value> storeIds;
  // The ports for loads come first and consist only of addresses.
  // Stores have both addresses and data
  size_t id = 0;
  for (size_t i = 0; i < op->get_nloads(); ++i)
  {
    auto bundle = GetInPort(module, i);
    loadAddrReadys.push_back(GetSubfield(body, bundle, "ready"));
    loadAddrValids.push_back(GetSubfield(body, bundle, "valid"));
    loadAddrDatas.push_back(GetSubfield(body, bundle, "data"));
    loadIds.push_back(GetConstant(body, 8, id));
    id++;
  }
  for (size_t i = op->get_nloads(); i < node->ninputs(); ++i)
  {
    // Store
    auto addrBundle = GetInPort(module, i);
    storeAddrReadys.push_back(GetSubfield(body, addrBundle, "ready"));
    storeAddrValids.push_back(GetSubfield(body, addrBundle, "valid"));
    storeAddrDatas.push_back(GetSubfield(body, addrBundle, "data"));
    i++;
    auto dataBundle = GetInPort(module, i);
    storeDataReadys.push_back(GetSubfield(body, dataBundle, "ready"));
    storeDataValids.push_back(GetSubfield(body, dataBundle, "valid"));
    storeDataDatas.push_back(GetSubfield(body, dataBundle, "data"));
    storeIds.push_back(GetConstant(body, 8, id));
    id++;
  }

  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);
  ::llvm::SmallVector<mlir::Value> loadGranted(loadTypes->size(), zeroBitValue);
  ::llvm::SmallVector<mlir::Value> storeGranted(storeTypes->size(), zeroBitValue);
  for (size_t j = 0; j < node->noutputs(); ++j)
  {
    auto reqType = util::AssertedCast<const BundleType>(node->output(j)->Type().get());
    auto hasWrite = reqType->elements_.size() == 5;
    mlir::BlockArgument memReq = GetOutPort(module, j);
    mlir::Value memReqData;
    mlir::Value memReqWrite;
    auto memReqReady = GetSubfield(body, memReq, "ready");
    auto memReqValid = GetSubfield(body, memReq, "valid");
    auto memReqBundle = GetSubfield(body, memReq, "data");
    auto memReqAddr = GetSubfield(body, memReqBundle, "addr");
    auto memReqSize = GetSubfield(body, memReqBundle, "size");
    auto memReqId = GetSubfield(body, memReqBundle, "id");
    if (hasWrite)
    {
      memReqData = GetSubfield(body, memReqBundle, "data");
      memReqWrite = GetSubfield(body, memReqBundle, "write");
    }
    // Default request connection
    Connect(body, memReqValid, zeroBitValue);
    ConnectInvalid(body, memReqBundle);
    mlir::Value previousGranted = zeroBitValue;
    for (size_t i = 0; i < loadTypes->size(); ++i)
    {
      if (j == 0)
      {
        Connect(body, loadAddrReadys[i], zeroBitValue);
      }
      auto canGrant = AddNotOp(body, AddOrOp(body, previousGranted, loadGranted[i]));
      auto grant = AddAndOp(body, canGrant, loadAddrValids[i]);
      auto whenOp = AddWhenOp(body, grant, false);
      auto thenBody = whenOp.getThenBodyBuilder().getBlock();
      Connect(thenBody, loadAddrReadys[i], memReqReady);
      Connect(thenBody, memReqValid, loadAddrValids[i]);
      Connect(thenBody, memReqAddr, loadAddrDatas[i]);
      Connect(thenBody, memReqId, loadIds[i]);
      // No data or write
      auto loadType = loadTypes->at(i).get();
      int bitWidth = JlmSize(loadType);
      int log2Bytes = log2(bitWidth / 8);
      Connect(thenBody, memReqSize, GetConstant(thenBody, 3, log2Bytes));
      if (hasWrite)
      {
        Connect(thenBody, memReqWrite, zeroBitValue);
      }
      // Update for next iteration
      previousGranted = AddOrOp(body, previousGranted, grant);
      loadGranted[i] = AddOrOp(body, loadGranted[i], grant);
    }
    // Stores
    for (size_t i = 0; hasWrite && i < storeTypes->size(); ++i)
    {
      if (j == 0)
      {
        Connect(body, storeAddrReadys[i], zeroBitValue);
        Connect(body, storeDataReadys[i], zeroBitValue);
      }
      auto notOp = AddNotOp(body, AddOrOp(body, previousGranted, storeGranted[i]));
      auto grant = AddAndOp(body, notOp, storeAddrValids[i]);
      grant = AddAndOp(body, grant, storeDataValids[i]);
      auto whenOp = AddWhenOp(body, grant, false);
      auto thenBody = whenOp.getThenBodyBuilder().getBlock();
      Connect(thenBody, storeAddrReadys[i], memReqReady);
      Connect(thenBody, storeDataReadys[i], memReqReady);
      Connect(thenBody, memReqValid, storeAddrValids[i]);
      Connect(thenBody, memReqAddr, storeAddrDatas[i]);
      Connect(thenBody, memReqData, storeDataDatas[i]);
      // TODO: pad
      //      auto portWidth =
      //      memReqData.getType().cast<circt::firrtl::IntType>().getWidth().value();
      Connect(thenBody, memReqId, storeIds[i]);
      // No data or write
      auto storeType = storeTypes->at(i).get();
      int bitWidth = JlmSize(storeType);
      int log2Bytes = log2(bitWidth / 8);
      Connect(thenBody, memReqSize, GetConstant(thenBody, 3, log2Bytes));
      Connect(thenBody, memReqWrite, oneBitValue);
      // Update for next iteration
      previousGranted = AddOrOp(body, previousGranted, grant);
      storeGranted[i] = AddOrOp(body, storeGranted[i], grant);
    }
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsLoad(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, false);
  auto body = module.getBodyBlock();

  auto load = dynamic_cast<const LoadOperation *>(&(node->GetOperation()));
  auto local_load = dynamic_cast<const local_load_op *>(&(node->GetOperation()));
  JLM_ASSERT(load || local_load);

  // Input signals
  auto inBundleAddr = GetInPort(module, 0);
  auto inReadyAddr = GetSubfield(body, inBundleAddr, "ready");
  auto inValidAddr = GetSubfield(body, inBundleAddr, "valid");
  auto inDataAddr = GetSubfield(body, inBundleAddr, "data");

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inReadyStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inValidStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inDataStates;
  for (size_t i = 1; i < node->ninputs() - 1; ++i)
  {
    auto bundle = GetInPort(module, i);
    inReadyStates.push_back(GetSubfield(body, bundle, "ready"));
    inValidStates.push_back(GetSubfield(body, bundle, "valid"));
    inDataStates.push_back(GetSubfield(body, bundle, "data"));
  }

  auto inBundleMemData = GetInPort(module, node->ninputs() - 1);
  auto inReadyMemData = GetSubfield(body, inBundleMemData, "ready");
  auto inValidMemData = GetSubfield(body, inBundleMemData, "valid");
  auto inDataMemData = GetSubfield(body, inBundleMemData, "data");

  // Output signals
  auto outBundleData = GetOutPort(module, 0);
  auto outReadyData = GetSubfield(body, outBundleData, "ready");
  auto outValidData = GetSubfield(body, outBundleData, "valid");
  auto outDataData = GetSubfield(body, outBundleData, "data");

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outReadyStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outValidStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outDataStates;
  for (size_t i = 1; i < node->noutputs() - 1; ++i)
  {
    auto bundle = GetOutPort(module, i);
    outReadyStates.push_back(GetSubfield(body, bundle, "ready"));
    outValidStates.push_back(GetSubfield(body, bundle, "valid"));
    outDataStates.push_back(GetSubfield(body, bundle, "data"));
  }

  auto outBundleMemAddr = GetOutPort(module, node->noutputs() - 1);
  auto outReadyMemAddr = GetSubfield(body, outBundleMemAddr, "ready");
  auto outValidMemAddr = GetSubfield(body, outBundleMemAddr, "valid");
  auto outDataMemAddr = GetSubfield(body, outBundleMemAddr, "data");

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  // Registers
  ::llvm::SmallVector<circt::firrtl::RegResetOp> oValidRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> oDataRegs;
  for (size_t i = 0; i < node->noutputs() - 1; i++)
  {
    std::string validName("o");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    oValidRegs.push_back(validReg);

    auto zeroValue = GetConstant(body, JlmSize(node->output(i)->Type().get()), 0);
    std::string dataName("o");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(node->output(i)->Type().get()),
        clock,
        reset,
        zeroValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    oDataRegs.push_back(dataReg);
  }
  auto sentReg = Builder_->create<circt::firrtl::RegResetOp>(
      Builder_->getUnknownLoc(),
      GetIntType(1),
      clock,
      reset,
      zeroBitValue,
      Builder_->getStringAttr("sent_reg"));
  body->push_back(sentReg);

  //    mlir::Value canRequest = AddOrOp(body, AddNotOp(body, sentReg), AddAndOp(body,
  //    inValidMemData, outReadyData));
  mlir::Value canRequest = AddNotOp(body, sentReg.getResult());
  canRequest = AddAndOp(body, canRequest, inValidAddr);
  for (auto vld : inValidStates)
  {
    canRequest = AddAndOp(body, canRequest, vld);
  }
  //    canRequest = AddAndOp(body, canRequest, AddOrOp(body, AddNotOp(body, oValidRegs[0]),
  //    outReadyData));
  canRequest = AddAndOp(body, canRequest, AddNotOp(body, oValidRegs[0].getResult()));
  for (size_t i = 1; i < oValidRegs.size(); i++)
  {
    //        canRequest = AddAndOp(body, canRequest, AddOrOp(body, AddNotOp(body, oValidRegs[i]),
    //        outReadyStates[i-1]));
    canRequest = AddAndOp(body, canRequest, AddNotOp(body, oValidRegs[i].getResult()));
  }

  // Block until all inputs and no outputs are valid
  Connect(body, outValidMemAddr, canRequest);
  Connect(body, outDataMemAddr, inDataAddr);

  Connect(body, outValidData, oValidRegs[0].getResult());
  Connect(body, outDataData, oDataRegs[0].getResult());

  for (size_t i = 1; i < node->noutputs() - 1; ++i)
  {
    Connect(body, outValidStates[i - 1], oValidRegs[i].getResult());
    Connect(body, outDataStates[i - 1], oDataRegs[i].getResult());
    auto andOp2 = AddAndOp(body, outReadyStates[i - 1], outValidStates[i - 1]);
    Connect(
        // When o1 fires
        AddWhenOp(body, andOp2, false).getThenBodyBuilder().getBlock(),
        oValidRegs[i].getResult(),
        zeroBitValue);
  }

  // mem_res fire
  auto whenResFireOp = AddWhenOp(body, AddAndOp(body, sentReg.getResult(), inValidMemData), false);
  auto whenResFireBody = whenResFireOp.getThenBodyBuilder().getBlock();
  Connect(whenResFireBody, sentReg.getResult(), zeroBitValue);
  Connect(whenResFireBody, oDataRegs[0].getResult(), inDataMemData);
  Connect(whenResFireBody, oValidRegs[0].getResult(), oneBitValue);
  Connect(whenResFireBody, outDataData, inDataMemData);
  Connect(whenResFireBody, outValidData, oneBitValue);

  // mem_req fire
  auto whenReqFireOp = AddWhenOp(body, outReadyMemAddr, false);
  auto whenReqFireBody = whenReqFireOp.getThenBodyBuilder().getBlock();
  Connect(whenReqFireBody, sentReg.getResult(), oneBitValue);
  for (size_t i = 1; i < node->noutputs() - 1; ++i)
  {
    Connect(whenReqFireBody, oValidRegs[i].getResult(), oneBitValue);
    Connect(whenReqFireBody, oDataRegs[i].getResult(), inDataStates[i - 1]);
  }

  // Handshaking
  Connect(body, inReadyAddr, outReadyMemAddr);
  for (size_t i = 1; i < node->ninputs() - 1; ++i)
  {
    Connect(body, inReadyStates[i - 1], outReadyMemAddr);
  }
  Connect(body, inReadyMemData, sentReg.getResult());

  auto andOp = AddAndOp(body, outReadyData, outValidData);
  Connect(
      // When o0 fires
      AddWhenOp(body, andOp, false).getThenBodyBuilder().getBlock(),
      oValidRegs[0].getResult(),
      zeroBitValue);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsDLoad(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, false);
  auto body = module.getBodyBlock();

  auto load = dynamic_cast<const decoupled_load_op *>(&(node->GetOperation()));
  JLM_ASSERT(load);

  // Input signals
  auto inBundleAddr = GetInPort(module, 0);
  auto inReadyAddr = GetSubfield(body, inBundleAddr, "ready");
  auto inValidAddr = GetSubfield(body, inBundleAddr, "valid");
  auto inDataAddr = GetSubfield(body, inBundleAddr, "data");

  auto inBundleMemData = GetInPort(module, node->ninputs() - 1);
  auto inReadyMemData = GetSubfield(body, inBundleMemData, "ready");
  auto inValidMemData = GetSubfield(body, inBundleMemData, "valid");
  auto inDataMemData = GetSubfield(body, inBundleMemData, "data");

  // Output signals
  auto outBundleData = GetOutPort(module, 0);
  auto outReadyData = GetSubfield(body, outBundleData, "ready");
  auto outValidData = GetSubfield(body, outBundleData, "valid");
  auto outDataData = GetSubfield(body, outBundleData, "data");

  auto outBundleMemAddr = GetOutPort(module, node->noutputs() - 1);
  auto outReadyMemAddr = GetSubfield(body, outBundleMemAddr, "ready");
  auto outValidMemAddr = GetSubfield(body, outBundleMemAddr, "valid");
  auto outDataMemAddr = GetSubfield(body, outBundleMemAddr, "data");

  // Block until all inputs and no outputs are valid
  Connect(body, outValidMemAddr, inValidAddr);
  Connect(body, outDataMemAddr, inDataAddr);

  // Handshaking
  Connect(body, inReadyAddr, outReadyMemAddr);
  Connect(body, inReadyMemData, outReadyData);

  Connect(body, outValidData, inValidMemData);
  Connect(body, outDataData, inDataMemData);
  AddAndOp(body, outReadyData, outValidData);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsLocalMem(const jlm::rvsdg::SimpleNode * node)
{
  auto lmem_op = dynamic_cast<const local_mem_op *>(&(node->GetOperation()));
  JLM_ASSERT(lmem_op);
  auto res_node = rvsdg::TryGetOwnerNode<rvsdg::Node>(**node->output(0)->begin());
  auto res_op = dynamic_cast<const local_mem_resp_op *>(&res_node->GetOperation());
  JLM_ASSERT(res_op);
  auto req_node = rvsdg::TryGetOwnerNode<rvsdg::Node>(**node->output(1)->begin());
  auto req_op = dynamic_cast<const local_mem_req_op *>(&req_node->GetOperation());
  JLM_ASSERT(req_op);
  // Create the module and its input/output ports - we use a non-standard way here
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;
  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);
  // Input bundle port
  // virtual in/outputs based on request/reponse ports
  for (size_t i = 1; i < req_node->ninputs(); ++i)
  {
    std::string name("i");
    name.append(std::to_string(i - 1));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::In,
        name,
        GetFirrtlType(req_node->input(i)->Type().get()));
  }
  for (size_t i = 0; i < res_node->noutputs(); ++i)
  {
    std::string name("o");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::Out,
        name,
        GetFirrtlType(res_node->output(i)->Type().get()));
  }

  // Creat a name for the module
  auto nodeName = GetModuleName(node);
  mlir::StringAttr name = Builder_->getStringAttr(nodeName);
  // Create the module
  auto module = Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      name,
      circt::firrtl::ConventionAttr::get(
          Builder_->getContext(),
          circt::firrtl::Convention::Internal),
      ports);

  auto body = module.getBodyBlock();

  size_t loads = rvsdg::TryGetOwnerNode<rvsdg::Node>(**node->output(0)->begin())->noutputs();

  // Input signals
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadAddrDatas;

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeAddrDatas;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> storeDataDatas;
  // the ports for loads come first and consist only of addresses. Stores have both addresses and
  // data
  for (size_t i = 1; i < req_node->ninputs(); ++i)
  {
    if (i - 1 < loads)
    {
      // Load
      JLM_ASSERT(storeAddrReadys.empty()); // no stores yet
      auto bundle = GetInPort(module, i - 1);
      loadAddrReadys.push_back(GetSubfield(body, bundle, "ready"));
      loadAddrValids.push_back(GetSubfield(body, bundle, "valid"));
      loadAddrDatas.push_back(GetSubfield(body, bundle, "data"));
    }
    else
    {
      // Store
      auto addrBundle = GetInPort(module, i - 1);
      storeAddrReadys.push_back(GetSubfield(body, addrBundle, "ready"));
      storeAddrValids.push_back(GetSubfield(body, addrBundle, "valid"));
      storeAddrDatas.push_back(GetSubfield(body, addrBundle, "data"));
      i++;
      auto dataBundle = GetInPort(module, i - 1);
      storeDataReadys.push_back(GetSubfield(body, dataBundle, "ready"));
      storeDataValids.push_back(GetSubfield(body, dataBundle, "valid"));
      storeDataDatas.push_back(GetSubfield(body, dataBundle, "data"));
    }
  }

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadDataReadys;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadDataValids;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> loadDataDatas;
  for (size_t i = 0; i < res_node->noutputs(); ++i)
  {
    auto bundle = GetOutPort(module, i);
    loadDataReadys.push_back(GetSubfield(body, bundle, "ready"));
    loadDataValids.push_back(GetSubfield(body, bundle, "valid"));
    loadDataDatas.push_back(GetSubfield(body, bundle, "data"));
  }

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  // memory
  auto arraytype = std::dynamic_pointer_cast<const llvm::ArrayType>(lmem_op->result(0));
  size_t depth = arraytype->nelements();
  auto dataType = GetFirrtlType(&arraytype->element_type());
  ::llvm::SmallVector<mlir::Type> memTypes;
  ::llvm::SmallVector<mlir::Attribute> memNames;
  memTypes.push_back(circt::firrtl::MemOp::getTypeForPort(
      depth,
      dataType,
      circt::firrtl::MemOp::PortKind::ReadWrite));
  memNames.push_back(Builder_->getStringAttr("rw0"));
  //    memTypes.push_back(circt::firrtl::MemOp::getTypeForPort(depth, dataType,
  //    circt::firrtl::MemOp::PortKind::ReadWrite));
  //    memNames.push_back(Builder_->getStringAttr("rw1"));
  // TODO: figure out why writeLatency is wrong here
  auto memory = Builder_->create<circt::firrtl::MemOp>(
      Builder_->getUnknownLoc(),
      memTypes,
      2,
      1,
      depth,
      circt::firrtl::RUWAttr::New,
      memNames,
      "mem");
  body->push_back(memory);
  auto rw0 = memory.getPortNamed("rw0");
  Connect(body, GetSubfield(body, rw0, "clk"), clock);
  auto rw0_wmode = GetSubfield(body, rw0, "wmode");
  Connect(body, GetSubfield(body, rw0, "en"), oneBitValue);
  Connect(body, GetSubfield(body, rw0, "wmask"), oneBitValue);
  auto rw0_addr = GetSubfield(body, rw0, "addr");
  auto rw0_rdata = GetSubfield(body, rw0, "rdata");
  auto rw0_wdata = GetSubfield(body, rw0, "wdata");
  Connect(body, rw0_wdata, GetConstant(body, JlmSize(&arraytype->element_type()), 0));
  //    auto rw1 = memory.getPortNamed("rw1");
  //    Connect(body, GetSubfield(body, rw1, "clk"), clock);
  int addrwidth = ceil(log2(depth));

  // do stores first, because they pass state edges on directly; having loads first might create a
  // combinatorial cycle
  for (size_t i = 0; i < storeDataReadys.size(); ++i)
  {
    Connect(body, storeDataReadys[i], zeroBitValue);
    Connect(body, storeAddrReadys[i], zeroBitValue);
  }
  ::llvm::SmallVector<circt::firrtl::RegResetOp> loadValidRegs;
  for (size_t i = 0; i < loadAddrReadys.size(); ++i)
  {
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr("load_valid_" + std::to_string(i)));
    body->push_back(validReg);
    loadValidRegs.push_back(validReg);
    Connect(body, validReg.getResult(), zeroBitValue);
    Connect(body, loadDataValids[i], validReg.getResult());
    Connect(body, loadDataDatas[i], rw0_rdata);
    Connect(body, loadAddrReadys[i], zeroBitValue);
  }
  //    mlir::Value assigned = zeroBitValue;
  mlir::Block * elsewhen = body;
  for (size_t i = 0; i < storeDataReadys.size(); ++i)
  {
    auto whenReqFireOp =
        AddWhenOp(elsewhen, AddAndOp(elsewhen, storeAddrValids[i], storeDataValids[i]), true);
    auto whenReqFireBody = whenReqFireOp.getThenBodyBuilder().getBlock();
    Connect(whenReqFireBody, storeDataReadys[i], oneBitValue);
    Connect(whenReqFireBody, storeAddrReadys[i], oneBitValue);
    Connect(whenReqFireBody, rw0_wmode, oneBitValue);
    Connect(whenReqFireBody, rw0_wdata, storeDataDatas[i]);
    Connect(
        whenReqFireBody,
        rw0_addr,
        AddBitsOp(whenReqFireBody, storeAddrDatas[i], addrwidth - 1, 0));
    elsewhen = whenReqFireOp.getElseBodyBuilder().getBlock();
  }
  for (size_t i = 0; i < loadAddrReadys.size(); ++i)
  {
    auto whenReqFireOp = AddWhenOp(elsewhen, loadAddrValids[i], true);
    auto whenReqFireBody = whenReqFireOp.getThenBodyBuilder().getBlock();
    Connect(whenReqFireBody, loadAddrReadys[i], oneBitValue);
    Connect(whenReqFireBody, rw0_wmode, zeroBitValue);
    Connect(
        whenReqFireBody,
        rw0_addr,
        AddBitsOp(whenReqFireBody, loadAddrDatas[i], addrwidth - 1, 0));
    Connect(whenReqFireBody, loadValidRegs[i].getResult(), oneBitValue);
    elsewhen = whenReqFireOp.getElseBodyBuilder().getBlock();
  }
  Connect(elsewhen, rw0_wmode, zeroBitValue);
  Connect(elsewhen, rw0_addr, GetConstant(elsewhen, addrwidth, 0));

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenHlsStore(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, false);
  auto body = module.getBodyBlock();

  auto store = dynamic_cast<const store_op *>(&(node->GetOperation()));
  auto local_store = dynamic_cast<const local_store_op *>(&(node->GetOperation()));
  JLM_ASSERT(store || local_store);

  // Input signals
  auto inBundleAddr = GetInPort(module, 0);
  auto inReadyAddr = GetSubfield(body, inBundleAddr, "ready");
  auto inValidAddr = GetSubfield(body, inBundleAddr, "valid");
  auto inDataAddr = GetSubfield(body, inBundleAddr, "data");

  auto inBundleData = GetInPort(module, 1);
  auto inReadyData = GetSubfield(body, inBundleData, "ready");
  auto inValidData = GetSubfield(body, inBundleData, "valid");
  auto inDataData = GetSubfield(body, inBundleData, "data");

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inReadyStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inValidStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> inDataStates;
  for (size_t i = 2; i < node->ninputs() - 1; ++i)
  {
    auto bundle = GetInPort(module, i);
    inReadyStates.push_back(GetSubfield(body, bundle, "ready"));
    inValidStates.push_back(GetSubfield(body, bundle, "valid"));
    inDataStates.push_back(GetSubfield(body, bundle, "data"));
  }

  auto inBundleResp = GetInPort(module, node->ninputs() - 1);
  auto inReadyResp = GetSubfield(body, inBundleResp, "ready");
  auto inValidResp = GetSubfield(body, inBundleResp, "valid");

  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outReadyStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outValidStates;
  ::llvm::SmallVector<circt::firrtl::SubfieldOp> outDataStates;
  for (size_t i = 0; i < node->noutputs() - 2; ++i)
  {
    auto bundle = GetOutPort(module, i);
    outReadyStates.push_back(GetSubfield(body, bundle, "ready"));
    outValidStates.push_back(GetSubfield(body, bundle, "valid"));
    outDataStates.push_back(GetSubfield(body, bundle, "data"));
  }

  auto outBundleMemAddr = GetOutPort(module, node->noutputs() - 2);
  auto outReadyMemAddr = GetSubfield(body, outBundleMemAddr, "ready");
  auto outValidMemAddr = GetSubfield(body, outBundleMemAddr, "valid");
  auto outDataMemAddr = GetSubfield(body, outBundleMemAddr, "data");

  // Output signals
  auto outBundleMemData = GetOutPort(module, node->noutputs() - 1);
  auto outValidMemData = GetSubfield(body, outBundleMemData, "valid");
  auto outDataMemData = GetSubfield(body, outBundleMemData, "data");

  auto oneBitValue = GetConstant(body, 1, 1);

  mlir::Value canRequest = inValidAddr;
  canRequest = AddAndOp(body, canRequest, inValidData);
  for (auto vld : inValidStates)
  {
    canRequest = AddAndOp(body, canRequest, vld);
  }
  // TODO: for now just assume that there is always room for state edges
  //  for (size_t i = 0; i < oValidRegs.size(); ++i)
  //  {
  //    // register is empty or being drained
  //    //        canRequest = AddAndOp(body, canRequest, AddOrOp(body, AddNotOp(body,
  //    oValidRegs[i]),
  //    //        outReadyStates[i]));
  //    canRequest = AddAndOp(body, canRequest, AddNotOp(body, oValidRegs[i].getResult()));
  //  }

  // Block until all inputs and no outputs are valid
  Connect(body, outValidMemAddr, canRequest);
  Connect(body, outDataMemAddr, inDataAddr);
  Connect(body, outValidMemData, canRequest);
  Connect(body, outDataMemData, inDataData);

  mlir::Value outStatesReady = oneBitValue;
  for (size_t i = 0; i < node->noutputs() - 2; ++i)
  {
    Connect(body, outValidStates[i], inValidResp);
    ConnectInvalid(body, outDataStates[i]);
    outStatesReady = AddAndOp(body, outReadyStates[i], outStatesReady);
  }
  Connect(body, inReadyResp, outStatesReady);

  // Handshaking
  Connect(body, inReadyAddr, outReadyMemAddr);
  // TODO: check readyness seperately?
  Connect(body, inReadyData, outReadyMemAddr);
  for (size_t i = 2; i < node->ninputs() - 1; ++i)
  {
    Connect(body, inReadyStates[i - 2], outReadyMemAddr);
  }
  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenMem(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, true);
  auto body = module.getBodyBlock();

  // Check if it's a load or store GetOperation
  bool store = dynamic_cast<const llvm::StoreNonVolatileOperation *>(&(node->GetOperation()));

  InitializeMemReq(module);
  // Input signals
  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  auto inData0 = GetSubfield(body, inBundle0, "data");

  auto inBundle1 = GetInPort(module, 1);
  auto inReady1 = GetSubfield(body, inBundle1, "ready");
  auto inValid1 = GetSubfield(body, inBundle1, "valid");
  auto inData1 = GetSubfield(body, inBundle1, "data");

  // Stores also have a data input that needs to be handled
  // The input is not used by loads but code below reference
  // these variables so we need to define them
  mlir::BlockArgument inBundle2 = NULL;
  circt::firrtl::SubfieldOp inReady2 = NULL;
  circt::firrtl::SubfieldOp inValid2 = NULL;
  circt::firrtl::SubfieldOp inData2 = NULL;
  if (store)
  {
    inBundle2 = GetInPort(module, 2);
    inReady2 = GetSubfield(body, inBundle2, "ready");
    inValid2 = GetSubfield(body, inBundle2, "valid");
    inData2 = GetSubfield(body, inBundle2, "data");
  }

  // Output signals
  auto outBundle0 = GetOutPort(module, 0);
  auto outReady0 = GetSubfield(body, outBundle0, "ready");
  auto outValid0 = GetSubfield(body, outBundle0, "valid");
  auto outData0 = GetSubfield(body, outBundle0, "data");

  // Mem signals
  mlir::BlockArgument memReq = GetPort(module, "mem_req");
  mlir::BlockArgument memRes = GetPort(module, "mem_res");

  auto memReqReady = GetSubfield(body, memReq, "ready");
  auto memReqValid = GetSubfield(body, memReq, "valid");
  auto memReqAddr = GetSubfield(body, memReq, "addr");
  auto memReqData = GetSubfield(body, memReq, "data");
  auto memReqWrite = GetSubfield(body, memReq, "write");
  auto memReqWidth = GetSubfield(body, memReq, "width");

  auto memResValid = GetSubfield(body, memRes, "valid");
  auto memResData = GetSubfield(body, memRes, "data");

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  // Registers
  ::llvm::SmallVector<circt::firrtl::RegResetOp> oValidRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> oDataRegs;
  for (size_t i = 0; i < node->noutputs(); i++)
  {
    std::string validName("o");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    oValidRegs.push_back(validReg);

    auto zeroValue = GetConstant(body, JlmSize(node->output(i)->Type().get()), 0);
    std::string dataName("o");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(node->output(i)->Type().get()),
        clock,
        reset,
        zeroValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    oDataRegs.push_back(dataReg);
  }
  auto sentReg = Builder_->create<circt::firrtl::RegResetOp>(
      Builder_->getUnknownLoc(),
      GetIntType(1),
      clock,
      reset,
      zeroBitValue,
      Builder_->getStringAttr("sent_reg"));
  body->push_back(sentReg);

  mlir::Value canRequest = AddNotOp(body, sentReg.getResult());
  canRequest = AddAndOp(body, canRequest, inValid0);
  canRequest = AddAndOp(body, canRequest, inValid1);
  if (store)
  {
    canRequest = AddAndOp(body, canRequest, inValid2);
  }
  for (size_t i = 0; i < node->noutputs(); i++)
  {
    canRequest = AddAndOp(body, canRequest, AddNotOp(body, oValidRegs[i].getResult()));
  }

  // Block until all inputs and no outputs are valid
  Connect(body, memReqValid, canRequest);
  Connect(body, memReqAddr, inData0);

  int bitWidth;
  if (store)
  {
    Connect(body, memReqWrite, oneBitValue);
    Connect(body, memReqData, inData1);
    bitWidth = std::dynamic_pointer_cast<const rvsdg::bittype>(node->input(1)->Type())->nbits();
  }
  else
  {
    Connect(body, memReqWrite, zeroBitValue);
    auto invalid = GetInvalid(body, 32);
    Connect(body, memReqData, invalid);
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::bittype>(node->output(0)->Type()))
    {
      bitWidth = bitType->nbits();
    }
    else if (rvsdg::is<llvm::PointerType>(node->output(0)->Type()))
    {
      bitWidth = GetPointerSizeInBits();
    }
    else
    {
      throw jlm::util::error("unknown width for mem request");
    }
  }

  int log2Bytes = log2(bitWidth / 8);
  Connect(body, memReqWidth, GetConstant(body, 3, log2Bytes));

  // mem_req fire
  auto whenReqFireOp = AddWhenOp(body, memReqReady, false);
  auto whenReqFireBody = whenReqFireOp.getThenBodyBuilder().getBlock();
  Connect(whenReqFireBody, sentReg.getResult(), oneBitValue);
  if (store)
  {
    Connect(whenReqFireBody, oValidRegs[0].getResult(), oneBitValue);
    Connect(whenReqFireBody, oDataRegs[0].getResult(), inData2);
  }
  else
  {
    Connect(whenReqFireBody, oValidRegs[1].getResult(), oneBitValue);
    Connect(whenReqFireBody, oDataRegs[1].getResult(), inData1);
  }

  // mem_res fire
  auto whenResFireOp = AddWhenOp(body, AddAndOp(body, sentReg.getResult(), memResValid), false);
  auto whenResFireBody = whenResFireOp.getThenBodyBuilder().getBlock();
  Connect(whenResFireBody, sentReg.getResult(), zeroBitValue);
  if (!store)
  {
    Connect(whenResFireBody, oValidRegs[0].getResult(), oneBitValue);
    if (bitWidth != 64)
    {
      auto bitsOp = AddBitsOp(whenResFireBody, memResData, bitWidth - 1, 0);
      Connect(whenResFireBody, oDataRegs[0].getResult(), bitsOp);
    }
    else
    {
      Connect(whenResFireBody, oDataRegs[0].getResult(), memResData);
    }
  }

  // Handshaking
  Connect(body, inReady0, memReqReady);
  Connect(body, inReady1, memReqReady);
  if (store)
  {
    Connect(body, inReady2, memReqReady);
  }

  Connect(body, outValid0, oValidRegs[0].getResult());
  Connect(body, outData0, oDataRegs[0].getResult());
  auto andOp = AddAndOp(body, outReady0, outValid0);
  Connect(
      // When o0 fires
      AddWhenOp(body, andOp, false).getThenBodyBuilder().getBlock(),
      oValidRegs[0].getResult(),
      zeroBitValue);
  if (!store)
  {
    auto outBundle1 = GetOutPort(module, 1);
    auto outReady1 = GetSubfield(body, outBundle1, "ready");
    auto outValid1 = GetSubfield(body, outBundle1, "valid");
    auto outData1 = GetSubfield(body, outBundle1, "data");

    Connect(body, outValid1, oValidRegs[1].getResult());
    Connect(body, outData1, oDataRegs[1].getResult());
    auto andOp = AddAndOp(body, outReady1, outValid1);
    Connect(
        // When o1 fires
        AddWhenOp(body, andOp, false).getThenBodyBuilder().getBlock(),
        oValidRegs[1].getResult(),
        zeroBitValue);
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenTrigger(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  // Input signals
  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  // auto inData0  = GetSubfield(body, inBundle0, "data");
  auto inBundle1 = GetInPort(module, 1);
  auto inReady1 = GetSubfield(body, inBundle1, "ready");
  auto inValid1 = GetSubfield(body, inBundle1, "valid");
  auto inData1 = GetSubfield(body, inBundle1, "data");
  // Output signals
  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto outData = GetSubfield(body, outBundle, "data");

  auto andOp0 = AddAndOp(body, outReady, inValid1);
  auto andOp1 = AddAndOp(body, outReady, inValid0);
  auto andOp2 = AddAndOp(body, inValid0, inValid1);

  Connect(body, inReady0, andOp0);
  Connect(body, inReady1, andOp1);
  Connect(body, outValid, andOp2);
  Connect(body, outData, inData1);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenPrint(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);

  // Input signals
  auto inBundle = GetInPort(module, 0);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");
  // Output signals
  auto outBundle = GetOutPort(module, 0);
  Connect(body, outBundle, inBundle);
  auto trigger = AddAndOp(body, AddAndOp(body, inReady, inValid), AddNotOp(body, reset));
  auto pn = dynamic_cast<const PrintOperation *>(&node->GetOperation());
  auto formatString = "print node " + std::to_string(pn->id()) + ": %x\n";
  auto name = "print_node_" + std::to_string(pn->id());
  auto printValue = AddPadOp(body, inData, 64);
  ::llvm::SmallVector<mlir::Value> operands;
  operands.push_back(printValue);
  body->push_back(Builder_->create<circt::firrtl::PrintFOp>(
      Builder_->getUnknownLoc(),
      clock,
      trigger,
      formatString,
      operands,
      name));
  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenPredicationBuffer(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  std::string validName("buf_valid_reg");
  auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
      Builder_->getUnknownLoc(),
      GetIntType(1),
      clock,
      reset,
      oneBitValue,
      Builder_->getStringAttr(validName));
  body->push_back(validReg);

  std::string dataName("buf_data_reg");
  auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
      Builder_->getUnknownLoc(),
      GetIntType(node->input(0)->Type().get()),
      clock,
      reset,
      zeroBitValue,
      Builder_->getStringAttr(dataName));
  body->push_back(dataReg);

  auto inBundle = GetInPort(module, 0);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");

  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto outData = GetSubfield(body, outBundle, "data");

  auto orOp = AddOrOp(body, validReg.getResult(), inValid);
  Connect(body, outValid, orOp);
  auto muxOp = AddMuxOp(body, validReg.getResult(), dataReg.getResult(), inData);
  Connect(body, outData, muxOp);
  auto notOp = AddNotOp(body, validReg.getResult());
  Connect(body, inReady, notOp);

  // When
  auto condition = AddAndOp(body, inValid, inReady);
  auto whenOp = AddWhenOp(body, condition, false);
  auto thenBody = whenOp.getThenBodyBuilder().getBlock();
  Connect(thenBody, validReg.getResult(), oneBitValue);
  Connect(thenBody, dataReg.getResult(), inData);

  // When
  condition = AddAndOp(body, outValid, outReady);
  whenOp = AddWhenOp(body, condition, false);
  thenBody = whenOp.getThenBodyBuilder().getBlock();
  Connect(thenBody, validReg.getResult(), zeroBitValue);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenBuffer(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto op = dynamic_cast<const BufferOperation *>(&(node->GetOperation()));
  auto capacity = op->capacity;

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto zeroValue = GetConstant(body, JlmSize(node->input(0)->Type().get()), 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  // Registers
  ::llvm::SmallVector<circt::firrtl::RegResetOp> validRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> dataRegs;
  for (size_t i = 0; i <= capacity; i++)
  {
    std::string validName("buf");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    validRegs.push_back(validReg);

    std::string dataName("buf");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(node->input(0)->Type().get()),
        clock,
        reset,
        zeroValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    dataRegs.push_back(dataReg);
  }
  // FIXME
  // Resource waste as the registers will constantly be set to zero
  // This simplifies the code below but might waste resources unless
  // the tools are clever anough to replace it with a constant
  Connect(body, validRegs[capacity].getResult(), zeroBitValue);
  Connect(body, dataRegs[capacity].getResult(), zeroValue);

  // Add wires
  ::llvm::SmallVector<circt::firrtl::WireOp> shiftWires;
  ::llvm::SmallVector<circt::firrtl::WireOp> consumedWires;
  for (size_t i = 0; i <= capacity; i++)
  {
    std::string shiftName("shift_out");
    shiftName.append(std::to_string(i));
    shiftWires.push_back(AddWireOp(body, shiftName, 1));
    std::string consumedName("in_consumed");
    consumedName.append(std::to_string(i));
    consumedWires.push_back(AddWireOp(body, consumedName, 1));
  }

  auto inBundle = GetInPort(module, 0);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");

  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto outData = GetSubfield(body, outBundle, "data");

  // Connect out to buf0
  Connect(body, outValid, validRegs[0].getResult());
  Connect(body, outData, dataRegs[0].getResult());
  auto andOp = AddAndOp(body, outReady, outValid);
  Connect(body, shiftWires[0].getResult(), andOp);
  if (op->pass_through)
  {
    auto notOp = AddNotOp(body, validRegs[0].getResult());
    andOp = AddAndOp(body, notOp, outReady);
    Connect(body, consumedWires[0].getResult(), andOp);
    auto whenOp = AddWhenOp(body, notOp, false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, outData, inData);
    Connect(thenBody, outValid, inValid);
  }
  else
  {
    Connect(body, consumedWires[0].getResult(), zeroBitValue);
  }

  // The buffer is ready if the last one is empty
  auto notOp = AddNotOp(body, validRegs[capacity - 1].getResult());
  Connect(body, inReady, notOp);

  andOp = AddAndOp(body, inReady, inValid);
  for (size_t i = 0; i < capacity; ++i)
  {
    Connect(body, consumedWires[i + 1].getResult(), consumedWires[i].getResult());
    Connect(body, shiftWires[i + 1].getResult(), zeroBitValue);

    // When valid reg
    auto whenOp = AddWhenOp(body, shiftWires[i].getResult(), false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validRegs[i].getResult(), zeroBitValue);

    // When will be empty
    auto notOp = AddNotOp(body, validRegs[i].getResult());
    auto condition = AddOrOp(body, shiftWires[i].getResult(), notOp);
    whenOp = AddWhenOp(body, condition, false);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    // Create the condition needed in nested when
    notOp = AddNotOp(thenBody, consumedWires[i].getResult());
    auto elseCondition = AddAndOp(thenBody, andOp, notOp);

    // Nested when valid reg
    whenOp = AddWhenOp(thenBody, validRegs[i + 1].getResult(), true);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validRegs[i].getResult(), oneBitValue);
    Connect(thenBody, dataRegs[i].getResult(), dataRegs[i + 1].getResult());
    Connect(thenBody, shiftWires[i + 1].getResult(), oneBitValue);

    // Nested else in available
    auto elseBody = whenOp.getElseBodyBuilder().getBlock();
    auto nestedWhen = AddWhenOp(elseBody, elseCondition, false);
    thenBody = nestedWhen.getThenBodyBuilder().getBlock();
    Connect(thenBody, consumedWires[i + 1].getResult(), oneBitValue);
    Connect(thenBody, validRegs[i].getResult(), oneBitValue);
    Connect(thenBody, dataRegs[i].getResult(), inData);
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenAddrQueue(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto op = dynamic_cast<const hls::addr_queue_op *>(&(node->GetOperation()));
  auto capacity = op->capacity;

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto zeroValue = GetConstant(body, JlmSize(node->input(0)->Type().get()), 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  // Registers
  ::llvm::SmallVector<circt::firrtl::RegResetOp> validRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> dataRegs;
  for (size_t i = 0; i <= capacity; i++)
  {
    std::string validName("buf");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    validRegs.push_back(validReg);

    std::string dataName("buf");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(node->input(0)->Type().get()),
        clock,
        reset,
        zeroValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    dataRegs.push_back(dataReg);
  }
  // FIXME
  // Resource waste as the registers will constantly be set to zero
  // This simplifies the code below but might waste resources unless
  // the tools are clever anough to replace it with a constant
  Connect(body, validRegs[capacity].getResult(), zeroBitValue);
  Connect(body, dataRegs[capacity].getResult(), zeroValue);

  // Add wires
  ::llvm::SmallVector<circt::firrtl::WireOp> shiftWires;
  ::llvm::SmallVector<circt::firrtl::WireOp> consumedWires;
  for (size_t i = 0; i <= capacity; i++)
  {
    std::string shiftName("shift_out");
    shiftName.append(std::to_string(i));
    shiftWires.push_back(AddWireOp(body, shiftName, 1));
    std::string consumedName("in_consumed");
    consumedName.append(std::to_string(i));
    consumedWires.push_back(AddWireOp(body, consumedName, 1));
  }

  auto checkBundle = GetInPort(module, 0);
  auto checkReady = GetSubfield(body, checkBundle, "ready");
  auto checkValid = GetSubfield(body, checkBundle, "valid");
  auto checkData = GetSubfield(body, checkBundle, "data");

  auto enqBundle = GetInPort(module, 1);
  auto enqReady = GetSubfield(body, enqBundle, "ready");
  auto enqValid = GetSubfield(body, enqBundle, "valid");
  auto enqData = GetSubfield(body, enqBundle, "data");

  auto deqBundle = GetInPort(module, 2);
  auto deqReady = GetSubfield(body, deqBundle, "ready");
  auto deqValid = GetSubfield(body, deqBundle, "valid");

  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto outData = GetSubfield(body, outBundle, "data");

  // Connect out to addr
  auto addr_in_queue_wire = AddWireOp(body, "addr_in_queue", 1);
  auto addr_out_valid = AddAndOp(body, checkValid, AddNotOp(body, addr_in_queue_wire.getResult()));
  Connect(body, outValid, addr_out_valid);
  Connect(body, outData, checkData);
  Connect(body, checkReady, AddAndOp(body, outReady, addr_out_valid));
  auto andOp = AddAndOp(body, deqReady, deqValid);

  Connect(body, deqReady, validRegs[0].getResult());
  // deq fire
  Connect(body, shiftWires[0].getResult(), andOp);
  //	if (op->pass_through) {
  //		auto notOp = AddNotOp(body, validRegs[0]);
  //		andOp = AddAndOp(body, notOp, outReady);
  //		Connect(body, consumedWires[0], andOp);
  //		auto whenOp = AddWhenOp(body, notOp, false);
  //		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
  //		Connect(thenBody, outData, inData);
  //		Connect(thenBody, outValid, inValid);
  //	} else {
  Connect(body, consumedWires[0].getResult(), zeroBitValue);
  //	}

  // The buffer is ready if the last one is empty
  auto notOp = AddNotOp(body, validRegs[capacity - 1].getResult());
  Connect(body, enqReady, notOp);

  andOp = AddAndOp(body, enqReady, enqValid);
  mlir::Value addr_in_queue = zeroBitValue;
  for (size_t i = 0; i < capacity; ++i)
  {
    Connect(body, consumedWires[i + 1].getResult(), consumedWires[i].getResult());
    Connect(body, shiftWires[i + 1].getResult(), zeroBitValue);

    // When valid reg
    auto whenOp = AddWhenOp(body, shiftWires[i].getResult(), false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validRegs[i].getResult(), zeroBitValue);

    // When will be empty
    auto notOp = AddNotOp(body, validRegs[i].getResult());
    auto condition = AddOrOp(body, shiftWires[i].getResult(), notOp);
    whenOp = AddWhenOp(body, condition, false);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    // Create the condition needed in nested when
    notOp = AddNotOp(thenBody, consumedWires[i].getResult());
    auto elseCondition = AddAndOp(thenBody, andOp, notOp);

    // Nested when valid reg
    whenOp = AddWhenOp(thenBody, validRegs[i + 1].getResult(), true);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validRegs[i].getResult(), oneBitValue);
    Connect(thenBody, dataRegs[i].getResult(), dataRegs[i + 1].getResult());
    Connect(thenBody, shiftWires[i + 1].getResult(), oneBitValue);

    // Nested else in available
    auto elseBody = whenOp.getElseBodyBuilder().getBlock();
    auto nestedWhen = AddWhenOp(elseBody, elseCondition, false);
    thenBody = nestedWhen.getThenBodyBuilder().getBlock();
    Connect(thenBody, consumedWires[i + 1].getResult(), oneBitValue);
    Connect(thenBody, validRegs[i].getResult(), oneBitValue);
    Connect(thenBody, dataRegs[i].getResult(), enqData);

    addr_in_queue = AddOrOp(
        body,
        addr_in_queue,
        AddAndOp(
            body,
            validRegs[i].getResult(),
            AddEqOp(body, dataRegs[i].getResult(), checkData)));
  }
  if (op->combinatorial)
  {
    // may not be the same as addr enqueued in same cycle
    addr_in_queue =
        AddOrOp(body, addr_in_queue, AddAndOp(body, enqValid, AddEqOp(body, enqData, checkData)));
  }
  Connect(body, addr_in_queue_wire.getResult(), addr_in_queue);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenDMux(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto zeroBitValue = GetConstant(body, 1, 0);

  auto inputs = node->ninputs();
  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  // Out valid
  auto outValid = GetSubfield(body, outBundle, "valid");
  Connect(body, outValid, zeroBitValue);
  // Out data
  auto invalid = GetInvalid(body, JlmSize(node->output(0)->Type().get()));
  auto outData = GetSubfield(body, outBundle, "data");
  Connect(body, outData, invalid);
  // Input ready 0
  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  auto inData0 = GetSubfield(body, inBundle0, "data");

  // Add discard registers
  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);

  int ctr_bits = 4;
  auto ctr_zero = GetConstant(body, ctr_bits, 0);
  auto ctr_one = GetConstant(body, ctr_bits, 1);
  auto ctr_max = GetConstant(body, ctr_bits, (1 << ctr_bits) - 1);

  ::llvm::SmallVector<mlir::Value> discard_queueds;
  ::llvm::SmallVector<circt::firrtl::WireOp> discardWires;
  mlir::Value any_discard_full = GetConstant(body, 1, 0);
  // each input has a counter that tracks how many tokens to discard
  // the discardWires are used to increase these counters
  for (size_t i = 1; i < inputs; i++)
  {
    auto inBundle = GetInPort(module, i);
    auto inReady = GetSubfield(body, inBundle, "ready");
    auto inValid = GetSubfield(body, inBundle, "valid");

    std::string regName("i");
    regName.append(std::to_string(i));
    regName.append("_discard_ctr");
    auto discard_ctr_reg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(ctr_bits),
        clock,
        reset,
        ctr_zero,
        Builder_->getStringAttr(regName));
    body->push_back(discard_ctr_reg);

    std::string wireName("i");
    wireName.append(std::to_string(i));
    wireName.append("_discard");
    auto discard_wire = AddWireOp(body, wireName, 1);
    discardWires.push_back(discard_wire);
    Connect(body, discard_wire.getResult(), zeroBitValue);
    auto discard_queued = AddNeqOp(body, discard_ctr_reg.getResult(), ctr_zero);
    discard_queueds.push_back(discard_queued);
    auto discard_full = AddEqOp(body, discard_ctr_reg.getResult(), ctr_max);
    any_discard_full = AddOrOp(body, any_discard_full, discard_full);
    auto fire = AddAndOp(body, inReady, inValid);
    Connect(body, inReady, AddOrOp(body, discard_queued, discard_wire.getResult()));
    auto whenOp = AddWhenOp(
        body,
        AddAndOp(
            body,
            AddAndOp(body, discard_queued, fire),
            AddNotOp(body, discard_wire.getResult())),
        true);
    // This connect was a partial connect and is likely to not work
    Connect(
        &whenOp.getThenBlock(),
        discard_ctr_reg.getResult(),
        DropMSBs(
            &whenOp.getThenBlock(),
            AddSubOp(&whenOp.getThenBlock(), discard_ctr_reg.getResult(), ctr_one),
            1));
    auto elseWhenOp = AddWhenOp(
        &whenOp.getElseBlock(),
        AddAndOp(
            &whenOp.getElseBlock(),
            discard_wire.getResult(),
            AddNotOp(&whenOp.getElseBlock(), fire)),
        false);
    // This connect was a partial connect and is likely to not work
    Connect(
        &elseWhenOp.getThenBlock(),
        discard_ctr_reg.getResult(),
        DropMSBs(
            &elseWhenOp.getThenBlock(),
            AddAddOp(&elseWhenOp.getThenBlock(), discard_ctr_reg.getResult(), ctr_one),
            1));
  }

  auto out_fire = AddAndOp(body, outReady, outValid);
  Connect(body, inReady0, out_fire);

  auto matchBlock =
      &AddWhenOp(body, AddAndOp(body, inValid0, AddNotOp(body, any_discard_full)), false)
           .getThenBlock();
  for (size_t i = 1; i < inputs; i++)
  {
    auto inBundle = GetInPort(module, i);
    auto inReady = GetSubfield(matchBlock, inBundle, "ready");
    auto inValid = GetSubfield(matchBlock, inBundle, "valid");
    auto inData = GetSubfield(matchBlock, inBundle, "data");

    auto whenBlock = &AddWhenOp(
                          matchBlock,
                          AddAndOp(
                              matchBlock,
                              AddEqOp(matchBlock, inData0, GetConstant(matchBlock, 64, i - 1)),
                              AddNotOp(matchBlock, discard_queueds[i - 1])),
                          false)
                          .getThenBlock();
    Connect(whenBlock, outValid, inValid);
    Connect(whenBlock, outData, inData);
    Connect(whenBlock, inReady, outReady);
    for (size_t j = 1; j < inputs; j++)
    {
      if (i == j)
      {
        continue;
      }
      Connect(whenBlock, discardWires[j - 1].getResult(), out_fire);
    }
  }

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenNDMux(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto inputs = node->ninputs();
  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  // Out valid
  auto outValid = GetSubfield(body, outBundle, "valid");
  auto zeroBitValue = GetConstant(body, 1, 0);
  Connect(body, outValid, zeroBitValue);
  // Out data
  auto invalid = GetInvalid(body, JlmSize(node->output(0)->Type().get()));
  auto outData = GetSubfield(body, outBundle, "data");
  Connect(body, outData, invalid);

  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  Connect(body, inReady0, zeroBitValue);
  auto inData0 = GetSubfield(body, inBundle0, "data");

  // We have already handled the first input (i.e., i == 0)
  for (size_t i = 1; i < inputs; i++)
  {
    auto inBundle = GetInPort(module, i);
    auto inReady = GetSubfield(body, inBundle, "ready");
    auto inValid = GetSubfield(body, inBundle, "valid");
    auto inData = GetSubfield(body, inBundle, "data");
    Connect(body, inReady, zeroBitValue);
    auto constant = GetConstant(body, JlmSize(node->input(0)->Type().get()), i - 1);
    auto eqOp = AddEqOp(body, inData0, constant);
    auto andOp = AddAndOp(body, inValid0, eqOp);
    auto whenOp = AddWhenOp(body, andOp, false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, outValid, inValid);
    Connect(thenBody, outData, inData);
    Connect(thenBody, inReady, outReady);
    auto whenAnd = AddAndOp(thenBody, outReady, inValid);
    Connect(thenBody, inReady0, whenAnd);
  }
  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenBranch(const jlm::rvsdg::SimpleNode * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto zeroBitValue = GetConstant(body, 1, 0);

  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  auto inData0 = GetSubfield(body, inBundle0, "data");

  auto inBundle1 = GetInPort(module, 1);
  auto inReady1 = GetSubfield(body, inBundle1, "ready");
  auto inValid1 = GetSubfield(body, inBundle1, "valid");
  auto inData1 = GetSubfield(body, inBundle1, "data");

  Connect(body, inReady0, zeroBitValue);
  Connect(body, inReady1, zeroBitValue);

  auto invalid = GetInvalid(body, 1);
  for (size_t i = 0; i < node->noutputs(); i++)
  {
    auto outBundle = GetOutPort(module, i);
    auto outReady = GetSubfield(body, outBundle, "ready");
    auto outValid = GetSubfield(body, outBundle, "valid");
    auto outData = GetSubfield(body, outBundle, "data");
    Connect(body, outValid, zeroBitValue);
    Connect(body, outData, invalid);

    auto constant = GetConstant(body, JlmSize(node->input(0)->Type().get()), i);
    auto eqOp = AddEqOp(body, inData0, constant);
    auto condition = AddAndOp(body, inValid0, eqOp);
    auto whenOp = AddWhenOp(body, condition, false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, inReady1, outReady);
    auto andOp = AddAndOp(thenBody, outReady, inValid1);
    Connect(thenBody, inReady0, andOp);
    Connect(thenBody, outValid, inValid1);
    Connect(thenBody, outData, inData1);
  }

  return module;
}

circt::firrtl::FModuleLike
RhlsToFirrtlConverter::MlirGen(const jlm::rvsdg::SimpleNode * node)
{
  if (dynamic_cast<const hls::SinkOperation *>(&(node->GetOperation())))
  {
    return MlirGenSink(node);
  }
  else if (dynamic_cast<const ForkOperation *>(&(node->GetOperation())))
  {
    return MlirGenFork(node);
  }
  else if (rvsdg::is<LoopConstantBufferOperation>(node))
  {
    return MlirGenLoopConstBuffer(node);
    //	} else if (dynamic_cast<const jlm::LoadOperation *>(&(node->GetOperation()))) {
    //		return MlirGenMem(node);
    //	} else if (dynamic_cast<const jlm::StoreOperation *>(&(node->GetOperati()))) {
    //		return MlirGenMem(node);
  }
  else if (dynamic_cast<const LoadOperation *>(&(node->GetOperation())))
  {
    return MlirGenHlsLoad(node);
  }
  else if (dynamic_cast<const hls::decoupled_load_op *>(&(node->GetOperation())))
  {
    return MlirGenExtModule(node);
  }
  else if (dynamic_cast<const hls::store_op *>(&(node->GetOperation())))
  {
    return MlirGenHlsStore(node);
  }
  else if (dynamic_cast<const hls::local_load_op *>(&(node->GetOperation())))
  {
    // same as normal load for now, but with index instead of address
    return MlirGenHlsLoad(node);
  }
  else if (dynamic_cast<const hls::local_store_op *>(&(node->GetOperation())))
  {
    // same as normal store for now, but with index instead of address
    return MlirGenHlsStore(node);
  }
  else if (dynamic_cast<const hls::local_mem_op *>(&(node->GetOperation())))
  {
    return MlirGenHlsLocalMem(node);
  }
  else if (dynamic_cast<const hls::mem_resp_op *>(&(node->GetOperation())))
  {
    return MlirGenHlsMemResp(node);
  }
  else if (dynamic_cast<const hls::mem_req_op *>(&(node->GetOperation())))
  {
    return MlirGenHlsMemReq(node);
  }
  else if (jlm::rvsdg::is<const PredicateBufferOperation>(node->GetOperation()))
  {
    return MlirGenPredicationBuffer(node);
  }
  else if (auto b = dynamic_cast<const BufferOperation *>(&node->GetOperation()))
  {
    JLM_ASSERT(b->capacity);
    return MlirGenExtModule(node);
  }
  else if (dynamic_cast<const hls::BranchOperation *>(&(node->GetOperation())))
  {
    return MlirGenBranch(node);
  }
  else if (rvsdg::is<TriggerOperation>(node))
  {
    return MlirGenTrigger(node);
  }
  else if (dynamic_cast<const hls::state_gate_op *>(&(node->GetOperation())))
  {
    return MlirGenStateGate(node);
  }
  else if (dynamic_cast<const PrintOperation *>(&(node->GetOperation())))
  {
    return MlirGenPrint(node);
  }
  else if (dynamic_cast<const hls::addr_queue_op *>(&(node->GetOperation())))
  {
    return MlirGenAddrQueue(node);
  }
  else if (dynamic_cast<const hls::merge_op *>(&(node->GetOperation())))
  {
    // return merge_to_firrtl(n);
    throw std::logic_error(node->DebugString() + " not implemented!");
  }
  else if (auto o = dynamic_cast<const MuxOperation *>(&(node->GetOperation())))
  {
    if (o->discarding)
    {
      return MlirGenSimpleNode(node);
    }
    else
    {
      return MlirGenNDMux(node);
    }
  }
  bool is_float = false;
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    is_float = is_float || rvsdg::is<const llvm::FloatingPointType>(node->input(i)->Type());
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    is_float = is_float || rvsdg::is<const llvm::FloatingPointType>(node->output(i)->Type());
  }
  if (is_float)
  {
    return MlirGenExtModule(node);
  }
  return MlirGenSimpleNode(node);
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGen(hls::loop_node * loopNode, mlir::Block * circuitBody)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(loopNode);
  auto body = module.getBodyBlock();

  auto srModule = MlirGen(loopNode->subregion(), circuitBody);
  // Instantiate the region
  auto instance =
      Builder_->create<circt::firrtl::InstanceOp>(Builder_->getUnknownLoc(), srModule, "sr");
  body->push_back(instance);
  // Connect the Clock
  auto clock = GetClockSignal(module);
  Connect(body, GetInstancePort(instance, "clk"), clock);
  // Connect the Reset
  auto reset = GetResetSignal(module);
  Connect(body, GetInstancePort(instance, "reset"), reset);
  JLM_ASSERT(instance.getNumResults() == module.getNumPorts());

  const size_t clockAndResetOffset = 2;
  for (size_t i = 0; i < loopNode->ninputs(); ++i)
  {
    auto arg = loopNode->input(i)->arguments.begin().ptr();
    auto sourcePort = body->getArgument(i + clockAndResetOffset);
    Connect(body, GetInstancePort(instance, get_port_name(arg)), sourcePort);
  }
  for (size_t i = 0; i < loopNode->noutputs(); ++i)
  {
    auto res = loopNode->output(i)->results.begin().ptr();
    auto sinkPort = body->getArgument(i + loopNode->ninputs() + clockAndResetOffset);
    Connect(body, sinkPort, GetInstancePort(instance, get_port_name(res)));
  }
  return module;
}

circt::firrtl::BitsPrimOp
RhlsToFirrtlConverter::DropMSBs(mlir::Block * body, mlir::Value value, int amount)
{
  auto type = value.getType().cast<circt::firrtl::UIntType>();
  auto width = type.getWidth();
  auto result = AddBitsOp(body, value, width.value() - 1 - amount, 0);
  return result;
}

// Trace the argument back to the "node" generating the value
// Returns the output of a node or the argument of a region that has
// been instantiated as a module
jlm::rvsdg::Output *
RhlsToFirrtlConverter::TraceArgument(rvsdg::RegionArgument * arg)
{
  // Check if the argument is part of a hls::loop_node
  auto region = arg->region();
  auto node = region->node();
  if (dynamic_cast<hls::loop_node *>(node))
  {
    if (auto ba = dynamic_cast<backedge_argument *>(arg))
    {
      return ba->result()->origin();
    }
    else
    {
      // Check if the argument is connected to an input,
      // i.e., if the argument exits the region
      JLM_ASSERT(arg->input() != nullptr);
      // Check if we are in a nested region and directly
      // connected to the outer regions argument
      auto origin = arg->input()->origin();
      if (auto o = dynamic_cast<rvsdg::RegionArgument *>(origin))
      {
        // Need to find the source of the outer regions argument
        return TraceArgument(o);
      }
      else if (auto o = dynamic_cast<rvsdg::StructuralOutput *>(origin))
      {
        // Check if we the input of one loop_node is connected to the output of another
        // StructuralNode, i.e., if the input is connected to the output of another loop_node
        return TraceStructuralOutput(o);
      }
      // Else we have reached the source
      return origin;
    }
  }
  // Reached the argument of a structural node that is not a hls::loop_node
  return arg;
}

circt::firrtl::FModuleLike
RhlsToFirrtlConverter::MlirGen(rvsdg::Region * subRegion, mlir::Block * circuitBody)
{
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;

  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);
  // Argument ports
  for (size_t i = 0; i < subRegion->narguments(); ++i)
  {
    if (!dynamic_cast<backedge_argument *>(subRegion->argument(i)))
    {
      AddBundlePort(
          &ports,
          circt::firrtl::Direction::In,
          get_port_name(subRegion->argument(i)),
          GetFirrtlType(subRegion->argument(i)->Type().get()));
    }
  }
  // Result ports
  for (size_t i = 0; i < subRegion->nresults(); ++i)
  {
    if (!dynamic_cast<backedge_result *>(subRegion->result(i)))
    {
      AddBundlePort(
          &ports,
          circt::firrtl::Direction::Out,
          get_port_name(subRegion->result(i)),
          GetFirrtlType(subRegion->result(i)->Type().get()));
    }
  }

  // Create a name for the module
  auto moduleName = Builder_->getStringAttr("subregion_mod_" + util::strfmt(subRegion));
  // Now when we have all the port information we can create the module
  auto module = Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      moduleName,
      circt::firrtl::ConventionAttr::get(
          Builder_->getContext(),
          circt::firrtl::Convention::Internal),
      ports);
  // Get the body of the module such that we can add contents to the module
  auto body = module.getBodyBlock();

  const size_t clockAndResetOffset = 2;

  std::unordered_map<rvsdg::Output *, mlir::Value> output_map;
  // Arguments
  for (size_t i = 0; i < subRegion->narguments(); ++i)
  {
    if (dynamic_cast<backedge_argument *>(subRegion->argument(i)))
    {
      auto bundleType = GetBundleType(GetFirrtlType(subRegion->argument(i)->Type().get()));
      auto op = Builder_->create<circt::firrtl::WireOp>(
          Builder_->getUnknownLoc(),
          bundleType,
          get_port_name(subRegion->argument(i)));
      body->push_back(op);
      output_map[subRegion->argument(i)] = op.getResult();
    }
    else
    {
      auto ix = i;
      // handle indices of lambdas, that have no inputs and loops, that have backedges
      if (!rvsdg::is<rvsdg::LambdaOperation>(subRegion->node()))
      {
        ix = subRegion->argument(i)->input()->index();
      }
      auto sourcePort = body->getArgument(ix + clockAndResetOffset);
      output_map[subRegion->argument(i)] = sourcePort;
    }
  }

  auto clock = body->getArgument(0);
  auto reset = body->getArgument(1);
  // create nod instances and connect their inputs
  for (const auto node : rvsdg::TopDownTraverser(subRegion))
  {
    auto instance = AddInstanceOp(circuitBody, node);
    body->push_back(instance);
    // Connect clock and reset to the instance
    Connect(body, instance->getResult(0), clock);
    Connect(body, instance->getResult(1), reset);
    // connect inputs
    for (size_t i = 0; i < node->ninputs(); ++i)
    {
      auto sourcePort = output_map[node->input(i)->origin()];
      auto sinkPort = instance->getResult(i + clockAndResetOffset);
      Connect(body, sinkPort, sourcePort);
    }
    // map outputs
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto outputPort = instance->getResult(i + node->ninputs() + clockAndResetOffset);
      output_map[node->output(i)] = outputPort;
    }
  }

  for (size_t i = 0; i < subRegion->nresults(); ++i)
  {
    mlir::Value resultSink;
    if (auto ber = dynamic_cast<backedge_result *>(subRegion->result(i)))
    {
      auto bundleType = GetBundleType(GetFirrtlType(subRegion->result(i)->Type().get()));
      auto op = Builder_->create<circt::firrtl::WireOp>(
          Builder_->getUnknownLoc(),
          bundleType,
          get_port_name(subRegion->result(i)));
      body->push_back(op);
      resultSink = op.getResult();
      // connect backedge to its argument
      Connect(body, output_map[ber->argument()], resultSink);
    }
    else
    {
      auto ix = i;
      // handle indices of lambdas, that have no outputs and loops, that have backedges
      if (!rvsdg::is<rvsdg::LambdaOperation>(subRegion->node()))
      {
        ix = subRegion->result(i)->output()->index();
      }
      resultSink = body->getArgument(ix + module.getNumInputPorts());
    }
    Connect(body, resultSink, output_map[subRegion->result(i)->origin()]);
  }
  circuitBody->push_back(module);
  return module;
}

// Trace a structural output back to the "node" generating the value
// Returns the output of the node
rvsdg::SimpleOutput *
RhlsToFirrtlConverter::TraceStructuralOutput(rvsdg::StructuralOutput * output)
{
  auto node = output->node();

  // We are only expecting hls::loop_node to have a structural output
  if (!dynamic_cast<hls::loop_node *>(node))
  {
    throw std::logic_error("Expected a hls::loop_node but found: " + node->DebugString());
  }
  JLM_ASSERT(output->results.size() == 1);
  auto origin = output->results.begin().ptr()->origin();
  if (auto o = dynamic_cast<rvsdg::StructuralOutput *>(origin))
  {
    // Need to trace the output of the nested structural node
    return TraceStructuralOutput(o);
  }

  if (auto o = dynamic_cast<rvsdg::SimpleOutput *>(origin))
  {
    // Found the source node
    return o;
  }
  else if (dynamic_cast<rvsdg::RegionArgument *>(origin))
  {
    throw std::logic_error("Encountered pass through argument - should be eliminated");
  }
  else
  {
    throw std::logic_error("Encountered an unexpected output type");
  }
}

// Emit a circuit
circt::firrtl::CircuitOp
RhlsToFirrtlConverter::MlirGen(const rvsdg::LambdaNode * lambdaNode)
{

  // Ensure consistent naming across runs
  create_node_names(lambdaNode->subregion());
  // The same name is used for the circuit and main module
  auto moduleName = Builder_->getStringAttr(
      dynamic_cast<llvm::LlvmLambdaOperation &>(lambdaNode->GetOperation()).name() + "_lambda_mod");
  // Create the top level FIRRTL circuit
  auto circuit = Builder_->create<circt::firrtl::CircuitOp>(Builder_->getUnknownLoc(), moduleName);
  // The body will be populated with a list of modules
  auto circuitBody = circuit.getBodyBlock();

  // Get the region of the function
  auto subRegion = lambdaNode->subregion();

  //
  //   Add ports
  //
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;

  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);

  auto reg_args = get_reg_args(*lambdaNode);
  auto reg_results = get_reg_results(*lambdaNode);

  // Input bundle
  using BundleElement = circt::firrtl::BundleType::BundleElement;
  ::llvm::SmallVector<BundleElement> inputElements;
  inputElements.push_back(GetReadyElement());
  inputElements.push_back(GetValidElement());

  for (size_t i = 0; i < reg_args.size(); ++i)
  {
    // don't generate ports for state edges
    if (rvsdg::is<rvsdg::StateType>(reg_args[i]->Type()))
      continue;
    std::string portName("data_");
    portName.append(std::to_string(i));
    inputElements.push_back(BundleElement(
        Builder_->getStringAttr(portName),
        false,
        GetIntType(reg_args[i]->Type().get())));
  }
  auto inputType = circt::firrtl::BundleType::get(Builder_->getContext(), inputElements);
  struct circt::firrtl::PortInfo iBundle = {
    Builder_->getStringAttr("i"), inputType, circt::firrtl::Direction::In, {},
    Builder_->getUnknownLoc(),
  };
  ports.push_back(iBundle);

  // Output bundle
  ::llvm::SmallVector<BundleElement> outputElements;
  outputElements.push_back(GetReadyElement());
  outputElements.push_back(GetValidElement());
  for (size_t i = 0; i < reg_results.size(); ++i)
  {
    // don't generate ports for state edges
    if (rvsdg::is<rvsdg::StateType>(reg_results[i]->Type()))
      continue;
    std::string portName("data_");
    portName.append(std::to_string(i));
    outputElements.push_back(BundleElement(
        Builder_->getStringAttr(portName),
        false,
        GetIntType(reg_results[i]->Type().get())));
  }
  auto outputType = circt::firrtl::BundleType::get(Builder_->getContext(), outputElements);
  struct circt::firrtl::PortInfo oBundle = {
    Builder_->getStringAttr("o"), outputType, circt::firrtl::Direction::Out, {},
    Builder_->getUnknownLoc(),
  };
  ports.push_back(oBundle);

  // Memory ports
  auto mem_reqs = get_mem_reqs(*lambdaNode);
  auto mem_resps = get_mem_resps(*lambdaNode);
  JLM_ASSERT(mem_resps.size() == mem_reqs.size());
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    ::llvm::SmallVector<BundleElement> memElements;

    ::llvm::SmallVector<BundleElement> reqElements;
    reqElements.push_back(GetReadyElement());
    reqElements.push_back(GetValidElement());
    reqElements.push_back(BundleElement(
        Builder_->getStringAttr("data"),
        false,
        GetFirrtlType(mem_reqs[i]->Type().get())));
    auto reqType = circt::firrtl::BundleType::get(Builder_->getContext(), reqElements);
    memElements.push_back(BundleElement(Builder_->getStringAttr("req"), false, reqType));

    ::llvm::SmallVector<BundleElement> resElements;
    resElements.push_back(GetReadyElement());
    resElements.push_back(GetValidElement());
    resElements.push_back(BundleElement(
        Builder_->getStringAttr("data"),
        false,
        GetFirrtlType(mem_resps[i]->Type().get())));
    auto resType = circt::firrtl::BundleType::get(Builder_->getContext(), resElements);
    memElements.push_back(BundleElement(Builder_->getStringAttr("res"), true, resType));

    auto memType = circt::firrtl::BundleType::get(Builder_->getContext(), memElements);
    struct circt::firrtl::PortInfo memBundle = {
      Builder_->getStringAttr("mem_" + std::to_string(i)),
      memType,
      circt::firrtl::Direction::Out,
      {},
      Builder_->getUnknownLoc(),
    };
    ports.push_back(memBundle);
  }

  // Now when we have all the port information we can create the module
  // The same name is used for the circuit and main module
  auto module = Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      moduleName,
      circt::firrtl::ConventionAttr::get(
          Builder_->getContext(),
          circt::firrtl::Convention::Internal),
      ports);
  // Get the body of the module such that we can add contents to the module
  auto body = module.getBodyBlock();

  // Create a module of the region
  auto srModule = MlirGen(subRegion, circuitBody);
  // Instantiate the region
  auto instance =
      Builder_->create<circt::firrtl::InstanceOp>(Builder_->getUnknownLoc(), srModule, "sr");
  body->push_back(instance);
  // Connect the Clock
  auto clock = GetClockSignal(module);
  Connect(body, GetInstancePort(instance, "clk"), clock);
  // Connect the Reset
  auto reset = GetResetSignal(module);
  Connect(body, GetInstancePort(instance, "reset"), reset);

  //
  // Add registers to the module
  //
  // Reset when low (0 == false) 1-bit
  auto zeroBitValue = GetConstant(body, 1, 0);

  // Input registers
  ::llvm::SmallVector<circt::firrtl::RegResetOp> inputValidRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> inputDataRegs;
  for (size_t i = 0; i < reg_args.size(); ++i)
  {
    std::string validName("i");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    inputValidRegs.push_back(validReg);

    std::string dataName("i");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(reg_args[i]->Type().get()),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    inputDataRegs.push_back(dataReg);

    auto port = GetInstancePort(instance, "a" + std::to_string(reg_args[i]->index()));
    auto portValid = GetSubfield(body, port, "valid");
    Connect(body, portValid, validReg.getResult());
    auto portData = GetSubfield(body, port, "data");
    Connect(body, portData, dataReg.getResult());

    // When statement
    auto portReady = GetSubfield(body, port, "ready");
    auto whenCondition = AddAndOp(body, portReady, portValid);
    auto whenOp = AddWhenOp(body, whenCondition, false);

    // getThenBlock() cause an error during commpilation
    // So we first get the builder and then its associated body
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validReg.getResult(), zeroBitValue);
  }

  // Output registers

  // Need to know the number of inputs so we can calculate the
  // correct index for outputs
  ::llvm::SmallVector<circt::firrtl::RegResetOp> outputValidRegs;
  ::llvm::SmallVector<circt::firrtl::RegResetOp> outputDataRegs;

  auto oneBitValue = GetConstant(body, 1, 1);
  for (size_t i = 0; i < reg_results.size(); ++i)
  {
    std::string validName("o");
    validName.append(std::to_string(i));
    validName.append("_valid_reg");
    auto validReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(validName));
    body->push_back(validReg);
    outputValidRegs.push_back(validReg);

    std::string dataName("o");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(reg_results[i]->Type().get()),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(dataName));
    body->push_back(dataReg);
    outputDataRegs.push_back(dataReg);

    // Get the bundle
    auto port = GetInstancePort(instance, "r" + std::to_string(reg_results[i]->index()));

    auto portReady = GetSubfield(body, port, "ready");
    auto notValidReg = Builder_->create<circt::firrtl::NotPrimOp>(
        Builder_->getUnknownLoc(),
        circt::firrtl::IntType::get(Builder_->getContext(), false, 1),
        validReg.getResult());
    body->push_back(notValidReg);
    Connect(body, portReady, notValidReg);

    // When statement
    auto portValid = GetSubfield(body, port, "valid");
    auto portData = GetSubfield(body, port, "data");
    auto whenCondition = AddAndOp(body, portReady, portValid);
    auto whenOp = AddWhenOp(body, whenCondition, false);

    // getThenBlock() cause an error during commpilation
    // So we first get the builder and then its associated body
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, validReg.getResult(), oneBitValue);
    Connect(thenBody, dataReg.getResult(), portData);
  }

  // Create the ready signal for the input bundle
  mlir::Value prevAnd = oneBitValue;
  for (size_t i = 0; i < inputValidRegs.size(); i++)
  {
    auto notReg = Builder_->create<circt::firrtl::NotPrimOp>(
        Builder_->getUnknownLoc(),
        circt::firrtl::IntType::get(Builder_->getContext(), false, 1),
        inputValidRegs[i].getResult());
    body->push_back(notReg);
    auto andOp = AddAndOp(body, notReg, prevAnd);
    prevAnd = andOp;
  }
  auto inBundle = GetPort(module, "i");
  auto inReady = GetSubfield(body, inBundle, "ready");
  Connect(body, inReady, prevAnd);

  // Create the valid signal for the output bundle
  prevAnd = oneBitValue;
  for (size_t i = 0; i < outputValidRegs.size(); i++)
  {
    auto andOp = AddAndOp(body, outputValidRegs[i].getResult(), prevAnd);
    prevAnd = andOp;
  }
  auto outBundle = GetPort(module, "o");
  auto outValid = GetSubfield(body, outBundle, "valid");
  Connect(body, outValid, prevAnd);

  // Connect output data signals
  for (size_t i = 0; i < outputDataRegs.size(); i++)
  {
    // don't generate ports for state edges
    if (rvsdg::is<rvsdg::StateType>(reg_results[i]->Type()))
      continue;
    auto outData = GetSubfield(body, outBundle, "data_" + std::to_string(i));
    Connect(body, outData, outputDataRegs[i].getResult());
  }

  if (inputValidRegs.size())
  { // avoid generating invalid firrtl for return of just a constant
    // Input when statement
    auto inValid = GetSubfield(body, inBundle, "valid");
    auto whenCondition = AddAndOp(body, inReady, inValid);
    auto whenOp = AddWhenOp(body, whenCondition, false);

    // getThenBlock() cause an error during commpilation
    // So we first get the builder and then its associated body
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    for (size_t i = 0; i < inputValidRegs.size(); i++)
    {
      Connect(thenBody, inputValidRegs[i].getResult(), oneBitValue);
      // don't generate ports for state edges
      if (rvsdg::is<rvsdg::StateType>(reg_args[i]->Type()))
        continue;
      auto inData = GetSubfield(thenBody, inBundle, "data_" + std::to_string(i));
      Connect(thenBody, inputDataRegs[i].getResult(), inData);
    }
  }

  // Output when statement
  auto outReady = GetSubfield(body, outBundle, "ready");
  auto whenCondition = AddAndOp(body, outReady, outValid);
  auto whenOp = AddWhenOp(body, whenCondition, false);
  // getThenBlock() cause an error during commpilation
  // So we first get the builder and then its associated body
  auto thenBody = whenOp.getThenBodyBuilder().getBlock();
  for (size_t i = 0; i < outputValidRegs.size(); i++)
  {
    Connect(thenBody, outputValidRegs[i].getResult(), zeroBitValue);
  }

  // Connect the memory ports
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    auto mem_port = GetPort(module, "mem_" + std::to_string(i));
    auto mem_req = GetSubfield(body, mem_port, "req");
    auto mem_res = GetSubfield(body, mem_port, "res");
    auto inst_req = GetInstancePort(instance, "r" + std::to_string(mem_reqs[i]->index()));
    auto inst_res = GetInstancePort(instance, "a" + std::to_string(mem_resps[i]->index()));
    Connect(body, mem_req, inst_req);
    Connect(body, inst_res, mem_res);
  }

  // Add the module to the body of the circuit
  circuitBody->push_back(module);

  return circuit;
}

/*
  Helper functions
*/

// Returns a PortInfo of ClockType
void
RhlsToFirrtlConverter::AddClockPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports)
{
  struct circt::firrtl::PortInfo port = {
    Builder_->getStringAttr("clk"), circt::firrtl::ClockType::get(Builder_->getContext()),
    circt::firrtl::Direction::In,   {},
    Builder_->getUnknownLoc(),
  };
  ports->push_back(port);
}

// Returns a PortInfo of unsigned IntType with width of 1
void
RhlsToFirrtlConverter::AddResetPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports)
{
  struct circt::firrtl::PortInfo port = {
    Builder_->getStringAttr("reset"), circt::firrtl::IntType::get(Builder_->getContext(), false, 1),
    circt::firrtl::Direction::In,     {},
    Builder_->getUnknownLoc(),
  };
  ports->push_back(port);
}

void
RhlsToFirrtlConverter::AddMemReqPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports)
{
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  ::llvm::SmallVector<BundleElement> memReqElements;
  memReqElements.push_back(GetReadyElement());
  memReqElements.push_back(GetValidElement());
  memReqElements.push_back(BundleElement(
      Builder_->getStringAttr("addr"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, GetPointerSizeInBits())));
  memReqElements.push_back(BundleElement(
      Builder_->getStringAttr("data"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 64)));
  memReqElements.push_back(BundleElement(
      Builder_->getStringAttr("write"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 1)));
  memReqElements.push_back(BundleElement(
      Builder_->getStringAttr("width"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 3)));

  auto memType = circt::firrtl::BundleType::get(Builder_->getContext(), memReqElements);
  struct circt::firrtl::PortInfo memBundle = {
    Builder_->getStringAttr("mem_req"), memType, circt::firrtl::Direction::Out, {},
    Builder_->getUnknownLoc(),
  };
  ports->push_back(memBundle);
}

void
RhlsToFirrtlConverter::AddMemResPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports)
{
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  ::llvm::SmallVector<BundleElement> memResElements;
  memResElements.push_back(GetValidElement());
  memResElements.push_back(BundleElement(
      Builder_->getStringAttr("data"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 64)));

  auto memResType = circt::firrtl::BundleType::get(Builder_->getContext(), memResElements);
  struct circt::firrtl::PortInfo memResBundle = {
    Builder_->getStringAttr("mem_res"), memResType, circt::firrtl::Direction::In, {},
    Builder_->getUnknownLoc(),
  };
  ports->push_back(memResBundle);
}

void
RhlsToFirrtlConverter::AddBundlePort(
    ::llvm::SmallVector<circt::firrtl::PortInfo> * ports,
    circt::firrtl::Direction direction,
    std::string name,
    circt::firrtl::FIRRTLBaseType type)
{
  auto bundleType = GetBundleType(type);
  struct circt::firrtl::PortInfo bundle = {
    Builder_->getStringAttr(name), bundleType, direction, {}, Builder_->getUnknownLoc(),
  };
  ports->push_back(bundle);
}

circt::firrtl::BundleType
RhlsToFirrtlConverter::GetBundleType(const circt::firrtl::FIRRTLBaseType & type)
{
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  ::llvm::SmallVector<BundleElement> elements;
  elements.push_back(this->GetReadyElement());
  elements.push_back(this->GetValidElement());
  elements.push_back(BundleElement(this->Builder_->getStringAttr("data"), false, type));

  return circt::firrtl::BundleType::get(this->Builder_->getContext(), elements);
}

circt::firrtl::SubfieldOp
RhlsToFirrtlConverter::GetSubfield(mlir::Block * body, mlir::Value value, int index)
{
  auto subfield =
      Builder_->create<circt::firrtl::SubfieldOp>(Builder_->getUnknownLoc(), value, index);
  body->push_back(subfield);
  return subfield;
}

circt::firrtl::SubfieldOp
RhlsToFirrtlConverter::GetSubfield(
    mlir::Block * body,
    mlir::Value value,
    ::llvm::StringRef fieldName)
{
  auto subfield =
      Builder_->create<circt::firrtl::SubfieldOp>(Builder_->getUnknownLoc(), value, fieldName);
  body->push_back(subfield);
  return subfield;
}

mlir::BlockArgument
RhlsToFirrtlConverter::GetPort(circt::firrtl::FModuleOp & module, std::string portName)
{
  for (size_t i = 0; i < module.getNumPorts(); ++i)
  {
    if (module.getPortName(i) == portName)
    {
      return module.getArgument(i);
    }
  }
  llvm_unreachable("port not found");
}

mlir::OpResult
RhlsToFirrtlConverter::GetInstancePort(circt::firrtl::InstanceOp & instance, std::string portName)
{
  for (size_t i = 0; i < instance.getNumResults(); ++i)
  {
    //        std::cout << instance.getPortName(i).str() << std::endl;
    if (instance.getPortName(i) == portName)
    {
      return instance->getResult(i);
    }
  }
  llvm_unreachable("port not found");
}

mlir::BlockArgument
RhlsToFirrtlConverter::GetInPort(circt::firrtl::FModuleOp & module, size_t portNr)
{
  return GetPort(module, "i" + std::to_string(portNr));
}

mlir::BlockArgument
RhlsToFirrtlConverter::GetOutPort(circt::firrtl::FModuleOp & module, size_t portNr)
{
  return GetPort(module, "o" + std::to_string(portNr));
}

void
RhlsToFirrtlConverter::Connect(mlir::Block * body, mlir::Value sink, mlir::Value source)
{
  body->push_back(
      Builder_->create<circt::firrtl::ConnectOp>(Builder_->getUnknownLoc(), sink, source));
}

circt::firrtl::BitsPrimOp
RhlsToFirrtlConverter::AddBitsOp(mlir::Block * body, mlir::Value value, int high, int low)
{
  auto intType = Builder_->getIntegerType(32);
  auto op = Builder_->create<circt::firrtl::BitsPrimOp>(
      Builder_->getUnknownLoc(),
      value,
      Builder_->getIntegerAttr(intType, high),
      Builder_->getIntegerAttr(intType, low));
  body->push_back(op);
  return op;
}

circt::firrtl::AndPrimOp
RhlsToFirrtlConverter::AddAndOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::AndPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::NodeOp
RhlsToFirrtlConverter::AddNodeOp(mlir::Block * body, mlir::Value value, std::string name)
{
  auto op = Builder_->create<circt::firrtl::NodeOp>(Builder_->getUnknownLoc(), value, name);
  body->push_back(op);
  return op;
}

circt::firrtl::XorPrimOp
RhlsToFirrtlConverter::AddXorOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::XorPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::OrPrimOp
RhlsToFirrtlConverter::AddOrOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::OrPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::NotPrimOp
RhlsToFirrtlConverter::AddNotOp(mlir::Block * body, mlir::Value first)
{
  auto op = Builder_->create<circt::firrtl::NotPrimOp>(Builder_->getUnknownLoc(), first);
  body->push_back(op);
  return op;
}

circt::firrtl::AddPrimOp
RhlsToFirrtlConverter::AddAddOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::AddPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::SubPrimOp
RhlsToFirrtlConverter::AddSubOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::SubPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::MulPrimOp
RhlsToFirrtlConverter::AddMulOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::MulPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::DivPrimOp
RhlsToFirrtlConverter::AddDivOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::DivPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::DShrPrimOp
RhlsToFirrtlConverter::AddDShrOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::DShrPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::DShlPrimOp
RhlsToFirrtlConverter::AddDShlOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::DShlPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::RemPrimOp
RhlsToFirrtlConverter::AddRemOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::RemPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::EQPrimOp
RhlsToFirrtlConverter::AddEqOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::EQPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::NEQPrimOp
RhlsToFirrtlConverter::AddNeqOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::NEQPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::GTPrimOp
RhlsToFirrtlConverter::AddGtOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::GTPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::GEQPrimOp
RhlsToFirrtlConverter::AddGeqOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::GEQPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::LTPrimOp
RhlsToFirrtlConverter::AddLtOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::LTPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::LEQPrimOp
RhlsToFirrtlConverter::AddLeqOp(mlir::Block * body, mlir::Value first, mlir::Value second)
{
  auto op = Builder_->create<circt::firrtl::LEQPrimOp>(Builder_->getUnknownLoc(), first, second);
  body->push_back(op);
  return op;
}

circt::firrtl::MuxPrimOp
RhlsToFirrtlConverter::AddMuxOp(
    mlir::Block * body,
    mlir::Value select,
    mlir::Value high,
    mlir::Value low)
{
  auto op =
      Builder_->create<circt::firrtl::MuxPrimOp>(Builder_->getUnknownLoc(), select, high, low);
  body->push_back(op);
  return op;
}

circt::firrtl::AsSIntPrimOp
RhlsToFirrtlConverter::AddAsSIntOp(mlir::Block * body, mlir::Value value)
{
  auto op = Builder_->create<circt::firrtl::AsSIntPrimOp>(Builder_->getUnknownLoc(), value);
  body->push_back(op);
  return op;
}

circt::firrtl::AsUIntPrimOp
RhlsToFirrtlConverter::AddAsUIntOp(mlir::Block * body, mlir::Value value)
{
  auto op = Builder_->create<circt::firrtl::AsUIntPrimOp>(Builder_->getUnknownLoc(), value);
  body->push_back(op);
  return op;
}

circt::firrtl::PadPrimOp
RhlsToFirrtlConverter::AddPadOp(mlir::Block * body, mlir::Value value, int amount)
{
  auto op = Builder_->create<circt::firrtl::PadPrimOp>(Builder_->getUnknownLoc(), value, amount);
  body->push_back(op);
  return op;
}

circt::firrtl::CvtPrimOp
RhlsToFirrtlConverter::AddCvtOp(mlir::Block * body, mlir::Value value)
{
  auto op = Builder_->create<circt::firrtl::CvtPrimOp>(Builder_->getUnknownLoc(), value);
  body->push_back(op);
  return op;
}

circt::firrtl::WireOp
RhlsToFirrtlConverter::AddWireOp(mlir::Block * body, std::string name, int size)
{
  auto op =
      Builder_->create<circt::firrtl::WireOp>(Builder_->getUnknownLoc(), GetIntType(size), name);
  body->push_back(op);
  return op;
}

circt::firrtl::WhenOp
RhlsToFirrtlConverter::AddWhenOp(mlir::Block * body, mlir::Value condition, bool elseStatement)
{
  auto op =
      Builder_->create<circt::firrtl::WhenOp>(Builder_->getUnknownLoc(), condition, elseStatement);
  body->push_back(op);
  return op;
}

void
check_may_not_depend_on(
    mlir::Value value,
    ::llvm::SmallPtrSet<mlir::Value, 16> & forbiddenDependencies,
    ::llvm::SmallPtrSet<mlir::Value, 16> & visited)
{
  if (visited.contains(value))
  {
    return;
  }
  visited.insert(value);
  if (forbiddenDependencies.contains(value))
  {
    throw jlm::util::error("forbidden dependency detected");
  }
  auto op = value.getDefiningOp();
  // don't check anything for registers - connects don't count since they don't form combinatorial
  // circuits
  if (mlir::dyn_cast<circt::firrtl::RegResetOp>(op))
  {
    return;
  }
  else if (mlir::dyn_cast<circt::firrtl::RegOp>(op))
  {
    return;
  }
  // check uses because of connects
  for (auto & use : value.getUses())
  {
    auto * user = use.getOwner();
    if (auto connectOp = mlir::dyn_cast<circt::firrtl::ConnectOp>(user))
    {
      if (connectOp.getDest() == value)
      {
        check_may_not_depend_on(connectOp.getSrc(), forbiddenDependencies, visited);
      }
    }
    else
    {
    }
  }
  // stop at port level
  if (mlir::dyn_cast<circt::firrtl::SubfieldOp>(op))
  {
    return;
  }
  JLM_ASSERT(op->getNumResults() == 1);
  for (size_t i = 0; i < op->getNumOperands(); ++i)
  {
    check_may_not_depend_on(op->getOperand(i), forbiddenDependencies, visited);
  }
}

void
check_oValids(
    ::llvm::SmallVector<mlir::Value> & oReadys,
    ::llvm::SmallVector<mlir::Value> & oValids)
{
  ::llvm::SmallPtrSet<mlir::Value, 16> forbiddenDependencies(oReadys.begin(), oReadys.end());
  for (auto oValid : oValids)
  {
    ::llvm::SmallPtrSet<mlir::Value, 16> visited;
    check_may_not_depend_on(oValid, forbiddenDependencies, visited);
  }
}

void
RhlsToFirrtlConverter::check_module(circt::firrtl::FModuleOp & module)
{
  // check if module/node obeys ready/valid semantics at the circuit level

  // compile time: ovalid and odata may not depend on oready
  ::llvm::SmallVector<mlir::Value> oReadys;
  ::llvm::SmallVector<mlir::Value> oValids;
  ::llvm::SmallVector<mlir::Value> oDatas;
  for (size_t i = 0; i < module.getNumPorts(); ++i)
  {
    auto portName = module.getPortName(i);
    auto port = module.getArgument(i);
    if (portName.starts_with("o"))
    {
      // out port
      for (auto & use : port.getUses())
      {
        auto * user = use.getOwner();
        if (auto subfieldOp = mlir::dyn_cast<circt::firrtl::SubfieldOp>(user))
        {
          auto subfieldName =
              subfieldOp.getInput().getType().cast<circt::firrtl::BundleType>().getElementName(
                  subfieldOp.getFieldIndex());
          if (subfieldName == "ready")
          {
            oReadys.push_back(subfieldOp);
          }
          else if (subfieldName == "valid")
          {
            oValids.push_back(subfieldOp);
          }
          else if (subfieldName == "data")
          {
            oDatas.push_back(subfieldOp);
          }
        }
        else
        {
          user->print(::llvm::outs());
          llvm_unreachable("unexpected GetOperation");
        }
      }
    }
  }
  check_oValids(oReadys, oValids);
  check_oValids(oReadys, oDatas);

#ifdef FIRRTL_RUNTIME_ASSERTIONS
  // run time: valid/ready may not go down without firing once they are up - insert assertions
  auto body = &module.getBody().back();
  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  for (size_t i = 0; i < module.getNumPorts(); ++i)
  {
    auto portName = module.getPortName(i);
    auto port = module.getArgument(i);
    if (portName.starts_with("o") || portName.starts_with("i"))
    {
      auto ready = GetSubfield(body, port, "ready");
      auto valid = GetSubfield(body, port, "valid");
      auto data = GetSubfield(body, port, "data");
      if (data.getResult().getType().dyn_cast<circt::firrtl::BundleType>())
      {
        // skip memory ports
        continue;
      }
      auto fire = AddAndOp(body, ready, valid);
      auto prev_ready_reg = Builder_->create<circt::firrtl::RegResetOp>(
          Builder_->getUnknownLoc(),
          GetIntType(1),
          clock,
          reset,
          zeroBitValue,
          std::string(portName) + "_prev_ready_reg");
      body->push_back(prev_ready_reg);
      auto prev_valid_reg = Builder_->create<circt::firrtl::RegResetOp>(
          Builder_->getUnknownLoc(),
          GetIntType(1),
          clock,
          reset,
          zeroBitValue,
          std::string(portName) + "_prev_valid_reg");
      body->push_back(prev_valid_reg);
      auto prev_data_reg = Builder_->create<circt::firrtl::RegOp>(
          Builder_->getUnknownLoc(),
          data.getResult().getType(),
          clock,
          std::string(portName) + "_prev_data_reg");
      body->push_back(prev_data_reg);
      Connect(body, prev_ready_reg.getResult(), ready);
      Connect(body, prev_valid_reg.getResult(), valid);
      Connect(body, prev_data_reg.getResult(), data);
      auto fireBody = &AddWhenOp(body, fire, false).getThenBlock();
      Connect(fireBody, prev_ready_reg.getResult(), zeroBitValue);
      Connect(fireBody, prev_valid_reg.getResult(), zeroBitValue);

      auto valid_assert = Builder_->create<circt::firrtl::AssertOp>(
          Builder_->getUnknownLoc(),
          clock,
          AddNotOp(body, AddAndOp(body, prev_valid_reg.getResult(), AddNotOp(body, valid))),
          AddNotOp(body, reset),
          std::string(portName) + "_valid went down without firing",
          mlir::ValueRange(),
          std::string(portName) + "_valid_assert");
      body->push_back(valid_assert);

      auto ready_assert = Builder_->create<circt::firrtl::AssertOp>(
          Builder_->getUnknownLoc(),
          clock,
          AddNotOp(body, AddAndOp(body, prev_ready_reg.getResult(), AddNotOp(body, ready))),
          AddNotOp(body, reset),
          std::string(portName) + "_ready went down without firing",
          mlir::ValueRange(),
          std::string(portName) + "_ready_assert");
      body->push_back(ready_assert);

      auto data_assert = Builder_->create<circt::firrtl::AssertOp>(
          Builder_->getUnknownLoc(),
          clock,
          AddNotOp(
              body,
              AddAndOp(
                  body,
                  prev_valid_reg.getResult(),
                  AddNeqOp(body, prev_data_reg.getResult(), data))),
          AddNotOp(body, reset),
          std::string(portName) + "_data changed without firing",
          mlir::ValueRange(),
          std::string(portName) + "_data_assert");
      body->push_back(data_assert);
    }
  }
#endif // FIRRTL_RUNTIME_ASSERTIONS
}

circt::firrtl::InstanceOp
RhlsToFirrtlConverter::AddInstanceOp(mlir::Block * circuitBody, jlm::rvsdg::Node * node)
{
  auto name = GetModuleName(node);
  // Check if the module has already been instantiated else we need to generate it
  if (auto sn = dynamic_cast<rvsdg::SimpleNode *>(node))
  {
    if (!modules[name])
    {
      auto module = MlirGen(sn);
      if (circt::isa<circt::firrtl::FModuleOp>(module))
        check_module(circt::cast<circt::firrtl::FModuleOp>(module));
      modules[name] = module;
      circuitBody->push_back(module);
    }
  }
  else
  {
    auto ln = dynamic_cast<loop_node *>(node);
    JLM_ASSERT(ln);
    auto module = MlirGen(ln, circuitBody);
    modules[name] = module;
    circuitBody->push_back(module);
  }
  // We increment a counter for each node that is instantiated
  // to assure the name is unique while still being relatively
  // easy to read (which helps when debugging).
  auto node_name = get_node_name(node);
  return Builder_->create<circt::firrtl::InstanceOp>(
      Builder_->getUnknownLoc(),
      modules[name],
      node_name);
}

circt::firrtl::ConstantOp
RhlsToFirrtlConverter::GetConstant(mlir::Block * body, int size, int value)
{
  auto intType = GetIntType(size);
  auto constant = Builder_->create<circt::firrtl::ConstantOp>(
      Builder_->getUnknownLoc(),
      intType,
      ::llvm::APInt(size, value));
  body->push_back(constant);
  return constant;
}

circt::firrtl::InvalidValueOp
RhlsToFirrtlConverter::GetInvalid(mlir::Block * body, int size)
{

  auto invalid =
      Builder_->create<circt::firrtl::InvalidValueOp>(Builder_->getUnknownLoc(), GetIntType(size));
  body->push_back(invalid);
  return invalid;
}

void
RhlsToFirrtlConverter::ConnectInvalid(mlir::Block * body, mlir::Value value)
{

  auto invalid =
      Builder_->create<circt::firrtl::InvalidValueOp>(Builder_->getUnknownLoc(), value.getType());
  body->push_back(invalid);
  return Connect(body, value, invalid);
}

// Get the clock signal in the module
mlir::BlockArgument
RhlsToFirrtlConverter::GetClockSignal(circt::firrtl::FModuleOp module)
{
  auto clock = module.getArgument(0);
  auto ctype = clock.getType().cast<circt::firrtl::FIRRTLType>();
  if (!ctype.isa<circt::firrtl::ClockType>())
  {
    JLM_ASSERT("Not a ClockType");
  }
  return clock;
}

// Get the reset signal in the module
mlir::BlockArgument
RhlsToFirrtlConverter::GetResetSignal(circt::firrtl::FModuleOp module)
{
  auto reset = module.getArgument(1);
  auto rtype = reset.getType().cast<circt::firrtl::FIRRTLType>();
  if (!rtype.isa<circt::firrtl::ResetType>())
  {
    JLM_ASSERT("Not a ResetType");
  }
  return reset;
}

circt::firrtl::BundleType::BundleElement
RhlsToFirrtlConverter::GetReadyElement()
{
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  return BundleElement(
      Builder_->getStringAttr("ready"),
      true,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 1));
}

circt::firrtl::BundleType::BundleElement
RhlsToFirrtlConverter::GetValidElement()
{
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  return BundleElement(
      Builder_->getStringAttr("valid"),
      false,
      circt::firrtl::IntType::get(Builder_->getContext(), false, 1));
}

void
RhlsToFirrtlConverter::InitializeMemReq(circt::firrtl::FModuleOp module)
{
  mlir::BlockArgument mem = GetPort(module, "mem_req");
  mlir::Block * body = module.getBodyBlock();

  auto zeroBitValue = GetConstant(body, 1, 0);
  auto invalid1 = GetInvalid(body, 1);
  auto invalid3 = GetInvalid(body, 3);
  auto invalidPtr = GetInvalid(body, GetPointerSizeInBits());
  auto invalid64 = GetInvalid(body, 64);

  auto memValid = GetSubfield(body, mem, "valid");
  auto memAddr = GetSubfield(body, mem, "addr");
  auto memData = GetSubfield(body, mem, "data");
  auto memWrite = GetSubfield(body, mem, "write");
  auto memWidth = GetSubfield(body, mem, "width");

  Connect(body, memValid, zeroBitValue);
  Connect(body, memAddr, invalidPtr);
  Connect(body, memData, invalid64);
  Connect(body, memWrite, invalid1);
  Connect(body, memWidth, invalid3);
}

// Takes a jlm::rvsdg::Node and creates a firrtl module with an input
// bundle for each node input and output bundle for each node output
// Returns a circt::firrtl::FModuleOp with an empty body
circt::firrtl::FModuleOp
RhlsToFirrtlConverter::nodeToModule(const jlm::rvsdg::Node * node, bool mem)
{
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;

  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);
  // Input bundle port
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    std::string name("i");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::In,
        name,
        GetFirrtlType(node->input(i)->Type().get()));
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    std::string name("o");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::Out,
        name,
        GetFirrtlType(node->output(i)->Type().get()));
  }

  if (mem)
  {
    AddMemReqPort(&ports);
    AddMemResPort(&ports);
  }

  // Creat a name for the module
  auto nodeName = GetModuleName(node);
  mlir::StringAttr name = Builder_->getStringAttr(nodeName);
  // Create the module
  return Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      name,
      circt::firrtl::ConventionAttr::get(
          Builder_->getContext(),
          circt::firrtl::Convention::Internal),
      ports);
}

//
// HLS only works with wires so all types are represented as unsigned integers
//

// Returns IntType of the specified width
circt::firrtl::IntType
RhlsToFirrtlConverter::GetIntType(int size)
{
  return circt::firrtl::IntType::get(Builder_->getContext(), false, size);
}

// Return unsigned IntType with the bit width specified by the
// jlm::rvsdg::type. The extend argument extends the width of the IntType,
// which is useful for, e.g., additions where the result has to be 1
// larger than the operands to accommodate for the carry.
circt::firrtl::IntType
RhlsToFirrtlConverter::GetIntType(const jlm::rvsdg::Type * type, int extend)
{
  return circt::firrtl::IntType::get(Builder_->getContext(), false, JlmSize(type) + extend);
}

circt::firrtl::FIRRTLBaseType
RhlsToFirrtlConverter::GetFirrtlType(const jlm::rvsdg::Type * type)
{
  if (auto bt = dynamic_cast<const BundleType *>(type))
  {
    using BundleElement = circt::firrtl::BundleType::BundleElement;
    ::llvm::SmallVector<BundleElement> elements;
    for (size_t i = 0; i < bt->elements_.size(); ++i)
    {
      auto t = &bt->elements_.at(i);
      elements.push_back(
          BundleElement(Builder_->getStringAttr(t->first), false, GetFirrtlType(t->second.get())));
    }
    return circt::firrtl::BundleType::get(Builder_->getContext(), elements);
  }
  else
  {
    return GetIntType(type);
  }
}

std::string
RhlsToFirrtlConverter::GetModuleName(const rvsdg::Node * node)
{

  std::string append = "";
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    append.append("_I");
    append.append(std::to_string(JlmSize(node->input(i)->Type().get())));
    append.append("W");
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    append.append("_O");
    append.append(std::to_string(JlmSize(node->output(i)->Type().get())));
    append.append("W");
  }
  if (auto op = dynamic_cast<const llvm::GetElementPtrOperation *>(&node->GetOperation()))
  {
    const jlm::rvsdg::Type * pointeeType = &op->GetPointeeType();
    for (size_t i = 1; i < node->ninputs(); i++)
    {
      int bits = JlmSize(pointeeType);
      if (dynamic_cast<const jlm::rvsdg::bittype *>(pointeeType)
          || dynamic_cast<const llvm::FloatingPointType *>(pointeeType))
      {
        pointeeType = nullptr;
      }
      else if (auto arrayType = dynamic_cast<const llvm::ArrayType *>(pointeeType))
      {
        pointeeType = &arrayType->element_type();
      }
      else if (auto vectorType = dynamic_cast<const llvm::VectorType *>(pointeeType))
      {
        pointeeType = vectorType->Type().get();
      }
      else
      {
        throw std::logic_error(pointeeType->debug_string() + " pointer not implemented!");
      }
      int bytes = bits / 8;
      append.append("_");
      append.append(std::to_string(bytes));
    }
  }
  if (auto op = dynamic_cast<const mem_req_op *>(&node->GetOperation()))
  {
    auto loadTypes = op->GetLoadTypes();
    for (size_t i = 0; i < loadTypes->size(); i++)
    {
      auto loadType = loadTypes->at(i).get();
      int bitWidth = JlmSize(loadType);
      append.append("_");
      append.append(std::to_string(bitWidth));
    }
  }
  if (auto op = dynamic_cast<const local_mem_op *>(&node->GetOperation()))
  {
    append.append("_S");
    append.append(std::to_string(
        std::dynamic_pointer_cast<const llvm::ArrayType>(op->result(0))->nelements()));
    append.append("_L");
    size_t loads = rvsdg::TryGetOwnerNode<rvsdg::Node>(**node->output(0)->begin())->noutputs();
    append.append(std::to_string(loads));
    append.append("_S");
    size_t stores =
        (rvsdg::TryGetOwnerNode<rvsdg::Node>(**node->output(1)->begin())->ninputs() - 1 - loads)
        / 2;
    append.append(std::to_string(stores));
  }
  if (dynamic_cast<const LoopOperation *>(&node->GetOperation()))
  {
    append.append("_");
    append.append(util::strfmt(node));
  }
  auto name = jlm::util::strfmt("op_", node->DebugString() + append);
  // Remove characters that are not valid in firrtl module names
  std::replace_if(name.begin(), name.end(), isForbiddenChar, '_');
  return name;
}

bool
RhlsToFirrtlConverter::IsIdentityMapping(const jlm::rvsdg::match_op & op)
{
  for (const auto & pair : op)
  {
    if (pair.first != pair.second)
      return false;
  }

  return true;
}

// Used for debugging a module by wrapping it in a circuit and writing it to a file
// Node is simply a convenience for generating the circuit name
void
RhlsToFirrtlConverter::WriteModuleToFile(
    const circt::firrtl::FModuleOp fModuleOp,
    const rvsdg::Node * node)
{
  if (!fModuleOp)
    return;

  auto name = GetModuleName(node);
  auto moduleName = Builder_->getStringAttr(name);

  // Adde the fModuleOp to a circuit
  auto circuit = Builder_->create<circt::firrtl::CircuitOp>(Builder_->getUnknownLoc(), moduleName);
  auto body = circuit.getBodyBlock();
  body->push_back(fModuleOp);

  WriteCircuitToFile(circuit, name);
}

// Verifies the circuit and writes the FIRRTL to a file
void
RhlsToFirrtlConverter::WriteCircuitToFile(const circt::firrtl::CircuitOp circuit, std::string name)
{
  // Add the circuit to a top module
  auto module = mlir::ModuleOp::create(Builder_->getUnknownLoc());
  module.push_back(circuit);

  // Verify the module
  if (failed(mlir::verify(module)))
  {
    module.emitError("module verification error");
    throw std::logic_error("Verification of firrtl failed");
  }
  // Print the FIRRTL IR
  module.print(::llvm::outs());

  // Write the module to file
  std::string fileName = name + extension();
  std::error_code EC;
  ::llvm::raw_fd_ostream output(fileName, EC);
  size_t targetLineLength = 100;
  auto status = circt::firrtl::exportFIRFile(module, output, targetLineLength, DefaultFIRVersion_);

  if (status.failed())
  {
    throw jlm::util::error("Exporting of FIRRTL failed");
  }

  output.close();
  std::cout << "\nWritten firrtl to " << fileName << "\n";
}

std::string
RhlsToFirrtlConverter::toString(const circt::firrtl::CircuitOp circuit)
{
  // Add the circuit to a top module
  auto module = mlir::ModuleOp::create(Builder_->getUnknownLoc());
  module.push_back(circuit);

  // Verify the module
  if (failed(mlir::verify(module)))
  {
    module.emitError("module verification error");
    module.print(::llvm::outs());
    throw std::logic_error("Verification of firrtl failed");
  }

  // Export FIRRTL to string
  std::string outputString;
  ::llvm::raw_string_ostream output(outputString);

  size_t targetLineLength = 100;
  auto status = circt::firrtl::exportFIRFile(module, output, targetLineLength, DefaultFIRVersion_);
  if (status.failed())
    throw std::logic_error("Exporting of firrtl failed");

  return outputString;
}

circt::firrtl::FExtModuleOp
RhlsToFirrtlConverter::MlirGenExtModule(const jlm::rvsdg::SimpleNode * node)
{
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;

  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);
  // Input bundle port
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    std::string name("i");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::In,
        name,
        GetFirrtlType(node->input(i)->Type().get()));
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    std::string name("o");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::Out,
        name,
        GetFirrtlType(node->output(i)->Type().get()));
  }

  // Creat a name for the module
  auto nodeName = GetModuleName(node);
  mlir::StringAttr name = Builder_->getStringAttr(nodeName);
  // Create the module
  return Builder_->create<circt::firrtl::FExtModuleOp>(
      Builder_->getUnknownLoc(),
      name,
      circt::firrtl::ConventionAttr::get(
          Builder_->getContext(),
          circt::firrtl::Convention::Internal),
      ports);
}
} // namespace jlm::hls
