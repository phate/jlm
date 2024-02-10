/*
 * Copyright 2021 Magnus Sjalander <work@sjalander.com> and
 * David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp"
#include "jlm/llvm/opt/alias-analyses/Operators.hpp"

namespace jlm::hls
{

// Handles nodes with 2 inputs and 1 output
circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenSimpleNode(const jlm::rvsdg::simple_node * node)
{
  // Only handles nodes with a single output
  if (node->noutputs() != 1)
  {
    throw std::logic_error(node->operation().debug_string() + " has more than 1 output");
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

  if (dynamic_cast<const jlm::rvsdg::bitadd_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddAddOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsub_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddSubOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitand_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddAndOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitxor_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddXorOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitor_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddOrOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitmul_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddMulOp(body, input0, input1);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsdiv_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto divOp = AddDivOp(body, sIntOp0, sIntOp1);
    auto uIntOp = AddAsUIntOp(body, divOp);
    // Connect the op to the output data
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, uIntOp, outSize - 1, 0);
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitshr_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddDShrOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitashr_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto shrOp = AddDShrOp(body, sIntOp0, input1);
    auto uIntOp = AddAsUIntOp(body, shrOp);
    // Connect the op to the output data
    Connect(body, outData, uIntOp);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitshl_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto bitsOp = AddBitsOp(body, input1, 7, 0);
    auto op = AddDShlOp(body, input0, bitsOp);
    int outSize = JlmSize(&node->output(0)->type());
    auto slice = AddBitsOp(body, op, outSize - 1, 0);
    // Connect the op to the output data
    Connect(body, outData, slice);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsmod_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto remOp = AddRemOp(body, sIntOp0, sIntOp1);
    auto uIntOp = AddAsUIntOp(body, remOp);
    Connect(body, outData, uIntOp);
  }
  else if (dynamic_cast<const jlm::rvsdg::biteq_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddEqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitne_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddNeqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsgt_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddGtOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitult_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddLtOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitule_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddLeqOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitugt_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto op = AddGtOp(body, input0, input1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsge_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddGeqOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitsle_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sIntOp0 = AddAsSIntOp(body, input0);
    auto sIntOp1 = AddAsSIntOp(body, input1);
    auto op = AddLeqOp(body, sIntOp0, sIntOp1);
    // Connect the op to the output data
    Connect(body, outData, op);
  }
  else if (dynamic_cast<const llvm::zext_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, input0);
  }
  else if (dynamic_cast<const llvm::trunc_op *>(&(node->operation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    int outSize = JlmSize(&node->output(0)->type());
    Connect(body, outData, AddBitsOp(body, inData, outSize - 1, 0));
  }
  else if (dynamic_cast<const llvm::aa::LambdaExitMemStateOperator *>(&(node->operation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, inData);
  }
  else if (dynamic_cast<const llvm::MemStateMergeOperator *>(&(node->operation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    Connect(body, outData, inData);
  }
  else if (auto op = dynamic_cast<const llvm::sext_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto sintOp = AddAsSIntOp(body, input0);
    auto padOp = AddPadOp(body, sintOp, op->ndstbits());
    auto uintOp = AddAsUIntOp(body, padOp);
    Connect(body, outData, uintOp);
  }
  else if (auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&(node->operation())))
  {
    auto value = op->value();
    auto size = value.nbits();
    // Create a constant of UInt<size>(value) and connect to output data
    auto constant = GetConstant(body, size, value.to_uint());
    Connect(body, outData, constant);
  }
  else if (auto op = dynamic_cast<const jlm::rvsdg::ctlconstant_op *>(&(node->operation())))
  {
    auto value = op->value().alternative();
    auto size = ceil(log2(op->value().nalternatives()));
    auto constant = GetConstant(body, size, value);
    Connect(body, outData, constant);
  }
  else if (dynamic_cast<const jlm::rvsdg::bitslt_op *>(&(node->operation())))
  {
    auto input0 = GetSubfield(body, inBundles[0], "data");
    auto input1 = GetSubfield(body, inBundles[1], "data");
    auto sInt0 = AddAsSIntOp(body, input0);
    auto sInt1 = AddAsSIntOp(body, input1);
    auto op = AddLtOp(body, sInt0, sInt1);
    Connect(body, outData, op);
  }
  else if (auto op = dynamic_cast<const jlm::rvsdg::match_op *>(&(node->operation())))
  {
    auto inData = GetSubfield(body, inBundles[0], "data");
    auto outData = GetSubfield(body, outBundle, "data");
    int inSize = JlmSize(&node->input(0)->type());
    int outSize = JlmSize(&node->output(0)->type());
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
  else if (auto op = dynamic_cast<const llvm::GetElementPtrOperation *>(&(node->operation())))
  {
    // Start of with base pointer
    auto input0 = GetSubfield(body, inBundles[0], "data");
    mlir::Value result = AddCvtOp(body, input0);

    // TODO: support structs
    const jlm::rvsdg::type * pointeeType = &op->GetPointeeType();
    for (size_t i = 1; i < node->ninputs(); i++)
    {
      int bits = JlmSize(pointeeType);
      if (dynamic_cast<const jlm::rvsdg::bittype *>(pointeeType))
      {
        ;
      }
      else if (auto arrayType = dynamic_cast<const llvm::arraytype *>(pointeeType))
      {
        pointeeType = &arrayType->element_type();
      }
      else
      {
        throw std::logic_error(pointeeType->debug_string() + " pointer not implemented!");
      }
      // GEP inputs are signed
      auto input = GetSubfield(body, inBundles[i], "data");
      auto asSInt = AddAsSIntOp(body, input);
      int bytes = bits / 8;
      auto constantOp = GetConstant(body, 64, bytes);
      auto cvtOp = AddCvtOp(body, constantOp);
      auto offset = AddMulOp(body, asSInt, cvtOp);
      result = AddAddOp(body, result, offset);
    }
    auto asUInt = AddAsUIntOp(body, result);
    Connect(body, outData, AddBitsOp(body, asUInt, 63, 0));
  }
  else if (dynamic_cast<const llvm::UndefValueOperation *>(&(node->operation())))
  {
    Connect(body, outData, GetConstant(body, 1, 0));
  }
  else
  {
    throw std::logic_error("Simple node " + node->operation().debug_string() + " not implemented!");
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
RhlsToFirrtlConverter::MlirGenSink(const jlm::rvsdg::simple_node * node)
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
RhlsToFirrtlConverter::MlirGenFork(const jlm::rvsdg::simple_node * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  // Input signals
  auto inBundle = GetInPort(module, 0);
  auto inReady = GetSubfield(body, inBundle, "ready");
  auto inValid = GetSubfield(body, inBundle, "valid");
  auto inData = GetSubfield(body, inBundle, "data");

  //
  // Output registers
  //
  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  ::llvm::SmallVector<circt::firrtl::RegResetOp> firedRegs;
  ::llvm::SmallVector<circt::firrtl::AndPrimOp> whenConditions;
  auto oneBitValue = GetConstant(body, 1, 1);
  auto zeroBitValue = GetConstant(body, 1, 0);
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
    auto port = GetOutPort(module, i);
    auto portReady = GetSubfield(body, port, "ready");
    auto portValid = GetSubfield(body, port, "valid");
    auto portData = GetSubfield(body, port, "data");

    auto notFiredReg = AddNotOp(body, firedReg.getResult());
    auto andOp = AddAndOp(body, inValid, notFiredReg);
    Connect(body, portValid, andOp);
    Connect(body, portData, inData);

    auto orOp = AddOrOp(body, portReady, firedReg.getResult());
    allFired = AddAndOp(body, allFired, orOp);

    // Conditions needed for the when statements
    whenConditions.push_back(AddAndOp(body, portReady, portValid));
  }
  allFired = AddNodeOp(body, allFired, "all_fired")->getResult(0);
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

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenMem(const jlm::rvsdg::simple_node * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node, true);
  auto body = module.getBodyBlock();

  // Check if it's a load or store operation
  bool store = dynamic_cast<const llvm::StoreOperation *>(&(node->operation()));

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

    auto zeroValue = GetConstant(body, JlmSize(&node->output(i)->type()), 0);
    std::string dataName("o");
    dataName.append(std::to_string(i));
    dataName.append("_data_reg");
    auto dataReg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(&node->output(i)->type()),
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
    bitWidth = dynamic_cast<const jlm::rvsdg::bittype *>(&node->input(1)->type())->nbits();
  }
  else
  {
    Connect(body, memReqWrite, zeroBitValue);
    auto invalid = GetInvalid(body, 32);
    Connect(body, memReqData, invalid);
    if (auto bitType = dynamic_cast<const jlm::rvsdg::bittype *>(&node->output(0)->type()))
    {
      bitWidth = bitType->nbits();
    }
    else if (dynamic_cast<const llvm::PointerType *>(&node->output(0)->type()))
    {
      bitWidth = 64;
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
RhlsToFirrtlConverter::MlirGenTrigger(const jlm::rvsdg::simple_node * node)
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
RhlsToFirrtlConverter::MlirGenPrint(const jlm::rvsdg::simple_node * node)
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
  auto pn = dynamic_cast<const print_op *>(&node->operation());
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
RhlsToFirrtlConverter::MlirGenPredicationBuffer(const jlm::rvsdg::simple_node * node)
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
      GetIntType(&node->input(0)->type()),
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
RhlsToFirrtlConverter::MlirGenBuffer(const jlm::rvsdg::simple_node * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto op = dynamic_cast<const hls::buffer_op *>(&(node->operation()));
  auto capacity = op->capacity;

  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  auto zeroBitValue = GetConstant(body, 1, 0);
  auto zeroValue = GetConstant(body, JlmSize(&node->input(0)->type()), 0);
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
        GetIntType(&node->input(0)->type()),
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
RhlsToFirrtlConverter::MlirGenDMux(const jlm::rvsdg::simple_node * node)
{
  // Create the module and its input/output ports
  auto module = nodeToModule(node);
  auto body = module.getBodyBlock();

  auto zeroBitValue = GetConstant(body, 1, 0);
  auto oneBitValue = GetConstant(body, 1, 1);

  auto inputs = node->ninputs();
  auto outBundle = GetOutPort(module, 0);
  auto outReady = GetSubfield(body, outBundle, "ready");
  // Out valid
  auto outValid = GetSubfield(body, outBundle, "valid");
  Connect(body, outValid, zeroBitValue);
  // Out data
  auto invalid = GetInvalid(body, JlmSize(&node->output(0)->type()));
  auto outData = GetSubfield(body, outBundle, "data");
  Connect(body, outData, invalid);
  // Input ready 0
  auto inBundle0 = GetInPort(module, 0);
  auto inReady0 = GetSubfield(body, inBundle0, "ready");
  auto inValid0 = GetSubfield(body, inBundle0, "valid");
  auto inData0 = GetSubfield(body, inBundle0, "data");
  Connect(body, inReady0, zeroBitValue);

  // Add discard registers
  auto clock = GetClockSignal(module);
  auto reset = GetResetSignal(module);
  ::llvm::SmallVector<circt::firrtl::RegResetOp> discardRegs;
  ::llvm::SmallVector<circt::firrtl::WireOp> discardWires;
  mlir::Value anyDiscardReg = GetConstant(body, 1, 0);
  for (size_t i = 1; i < inputs; i++)
  {
    std::string regName("i");
    regName.append(std::to_string(i));
    regName.append("_discard_reg");
    auto reg = Builder_->create<circt::firrtl::RegResetOp>(
        Builder_->getUnknownLoc(),
        GetIntType(1),
        clock,
        reset,
        zeroBitValue,
        Builder_->getStringAttr(regName));
    body->push_back(reg);
    discardRegs.push_back(reg);
    anyDiscardReg = AddOrOp(body, anyDiscardReg, reg.getResult());

    std::string wireName("i");
    wireName.append(std::to_string(i));
    wireName.append("_discard");
    auto wire = AddWireOp(body, wireName, 1);
    discardWires.push_back(wire);
    Connect(body, wire.getResult(), reg.getResult());
    Connect(body, reg.getResult(), wire.getResult());
  }
  auto notAnyDiscardReg = AddNotOp(body, anyDiscardReg);

  auto processedReg = Builder_->create<circt::firrtl::RegResetOp>(
      Builder_->getUnknownLoc(),
      GetIntType(1),
      clock,
      reset,
      zeroBitValue,
      Builder_->getStringAttr("processed_reg"));
  body->push_back(processedReg);
  auto notProcessedReg = AddNotOp(body, processedReg.getResult());

  for (size_t i = 1; i < inputs; i++)
  {
    auto inBundle = GetInPort(module, i);
    auto inReady = GetSubfield(body, inBundle, "ready");
    auto inValid = GetSubfield(body, inBundle, "valid");
    auto inData = GetSubfield(body, inBundle, "data");

    Connect(body, inReady, discardWires[i - 1].getResult());

    // First when
    auto andOp = AddAndOp(body, inReady, inValid);
    auto whenOp = AddWhenOp(body, andOp, false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, discardRegs[i - 1].getResult(), zeroBitValue);

    // Second when
    auto constant = GetConstant(body, 64, i - 1);
    auto eqOp = AddEqOp(body, inData0, constant);
    auto andOp0 = AddAndOp(body, inValid0, eqOp);
    auto andOp1 = AddAndOp(body, notAnyDiscardReg.getResult(), andOp0);
    whenOp = AddWhenOp(body, andOp1, false);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, outValid, inValid);
    Connect(thenBody, outData, inData);
    Connect(thenBody, inReady, outReady);
    auto andOp2 = AddAndOp(thenBody, outReady, inValid);
    Connect(thenBody, inReady0, andOp2);

    // Nested when
    whenOp = AddWhenOp(thenBody, notProcessedReg.getResult(), false);
    thenBody = whenOp.getThenBodyBuilder().getBlock();
    Connect(thenBody, processedReg.getResult(), oneBitValue);
    for (size_t j = 1; j < inputs; ++j)
    {
      if (i != j)
      {
        Connect(thenBody, discardWires[j - 1].getResult(), oneBitValue);
      }
    }
  }

  auto andOp = AddAndOp(body, outValid, outReady);
  auto whenOp = AddWhenOp(body, andOp, false);
  auto thenBody = whenOp.getThenBodyBuilder().getBlock();
  Connect(thenBody, processedReg.getResult(), zeroBitValue);

  return module;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGenNDMux(const jlm::rvsdg::simple_node * node)
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
  auto invalid = GetInvalid(body, JlmSize(&node->output(0)->type()));
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
    auto constant = GetConstant(body, JlmSize(&node->input(0)->type()), i - 1);
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
RhlsToFirrtlConverter::MlirGenBranch(const jlm::rvsdg::simple_node * node)
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

    auto constant = GetConstant(body, JlmSize(&node->input(0)->type()), i);
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

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGen(const jlm::rvsdg::simple_node * node)
{
  if (dynamic_cast<const hls::sink_op *>(&(node->operation())))
  {
    return MlirGenSink(node);
  }
  else if (dynamic_cast<const hls::fork_op *>(&(node->operation())))
  {
    return MlirGenFork(node);
  }
  else if (dynamic_cast<const llvm::LoadOperation *>(&(node->operation())))
  {
    return MlirGenMem(node);
  }
  else if (dynamic_cast<const llvm::StoreOperation *>(&(node->operation())))
  {
    return MlirGenMem(node);
  }
  else if (dynamic_cast<const hls::predicate_buffer_op *>(&(node->operation())))
  {
    return MlirGenPredicationBuffer(node);
  }
  else if (dynamic_cast<const hls::buffer_op *>(&(node->operation())))
  {
    return MlirGenBuffer(node);
  }
  else if (dynamic_cast<const hls::branch_op *>(&(node->operation())))
  {
    return MlirGenBranch(node);
  }
  else if (dynamic_cast<const hls::trigger_op *>(&(node->operation())))
  {
    return MlirGenTrigger(node);
  }
  else if (dynamic_cast<const hls::print_op *>(&(node->operation())))
  {
    return MlirGenPrint(node);
  }
  else if (dynamic_cast<const hls::merge_op *>(&(node->operation())))
  {
    // return merge_to_firrtl(n);
    throw std::logic_error(node->operation().debug_string() + " not implemented!");
  }
  else if (auto o = dynamic_cast<const hls::mux_op *>(&(node->operation())))
  {
    if (o->discarding)
    {
      return MlirGenDMux(node);
    }
    else
    {
      return MlirGenNDMux(node);
    }
  }
  return MlirGenSimpleNode(node);
}

std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp>
RhlsToFirrtlConverter::MlirGen(
    hls::loop_node * loopNode,
    mlir::Block * body,
    mlir::Block * circuitBody)
{
  auto subRegion = loopNode->subregion();
  return createInstances(subRegion, circuitBody, body);
}

// Trace the argument back to the "node" generating the value
// Returns the output of a node or the argument of a region that has
// been instantiated as a module
jlm::rvsdg::output *
RhlsToFirrtlConverter::TraceArgument(jlm::rvsdg::argument * arg)
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
      assert(arg->input() != nullptr);
      // Check if we are in a nested region and directly
      // connected to the outer regions argument
      auto origin = arg->input()->origin();
      if (auto o = dynamic_cast<jlm::rvsdg::argument *>(origin))
      {
        // Need to find the source of the outer regions argument
        return TraceArgument(o);
      }
      else if (auto o = dynamic_cast<jlm::rvsdg::structural_output *>(origin))
      {
        // Check if we the input of one loop_node is connected to the output of another
        // structural_node, i.e., if the input is connected to the output of another loop_node
        return TraceStructuralOutput(o);
      }
      // Else we have reached the source
      return origin;
    }
  }
  // Reached the argument of a structural node that is not a hls::loop_node
  return arg;
}

circt::firrtl::FModuleOp
RhlsToFirrtlConverter::MlirGen(jlm::rvsdg::region * subRegion, mlir::Block * circuitBody)
{
  // Generate a vector with all inputs and outputs of the module
  ::llvm::SmallVector<circt::firrtl::PortInfo> ports;

  // Clock and reset ports
  AddClockPort(&ports);
  AddResetPort(&ports);
  // Argument ports
  for (size_t i = 0; i < subRegion->narguments(); ++i)
  {
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::In,
        get_port_name(subRegion->argument(i)),
        GetIntType(&subRegion->argument(i)->type()));
  }
  // Result ports
  for (size_t i = 0; i < subRegion->nresults(); ++i)
  {
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::Out,
        get_port_name(subRegion->result(i)),
        GetIntType(&subRegion->result(i)->type()));
  }
  // Memory ports
  AddMemReqPort(&ports);
  AddMemResPort(&ports);

  // Create a name for the module
  auto moduleName = Builder_->getStringAttr("subregion_mod");
  // Now when we have all the port information we can create the module
  auto module = Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      moduleName,
      circt::firrtl::ConventionAttr::get(Builder_->getContext(), Convention::Internal),
      ports);
  // Get the body of the module such that we can add contents to the module
  auto body = module.getBodyBlock();

  // Initialize the signals of mem_req
  InitializeMemReq(module);

  // First we create and instantiate all the modules and keep them in a dictionary
  std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp> instances =
      createInstances(subRegion, circuitBody, body);

  // Need to keep track of memory operations such that they can be connected
  // to the main memory port.
  //
  // TODO: The use of unorderd_maps for tracking instances maybe can break the
  //       memory order, i.e., not adhear to WAR, RAW, and WAW
  std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp> memInstances;
  // Wire up the instances
  for (const auto & instance : instances)
  {
    // RVSDG node
    auto rvsdgNode = instance.first;
    // Corresponding InstanceOp
    auto sinkNode = instance.second;

    // Memory instances will need to be connected to the main memory ports
    // So we keep track of them to handle them later
    if (dynamic_cast<const llvm::LoadOperation *>(&(rvsdgNode->operation())))
    {
      memInstances.insert(instance);
    }
    else if (dynamic_cast<const llvm::StoreOperation *>(&(rvsdgNode->operation())))
    {
      memInstances.insert(instance);
    }

    // Go through each of the inputs of the RVSDG node and try to connect
    // the corresponding port on the InstanceOp
    for (size_t i = 0; i < rvsdgNode->ninputs(); i++)
    {
      // The port of the instance is connected to another instance

      // Get the RVSDG node that's the origin of this input
      jlm::rvsdg::simple_input * input = rvsdgNode->input(i);
      auto origin = input->origin();
      if (auto o = dynamic_cast<jlm::rvsdg::argument *>(origin))
      {
        origin = TraceArgument(o);
      }
      if (auto o = dynamic_cast<jlm::rvsdg::structural_output *>(origin))
      {
        // Need to trace through the region to find the source node
        origin = TraceStructuralOutput(o);
      }
      // now origin is either a simple_output or a top-level argument
      if (auto o = dynamic_cast<jlm::rvsdg::argument *>(origin))
      {
        // The port of the instance is connected to an argument
        // of the region
        // Calculate the result port of the instance:
        //   2 for clock and reset +
        //   The index of the input of the region
        auto sourceIndex = 2 + o->index();
        auto sourcePort = body->getArgument(sourceIndex);
        auto sinkPort = sinkNode->getResult(i + 2);
        Connect(body, sinkPort, sourcePort);
      }
      else if (auto o = dynamic_cast<jlm::rvsdg::simple_output *>(origin))
      {
        // Get RVSDG node of the source
        auto source = o->node();
        // Calculate the result port of the instance:
        //   2 for clock and reset +
        //   Number of inputs of the node +
        //   The index of the output of the node
        auto sourceIndex = 2 + source->ninputs() + o->index();
        // Get the corresponding InstanceOp
        auto sourceNode = instances[source];
        auto sourcePort = sourceNode->getResult(sourceIndex);
        auto sinkPort = sinkNode->getResult(i + 2);
        Connect(body, sinkPort, sourcePort);
      }
      else
      {
        throw std::logic_error("Unsupported output");
      }
    }
  }

  // Connect memory instances to the main memory ports
  mlir::Value previousGranted = GetConstant(body, 1, 0);
  for (const auto & instance : memInstances)
  {
    // RVSDG node
    auto rvsdgNode = instance.first;
    // Corresponding InstanceOp
    auto node = instance.second;

    // Get the index to the last port of the subregion and the node
    auto mainIndex = body->getArguments().size();
    auto nodeIndex = 2 + rvsdgNode->ninputs() + rvsdgNode->noutputs() - 1;

    // mem_res (last argument of the region and result of the instance)
    auto mainMemRes = body->getArgument(mainIndex - 1);
    auto nodeMemRes = node->getResult(nodeIndex + 2);
    Connect(body, nodeMemRes, mainMemRes);

    // mem_req (second to last argument of the region and result of the instance)
    // The arbitration is prioritized for now so the first memory operation
    // (as given by memInstances) that makes a request will be granted.
    auto mainMemReq = body->getArgument(mainIndex - 2);
    auto nodeMemReq = node->getResult(nodeIndex + 1);
    auto memReqReady = GetSubfield(body, nodeMemReq, "ready");
    Connect(body, memReqReady, GetConstant(body, 1, 0));
    auto memReqValid = GetSubfield(body, nodeMemReq, "valid");
    auto notOp = AddNotOp(body, previousGranted);
    auto condition = AddAndOp(body, notOp, memReqValid);
    auto whenOp = AddWhenOp(body, condition, false);
    auto thenBody = whenOp.getThenBodyBuilder().getBlock();
    // The direction is inverted compared to mem_res
    Connect(thenBody, mainMemReq, nodeMemReq);
    // update for next iteration
    previousGranted = AddOrOp(body, previousGranted, memReqValid);
  }

  // Connect the results of the region
  for (size_t i = 0; i < subRegion->nresults(); i++)
  {
    auto result = subRegion->result(i);
    auto origin = result->origin();
    jlm::rvsdg::simple_output * output;
    if (auto o = dynamic_cast<jlm::rvsdg::simple_output *>(origin))
    {
      // We have found the source output
      output = o;
    }
    else if (auto o = dynamic_cast<jlm::rvsdg::structural_output *>(origin))
    {
      // Need to trace through the region to find the source node
      output = TraceStructuralOutput(o);
    }
    else
    {
      throw std::logic_error("Unsupported output");
    }
    // Get the node of the output
    jlm::rvsdg::simple_node * source = output->node();
    // Get the corresponding InstanceOp
    auto sourceNode = instances[source];
    // Calculate the result port of the instance:
    //   2 for clock and reset +
    //   Number of inputs of the node +
    //   The index of the output of the node
    auto sourceIndex = 2 + source->ninputs() + output->index();
    auto sourcePort = sourceNode->getResult(sourceIndex);

    // Calculate the result port of the region:
    //   2 for clock and reset +
    //   Number of inputs of the region +
    //   The index of the result of the region (== i)
    auto sinkIndex = 2 + subRegion->narguments() + i;
    auto sinkPort = body->getArgument(sinkIndex);

    // Connect the InstanceOp output to the result of the region
    Connect(body, sinkPort, sourcePort);
  }

  return module;
}

std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp>
RhlsToFirrtlConverter::createInstances(
    jlm::rvsdg::region * subRegion,
    mlir::Block * circuitBody,
    mlir::Block * body)
{
  // create and instantiate all the modules and keep them in a dictionary
  auto clock = body->getArgument(0);
  auto reset = body->getArgument(1);
  std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp> instances;
  for (const auto node : jlm::rvsdg::topdown_traverser(subRegion))
  {
    if (auto sn = dynamic_cast<jlm::rvsdg::simple_node *>(node))
    {
      if (dynamic_cast<const hls::local_mem_req_op *>(&(node->operation()))
          || dynamic_cast<const hls::local_mem_resp_op *>(&(node->operation())))
      {
        // these are virtual - connections go to local_mem instead
        continue;
      }
      instances[sn] = AddInstanceOp(circuitBody, sn);
      body->push_back(instances[sn]);
      // Connect clock and reset to the instance
      Connect(body, instances[sn]->getResult(0), clock);
      Connect(body, instances[sn]->getResult(1), reset);
    }
    else if (auto oln = dynamic_cast<loop_node *>(node))
    {
      auto inst = MlirGen(oln, body, circuitBody);
      instances.merge(inst);
    }
    else
    {
      throw jlm::util::error(
          "Unimplemented op (unexpected structural node) : " + node->operation().debug_string());
    }
  }
  return instances;
}

// Trace a structural output back to the "node" generating the value
// Returns the output of the node
jlm::rvsdg::simple_output *
RhlsToFirrtlConverter::TraceStructuralOutput(jlm::rvsdg::structural_output * output)
{
  auto node = output->node();

  // We are only expecting hls::loop_node to have a structural output
  if (!dynamic_cast<hls::loop_node *>(node))
  {
    throw std::logic_error(
        "Expected a hls::loop_node but found: " + node->operation().debug_string());
  }
  assert(output->results.size() == 1);
  auto origin = output->results.begin().ptr()->origin();
  if (auto o = dynamic_cast<jlm::rvsdg::structural_output *>(origin))
  {
    // Need to trace the output of the nested structural node
    return TraceStructuralOutput(o);
  }
  else if (auto o = dynamic_cast<jlm::rvsdg::simple_output *>(origin))
  {
    // Found the source node
    return o;
  }
  else
  {
    throw std::logic_error("Encountered an unexpected output type");
  }
}

// Emit a circuit
circt::firrtl::CircuitOp
RhlsToFirrtlConverter::MlirGen(const llvm::lambda::node * lambdaNode)
{

  // Ensure consistent naming across runs
  create_node_names(lambdaNode->subregion());
  // The same name is used for the circuit and main module
  auto moduleName = Builder_->getStringAttr(lambdaNode->name() + "_lambda_mod");
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

  auto reg_args = get_reg_args(lambdaNode);
  auto reg_results = get_reg_results(lambdaNode);

  // Input bundle
  using BundleElement = circt::firrtl::BundleType::BundleElement;
  ::llvm::SmallVector<BundleElement> inputElements;
  inputElements.push_back(GetReadyElement());
  inputElements.push_back(GetValidElement());

  for (size_t i = 0; i < reg_args.size(); ++i)
  {
    std::string portName("data_");
    portName.append(std::to_string(i));
    inputElements.push_back(
        BundleElement(Builder_->getStringAttr(portName), false, GetIntType(&reg_args[i]->type())));
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
    std::string portName("data_");
    portName.append(std::to_string(i));
    outputElements.push_back(BundleElement(
        Builder_->getStringAttr(portName),
        false,
        GetIntType(&reg_results[i]->type())));
  }
  auto outputType = circt::firrtl::BundleType::get(Builder_->getContext(), outputElements);
  struct circt::firrtl::PortInfo oBundle = {
    Builder_->getStringAttr("o"), outputType, circt::firrtl::Direction::Out, {},
    Builder_->getUnknownLoc(),
  };
  ports.push_back(oBundle);

  // Memory ports
  AddMemReqPort(&ports);
  AddMemResPort(&ports);

  auto mem_reqs = get_mem_reqs(lambdaNode);
  auto mem_resps = get_mem_resps(lambdaNode);
  assert(mem_resps.size() == mem_reqs.size());
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    ::llvm::SmallVector<BundleElement> memElements;

    ::llvm::SmallVector<BundleElement> reqElements;
    reqElements.push_back(GetReadyElement());
    reqElements.push_back(GetValidElement());
    reqElements.push_back(
        BundleElement(Builder_->getStringAttr("data"), false, GetFirrtlType(&mem_reqs[i]->type())));
    auto reqType = circt::firrtl::BundleType::get(Builder_->getContext(), reqElements);
    memElements.push_back(BundleElement(Builder_->getStringAttr("req"), false, reqType));

    ::llvm::SmallVector<BundleElement> resElements;
    resElements.push_back(GetReadyElement());
    resElements.push_back(GetValidElement());
    resElements.push_back(BundleElement(
        Builder_->getStringAttr("data"),
        false,
        GetFirrtlType(&mem_resps[i]->type())));
    auto resType = circt::firrtl::BundleType::get(Builder_->getContext(), resElements);
    memElements.push_back(BundleElement(Builder_->getStringAttr("res"), true, resType));

    auto memType = circt::firrtl::BundleType::get(Builder_->getContext(), memElements);
    struct circt::firrtl::PortInfo memBundle = {
      Builder_->getStringAttr("mem_" + std::to_string(i)),
      memType,
      circt::firrtl::Direction::Out,
      { Builder_->getStringAttr("") },
      Builder_->getUnknownLoc(),
    };
    ports.push_back(memBundle);
  }

  // Now when we have all the port information we can create the module
  // The same name is used for the circuit and main module
  auto module = Builder_->create<circt::firrtl::FModuleOp>(
      Builder_->getUnknownLoc(),
      moduleName,
      circt::firrtl::ConventionAttr::get(Builder_->getContext(), Convention::Internal),
      ports);
  // Get the body of the module such that we can add contents to the module
  auto body = module.getBodyBlock();

  // Initialize the signals of mem_req
  InitializeMemReq(module);

  // Create a module of the region
  auto srModule = MlirGen(subRegion, circuitBody);
  circuitBody->push_back(srModule);
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
        GetIntType(&reg_args[i]->type()),
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
        GetIntType(&reg_results[i]->type()),
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
  auto args = body->getArguments().size();
  auto memResBundle = body->getArgument(args - 1);
  auto memResValid = GetSubfield(body, memResBundle, "valid");
  auto memResData = GetSubfield(body, memResBundle, "data");

  auto memReqBundle = body->getArgument(args - 2);
  auto memReqValid = GetSubfield(body, memReqBundle, "valid");
  auto memReqAddr = GetSubfield(body, memReqBundle, "addr");
  auto memReqData = GetSubfield(body, memReqBundle, "data");
  auto memReqWrite = GetSubfield(body, memReqBundle, "write");
  auto memReqWidth = GetSubfield(body, memReqBundle, "width");

  auto srArgs = instance.getResults().size();
  auto srMemResBundle = instance->getResult(srArgs - 1);
  auto srMemResValid = GetSubfield(body, srMemResBundle, "valid");
  auto srMemResData = GetSubfield(body, srMemResBundle, "data");

  auto srMemReqBundle = instance->getResult(srArgs - 2);
  auto srMemReqReady = GetSubfield(body, srMemReqBundle, "ready");
  auto srMemReqValid = GetSubfield(body, srMemReqBundle, "valid");
  auto srMemReqAddr = GetSubfield(body, srMemReqBundle, "addr");
  auto srMemReqData = GetSubfield(body, srMemReqBundle, "data");
  auto srMemReqWrite = GetSubfield(body, srMemReqBundle, "write");
  auto srMemReqWidth = GetSubfield(body, srMemReqBundle, "width");

  Connect(body, srMemResValid, memResValid);
  Connect(body, srMemResData, memResData);
  Connect(body, srMemReqReady, zeroBitValue);

  // When statement
  whenOp = AddWhenOp(body, srMemReqValid, false);
  // getThenBlock() cause an error during commpilation
  // So we first get the builder and then its associated body
  thenBody = whenOp.getThenBodyBuilder().getBlock();
  Connect(thenBody, srMemReqReady, oneBitValue);
  Connect(thenBody, memReqValid, oneBitValue);
  Connect(thenBody, memReqAddr, srMemReqAddr);
  Connect(thenBody, memReqData, srMemReqData);
  Connect(thenBody, memReqWrite, srMemReqWrite);
  Connect(thenBody, memReqWidth, srMemReqWidth);

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
    Builder_->getStringAttr("clk"),
    circt::firrtl::ClockType::get(Builder_->getContext()),
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
    Builder_->getStringAttr("reset"),
    circt::firrtl::IntType::get(Builder_->getContext(), false, 1),
    circt::firrtl::Direction::In,
    {},
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
      circt::firrtl::IntType::get(Builder_->getContext(), false, 64)));
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
    Builder_->getStringAttr("mem_req"),
    memType, circt::firrtl::Direction::Out,
    {},
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
  using BundleElement = circt::firrtl::BundleType::BundleElement;

  ::llvm::SmallVector<BundleElement> elements;
  elements.push_back(GetReadyElement());
  elements.push_back(GetValidElement());
  elements.push_back(BundleElement(Builder_->getStringAttr("data"), false, type));

  auto bundleType = circt::firrtl::BundleType::get(Builder_->getContext(), elements);
  struct circt::firrtl::PortInfo bundle = {
    Builder_->getStringAttr(name), bundleType, direction, {}, Builder_->getUnknownLoc(),
  };
  ports->push_back(bundle);
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

circt::firrtl::InstanceOp
RhlsToFirrtlConverter::AddInstanceOp(mlir::Block * body, jlm::rvsdg::simple_node * node)
{
  auto name = GetModuleName(node);
  // Check if the module has already been instantiated else we need to generate it
  if (!modules[name])
  {
    auto module = MlirGen(node);
    modules[name] = module;
    body->push_back(module);
  }
  // We increment a counter for each node that is instantiated
  // to assure the name is unique while still being relatively
  // easy to ready (which helps when debugging).
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

// Get the clock signal in the module
mlir::BlockArgument
RhlsToFirrtlConverter::GetClockSignal(circt::firrtl::FModuleOp module)
{
  auto clock = module.getArgument(0);
  auto ctype = clock.getType().cast<circt::firrtl::FIRRTLType>();
  if (!ctype.isa<circt::firrtl::ClockType>())
  {
    assert("Not a ClockType");
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
    assert("Not a ResetType");
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
  auto invalid64 = GetInvalid(body, 64);

  auto memValid = GetSubfield(body, mem, "valid");
  auto memAddr = GetSubfield(body, mem, "addr");
  auto memData = GetSubfield(body, mem, "data");
  auto memWrite = GetSubfield(body, mem, "write");
  auto memWidth = GetSubfield(body, mem, "width");

  Connect(body, memValid, zeroBitValue);
  Connect(body, memAddr, invalid64);
  Connect(body, memData, invalid64);
  Connect(body, memWrite, invalid1);
  Connect(body, memWidth, invalid3);
}

// Takes a jlm::rvsdg::simple_node and creates a firrtl module with an input
// bundle for each node input and output bundle for each node output
// Returns a circt::firrtl::FModuleOp with an empty body
circt::firrtl::FModuleOp
RhlsToFirrtlConverter::nodeToModule(const jlm::rvsdg::simple_node * node, bool mem)
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
        GetFirrtlType(&node->input(i)->type()));
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    std::string name("o");
    name.append(std::to_string(i));
    AddBundlePort(
        &ports,
        circt::firrtl::Direction::Out,
        name,
        GetFirrtlType(&node->output(i)->type()));
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
      circt::firrtl::ConventionAttr::get(Builder_->getContext(), Convention::Internal),
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
RhlsToFirrtlConverter::GetIntType(const jlm::rvsdg::type * type, int extend)
{
  return circt::firrtl::IntType::get(Builder_->getContext(), false, JlmSize(type) + extend);
}

circt::firrtl::FIRRTLBaseType
RhlsToFirrtlConverter::GetFirrtlType(const jlm::rvsdg::type * type)
{
  if (auto bt = dynamic_cast<const bundletype *>(type))
  {
    using BundleElement = circt::firrtl::BundleType::BundleElement;
    ::llvm::SmallVector<BundleElement> elements;
    for (size_t i = 0; i < bt->elements_->size(); ++i)
    {
      auto t = &bt->elements_->at(i);
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
RhlsToFirrtlConverter::GetModuleName(const jlm::rvsdg::node * node)
{

  std::string append = "";
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    append.append("_I");
    append.append(std::to_string(JlmSize(&node->input(i)->type())));
    append.append("W");
  }
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    append.append("_O");
    append.append(std::to_string(JlmSize(&node->output(i)->type())));
    append.append("W");
  }
  if (auto op = dynamic_cast<const llvm::GetElementPtrOperation *>(&node->operation()))
  {
    const jlm::rvsdg::type * pointeeType = &op->GetPointeeType();
    for (size_t i = 1; i < node->ninputs(); i++)
    {
      int bits = JlmSize(pointeeType);
      if (dynamic_cast<const jlm::rvsdg::bittype *>(pointeeType))
      {
        ;
      }
      else if (auto arrayType = dynamic_cast<const llvm::arraytype *>(pointeeType))
      {
        pointeeType = &arrayType->element_type();
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
  if (auto op = dynamic_cast<const mem_req_op *>(&node->operation()))
  {
    for (auto lt : op->load_types)
    {
      auto dataType = lt;
      int bitWidth;
      if (auto bitType = dynamic_cast<const jlm::rvsdg::bittype *>(dataType))
      {
        bitWidth = bitType->nbits();
      }
      else if (dynamic_cast<const llvm::PointerType *>(dataType))
      {
        bitWidth = 64;
      }
      else
      {
        throw jlm::util::error("unknown width for mem request");
      }
      append.append("_");
      append.append(std::to_string(bitWidth));
    }
  }
  if (auto op = dynamic_cast<const local_mem_op *>(&node->operation()))
  {
    append.append("_S");
    append.append(
        std::to_string(dynamic_cast<const llvm::arraytype *>(&op->result(0).type())->nelements()));
    append.append("_L");
    size_t loads = llvm::input_node(*node->output(0)->begin())->noutputs();
    append.append(std::to_string(loads));
    append.append("_S");
    size_t stores = (llvm::input_node(*node->output(1)->begin())->ninputs() - 1 - loads) / 2;
    append.append(std::to_string(stores));
  }
  auto name = jlm::util::strfmt("op_", node->operation().debug_string() + append);
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
    const jlm::rvsdg::node * node)
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
  auto status = circt::firrtl::exportFIRFile(module, output);
  if (status.failed())
    throw jlm::util::error("Exporting of FIRRTL failed");
  output.close();
  std::cout << "\nWritten firrtl to " << fileName << "\n";
}

std::string
RhlsToFirrtlConverter::ToString(llvm::RvsdgModule & rvsdgModule)
{
  auto region = rvsdgModule.Rvsdg().root();
  auto lambdaNode = dynamic_cast<const llvm::lambda::node *>(region->nodes.begin().ptr());
  // Region should consist of a single lambdaNode
  JLM_ASSERT(region->nnodes() == 1 && lambdaNode);
  auto circuit = MlirGen(lambdaNode);

  // Add the circuit to a top module
  auto module = mlir::ModuleOp::create(Builder_->getUnknownLoc());
  module.push_back(circuit);

  // Verify the module
  if (failed(mlir::verify(module)))
  {
    module.emitError("module verification error");
    throw std::logic_error("Verification of firrtl failed");
  }

  // Export FIRRTL to string
  std::string outputString;
  ::llvm::raw_string_ostream output(outputString);
  auto status = circt::firrtl::exportFIRFile(module, output);
  if (status.failed())
    throw std::logic_error("Exporting of firrtl failed");

  return outputString;
}

}
