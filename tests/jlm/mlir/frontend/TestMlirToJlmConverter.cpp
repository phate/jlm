/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/rvsdg/view.hpp>

static int
TestLambda()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(Builder_->getIntegerType(32));
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    auto constOp = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 1, 32);
    lambdaBlock->push_back(constOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(constOp);
    regionResults.push_back(lambdaBlock->getArgument(0));
    regionResults.push_back(lambdaBlock->getArgument(1));

    // Handle the result of the lambda
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::rvsdg;
      std::cout << "Checking the result" << std::endl;

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<bitconstant_op>(convertedLambda->subregion()->Nodes().begin().ptr()));
    }
  }
  return 0;
}

/** \brief TestDivOperation
 *
 * This test is similar to TestLambda, but it adds a division operation to the
 * lambda block and do a graph traversal.
 * This function is similar to the TestAddOperation function in the backend tests.
 */
static int
TestDivOperation()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(Builder_->getIntegerType(32));
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(Builder_->getIntegerType(32));
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(Builder_->getIntegerType(32), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    // ConstOp1 is not connected to anything
    auto constOp1 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 20, 32);
    lambdaBlock->push_back(constOp1);

    // ConstOp2 is connected as second argument of the divide operation
    auto constOp2 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 5, 32);
    lambdaBlock->push_back(constOp2);

    // lambdaBlock->getArguments();
    for (unsigned int i = 0; i < lambdaBlock->getNumArguments(); ++i)
    {
      auto arg = lambdaBlock->getArgument(i);
      if (arg.getType().isa<IOStateEdgeType>())
      {
        std::cout << "Argument " << i << " is an IOStateEdgeType" << std::endl;
      }
      else if (arg.getType().isa<MemStateEdgeType>())
      {
        std::cout << "Argument " << i << " is a MemStateEdgeType" << std::endl;
      }
      else if (arg.getType().isa<mlir::IntegerType>())
      {
        std::cout << "Argument " << i << " is an IntegerType" << std::endl;
      }
    }

    //! The divide op has to be connected to a lambda block argument and not only to constants
    //! because the rvsdg Builder_ has a constant propagation pass
    auto divideOp = Builder_->create<mlir::arith::DivUIOp>(
        Builder_->getUnknownLoc(),
        lambdaBlock->getArgument(0),
        constOp2);
    lambdaBlock->push_back(divideOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(divideOp->getResult(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));

    // Handle the result of the lambda
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    jlm::rvsdg::view(region, stdout);

    {
      using namespace jlm::rvsdg;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      // 2 Constants + 1 DivUIOp
      assert(convertedLambda->subregion()->nnodes() == 3);

      // Traverse the rvsgd graph upwards to check connections
      jlm::rvsdg::node_output * lambdaResultOriginNodeOuput;
      assert(
          lambdaResultOriginNodeOuput = dynamic_cast<jlm::rvsdg::node_output *>(
              convertedLambda->subregion()->result(0)->origin()));
      Node * lambdaResultOriginNode = lambdaResultOriginNodeOuput->node();
      assert(is<bitudiv_op>(lambdaResultOriginNode->GetOperation()));
      assert(lambdaResultOriginNode->ninputs() == 2);

      // Check first input
      jlm::rvsdg::RegionArgument * DivInput0;
      assert(
          DivInput0 = dynamic_cast<jlm::rvsdg::RegionArgument *>(
              lambdaResultOriginNode->input(0)->origin()));
      assert(dynamic_cast<const bittype *>(&DivInput0->type()));
      assert(dynamic_cast<const bittype *>(&DivInput0->type())->nbits() == 32);

      // Check second input
      jlm::rvsdg::node_output * DivInput1NodeOuput;
      assert(
          DivInput1NodeOuput =
              dynamic_cast<jlm::rvsdg::node_output *>(lambdaResultOriginNode->input(1)->origin()));
      Node * DivInput1Node = DivInput1NodeOuput->node();
      assert(is<bitconstant_op>(DivInput1Node->GetOperation()));
      const jlm::rvsdg::bitconstant_op * DivInput1Constant =
          dynamic_cast<const bitconstant_op *>(&DivInput1Node->GetOperation());
      assert(DivInput1Constant->value() == 5);
      assert(is<const bittype>(DivInput1Constant->result(0)));
      assert(std::dynamic_pointer_cast<const bittype>(DivInput1Constant->result(0))->nbits() == 32);
    }
  }
  return 0;
}

/** \brief TestCompZeroExt
 *
 * This test is similar to TestLambda, but it adds an add operation, a comparison operation and a
 * zero extension operation to the lambda block and do a graph traversal check. This function is
 * similar to the TestComZeroExt function in the backend tests.
 *
 */
static int
TestCompZeroExt()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(Builder_->getIntegerType(32));
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(Builder_->getIntegerType(32));
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(Builder_->getIntegerType(32), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    // ConstOp1 is connected to the second argument of the add operation
    auto constOp1 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 20, 32);
    lambdaBlock->push_back(constOp1);

    // ConstOp2 is connected as second argument of the compare operation
    auto constOp2 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 5, 32);
    lambdaBlock->push_back(constOp2);

    //! The divide op has to be connected to a lambda block argument and not only to constants
    //! because the rvsdg Builder_ has a constant propagation pass
    auto AddOp = Builder_->create<mlir::arith::AddIOp>(
        Builder_->getUnknownLoc(),
        lambdaBlock->getArgument(0),
        constOp1);
    lambdaBlock->push_back(AddOp);

    auto compOp = Builder_->create<mlir::arith::CmpIOp>(
        Builder_->getUnknownLoc(),
        mlir::arith::CmpIPredicate::eq,
        AddOp.getResult(),
        constOp2);
    lambdaBlock->push_back(compOp);

    auto zeroExtOp = Builder_->create<mlir::arith::ExtUIOp>(
        Builder_->getUnknownLoc(),
        Builder_->getIntegerType(32),
        compOp.getResult());
    lambdaBlock->push_back(zeroExtOp);

    // Handle the result of the lambda
    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(zeroExtOp->getResult(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::rvsdg;

      std::cout << "Checking the result" << std::endl;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      // 2 Constants + AddOp + CompOp + ZeroExtOp
      assert(convertedLambda->subregion()->nnodes() == 5);

      // Traverse the rvsgd graph upwards to check connections
      std::cout << "Testing lambdaResultOriginNodeOuput\n";
      jlm::rvsdg::node_output * lambdaResultOriginNodeOuput;
      assert(
          lambdaResultOriginNodeOuput = dynamic_cast<jlm::rvsdg::node_output *>(
              convertedLambda->subregion()->result(0)->origin()));
      Node * ZExtNode = lambdaResultOriginNodeOuput->node();
      assert(is<jlm::llvm::zext_op>(ZExtNode->GetOperation()));
      assert(ZExtNode->ninputs() == 1);

      // Check ZExt
      const jlm::llvm::zext_op * ZExtOp =
          dynamic_cast<const jlm::llvm::zext_op *>(&ZExtNode->GetOperation());
      assert(ZExtOp->nsrcbits() == 1);
      assert(ZExtOp->ndstbits() == 32);

      // Check ZExt input
      std::cout << "Testing input 0\n";
      jlm::rvsdg::node_output * ZExtInput0;
      assert(ZExtInput0 = dynamic_cast<jlm::rvsdg::node_output *>(ZExtNode->input(0)->origin()));
      Node * BitEqNode = ZExtInput0->node();
      assert(is<jlm::rvsdg::biteq_op>(BitEqNode->GetOperation()));

      // Check BitEq
      assert(
          dynamic_cast<const jlm::rvsdg::biteq_op *>(&BitEqNode->GetOperation())->type().nbits()
          == 32);
      assert(BitEqNode->ninputs() == 2);

      // Check BitEq input 0
      jlm::rvsdg::node_output * AddOuput;
      assert(AddOuput = dynamic_cast<jlm::rvsdg::node_output *>(BitEqNode->input(0)->origin()));
      Node * AddNode = AddOuput->node();
      assert(is<bitadd_op>(AddNode->GetOperation()));
      assert(AddNode->ninputs() == 2);

      // Check BitEq input 1
      jlm::rvsdg::node_output * Const2Ouput;
      assert(Const2Ouput = dynamic_cast<jlm::rvsdg::node_output *>(BitEqNode->input(1)->origin()));
      Node * Const2Node = Const2Ouput->node();
      assert(is<bitconstant_op>(Const2Node->GetOperation()));

      // Check Const2
      const jlm::rvsdg::bitconstant_op * Const2Op =
          dynamic_cast<const bitconstant_op *>(&Const2Node->GetOperation());
      assert(Const2Op->value() == 5);
      assert(is<const bittype>(Const2Op->result(0)));
      assert(std::dynamic_pointer_cast<const bittype>(Const2Op->result(0))->nbits() == 32);

      // Check add op
      const jlm::rvsdg::bitadd_op * AddOp =
          dynamic_cast<const bitadd_op *>(&AddNode->GetOperation());
      assert(AddOp->type().nbits() == 32);

      // Check add input0
      jlm::rvsdg::RegionArgument * AddInput0;
      assert(AddInput0 = dynamic_cast<jlm::rvsdg::RegionArgument *>(AddNode->input(0)->origin()));
      assert(dynamic_cast<const bittype *>(&AddInput0->type()));
      assert(dynamic_cast<const bittype *>(&AddInput0->type())->nbits() == 32);

      // Check add input1
      jlm::rvsdg::node_output * Const1Output;
      assert(Const1Output = dynamic_cast<jlm::rvsdg::node_output *>(AddNode->input(1)->origin()));
      Node * Const1Node = Const1Output->node();
      assert(is<bitconstant_op>(Const1Node->GetOperation()));

      // Check Const1
      const jlm::rvsdg::bitconstant_op * Const1Op =
          dynamic_cast<const bitconstant_op *>(&Const1Node->GetOperation());
      assert(Const1Op->value() == 20);
      assert(is<const bittype>(Const1Op->result(0)));
      assert(std::dynamic_pointer_cast<const bittype>(Const1Op->result(0))->nbits() == 32);
    }
  }
  return 0;
}

/** \brief TestMatchOp
 *
 * This function tests the Match operation. It creates a lambda block with a Match operation.
 *
 */
static int
TestMatchOp()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(Builder_->getIntegerType(32));
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(::mlir::rvsdg::RVSDG_CTRLType::get(Builder_->getContext(), 4));
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(Builder_->getIntegerType(32), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    ::llvm::SmallVector<::mlir::Attribute> mappingVector;

    mappingVector.push_back(::mlir::rvsdg::MatchRuleAttr::get(
        Builder_->getContext(),
        ::llvm::ArrayRef(static_cast<int64_t>(0)),
        4));
    mappingVector.push_back(::mlir::rvsdg::MatchRuleAttr::get(
        Builder_->getContext(),
        ::llvm::ArrayRef(static_cast<int64_t>(1)),
        5));
    mappingVector.push_back(::mlir::rvsdg::MatchRuleAttr::get(
        Builder_->getContext(),
        ::llvm::ArrayRef(static_cast<int64_t>(1)),
        6));
    //! The default alternative has an empty mapping
    mappingVector.push_back(
        ::mlir::rvsdg::MatchRuleAttr::get(Builder_->getContext(), ::llvm::ArrayRef<int64_t>(), 2));

    auto Match = Builder_->create<::mlir::rvsdg::Match>(
        Builder_->getUnknownLoc(),
        ::mlir::rvsdg::RVSDG_CTRLType::get(
            Builder_->getContext(),
            mappingVector.size()), // Control, ouput type
        // omegaBlock->getArgument(0),                           // input
        lambdaBlock->getArgument(0), // input
        ::mlir::ArrayAttr::get(Builder_->getContext(), ::llvm::ArrayRef(mappingVector)));
    lambdaBlock->push_back(Match);

    // Handle the result of the lambda
    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(Match->getResult(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::rvsdg;

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      auto lambdaRegion = convertedLambda->subregion();

      jlm::rvsdg::node_output * matchOutput;
      assert(
          matchOutput = dynamic_cast<jlm::rvsdg::node_output *>(lambdaRegion->result(0)->origin()));
      Node * matchNode = matchOutput->node();
      assert(is<match_op>(matchNode->GetOperation()));

      auto matchOp = dynamic_cast<const match_op *>(&matchNode->GetOperation());
      assert(matchOp->narguments() == 1);
      assert(is<const bittype>(matchOp->argument(0)));
      assert(std::dynamic_pointer_cast<const bittype>(matchOp->argument(0))->nbits() == 32);

      // 3 alternatives + default
      assert(matchOp->nalternatives() == 4);

      assert(matchOp->default_alternative() == 2);

      for (auto mapping : *matchOp)
      {
        assert(
            (mapping.first == 0 && mapping.second == 4)
            || (mapping.first == 1 && mapping.second == 5)
            || (mapping.first == 1 && mapping.second == 6));
      }
    }
  }
  return 0;
}

/** \brief TestMatchOp
 *
 * This function tests the Gamma operation. It creates a lambda block with a Gamma operation.
 *
 */
static int
TestGammaOp()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(::mlir::rvsdg::RVSDG_CTRLType::get(Builder_->getContext(), 3));
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(Builder_->getIntegerType(32));
    results.push_back(Builder_->getIntegerType(32));
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(
        ::mlir::rvsdg::RVSDG_CTRLType::get(Builder_->getContext(), 3),
        Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    auto entryVar1 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 5, 32);
    lambdaBlock->push_back(entryVar1);
    auto entryVar2 = Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 6, 32);
    lambdaBlock->push_back(entryVar2);

    ::llvm::SmallVector<::mlir::Type> typeRangeOuput;
    typeRangeOuput.push_back(::mlir::IntegerType::get(Builder_->getContext(), 32));
    typeRangeOuput.push_back(::mlir::IntegerType::get(Builder_->getContext(), 32));
    ::mlir::rvsdg::GammaNode gamma = Builder_->create<::mlir::rvsdg::GammaNode>(
        Builder_->getUnknownLoc(),
        ::mlir::TypeRange(::llvm::ArrayRef(typeRangeOuput)), // Ouputs types
        lambdaBlock->getArgument(0),                         // predicate
        ::mlir::ValueRange(::llvm::ArrayRef<::mlir::Value>({ entryVar1, entryVar2 })), // Inputs
        static_cast<unsigned>(3) // regionsCount
    );
    lambdaBlock->push_back(gamma);

    for (size_t i = 0; i < gamma.getNumRegions(); ++i)
    {
      auto & gammaBlock = gamma.getRegion(i).emplaceBlock();
      auto exitvar1 =
          Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), i + 1, 32);
      gammaBlock.push_back(exitvar1);
      auto exitvar2 =
          Builder_->create<mlir::arith::ConstantIntOp>(Builder_->getUnknownLoc(), 10 * (i + 1), 32);
      gammaBlock.push_back(exitvar2);
      auto gammaResult = Builder_->create<::mlir::rvsdg::GammaResult>(
          Builder_->getUnknownLoc(),
          ::llvm::SmallVector<mlir::Value>({ exitvar1, exitvar2 }));
      gammaBlock.push_back(gammaResult);
    }

    // Handle the result of the lambda
    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(gamma->getResult(0));
    regionResults.push_back(gamma->getResult(1));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::rvsdg;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      auto lambdaRegion = convertedLambda->subregion();

      // 2 constants + gamma
      assert(lambdaRegion->nnodes() == 3);

      jlm::rvsdg::node_output * gammaOutput;
      assert(
          gammaOutput = dynamic_cast<jlm::rvsdg::node_output *>(lambdaRegion->result(0)->origin()));
      Node * gammaNode = gammaOutput->node();
      assert(is<GammaOperation>(gammaNode->GetOperation()));

      std::cout << "Checking gamma operation" << std::endl;
      auto gammaOp = dynamic_cast<const GammaOperation *>(&gammaNode->GetOperation());
      assert(gammaNode->ninputs() == 3);
      assert(gammaOp->nalternatives() == 3);
      assert(gammaNode->noutputs() == 2);
    }
  }
  return 0;
}

/** \brief TestThetaOp
 *
 * This function tests the Theta operation. It creates a lambda block with a Theta operation.
 *
 */
static int
TestThetaOp()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
    std::cout << "Creating MLIR context" << std::endl;
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<RVSDGDialect>();
    context->getOrLoadDialect<JLMDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto Builder_ = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = Builder_->create<OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(Builder_->getType<IOStateEdgeType>());
    arguments.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(Builder_->getType<IOStateEdgeType>());
    results.push_back(Builder_->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = Builder_->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = Builder_->getStringAttr("sym_name");
    auto attributeValue = Builder_->getStringAttr("test");
    auto symbolName = Builder_->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        Builder_->create<LambdaNode>(Builder_->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(Builder_->getType<IOStateEdgeType>(), Builder_->getUnknownLoc());
    lambdaBlock->addArgument(Builder_->getType<MemStateEdgeType>(), Builder_->getUnknownLoc());

    ::llvm::SmallVector<::mlir::NamedAttribute> thetaAttributes;
    ::llvm::SmallVector<::mlir::Type> typeRangeOuput;
    typeRangeOuput.push_back(Builder_->getType<IOStateEdgeType>());
    typeRangeOuput.push_back(Builder_->getType<MemStateEdgeType>());
    ::mlir::rvsdg::ThetaNode theta = Builder_->create<::mlir::rvsdg::ThetaNode>(
        Builder_->getUnknownLoc(),
        ::mlir::TypeRange(::llvm::ArrayRef(typeRangeOuput)), // Ouputs types
        ::mlir::ValueRange(::llvm::ArrayRef<::mlir::Value>(
            { lambdaBlock->getArgument(0), lambdaBlock->getArgument(1) })), // Inputs
        thetaAttributes);
    lambdaBlock->push_back(theta);

    auto & thetaBlock = theta.getRegion().emplaceBlock();
    auto predicate = Builder_->create<mlir::rvsdg::ConstantCtrl>(
        Builder_->getUnknownLoc(),
        Builder_->getType<::mlir::rvsdg::RVSDG_CTRLType>(2),
        0);
    thetaBlock.push_back(predicate);

    auto thetaResult = Builder_->create<::mlir::rvsdg::ThetaResult>(
        Builder_->getUnknownLoc(),
        predicate,
        ::llvm::SmallVector<mlir::Value>(theta.getInputs()));
    thetaBlock.push_back(thetaResult);

    // Handle the result of the lambda
    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(theta->getResult(0));
    regionResults.push_back(theta->getResult(1));
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = Builder_->create<LambdaResult>(Builder_->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = Builder_->create<OmegaResult>(Builder_->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::rvsdg;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::llvm::LlvmLambdaOperation>(convertedLambda));

      auto lambdaRegion = convertedLambda->subregion();

      // Just the theta node
      assert(lambdaRegion->nnodes() == 1);

      jlm::rvsdg::node_output * thetaOutput;
      assert(
          thetaOutput = dynamic_cast<jlm::rvsdg::node_output *>(lambdaRegion->result(0)->origin()));
      Node * node = thetaOutput->node();
      assert(is<ThetaOperation>(node->GetOperation()));
      auto thetaNode = dynamic_cast<const jlm::rvsdg::ThetaNode *>(node);

      std::cout << "Checking theta node" << std::endl;
      assert(thetaNode->ninputs() == 2);
      assert(thetaNode->GetLoopVars().size() == 2);
      assert(thetaNode->noutputs() == 2);
      assert(thetaNode->nsubregions() == 1);
      assert(is<jlm::rvsdg::ControlType>(thetaNode->predicate()->type()));
      auto predicateType =
          dynamic_cast<const jlm::rvsdg::ControlType *>(&thetaNode->predicate()->type());
      assert(predicateType->nalternatives() == 2);
      std::cout << predicate.getValue() << std::endl;
    }
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgLambdaGen", TestLambda)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgDivOperationGen", TestDivOperation)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgCompZeroExtGen", TestCompZeroExt)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestMatchGen", TestMatchOp)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestGammaGen", TestGammaOp)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestThetaGen", TestThetaOp)
