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
    auto builder = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = builder->create<OmegaNode>(builder->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(builder->getType<IOStateEdgeType>());
    arguments.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(builder->getIntegerType(32));
    results.push_back(builder->getType<IOStateEdgeType>());
    results.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = builder->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = builder->getStringAttr("sym_name");
    auto attributeValue = builder->getStringAttr("test");
    auto symbolName = builder->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        builder->create<LambdaNode>(builder->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(builder->getType<IOStateEdgeType>(), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<MemStateEdgeType>(), builder->getUnknownLoc());

    auto constOp = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 1, 32);
    lambdaBlock->push_back(constOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(constOp);
    regionResults.push_back(lambdaBlock->getArgument(0));
    regionResults.push_back(lambdaBlock->getArgument(1));

    // Handle the result of the lambda
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = builder->create<LambdaResult>(builder->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = builder->create<OmegaResult>(builder->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = rvsdgModule->Rvsdg().root();
    {
      using namespace jlm::rvsdg;
      std::cout << "Checking the result" << std::endl;

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::llvm::lambda::node>(region->nodes.first());
      assert(is<jlm::llvm::lambda::operation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<bitconstant_op>(convertedLambda->subregion()->nodes.first()));
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
    auto builder = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = builder->create<OmegaNode>(builder->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(builder->getIntegerType(32));
    arguments.push_back(builder->getType<IOStateEdgeType>());
    arguments.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(builder->getIntegerType(32));
    results.push_back(builder->getType<IOStateEdgeType>());
    results.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = builder->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = builder->getStringAttr("sym_name");
    auto attributeValue = builder->getStringAttr("test");
    auto symbolName = builder->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        builder->create<LambdaNode>(builder->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(builder->getIntegerType(32), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<IOStateEdgeType>(), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<MemStateEdgeType>(), builder->getUnknownLoc());

    // ConstOp1 is not connected to anything
    auto constOp1 = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 20, 32);
    lambdaBlock->push_back(constOp1);

    // ConstOp2 is connected as second argument of the divide operation
    auto constOp2 = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 5, 32);
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
    //! because the rvsdg builder has a constant propagation pass
    auto divideOp = builder->create<mlir::arith::DivUIOp>(
        builder->getUnknownLoc(),
        lambdaBlock->getArgument(0),
        constOp2);
    lambdaBlock->push_back(divideOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(divideOp->getResult(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));

    // Handle the result of the lambda
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = builder->create<LambdaResult>(builder->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = builder->create<OmegaResult>(builder->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = rvsdgModule->Rvsdg().root();

    jlm::rvsdg::view(region, stdout);

    {
      using namespace jlm::rvsdg;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::llvm::lambda::node>(region->nodes.first());
      assert(is<jlm::llvm::lambda::operation>(convertedLambda));

      // 2 Constants + 1 DivUIOp
      assert(convertedLambda->subregion()->nnodes() == 3);

      // Traverse the rvsgd graph upwards to check connections
      jlm::rvsdg::node_output * lambdaResultOriginNodeOuput;
      assert(
          lambdaResultOriginNodeOuput = dynamic_cast<jlm::rvsdg::node_output *>(
              convertedLambda->subregion()->result(0)->origin()));
      jlm::rvsdg::node * lambdaResultOriginNode = lambdaResultOriginNodeOuput->node();
      assert(is<bitudiv_op>(lambdaResultOriginNode->operation()));
      assert(lambdaResultOriginNode->ninputs() == 2);

      // Check first input
      jlm::rvsdg::argument * DivInput0;
      assert(
          DivInput0 =
              dynamic_cast<jlm::rvsdg::argument *>(lambdaResultOriginNode->input(0)->origin()));
      assert(dynamic_cast<const bittype *>(&DivInput0->type()));
      assert(dynamic_cast<const bittype *>(&DivInput0->type())->nbits() == 32);

      // Check second input
      jlm::rvsdg::node_output * DivInput1NodeOuput;
      assert(
          DivInput1NodeOuput =
              dynamic_cast<jlm::rvsdg::node_output *>(lambdaResultOriginNode->input(1)->origin()));
      jlm::rvsdg::node * DivInput1Node = DivInput1NodeOuput->node();
      assert(is<bitconstant_op>(DivInput1Node->operation()));
      const jlm::rvsdg::bitconstant_op * DivInput1Constant =
          dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&DivInput1Node->operation());
      assert(DivInput1Constant->value() == 5);
      assert(dynamic_cast<const bittype *>(&DivInput1Constant->result(0).type()));
      assert(dynamic_cast<const bittype *>(&DivInput1Constant->result(0).type())->nbits() == 32);
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
    auto builder = std::make_unique<mlir::OpBuilder>(context.get());

    auto omega = builder->create<OmegaNode>(builder->getUnknownLoc());
    auto & omegaRegion = omega.getRegion();
    auto * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    std::cout << "Creating function arguments" << std::endl;
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(builder->getIntegerType(32));
    arguments.push_back(builder->getType<IOStateEdgeType>());
    arguments.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    std::cout << "Creating function results" << std::endl;
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(builder->getIntegerType(32));
    results.push_back(builder->getType<IOStateEdgeType>());
    results.push_back(builder->getType<MemStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    std::cout << "Creating LambdaRefType" << std::endl;
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = builder->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    std::cout << "Creating function attributes" << std::endl;
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = builder->getStringAttr("sym_name");
    auto attributeValue = builder->getStringAttr("test");
    auto symbolName = builder->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    std::cout << "Creating LambdaNode" << std::endl;
    auto lambda =
        builder->create<LambdaNode>(builder->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    std::cout << "Adding arguments to the region" << std::endl;
    lambdaBlock->addArgument(builder->getIntegerType(32), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<IOStateEdgeType>(), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<MemStateEdgeType>(), builder->getUnknownLoc());

    // ConstOp1 is connected to the second argument of the add operation
    auto constOp1 = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 20, 32);
    lambdaBlock->push_back(constOp1);

    // ConstOp2 is connected as second argument of the compare operation
    auto constOp2 = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 5, 32);
    lambdaBlock->push_back(constOp2);

    //! The divide op has to be connected to a lambda block argument and not only to constants
    //! because the rvsdg builder has a constant propagation pass
    auto AddOp = builder->create<mlir::arith::AddIOp>(
        builder->getUnknownLoc(),
        lambdaBlock->getArgument(0),
        constOp1);
    lambdaBlock->push_back(AddOp);

    auto compOp = builder->create<mlir::arith::CmpIOp>(
        builder->getUnknownLoc(),
        mlir::arith::CmpIPredicate::eq,
        AddOp.getResult(),
        constOp2);
    lambdaBlock->push_back(compOp);

    auto zeroExtOp = builder->create<mlir::arith::ExtUIOp>(
        builder->getUnknownLoc(),
        builder->getIntegerType(32),
        compOp.getResult());
    lambdaBlock->push_back(zeroExtOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(zeroExtOp->getResult(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));

    // Handle the result of the lambda
    std::cout << "Creating LambdaResult" << std::endl;
    auto lambdaResult = builder->create<LambdaResult>(builder->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    std::cout << "Creating OmegaResult" << std::endl;
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = builder->create<OmegaResult>(builder->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = rvsdgModule->Rvsdg().root();

    {
      using namespace jlm::rvsdg;

      std::cout << "Checking the result" << std::endl;

      assert(region->nnodes() == 1);

      // Get the lambda block
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::llvm::lambda::node>(region->nodes.first());
      assert(is<jlm::llvm::lambda::operation>(convertedLambda));

      // 2 Constants + AddOp + CompOp + ZeroExtOp
      assert(convertedLambda->subregion()->nnodes() == 5);

      // Traverse the rvsgd graph upwards to check connections
      std::cout << "Testing lambdaResultOriginNodeOuput\n";
      jlm::rvsdg::node_output * lambdaResultOriginNodeOuput;
      assert(
          lambdaResultOriginNodeOuput = dynamic_cast<jlm::rvsdg::node_output *>(
              convertedLambda->subregion()->result(0)->origin()));
      jlm::rvsdg::node * ZExtNode = lambdaResultOriginNodeOuput->node();
      assert(is<jlm::llvm::zext_op>(ZExtNode->operation()));
      assert(ZExtNode->ninputs() == 1);

      // Check ZExt
      const jlm::llvm::zext_op * ZExtOp =
          dynamic_cast<const jlm::llvm::zext_op *>(&ZExtNode->operation());
      assert(ZExtOp->nsrcbits() == 1);
      assert(ZExtOp->ndstbits() == 32);

      // Check ZExt input
      std::cout << "Testing input 0\n";
      jlm::rvsdg::node_output * ZExtInput0;
      assert(ZExtInput0 = dynamic_cast<jlm::rvsdg::node_output *>(ZExtNode->input(0)->origin()));
      jlm::rvsdg::node * BitEqNode = ZExtInput0->node();
      assert(is<jlm::rvsdg::biteq_op>(BitEqNode->operation()));

      // Check BitEq
      assert(
          dynamic_cast<const jlm::rvsdg::biteq_op *>(&BitEqNode->operation())->type().nbits()
          == 32);
      assert(BitEqNode->ninputs() == 2);

      // Check BitEq input 0
      jlm::rvsdg::node_output * AddOuput;
      assert(AddOuput = dynamic_cast<jlm::rvsdg::node_output *>(BitEqNode->input(0)->origin()));
      jlm::rvsdg::node * AddNode = AddOuput->node();
      assert(is<bitadd_op>(AddNode->operation()));
      assert(AddNode->ninputs() == 2);

      // Check BitEq input 1
      jlm::rvsdg::node_output * Const2Ouput;
      assert(Const2Ouput = dynamic_cast<jlm::rvsdg::node_output *>(BitEqNode->input(1)->origin()));
      jlm::rvsdg::node * Const2Node = Const2Ouput->node();
      assert(is<bitconstant_op>(Const2Node->operation()));

      // Check Const2
      const jlm::rvsdg::bitconstant_op * Const2Op =
          dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&Const2Node->operation());
      assert(Const2Op->value() == 5);
      assert(dynamic_cast<const bittype *>(&Const2Op->result(0).type()));
      assert(dynamic_cast<const bittype *>(&Const2Op->result(0).type())->nbits() == 32);

      // Check add op
      const jlm::rvsdg::bitadd_op * AddOp =
          dynamic_cast<const jlm::rvsdg::bitadd_op *>(&AddNode->operation());
      assert(AddOp->type().nbits() == 32);

      // Check add input0
      jlm::rvsdg::argument * AddInput0;
      assert(AddInput0 = dynamic_cast<jlm::rvsdg::argument *>(AddNode->input(0)->origin()));
      assert(dynamic_cast<const bittype *>(&AddInput0->type()));
      assert(dynamic_cast<const bittype *>(&AddInput0->type())->nbits() == 32);

      // Check add input1
      jlm::rvsdg::node_output * Const1Output;
      assert(Const1Output = dynamic_cast<jlm::rvsdg::node_output *>(AddNode->input(1)->origin()));
      jlm::rvsdg::node * Const1Node = Const1Output->node();
      assert(is<bitconstant_op>(Const1Node->operation()));

      // Check Const1
      const jlm::rvsdg::bitconstant_op * Const1Op =
          dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&Const1Node->operation());
      assert(Const1Op->value() == 20);
      assert(dynamic_cast<const bittype *>(&Const1Op->result(0).type()));
      assert(dynamic_cast<const bittype *>(&Const1Op->result(0).type())->nbits() == 32);
    }
  }
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgLambdaGen", TestLambda)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgDivOperationGen", TestDivOperation)
JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgCompZeroExtGen", TestCompZeroExt)
