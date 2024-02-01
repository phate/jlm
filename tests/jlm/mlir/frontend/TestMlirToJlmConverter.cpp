/*
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/traverser.hpp>

static void
TestLambda()
{
  {
    using namespace mlir::rvsdg;
    using namespace mlir::jlm;

    // Setup MLIR Context and load dialects
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
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(builder->getType<IOStateEdgeType>());
    arguments.push_back(builder->getType<MemStateEdgeType>());
    arguments.push_back(builder->getType<LoopStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(builder->getIntegerType(32));
    results.push_back(builder->getType<IOStateEdgeType>());
    results.push_back(builder->getType<MemStateEdgeType>());
    results.push_back(builder->getType<LoopStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = builder->getType<LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = builder->getStringAttr("sym_name");
    auto attributeValue = builder->getStringAttr("test");
    auto symbolName = builder->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    auto lambda =
        builder->create<LambdaNode>(builder->getUnknownLoc(), lambdaRef, inputs, attributesRef);
    omegaBlock->push_back(lambda);
    auto & lambdaRegion = lambda.getRegion();
    auto * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    lambdaBlock->addArgument(builder->getType<IOStateEdgeType>(), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<MemStateEdgeType>(), builder->getUnknownLoc());
    lambdaBlock->addArgument(builder->getType<LoopStateEdgeType>(), builder->getUnknownLoc());

    auto constOp = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 1, 32);
    lambdaBlock->push_back(constOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(constOp);
    regionResults.push_back(lambdaBlock->getArgument(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));

    // Handle the result of the lambda
    auto lambdaResult = builder->create<LambdaResult>(builder->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult = builder->create<OmegaResult>(builder->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG and check the result
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = rvsdgModule->Rvsdg().root();
    {
      using namespace jlm::rvsdg;

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::llvm::lambda::node>(region->nodes.first());
      assert(is<jlm::llvm::lambda::operation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<bitconstant_op>(convertedLambda->subregion()->nodes.first()));
    }
  }
}

static int
Test()
{
  TestLambda();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgGen", Test)
