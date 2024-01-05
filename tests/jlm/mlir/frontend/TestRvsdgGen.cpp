/*
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/backend/mlirgen.hpp>
#include <jlm/mlir/frontend/rvsdggen.hpp>
#include <jlm/rvsdg/traverser.hpp>

static void
TestLambda()
{
#ifdef MLIR_ENABLED

  {
    auto context = std::make_unique<mlir::MLIRContext>();
    // Load the RVSDG dialect
    context->getOrLoadDialect<mlir::rvsdg::RVSDGDialect>();
    // Load the JLM dialect
    context->getOrLoadDialect<mlir::jlm::JLMDialect>();
    // Load the Arith dialect
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto builder = std::make_unique<mlir::OpBuilder>(context.get());

    // Create the MLIR omega node
    mlir::rvsdg::OmegaNode omega =
        builder->create<mlir::rvsdg::OmegaNode>(builder->getUnknownLoc());

    // Create a block for the region as this is currently not done automatically
    mlir::Region & omegaRegion = omega.getRegion();
    mlir::Block * omegaBlock = new mlir::Block;
    omegaRegion.push_back(omegaBlock);

    // Handle function arguments
    ::llvm::SmallVector<mlir::Type> arguments;
    arguments.push_back(builder->getType<::mlir::rvsdg::IOStateEdgeType>());
    arguments.push_back(builder->getType<::mlir::rvsdg::MemStateEdgeType>());
    arguments.push_back(builder->getType<::mlir::rvsdg::LoopStateEdgeType>());
    ::llvm::ArrayRef argumentsArray(arguments);

    // Handle function results
    ::llvm::SmallVector<mlir::Type> results;
    results.push_back(builder->getIntegerType(32));
    results.push_back(builder->getType<::mlir::rvsdg::IOStateEdgeType>());
    results.push_back(builder->getType<::mlir::rvsdg::MemStateEdgeType>());
    results.push_back(builder->getType<::mlir::rvsdg::LoopStateEdgeType>());
    ::llvm::ArrayRef resultsArray(results);

    // LambdaNodes return a LambdaRefType
    ::llvm::SmallVector<mlir::Type> lambdaRef;
    auto refType = builder->getType<::mlir::rvsdg::LambdaRefType>(argumentsArray, resultsArray);
    lambdaRef.push_back(refType);

    // Add function attributes
    ::llvm::SmallVector<mlir::NamedAttribute> attributes;
    auto attributeName = builder->getStringAttr("sym_name");
    auto attributeValue = builder->getStringAttr("test");
    auto symbolName = builder->getNamedAttr(attributeName, attributeValue);
    attributes.push_back(symbolName);
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributesRef(attributes);

    // Add the inputs to the function
    ::llvm::SmallVector<mlir::Value> inputs;

    // Create the lambda node and add it to the region/block it resides in
    auto lambda = builder->create<mlir::rvsdg::LambdaNode>(
        builder->getUnknownLoc(),
        lambdaRef,
        inputs,
        attributesRef);
    omegaBlock->push_back(lambda);

    // Create a block for the region as this is not done automatically
    mlir::Region & lambdaRegion = lambda.getRegion();
    mlir::Block * lambdaBlock = new mlir::Block;
    lambdaRegion.push_back(lambdaBlock);

    // Add arguments to the region
    lambdaBlock->addArgument(
        builder->getType<::mlir::rvsdg::IOStateEdgeType>(),
        builder->getUnknownLoc());
    lambdaBlock->addArgument(
        builder->getType<::mlir::rvsdg::MemStateEdgeType>(),
        builder->getUnknownLoc());
    lambdaBlock->addArgument(
        builder->getType<::mlir::rvsdg::LoopStateEdgeType>(),
        builder->getUnknownLoc());

    auto constOp = builder->create<mlir::arith::ConstantIntOp>(builder->getUnknownLoc(), 1, 32);
    lambdaBlock->push_back(constOp);

    ::llvm::SmallVector<mlir::Value> regionResults;
    regionResults.push_back(constOp);
    regionResults.push_back(lambdaBlock->getArgument(0));
    regionResults.push_back(lambdaBlock->getArgument(1));
    regionResults.push_back(lambdaBlock->getArgument(2));

    // Handle the result of the lambda
    auto lambdaResult =
        builder->create<mlir::rvsdg::LambdaResult>(builder->getUnknownLoc(), regionResults);
    lambdaBlock->push_back(lambdaResult);

    // Handle the result of the omega
    ::llvm::SmallVector<mlir::Value> omegaRegionResults;
    omegaRegionResults.push_back(lambda);
    auto omegaResult =
        builder->create<mlir::rvsdg::OmegaResult>(builder->getUnknownLoc(), omegaRegionResults);
    omegaBlock->push_back(omegaResult);

    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);

    // Convert the MLIR to RVSDG
    jlm::mlirrvsdg::RVSDGGen rvsdggen;
    auto rvsdgModule = rvsdggen.convertMlir(rootBlock);

    auto & graph = rvsdgModule->Rvsdg();
    auto region = graph.root();

    assert(region->nnodes() == 1);
    assert(jlm::rvsdg::region::Contains<jlm::llvm::lambda::operation>(*region, false));
    assert(!jlm::rvsdg::region::Contains<jlm::rvsdg::bitconstant_op>(*region, false));
    assert(jlm::rvsdg::region::Contains<jlm::rvsdg::bitconstant_op>(*region, true));
  }

#endif // MLIR_ENABLED
}

static int
Test()
{
  TestLambda();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/frontend/TestRvsdgGen", Test)
