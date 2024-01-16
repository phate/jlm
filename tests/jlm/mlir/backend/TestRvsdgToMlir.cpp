/*
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/backend/RvsdgToMlir.hpp>

static void
TestLambda()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  {
    // Setup the function
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
        { &iOStateType, &memoryStateType, &loopStateType },
        { &jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    lambda->finalize({ constant, iOStateArgument, memoryStateArgument, loopStateArgument });

    // Convert the RVSDG to MLIR
    jlm::rvsdgmlir::RvsdgToMlir mlirgen;
    auto omega = mlirgen.convertModule(*rvsdgModule);

    // Validate the generated MLIR
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // Lamda + terminating operation
    assert(omegaBlock.getOperations().size() == 2);
    auto & mlirLambda = omegaBlock.front();
    assert(mlirLambda.getName().getStringRef() == LambdaNode::getOperationName());

    // Verify function name
    auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
    auto * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);
    auto string = functionName->getValue().str();
    assert(string == "test");

    // Verify function signature
    auto result = mlirLambda.getResult(0).getType();
    assert(result.getTypeID() == LambdaRefType::getTypeID());
    auto * lambdaRefType = static_cast<LambdaRefType *>(&result);
    std::vector<mlir::Type> arguments;
    for (auto argumentType : lambdaRefType->getParameterTypes())
    {
      arguments.push_back(argumentType);
    }
    assert(arguments[0].getTypeID() == IOStateEdgeType::getTypeID());
    assert(arguments[1].getTypeID() == MemStateEdgeType::getTypeID());
    assert(arguments[2].getTypeID() == LoopStateEdgeType::getTypeID());
    std::vector<mlir::Type> results;
    for (auto returnType : lambdaRefType->getReturnTypes())
    {
      results.push_back(returnType);
    }
    assert(results[0].getTypeID() == mlir::IntegerType::getTypeID());
    assert(results[1].getTypeID() == IOStateEdgeType::getTypeID());
    assert(results[2].getTypeID() == MemStateEdgeType::getTypeID());
    assert(results[3].getTypeID() == LoopStateEdgeType::getTypeID());

    auto & lambdaRegion = mlirLambda.getRegion(0);
    auto & lambdaBlock = lambdaRegion.front();
    // Bitconstant + terminating operation
    assert(lambdaBlock.getOperations().size() == 2);
    assert(
        lambdaBlock.front().getName().getStringRef()
        == mlir::arith::ConstantIntOp::getOperationName());

    omega->destroy();
  }
}

static int
Test()
{
  TestLambda();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirGen", Test)
