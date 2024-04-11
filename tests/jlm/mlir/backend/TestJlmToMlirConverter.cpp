/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>

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
    std::cout << "Function Setup" << std::endl;
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
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // Lamda + terminating operation
    assert(omegaBlock.getOperations().size() == 2);
    auto & mlirLambda = omegaBlock.front();
    assert(mlirLambda.getName().getStringRef().equals(LambdaNode::getOperationName()));

    // Verify function name
    std::cout << "Verify function name" << std::endl;
    auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
    auto * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);
    auto string = functionName->getValue().str();
    assert(string == "test");

    // Verify function signature
    std::cout << "Verify function signature" << std::endl;
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
    assert(lambdaBlock.front().getName().getStringRef().equals(
        mlir::arith::ConstantIntOp::getOperationName()));

    omega->destroy();
  }
}

/** \brief useChainsUpTraverse
 *
 * This function checks if the given operation matches the given definingOperations use chain
 * recursively. For each operation the operand 0 is checked until the definingOperations is empty.
 *
 * \param operation The starting operation to check. (the lambda result for example)
 * \param succesorOperations The trace of operations to check. The last operation is the direct user
 * of the given operation operand and the first operation is the last operation that will be checked
 * on the chain.
 */
static void
useChainsUpTraverse(mlir::Operation * operation, std::vector<llvm::StringRef> definingOperations)
{
  if (definingOperations.empty())
    return;
  std::cout << "Checking if operation: "
            << operation->getOperand(0).getDefiningOp()->getName().getStringRef().data()
            << " is equal to: " << definingOperations.back().data() << std::endl;
  assert(operation->getOperand(0).getDefiningOp()->getName().getStringRef().equals(
      definingOperations.back()));
  definingOperations.pop_back();
  useChainsUpTraverse(operation->getOperand(0).getDefiningOp(), definingOperations);
}

/** \brief TestAddOperation
 *
 * This test is similar to TestLambda, but it adds a add operation to the
 * lambda block and do a graph traversal.
 * This function is similar to the TestDivOperation function in the frontend tests.
 *
 * This function tests the generation of an add operation using 2 bit constants as operands in the
 * MLIR backend. The test checks the number of blocks and operations in the generated MLIR. It also
 * checks the types of the operations and the users chain upwards from the lambda result to the bit
 * constants. The users trace goes through the operation first operand user recursively to trace the
 * nodes.
 */
static void
TestAddOperation()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
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

    // Create add operation
    std::cout << "Add Operation" << std::endl;
    auto constant1 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto constant2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto add = jlm::rvsdg::bitadd_op::create(32, constant1, constant2);

    lambda->finalize({ add, iOStateArgument, memoryStateArgument, loopStateArgument });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Checking blocks and operations count
    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // Lamda + terminating operation
    assert(omegaBlock.getOperations().size() == 2);

    // Checking lambda block operations
    std::cout << "Checking lambda block operations" << std::endl;
    auto & mlirLambda = omegaBlock.front();
    auto & lambdaRegion = mlirLambda.getRegion(0);
    auto & lambdaBlock = lambdaRegion.front();
    // 2 Bits contants + add + terminating operation
    assert(lambdaBlock.getOperations().size() == 4);

    // Checking lambda block operations types
    std::cout << "Checking lambda block operations types" << std::endl;
    std::vector<mlir::Operation *> operations;
    for (auto & operation : lambdaBlock.getOperations())
    {
      operations.push_back(&operation);
    }

    int constCount = 0;
    for (auto & operation : operations)
    {
      if (operation->getName().getStringRef().equals(mlir::rvsdg::LambdaResult::getOperationName()))
        continue;
      if (operation->getName().getStringRef().equals(
              mlir::arith::ConstantIntOp::getOperationName()))
      {
        constCount++;
        continue;
      }
      // Checking add operation
      std::cout << "Checking add operation" << std::endl;
      assert(operation->getName().getStringRef().equals(
          mlir::LLVM::AddOp::getOperationName())); // Last remaining operation is the add operation
      assert(operation->getNumOperands() == 2);
      auto addOperand1 = operation->getOperand(0);
      auto addOperand2 = operation->getOperand(1);
      assert(addOperand1.getType().isInteger(32));
      assert(addOperand2.getType().isInteger(32));
    }
    assert(constCount == 2);

    useChainsUpTraverse(
        &lambdaBlock.getOperations().back(),
        { mlir::arith::ConstantIntOp::getOperationName(), mlir::LLVM::AddOp::getOperationName() });

    omega->destroy();
  }
}

/** \brief TestAddOperation
 *
 * This test is similar to previous tests, but uses a mul, zero extension
 * and comparison operation, test the types of the operations
 * and do the use chain traversal.
 */
static void
TestComZeroExt()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
        { &iOStateType, &memoryStateType, &loopStateType },
        { &jlm::rvsdg::bit1, &iOStateType, &memoryStateType, &loopStateType });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    // Create add operation
    std::cout << "Add Operation" << std::endl;
    auto constant1 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 8, 4);
    jlm::rvsdg::create_bitconstant(lambda->subregion(), 16, 5); // Unused constant
    jlm::rvsdg::create_bitconstant(lambda->subregion(), 16, 6); // Unused constant

    // zero extension of constant1
    auto zeroExtOp = jlm::llvm::zext_op(8, 16);
    auto zeroExt = jlm::rvsdg::simple_node::create_normalized(
        lambda->subregion(),
        zeroExtOp,
        { constant1 })[0];

    auto mul = jlm::rvsdg::bitmul_op::create(16, zeroExt, zeroExt);

    auto comp = jlm::rvsdg::bitsgt_op::create(16, mul, mul);

    lambda->finalize({ comp, iOStateArgument, memoryStateArgument, loopStateArgument });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Checking blocks and operations count
    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // Lamda + terminating operation
    assert(omegaBlock.getOperations().size() == 2);

    // Checking lambda block operations
    std::cout << "Checking lambda block operations" << std::endl;
    auto & mlirLambda = omegaBlock.front();
    auto & lambdaRegion = mlirLambda.getRegion(0);
    auto & lambdaBlock = lambdaRegion.front();
    // 3 Bits contants + ZeroExt + Mul + Comp + terminating operation
    assert(lambdaBlock.getOperations().size() == 7);

    // Checking lambda block operations types
    std::cout << "Checking lambda block operations types" << std::endl;
    std::vector<mlir::Operation *> operations;
    for (auto & operation : lambdaBlock.getOperations())
    {
      std::cout << "Operation: " << operation.getName().getStringRef().data() << std::endl;
      operations.push_back(&operation);
    }

    int constCount = 0;
    int extCount = 0;
    int mulCount = 0;
    int compCount = 0;
    for (auto & operation : operations)
    {
      if (operation->getName().getStringRef().equals(mlir::rvsdg::LambdaResult::getOperationName()))
        continue;
      if (operation->getName().getStringRef().equals(
              mlir::arith::ConstantIntOp::getOperationName()))
      {
        assert(
            operation->getResult(0).getType().isInteger(8)
            || operation->getResult(0).getType().isInteger(16));
        constCount++;
        continue;
      }
      if (operation->getName().getStringRef().equals(mlir::arith::ExtUIOp::getOperationName()))
      {
        assert(operation->getNumOperands() == 1);
        assert(operation->getOperand(0).getType().isInteger(8));
        assert(operation->getNumResults() == 1);
        assert(operation->getResult(0).getType().isInteger(16));
        extCount++;
        continue;
      }
      if (operation->getName().getStringRef().equals(mlir::arith::MulIOp::getOperationName()))
      {
        assert(operation->getNumOperands() == 2);
        assert(operation->getOperand(0).getType().isInteger(16));
        assert(operation->getOperand(1).getType().isInteger(16));
        assert(operation->getNumResults() == 1);
        assert(operation->getResult(0).getType().isInteger(16));
        mulCount++;
        continue;
      }
      if (operation->getName().getStringRef().equals(mlir::arith::CmpIOp::getOperationName()))
      {
        auto comparisonOp = mlir::cast<mlir::arith::CmpIOp>(operation);
        assert(comparisonOp.getPredicate() == mlir::arith::CmpIPredicate::sgt);
        assert(operation->getNumOperands() == 2);
        assert(operation->getOperand(0).getType().isInteger(16));
        assert(operation->getOperand(1).getType().isInteger(16));
        assert(operation->getNumResults() == 1);
        compCount++;
        continue;
      }
      assert(false);
    }

    // Check counts
    std::cout << "Checking counts" << std::endl;
    assert(constCount == 3);
    assert(extCount == 1);
    assert(mulCount == 1);
    assert(compCount == 1);

    useChainsUpTraverse(
        &lambdaBlock.getOperations().back(),
        { mlir::arith::ConstantIntOp::getOperationName(),
          mlir::arith::ExtUIOp::getOperationName(),
          mlir::arith::MulIOp::getOperationName(),
          mlir::arith::CmpIOp::getOperationName() });

    omega->destroy();
  }
}

static int
Test()
{
  std::cout << "*** Running TestLambda() ***" << std::endl;
  TestLambda();
  std::cout << "*** Running TestAddOperation() ***" << std::endl;
  TestAddOperation();
  std::cout << "*** Running TestComZeroExt() ***" << std::endl;
  TestComZeroExt();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirGen", Test)
