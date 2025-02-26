/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2024 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>

static int
TestLambda()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    lambda->finalize({ constant, iOStateArgument, memoryStateArgument });

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
    assert(mlirLambda.getName().getStringRef().equals(mlir::rvsdg::LambdaNode::getOperationName()));

    // Verify function name
    std::cout << "Verify function name" << std::endl;
    auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
    auto * functionName = static_cast<mlir::StringAttr *>(&functionNameAttribute);
    auto string = functionName->getValue().str();
    assert(string == "test");

    // Verify function signature
    std::cout << "Verify function signature" << std::endl;

    auto result = mlirLambda.getResult(0).getType();
    assert(result.getTypeID() == mlir::LLVM::LLVMPointerType::getTypeID());

    auto lambdaOp = ::mlir::dyn_cast<::mlir::rvsdg::LambdaNode>(&mlirLambda);

    auto lamdbaTerminator = lambdaOp.getRegion().front().getTerminator();
    auto lambdaResult = mlir::dyn_cast<mlir::rvsdg::LambdaResult>(lamdbaTerminator);
    assert(lambdaResult != nullptr);
    lambdaResult->dump();

    std::vector<mlir::Type> arguments;
    for (auto argument : lambdaOp->getRegion(0).getArguments())
    {
      arguments.push_back(argument.getType());
    }
    assert(arguments[0].getTypeID() == IOStateEdgeType::getTypeID());
    assert(arguments[1].getTypeID() == MemStateEdgeType::getTypeID());
    std::vector<mlir::Type> results;
    for (auto returnType : lambdaResult->getOperandTypes())
    {
      results.push_back(returnType);
    }
    assert(results[0].isa<mlir::IntegerType>());
    assert(results[1].isa<mlir::rvsdg::IOStateEdgeType>());
    assert(results[2].isa<mlir::rvsdg::MemStateEdgeType>());

    auto & lambdaRegion = mlirLambda.getRegion(0);
    auto & lambdaBlock = lambdaRegion.front();
    // Bitconstant + terminating operation
    assert(lambdaBlock.getOperations().size() == 2);
    assert(lambdaBlock.front().getName().getStringRef().equals(
        mlir::arith::ConstantIntOp::getOperationName()));

    omega->destroy();
  }
  return 0;
}

/** \brief useChainsUpTraverse
 *
 * This function checks if the given operation matches the given definingOperations use chain
 * recursively. For each operation the operand 0 is checked until the definingOperations is empty.
 *
 * \param operation The starting operation to check. (the lambda result for example)
 * \param definingOperations The trace of operations to check. The last operation is the direct user
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
 * lambda block and does a graph traversal.
 * This function is similar to the TestDivOperation function in the frontend tests.
 *
 * This function tests the generation of an add operation using 2 bit constants as operands in the
 * MLIR backend. The test checks the number of blocks and operations in the generated MLIR. It also
 * checks the types of the operations and the users chain upwards from the lambda result to the bit
 * constants. The users trace goes through the operation first operand user recursively to trace the
 * nodes.
 */
static int
TestAddOperation()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    // Create add operation
    std::cout << "Add Operation" << std::endl;
    auto constant1 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto constant2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto add = jlm::rvsdg::bitadd_op::create(32, constant1, constant2);

    lambda->finalize({ add, iOStateArgument, memoryStateArgument });

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
          mlir::arith::AddIOp::getOperationName())); // Last remaining operation is the add
                                                     // operation
      assert(operation->getNumOperands() == 2);
      auto addOperand1 = operation->getOperand(0);
      auto addOperand2 = operation->getOperand(1);
      assert(addOperand1.getType().isInteger(32));
      assert(addOperand2.getType().isInteger(32));
    }
    assert(constCount == 2);

    useChainsUpTraverse(
        &lambdaBlock.getOperations().back(),
        { mlir::arith::ConstantIntOp::getOperationName(),
          mlir::arith::AddIOp::getOperationName() });

    omega->destroy();
  }
  return 0;
}

/** \brief TestAddOperation
 *
 * This test is similar to previous tests, but uses a mul, zero extension
 * and comparison operation, it tests operations types
 * and does the use chain traversal.
 */
static int
TestComZeroExt()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(1), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    // Create add operation
    std::cout << "Add Operation" << std::endl;
    auto constant1 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 8, 4);
    jlm::rvsdg::create_bitconstant(lambda->subregion(), 16, 5); // Unused constant
    jlm::rvsdg::create_bitconstant(lambda->subregion(), 16, 6); // Unused constant

    // zero extension of constant1
    auto zeroExt = jlm::rvsdg::CreateOpNode<zext_op>({ constant1 }, 8, 16).output(0);

    auto mul = jlm::rvsdg::bitmul_op::create(16, zeroExt, zeroExt);

    auto comp = jlm::rvsdg::bitsgt_op::create(16, mul, mul);

    lambda->finalize({ comp, iOStateArgument, memoryStateArgument });

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
  return 0;
}

/** \brief TestMatch
 *
 * This test is similar to previous tests, but uses a match operation
 */
static int
TestMatch()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Setup the function
    std::cout << "Function Setup" << std::endl;
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::ControlType::Create(2), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    // Create a match operation
    std::cout << "Match Operation" << std::endl;
    auto predicateConst = jlm::rvsdg::create_bitconstant(lambda->subregion(), 8, 4);

    auto match =
        jlm::rvsdg::match_op::Create(*predicateConst, { { 4, 0 }, { 5, 1 }, { 6, 1 } }, 2, 2);

    lambda->finalize({ match, iOStateArgument, memoryStateArgument });

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
    // 1 Bits contants + Match + terminating operation
    assert(lambdaBlock.getOperations().size() == 3);

    bool matchFound = false;
    for (auto & operation : lambdaBlock.getOperations())
    {
      if (mlir::isa<mlir::rvsdg::Match>(operation))
      {
        matchFound = true;
        std::cout << "Checking match operation" << std::endl;
        auto matchOp = mlir::cast<mlir::rvsdg::Match>(operation);

        assert(mlir::isa<mlir::arith::ConstantIntOp>(matchOp.getInput().getDefiningOp()));
        auto constant = mlir::cast<mlir::arith::ConstantIntOp>(matchOp.getInput().getDefiningOp());
        assert(constant.value() == 4);
        assert(constant.getType().isInteger(8));

        auto mapping = matchOp.getMapping();
        mapping.dump();
        // 3 alternatives + default
        assert(mapping.size() == 4);

        // ** region check alternatives *$
        for (auto & attr : mapping)
        {
          assert(attr.isa<::mlir::rvsdg::MatchRuleAttr>());
          auto matchRuleAttr = attr.cast<::mlir::rvsdg::MatchRuleAttr>();
          if (matchRuleAttr.isDefault())
          {
            assert(matchRuleAttr.getIndex() == 2);
            assert(matchRuleAttr.getValues().empty());
            continue;
          }

          assert(matchRuleAttr.getValues().size() == 1);

          const int64_t value = matchRuleAttr.getValues().front();

          assert(
              (matchRuleAttr.getIndex() == 0 && value == 4)
              || (matchRuleAttr.getIndex() == 1 && (value == 5 || value == 6)));
        }
        // ** endregion check alternatives **
      }
    }
    assert(matchFound);

    omega->destroy();
  }
  return 0;
}

/** \brief TestGamma
 *
 * This test is similar to previous tests, but uses a gamma operation
 */
static int
TestGamma()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {

    // Create a gamma operation
    std::cout << "Gamma Operation" << std::endl;
    auto CtrlConstant = jlm::rvsdg::control_constant(&graph->GetRootRegion(), 3, 1);
    auto entryvar1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 5);
    auto entryvar2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 6);
    auto rvsdgGammaNode = jlm::rvsdg::GammaNode::create(
        CtrlConstant, // predicate
        3             // nalternatives
    );

    rvsdgGammaNode->AddEntryVar(entryvar1);
    rvsdgGammaNode->AddEntryVar(entryvar2);

    std::vector<jlm::rvsdg::output *> exitvars1;
    std::vector<jlm::rvsdg::output *> exitvars2;
    for (int i = 0; i < 3; i++)
    {
      exitvars1.push_back(jlm::rvsdg::create_bitconstant(rvsdgGammaNode->subregion(i), 32, i + 1));
      exitvars2.push_back(
          jlm::rvsdg::create_bitconstant(rvsdgGammaNode->subregion(i), 32, 10 * (i + 1)));
    }

    rvsdgGammaNode->AddExitVar(exitvars1);
    rvsdgGammaNode->AddExitVar(exitvars2);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Checking blocks and operations count
    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // 1 control + 2 constants + gamma + terminating operation
    assert(omegaBlock.getOperations().size() == 5);

    bool gammaFound = false;
    for (auto & operation : omegaBlock.getOperations())
    {
      if (mlir::isa<mlir::rvsdg::GammaNode>(operation))
      {
        gammaFound = true;
        std::cout << "Checking gamma operation" << std::endl;
        auto gammaOp = mlir::cast<mlir::rvsdg::GammaNode>(operation);
        assert(gammaOp.getNumRegions() == 3);
        // 1 predicate + 2 entryVars
        assert(gammaOp.getNumOperands() == 3);
        assert(gammaOp.getNumResults() == 2);

        std::cout << "Checking gamma predicate" << std::endl;
        assert(mlir::isa<mlir::rvsdg::ConstantCtrl>(gammaOp.getPredicate().getDefiningOp()));
        auto controlConstant =
            mlir::cast<mlir::rvsdg::ConstantCtrl>(gammaOp.getPredicate().getDefiningOp());
        assert(controlConstant.getValue() == 1);
        assert(mlir::isa<mlir::rvsdg::RVSDG_CTRLType>(controlConstant.getType()));
        auto ctrlType = mlir::cast<mlir::rvsdg::RVSDG_CTRLType>(controlConstant.getType());
        assert(ctrlType.getNumOptions() == 3);

        std::cout << "Checking gamma entryVars" << std::endl;
        //! getInputs() corresponds to the entryVars
        auto entryVars = gammaOp.getInputs();
        assert(entryVars.size() == 2);
        assert(mlir::isa<mlir::arith::ConstantIntOp>(entryVars[0].getDefiningOp()));
        assert(mlir::isa<mlir::arith::ConstantIntOp>(entryVars[1].getDefiningOp()));
        auto entryVar1 = mlir::cast<mlir::arith::ConstantIntOp>(entryVars[0].getDefiningOp());
        auto entryVar2 = mlir::cast<mlir::arith::ConstantIntOp>(entryVars[1].getDefiningOp());
        assert(entryVar1.value() == 5);
        assert(entryVar2.value() == 6);

        std::cout << "Checking gamma subregions" << std::endl;
        for (size_t i = 0; i < gammaOp.getNumRegions(); i++)
        {
          assert(gammaOp.getRegion(i).getBlocks().size() == 1);
          auto & gammaBlock = gammaOp.getRegion(i).front();
          // 2 bit constants + gamma result
          assert(gammaBlock.getOperations().size() == 3);

          std::cout << "Checking gamma exitVars" << std::endl;
          auto gammaResult = gammaBlock.getTerminator();
          assert(mlir::isa<mlir::rvsdg::GammaResult>(gammaResult));
          auto gammaResultOp = mlir::cast<mlir::rvsdg::GammaResult>(gammaResult);
          assert(gammaResultOp.getNumOperands() == 2);
          for (size_t j = 0; j < gammaResultOp.getNumOperands(); j++)
          {
            assert(
                mlir::isa<mlir::arith::ConstantIntOp>(gammaResultOp.getOperand(j).getDefiningOp()));
            auto constant =
                mlir::cast<mlir::arith::ConstantIntOp>(gammaResultOp.getOperand(j).getDefiningOp());
            assert(static_cast<size_t>(constant.value()) == (1 - j) * (i + 1) + 10 * (i + 1) * j);
          }
        }
      }
    }
    assert(gammaFound);
    omega->destroy();
  }
  return 0;
}

/** \brief TestTheta
 *
 * This test is similar to previous tests, but uses a theta operation
 */
static int
TestTheta()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create a theta operation
    std::cout << "Theta Operation" << std::endl;
    auto entryvar1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 5);
    auto entryvar2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 6);
    jlm::rvsdg::ThetaNode * rvsdgThetaNode = jlm::rvsdg::ThetaNode::create(&graph->GetRootRegion());

    auto predicate = jlm::rvsdg::control_constant(rvsdgThetaNode->subregion(), 2, 0);

    rvsdgThetaNode->AddLoopVar(entryvar1);
    rvsdgThetaNode->AddLoopVar(entryvar2);
    rvsdgThetaNode->set_predicate(predicate);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Checking blocks and operations count
    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // 1 theta + 1 predicate + 2 constants
    assert(omegaBlock.getOperations().size() == 4);

    bool thetaFound = false;
    for (auto & operation : omegaBlock.getOperations())
    {
      if (mlir::isa<mlir::rvsdg::ThetaNode>(operation))
      {
        thetaFound = true;
        std::cout << "Checking theta operation" << std::endl;
        auto thetaOp = mlir::cast<mlir::rvsdg::ThetaNode>(operation);
        // 2 loop vars
        assert(thetaOp.getNumOperands() == 2);
        assert(thetaOp.getNumResults() == 2);

        auto & thetaBlock = thetaOp.getRegion().front();
        auto thetaResult = thetaBlock.getTerminator();

        assert(mlir::isa<mlir::rvsdg::ThetaResult>(thetaResult));
        auto thetaResultOp = mlir::cast<mlir::rvsdg::ThetaResult>(thetaResult);

        std::cout << "Checking theta predicate" << std::endl;

        assert(mlir::isa<mlir::rvsdg::ConstantCtrl>(thetaResultOp.getPredicate().getDefiningOp()));
        auto controlConstant =
            mlir::cast<mlir::rvsdg::ConstantCtrl>(thetaResultOp.getPredicate().getDefiningOp());

        assert(controlConstant.getValue() == 0);

        assert(mlir::isa<mlir::rvsdg::RVSDG_CTRLType>(controlConstant.getType()));
        auto ctrlType = mlir::cast<mlir::rvsdg::RVSDG_CTRLType>(controlConstant.getType());
        assert(ctrlType.getNumOptions() == 2);

        std::cout << "Checking theta loop vars" << std::endl;
        //! getInputs() corresponds to the loop vars
        auto loopVars = thetaOp.getInputs();
        assert(loopVars.size() == 2);
        assert(mlir::isa<mlir::arith::ConstantIntOp>(loopVars[0].getDefiningOp()));
        assert(mlir::isa<mlir::arith::ConstantIntOp>(loopVars[1].getDefiningOp()));
        auto loopVar1 = mlir::cast<mlir::arith::ConstantIntOp>(loopVars[0].getDefiningOp());
        auto loopVar2 = mlir::cast<mlir::arith::ConstantIntOp>(loopVars[1].getDefiningOp());
        assert(loopVar1.value() == 5);
        assert(loopVar2.value() == 6);

        // Theta result, constant control predicate
        assert(thetaBlock.getOperations().size() == 2);

        std::cout << "Checking loop exitVars" << std::endl;
        std::cout << thetaResultOp.getNumOperands() << std::endl;

        std::cout << "Checking theta subregion" << std::endl;

        // Two arguments and predicate
        assert(thetaResultOp.getNumOperands() == 3);
      }
    }
    // }
    assert(thetaFound);
    omega->destroy();
  }
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirLambdaGen", TestLambda)
JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirAddOperationGen", TestAddOperation)
JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirComZeroExtGen", TestComZeroExt)
JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirMatchGen", TestMatch)
JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirGammaGen", TestGamma)
JLM_UNIT_TEST_REGISTER("jlm/mlir/backend/TestMlirThetaGen", TestTheta)
