/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

static void
TestArgumentIterators()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  {
    auto functionType = jlm::rvsdg::FunctionType::Create({ vt }, { vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    lambda->finalize({ lambda->GetFunctionArguments()[0] });

    std::vector<const jlm::rvsdg::output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    assert(
        functionArguments.size() == 1 && functionArguments[0] == lambda->GetFunctionArguments()[0]);
  }

  {
    auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

    auto nullaryNode = jlm::tests::create_testop(lambda->subregion(), {}, { vt });

    lambda->finalize({ nullaryNode });

    assert(lambda->GetFunctionArguments().empty());
  }

  {
    auto rvsdgImport = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), vt, "");

    auto functionType = jlm::rvsdg::FunctionType::Create({ vt, vt, vt }, { vt, vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

    auto cv = lambda->AddContextVar(*rvsdgImport).inner;

    lambda->finalize({ lambda->GetFunctionArguments()[0], cv });

    std::vector<const jlm::rvsdg::output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    assert(functionArguments.size() == 3);
    assert(functionArguments[0] == lambda->GetFunctionArguments()[0]);
    assert(functionArguments[1] == lambda->GetFunctionArguments()[1]);
    assert(functionArguments[2] == lambda->GetFunctionArguments()[2]);
  }
}

static void
TestInvalidOperandRegion()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg->GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  auto result = jlm::tests::create_testop(&rvsdg->GetRootRegion(), {}, { vt })[0];

  bool invalidRegionErrorCaught = false;
  try
  {
    lambdaNode->finalize({ result });
  }
  catch (jlm::util::error &)
  {
    invalidRegionErrorCaught = true;
  }

  assert(invalidRegionErrorCaught);
}

/**
 * Test LambdaNode::RemoveLambdaInputsWhere()
 */
static void
TestRemoveLambdaInputsWhere()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  auto lambdaBinder0 = lambdaNode->AddContextVar(*x);
  auto lambdaBinder1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<jlm::tests::test_op>(
                    { lambdaBinder1.inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act & Assert
  // Try to remove lambdaInput1 even though it is used
  auto numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == lambdaBinder1.input->index();
      });
  assert(numRemovedInputs == 0);
  assert(lambdaNode->ninputs() == 3);
  assert(lambdaNode->GetContextVars().size() == 3);

  // Remove lambdaInput2
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == 2;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 2);
  assert(lambdaNode->GetContextVars().size() == 2);
  assert(lambdaNode->input(0) == lambdaBinder0.input);
  assert(lambdaNode->input(1) == lambdaBinder1.input);

  // Remove lambdaInput0
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == 0;
      });
  assert(numRemovedInputs == 1);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(lambdaNode->input(0) == lambdaBinder1.input);
  assert(lambdaBinder1.input->index() == 0);
  assert(lambdaBinder1.inner->index() == 0);
}

/**
 * Test LambdaNode::PruneLambdaInputs()
 */
static void
TestPruneLambdaInputs()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  lambdaNode->AddContextVar(*x);
  auto lambdaInput1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<jlm::tests::test_op>(
                    { lambdaInput1.inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act
  auto numRemovedInputs = lambdaNode->PruneLambdaInputs();

  // Assert
  assert(numRemovedInputs == 2);
  assert(lambdaNode->ninputs() == 1);
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(lambdaNode->input(0) == lambdaInput1.input);
  assert(lambdaNode->GetContextVars()[0].inner == lambdaInput1.inner);
  assert(lambdaInput1.input->index() == 0);
  assert(lambdaInput1.inner->index() == 0);
}

static int
Test()
{
  TestArgumentIterators();
  TestInvalidOperandRegion();
  TestRemoveLambdaInputsWhere();
  TestPruneLambdaInputs();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestLambda", Test)
