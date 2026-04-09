/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(LambdaTests, TestArgumentIterators)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  {
    auto functionType = jlm::rvsdg::FunctionType::Create({ vt }, { vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
    lambda->finalize({ lambda->GetFunctionArguments()[0] });

    std::vector<const jlm::rvsdg::Output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    EXPECT_EQ(functionArguments.size(), 1u);
    EXPECT_EQ(functionArguments[0], lambda->GetFunctionArguments()[0]);
  }

  {
    auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

    auto nullaryNode = TestOperation::createNode(lambda->subregion(), {}, { vt })->output(0);

    lambda->finalize({ nullaryNode });

    EXPECT_TRUE(lambda->GetFunctionArguments().empty());
  }

  {
    auto rvsdgImport = &jlm::rvsdg::GraphImport::Create(rvsdgModule.Rvsdg(), vt, "");

    auto functionType = jlm::rvsdg::FunctionType::Create({ vt, vt, vt }, { vt, vt });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        rvsdgModule.Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

    auto cv = lambda->AddContextVar(*rvsdgImport).inner;

    lambda->finalize({ lambda->GetFunctionArguments()[0], cv });

    std::vector<const jlm::rvsdg::Output *> functionArguments;
    for (auto argument : lambda->GetFunctionArguments())
      functionArguments.push_back(argument);

    EXPECT_EQ(functionArguments.size(), 3u);
    EXPECT_EQ(functionArguments[0], lambda->GetFunctionArguments()[0]);
    EXPECT_EQ(functionArguments[1], lambda->GetFunctionArguments()[1]);
    EXPECT_EQ(functionArguments[2], lambda->GetFunctionArguments()[2]);
  }
}

TEST(LambdaTests, TestInvalidOperandRegion)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { vt });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg->GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto result = TestOperation::createNode(&rvsdg->GetRootRegion(), {}, { vt })->output(0);

  EXPECT_THROW(lambdaNode->finalize({ result }), jlm::util::Error);
}

/**
 * Test LambdaNode::RemoveLambdaInputsWhere()
 */
TEST(LambdaTests, TestRemoveLambdaInputsWhere)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { valueType });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto lambdaBinder0 = lambdaNode->AddContextVar(*x);
  auto lambdaBinder1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<TestOperation>(
                    { lambdaBinder1.inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act & Assert
  // Try to remove lambdaInput1 even though it is used
  auto numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == lambdaBinder1.input->index();
      });
  EXPECT_EQ(numRemovedInputs, 0u);
  EXPECT_EQ(lambdaNode->ninputs(), 3u);
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 3u);

  // Remove lambdaInput2
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == 2;
      });
  EXPECT_EQ(numRemovedInputs, 1u);
  EXPECT_EQ(lambdaNode->ninputs(), 2u);
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 2u);
  EXPECT_EQ(lambdaNode->input(0), lambdaBinder0.input);
  EXPECT_EQ(lambdaNode->input(1), lambdaBinder1.input);

  // Remove lambdaInput0
  numRemovedInputs = lambdaNode->RemoveLambdaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == 0;
      });
  EXPECT_EQ(numRemovedInputs, 1u);
  EXPECT_EQ(lambdaNode->ninputs(), 1u);
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 1u);
  EXPECT_EQ(lambdaNode->input(0), lambdaBinder1.input);
  EXPECT_EQ(lambdaBinder1.input->index(), 0u);
  EXPECT_EQ(lambdaBinder1.inner->index(), 0u);
}

/**
 * Test LambdaNode::PruneLambdaInputs()
 */
TEST(LambdaTests, TestPruneLambdaInputs)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create({}, { valueType });

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  lambdaNode->AddContextVar(*x);
  auto lambdaInput1 = lambdaNode->AddContextVar(*x);
  lambdaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<TestOperation>(
                    { lambdaInput1.inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  lambdaNode->finalize({ result });

  // Act
  auto numRemovedInputs = lambdaNode->PruneLambdaInputs();

  // Assert
  EXPECT_EQ(numRemovedInputs, 2u);
  EXPECT_EQ(lambdaNode->ninputs(), 1u);
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 1u);
  EXPECT_EQ(lambdaNode->input(0), lambdaInput1.input);
  EXPECT_EQ(lambdaNode->GetContextVars()[0].inner, lambdaInput1.inner);
  EXPECT_EQ(lambdaInput1.input->index(), 0u);
  EXPECT_EQ(lambdaInput1.inner->index(), 0u);
}
