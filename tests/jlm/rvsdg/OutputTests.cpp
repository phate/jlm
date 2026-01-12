/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>

TEST(OutputTests, TestOutputIterator)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i0 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i1 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i2 = &GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<TestOperation>(
      rootRegion,
      std::vector<std::shared_ptr<const Type>>(),
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  auto nodeIt = Output::Iterator(node.output(0));
  EXPECT_EQ(nodeIt.GetOutput(), node.output(0));
  EXPECT_EQ(nodeIt->index(), node.output(0)->index());
  EXPECT_EQ((*nodeIt).index(), node.output(0)->index());
  EXPECT_EQ(nodeIt, Output::Iterator(node.output(0)));
  EXPECT_NE(nodeIt, Output::Iterator(node.output(1)));

  nodeIt++;
  EXPECT_EQ(nodeIt.GetOutput(), node.output(1));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetOutput(), node.output(2));

  ++nodeIt;
  ++nodeIt;
  EXPECT_EQ(nodeIt.GetOutput(), node.output(4));

  ++nodeIt;
  EXPECT_EQ(nodeIt.GetOutput(), nullptr);

  auto regionIt = Output::Iterator(rootRegion.argument(0));
  EXPECT_EQ(regionIt.GetOutput(), i0);
  EXPECT_EQ(regionIt->index(), i0->index());
  EXPECT_EQ((*regionIt).index(), i0->index());
  EXPECT_EQ(regionIt, Output::Iterator(i0));
  EXPECT_NE(regionIt, Output::Iterator(i1));

  regionIt++;
  regionIt++;
  EXPECT_EQ(regionIt.GetOutput(), i2);

  regionIt++;
  EXPECT_EQ(regionIt.GetOutput(), nullptr);

  auto it = Input::Iterator(nullptr);
  it++;
  ++it;
  EXPECT_EQ(it.GetInput(), nullptr);
}

TEST(OutputTests, RouteToRegion_Gamma)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, controlType, "i1");

  const auto gammaNode = GammaNode::create(&i1, 2);

  // Act
  const auto & output = RouteToRegion(i0, *gammaNode->subregion(1));

  // Assert
  EXPECT_EQ(output.region(), gammaNode->subregion(1));
  EXPECT_EQ(gammaNode->GetEntryVars().size(), 1u);
  EXPECT_EQ(gammaNode->GetExitVars().size(), 0u);
}

TEST(OutputTests, RouteToRegion_Theta)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  // Act
  const auto & output = RouteToRegion(i0, *thetaNode->subregion());

  // Assert
  EXPECT_EQ(output.region(), thetaNode->subregion());
  EXPECT_EQ(thetaNode->GetLoopVars().size(), 1u);
  EXPECT_EQ(&output, thetaNode->GetLoopVars()[0].pre);
}

TEST(OutputTests, RouteToRegion_Lambda)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto lambdaNode =
      LambdaNode::Create(rvsdg.GetRootRegion(), std::make_unique<LambdaOperation>(functionType));

  // Act
  const auto & output = RouteToRegion(i0, *lambdaNode->subregion());

  // Assert
  EXPECT_EQ(output.region(), lambdaNode->subregion());
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 1u);
  EXPECT_EQ(&output, lambdaNode->GetContextVars()[0].inner);
}

TEST(OutputTests, RouteToRegion_Phi)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  PhiBuilder phiBuilder;
  phiBuilder.begin(&rvsdg.GetRootRegion());
  auto phiNode = phiBuilder.end();

  // Act
  const auto & output = RouteToRegion(i0, *phiNode->subregion());

  // Assert
  EXPECT_EQ(output.region(), phiNode->subregion());
  EXPECT_EQ(phiNode->GetContextVars().size(), 1u);
  EXPECT_EQ(&output, phiNode->GetContextVars()[0].inner);
}

TEST(OutputTests, RouteToRegion_Delta)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  auto deltaNode = DeltaNode::Create(
      &rvsdg.GetRootRegion(),
      std::make_unique<DeltaOperation>(valueType, true, valueType));

  // Act
  const auto & output = RouteToRegion(i0, *deltaNode->subregion());

  // Assert
  EXPECT_EQ(output.region(), deltaNode->subregion());
  EXPECT_EQ(deltaNode->GetContextVars().size(), 1u);
  EXPECT_EQ(&output, deltaNode->GetContextVars()[0].inner);
}

TEST(OutputTests, RouteToRegion_Nesting)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  auto valueType = TestType::createValueType();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto lambdaNode =
      LambdaNode::Create(rvsdg.GetRootRegion(), std::make_unique<LambdaOperation>(functionType));

  const auto controlConstant = &ControlConstantOperation::create(*lambdaNode->subregion(), 2, 0);
  const auto gammaNode = GammaNode::create(controlConstant, 2);

  // Act
  const auto & output = RouteToRegion(i0, *gammaNode->subregion(0));

  // Assert
  EXPECT_EQ(output.region(), gammaNode->subregion(0));
  EXPECT_EQ(gammaNode->GetEntryVars().size(), 1u);
  EXPECT_EQ(gammaNode->GetExitVars().size(), 0u);
  EXPECT_EQ(&output, gammaNode->GetEntryVars()[0].branchArgument[0]);

  auto origin = gammaNode->GetEntryVars()[0].input->origin();
  EXPECT_EQ(lambdaNode->GetContextVars().size(), 1u);
  EXPECT_EQ(origin, lambdaNode->GetContextVars()[0].inner);
}

TEST(OutputTests, RouteToRegion_Failure)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, controlType, "i1");

  const auto gammaNode = GammaNode::create(&i1, 2);
  auto entryVar = gammaNode->AddEntryVar(&i0);

  // Act & Assert
  EXPECT_THROW(
      RouteToRegion(*entryVar.branchArgument[0], *gammaNode->subregion(1)),
      std::logic_error);
}

TEST(OutputTests, DivertUsersWhere)
{
  using namespace jlm::rvsdg;

  // Assert
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");

  auto & x0 = GraphExport::Create(i0, "x0");
  auto & x1 = GraphExport::Create(i0, "x1");
  auto & x2 = GraphExport::Create(i0, "x2");
  auto & x3 = GraphExport::Create(i0, "x3");

  // Act & Assert

  // The new origin is the same as the old origin. Nothing should happen.
  auto numDivertedUsers = i0.divertUsersWhere(
      i0,
      [](const Input &)
      {
        return true;
      });
  EXPECT_EQ(numDivertedUsers, 0u);
  EXPECT_EQ(i0.nusers(), 4u);

  // Divert user x0 to new origin i0
  numDivertedUsers = i0.divertUsersWhere(
      i1,
      [&x0](const Input & user)
      {
        return &user == &x0;
      });
  EXPECT_EQ(numDivertedUsers, 1u);
  EXPECT_EQ(i0.nusers(), 3u);
  EXPECT_EQ(x0.origin(), &i1);

  // Nothing should happen as x0 is no longer a user of i0
  numDivertedUsers = i0.divertUsersWhere(
      i1,
      [&x0](const Input & user)
      {
        return &user == &x0;
      });
  EXPECT_EQ(numDivertedUsers, 0u);
  EXPECT_EQ(i0.nusers(), 3u);

  // Divert users x1 and x2 to i1
  numDivertedUsers = i0.divertUsersWhere(
      i1,
      [&x1, &x2](const Input & user)
      {
        return &user == &x1 || &user == &x2;
      });
  EXPECT_EQ(numDivertedUsers, 2u);
  EXPECT_EQ(i0.nusers(), 1u);
  EXPECT_EQ(x1.origin(), &i1);
  EXPECT_EQ(x2.origin(), &i1);

  // Finally, divert user x3 to i1
  numDivertedUsers = i0.divertUsersWhere(
      i1,
      [&x3](const Input & user)
      {
        return &user == &x3;
      });
  EXPECT_EQ(numDivertedUsers, 1u);
  EXPECT_EQ(i0.nusers(), 0u);
  EXPECT_EQ(x3.origin(), &i1);
}
