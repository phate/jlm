/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>

static void
TestOutputIterator()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i0 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i1 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i2 = &GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      rootRegion,
      std::vector<std::shared_ptr<const Type>>(),
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  auto nodeIt = Output::Iterator(node.output(0));
  assert(nodeIt.GetOutput() == node.output(0));
  assert(nodeIt->index() == node.output(0)->index());
  assert((*nodeIt).index() == node.output(0)->index());
  assert(nodeIt == Output::Iterator(node.output(0)));
  assert(nodeIt != Output::Iterator(node.output(1)));

  nodeIt++;
  assert(nodeIt.GetOutput() == node.output(1));

  ++nodeIt;
  assert(nodeIt.GetOutput() == node.output(2));

  ++nodeIt;
  ++nodeIt;
  assert(nodeIt.GetOutput() == node.output(4));

  ++nodeIt;
  assert(nodeIt.GetOutput() == nullptr);

  auto regionIt = Output::Iterator(rootRegion.argument(0));
  assert(regionIt.GetOutput() == i0);
  assert(regionIt->index() == i0->index());
  assert((*regionIt).index() == i0->index());
  assert(regionIt == Output::Iterator(i0));
  assert(regionIt != Output::Iterator(i1));

  regionIt++;
  regionIt++;
  assert(regionIt.GetOutput() == i2);

  regionIt++;
  assert(regionIt.GetOutput() == nullptr);

  auto it = Input::Iterator(nullptr);
  it++;
  ++it;
  assert(it.GetInput() == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-TestOutputIterator", TestOutputIterator)

static void
RouteToRegion_Gamma()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, controlType, "i1");

  const auto gammaNode = GammaNode::create(&i1, 2);

  // Act
  const auto & output = RouteToRegion(i0, *gammaNode->subregion(1));

  // Assert
  assert(output.region() == gammaNode->subregion(1));
  assert(gammaNode->GetEntryVars().size() == 1);
  assert(gammaNode->GetExitVars().size() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Gamma", RouteToRegion_Gamma)

static void
RouteToRegion_Theta()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  // Act
  const auto & output = RouteToRegion(i0, *thetaNode->subregion());

  // Assert
  assert(output.region() == thetaNode->subregion());
  assert(thetaNode->GetLoopVars().size() == 1);
  assert(&output == thetaNode->GetLoopVars()[0].pre);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Theta", RouteToRegion_Theta)

static void
RouteToRegion_Lambda()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto lambdaNode =
      LambdaNode::Create(rvsdg.GetRootRegion(), std::make_unique<LambdaOperation>(functionType));

  // Act
  const auto & output = RouteToRegion(i0, *lambdaNode->subregion());

  // Assert
  assert(output.region() == lambdaNode->subregion());
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(&output == lambdaNode->GetContextVars()[0].inner);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Lambda", RouteToRegion_Lambda)

static void
RouteToRegion_Phi()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  PhiBuilder phiBuilder;
  phiBuilder.begin(&rvsdg.GetRootRegion());
  auto phiNode = phiBuilder.end();

  // Act
  const auto & output = RouteToRegion(i0, *phiNode->subregion());

  // Assert
  assert(output.region() == phiNode->subregion());
  assert(phiNode->GetContextVars().size() == 1);
  assert(&output == phiNode->GetContextVars()[0].inner);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Phi", RouteToRegion_Phi)

static void
RouteToRegion_Delta()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  auto deltaNode = DeltaNode::Create(
      &rvsdg.GetRootRegion(),
      std::make_unique<DeltaOperation>(valueType, true, valueType));

  // Act
  const auto & output = RouteToRegion(i0, *deltaNode->subregion());

  // Assert
  assert(output.region() == deltaNode->subregion());
  assert(deltaNode->GetContextVars().size() == 1);
  assert(&output == deltaNode->GetContextVars()[0].inner);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Delta", RouteToRegion_Delta)

static void
RouteToRegion_Nesting()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  auto valueType = jlm::tests::ValueType::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");

  const auto lambdaNode =
      LambdaNode::Create(rvsdg.GetRootRegion(), std::make_unique<LambdaOperation>(functionType));

  const auto controlConstant = control_constant(lambdaNode->subregion(), 2, 0);
  const auto gammaNode = GammaNode::create(controlConstant, 2);

  // Act
  const auto & output = RouteToRegion(i0, *gammaNode->subregion(0));

  // Assert
  assert(output.region() == gammaNode->subregion(0));
  assert(gammaNode->GetEntryVars().size() == 1);
  assert(gammaNode->GetExitVars().size() == 0);
  assert(&output == gammaNode->GetEntryVars()[0].branchArgument[0]);

  auto origin = gammaNode->GetEntryVars()[0].input->origin();
  assert(lambdaNode->GetContextVars().size() == 1);
  assert(origin == lambdaNode->GetContextVars()[0].inner);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Nesting", RouteToRegion_Nesting)

static void
RouteToRegion_Failure()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, controlType, "i1");

  const auto gammaNode = GammaNode::create(&i1, 2);
  auto entryVar = gammaNode->AddEntryVar(&i0);

  // Act & Assert
  try
  {
    RouteToRegion(*entryVar.branchArgument[0], *gammaNode->subregion(1));
    assert(false);
  }
  catch (std::logic_error const &)
  {
    assert(true);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-RouteToRegion_Failure", RouteToRegion_Failure)
