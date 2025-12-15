/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/view.hpp>

#include <cassert>

static void
TraceOutputIntraProcedural_Gamma()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto controlType = ControlType::Create(2);
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, controlType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto gammaNode = GammaNode::create(&i0, 2);
  auto entryVar1 = gammaNode->AddEntryVar(&i1);
  auto entryVar2 = gammaNode->AddEntryVar(&i2);

  auto node = TestOperation::create(
      gammaNode->subregion(1),
      { entryVar2.branchArgument[1] },
      { valueType });

  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], entryVar1.branchArgument[1] });
  auto exitVar2 = gammaNode->AddExitVar({ entryVar2.branchArgument[0], node->output(0) });

  auto & x0 = GraphExport::Create(*exitVar1.output, "x0");
  auto & x1 = GraphExport::Create(*exitVar2.output, "x1");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*entryVar1.branchArgument[0]);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i1);
  assert(&tracedX1 == x1.origin());
  assert(&traceGammaEntry == &i1);
  assert(&tracedNodeInput == &i2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TraceOutputIntraProcedural_Gamma",
    TraceOutputIntraProcedural_Gamma)

static void
TraceOutputIntraProcedural_Theta()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar0 = thetaNode->AddLoopVar(&i0);
  auto loopVar1 = thetaNode->AddLoopVar(&i1);

  auto node = TestOperation::create(thetaNode->subregion(), { loopVar1.pre }, { valueType });
  loopVar1.post->divert_to(node->output(0));

  auto & x0 = GraphExport::Create(*loopVar0.output, "x0");
  auto & x1 = GraphExport::Create(*loopVar1.output, "x1");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*loopVar0.pre);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i0);
  assert(&tracedX1 == x1.origin());
  assert(&traceGammaEntry == &i0);
  assert(&tracedNodeInput == loopVar1.pre);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TraceOutputIntraProcedural_Theta",
    TraceOutputIntraProcedural_Theta)
