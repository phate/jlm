/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/view.hpp>

#include <cassert>

/**
 * Tests tracing out of and through a single gamma node.
 * One of the gamma's exit vars is trivially invariant, and can be traced through the gamma.
 */
static void
TestTraceOutputIntraProcedural_Gamma()
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
  auto entryVar1Copy = gammaNode->AddEntryVar(&i1);
  auto entryVar2 = gammaNode->AddEntryVar(&i2);

  auto node = TestOperation::create(
      gammaNode->subregion(1),
      { entryVar2.branchArgument[1] },
      { valueType });

  auto exitVar0 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], entryVar1Copy.branchArgument[1] });
  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });
  auto exitVar2 = gammaNode->AddExitVar({ entryVar2.branchArgument[0], node->output(0) });

  auto & x0 = GraphExport::Create(*exitVar0.output, "x0");
  auto & x1 = GraphExport::Create(*exitVar1.output, "x1");
  auto & x2 = GraphExport::Create(*exitVar2.output, "x2");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
  const auto & tracedX2 = traceOutputIntraProcedurally(*x2.origin());
  // Trace from within one of the context variables of the gamma
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*entryVar1.branchArgument[0]);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i1);
  assert(&tracedX1 == x1.origin());
  assert(&tracedX2 == x2.origin());
  assert(&traceGammaEntry == &i1);
  assert(&tracedNodeInput == &i2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TestTraceOutputIntraProcedural_Gamma",
    TestTraceOutputIntraProcedural_Gamma)

/**
 * Tests tracing out of and through a single theta node.
 */
static void
TestTraceOutputIntraProcedural_Theta()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  // loopVar0 is trivially loop invariant
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
  const auto & traceLoopVar0Pre = traceOutputIntraProcedurally(*loopVar0.pre);
  const auto & traceLoopVar1Pre = traceOutputIntraProcedurally(*loopVar1.pre);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin());

  // Assert
  assert(&tracedX0 == &i0);
  assert(&tracedX1 == x1.origin());
  assert(&traceLoopVar0Pre == &i0);
  assert(&traceLoopVar1Pre == loopVar1.pre);
  assert(&tracedNodeInput == loopVar1.pre);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TestTraceOutputIntraProcedural_Theta",
    TestTraceOutputIntraProcedural_Theta)

/**
 * Creates a graph with a gamma node inside a theta node, where some values are invariant.
 */
static void
TestTraceNestedStructuralNodes()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Assert
  const auto controlType = ControlType::Create(2);
  const auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  // loop variables 0 and 1 go through the inner gamma. loopVar2 is trivially loop invariant
  auto loopVar0 = thetaNode->AddLoopVar(&i0);
  auto loopVar1 = thetaNode->AddLoopVar(&i1);
  auto loopVar2 = thetaNode->AddLoopVar(&i2);

  // Create the gamma that sends loopVar0 and loopVar1 directly through
  const auto & undefNode =
      jlm::rvsdg::CreateOpNode<jlm::tests::NullaryOperation>(*thetaNode->subregion(), controlType);
  const auto gammaNode = GammaNode::create(undefNode.output(0), 2);
  auto entryVar0 = gammaNode->AddEntryVar(loopVar0.pre);
  auto entryVar1 = gammaNode->AddEntryVar(loopVar1.pre);
  auto exitVar0 =
      gammaNode->AddExitVar({ entryVar0.branchArgument[0], entryVar0.branchArgument[1] });
  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], entryVar1.branchArgument[1] });

  // Both loopVar0 and loopVar1 get loopVar0.pre as their indirect origin
  // This means only loopVar0 is loop invariant
  loopVar0.post->divert_to(exitVar0.output);
  loopVar1.post->divert_to(exitVar0.output);

  auto & x0 = GraphExport::Create(*loopVar0.output, "x0");
  auto & x1 = GraphExport::Create(*loopVar1.output, "x1");
  auto & x2 = GraphExport::Create(*loopVar2.output, "x2");

  view(&rvsdg.GetRootRegion(), stdout);

  // Act & Assert 1
  {
    const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin());
    const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin());
    const auto & tracedX2 = traceOutputIntraProcedurally(*x2.origin());
    const auto & traceExitVar0 = traceOutputIntraProcedurally(*exitVar0.output);
    const auto & traceExitVar1 = traceOutputIntraProcedurally(*exitVar1.output);
    const auto & traceBranchArgument0 = traceOutputIntraProcedurally(*entryVar0.branchArgument[0]);
    const auto & traceBranchArgument1 = traceOutputIntraProcedurally(*entryVar1.branchArgument[1]);
    const auto & traceLoopVar0Pre = traceOutputIntraProcedurally(*loopVar0.pre);
    const auto & traceLoopVar1Pre = traceOutputIntraProcedurally(*loopVar1.pre);
    const auto & traceLoopVar2Pre = traceOutputIntraProcedurally(*loopVar2.pre);

    assert(&tracedX0 == &i0);
    assert(&tracedX1 == loopVar1.output);
    assert(&tracedX2 == &i2);
    assert(&traceExitVar0 == &i0);
    assert(&traceExitVar1 == loopVar1.pre);
    assert(&traceBranchArgument0 == &i0);
    assert(&traceBranchArgument1 == loopVar1.pre);
    assert(&traceLoopVar0Pre == &i0);
    assert(&traceLoopVar1Pre == loopVar1.pre);
    assert(&traceLoopVar2Pre == &i2);
  }

  // Act & Assert 2
  {
    // Create an alternative tracer that does not perform deep tracing
    OutputTracer shallowTracer(false, false);

    const auto & tracedX0 = shallowTracer.trace(*x0.origin());
    const auto & tracedX1 = shallowTracer.trace(*x1.origin());
    const auto & tracedX2 = shallowTracer.trace(*x2.origin());
    const auto & traceExitVar0 = shallowTracer.trace(*exitVar0.output);
    const auto & traceExitVar1 = shallowTracer.trace(*exitVar1.output);
    const auto & traceBranchArgument0 = shallowTracer.trace(*entryVar0.branchArgument[0]);
    const auto & traceBranchArgument1 = shallowTracer.trace(*entryVar1.branchArgument[1]);
    const auto & traceLoopVar0Pre = shallowTracer.trace(*loopVar0.pre);
    const auto & traceLoopVar1Pre = shallowTracer.trace(*loopVar1.pre);
    const auto & traceLoopVar2Pre = shallowTracer.trace(*loopVar2.pre);

    assert(&tracedX0 == loopVar0.output);
    assert(&tracedX1 == loopVar1.output);
    assert(&tracedX2 == &i2); // loopVar2 can still be traced through as it is trivially invariant
    assert(&traceExitVar0 == loopVar0.pre);
    assert(&traceExitVar1 == loopVar1.pre);
    assert(&traceBranchArgument0 == loopVar0.pre);
    assert(&traceBranchArgument1 == loopVar1.pre);
    assert(&traceLoopVar0Pre == loopVar0.pre);
    assert(&traceLoopVar1Pre == loopVar1.pre);
    assert(&traceLoopVar2Pre == &i2);
  }
}
JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-nodes-TestTraceNestedStructuralNodes",
    TestTraceNestedStructuralNodes)
