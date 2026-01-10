/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/view.hpp>

/**
 * Tests tracing out of and through a single gamma node.
 * One of the gamma's exit vars is trivially invariant, and can be traced through the gamma.
 */
TEST(TraceTests, TestTraceOutputIntraProcedural_Gamma)
{
  using namespace jlm::rvsdg;

  // Assert
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, controlType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  const auto gammaNode = GammaNode::create(&i0, 2);
  auto entryVar1 = gammaNode->AddEntryVar(&i1);
  auto entryVar1Copy = gammaNode->AddEntryVar(&i1);
  auto entryVar2 = gammaNode->AddEntryVar(&i2);

  auto node = TestOperation::createNode(
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
  EXPECT_EQ(&tracedX0, &i1);
  EXPECT_EQ(&tracedX1, x1.origin());
  EXPECT_EQ(&tracedX2, x2.origin());
  EXPECT_EQ(&traceGammaEntry, &i1);
  EXPECT_EQ(&tracedNodeInput, &i2);
}

/**
 * Tests tracing out of and through a single theta node.
 */
TEST(TraceTests, TestTraceOutputIntraProcedural_Theta)
{
  using namespace jlm::rvsdg;

  // Assert
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i0");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  // loopVar0 is trivially loop invariant
  auto loopVar0 = thetaNode->AddLoopVar(&i0);
  auto loopVar1 = thetaNode->AddLoopVar(&i1);

  auto node = TestOperation::createNode(thetaNode->subregion(), { loopVar1.pre }, { valueType });
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
  EXPECT_EQ(&tracedX0, &i0);
  EXPECT_EQ(&tracedX1, x1.origin());
  EXPECT_EQ(&traceLoopVar0Pre, &i0);
  EXPECT_EQ(&traceLoopVar1Pre, loopVar1.pre);
  EXPECT_EQ(&tracedNodeInput, loopVar1.pre);
}

/**
 * Creates a graph with a gamma node inside a theta node, where some values are invariant.
 */
TEST(TraceTests, TestTraceNestedStructuralNodes)
{
  using namespace jlm::rvsdg;

  // Assert
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

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
  auto & undefNode =
      jlm::rvsdg::CreateOpNode<TestNullaryOperation>(*thetaNode->subregion(), controlType);
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

    EXPECT_EQ(&tracedX0, &i0);
    EXPECT_EQ(&tracedX1, loopVar1.output);
    EXPECT_EQ(&tracedX2, &i2);
    EXPECT_EQ(&traceExitVar0, &i0);
    EXPECT_EQ(&traceExitVar1, loopVar1.pre);
    EXPECT_EQ(&traceBranchArgument0, &i0);
    EXPECT_EQ(&traceBranchArgument1, loopVar1.pre);
    EXPECT_EQ(&traceLoopVar0Pre, &i0);
    EXPECT_EQ(&traceLoopVar1Pre, loopVar1.pre);
    EXPECT_EQ(&traceLoopVar2Pre, &i2);
  }

  // Act & Assert 2
  {
    // Create an alternative tracer that does not perform deep tracing
    OutputTracer shallowTracer;
    shallowTracer.setTraceThroughStructuralNodes(false);

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

    EXPECT_EQ(&tracedX0, loopVar0.output);
    EXPECT_EQ(&tracedX1, loopVar1.output);
    EXPECT_EQ(&tracedX2, &i2); // loopVar2 can still be traced through as it is trivially invariant
    EXPECT_EQ(&traceExitVar0, loopVar0.pre);
    EXPECT_EQ(&traceExitVar1, loopVar1.pre);
    EXPECT_EQ(&traceBranchArgument0, loopVar0.pre);
    EXPECT_EQ(&traceBranchArgument1, loopVar1.pre);
    EXPECT_EQ(&traceLoopVar0Pre, loopVar0.pre);
    EXPECT_EQ(&traceLoopVar1Pre, loopVar1.pre);
    EXPECT_EQ(&traceLoopVar2Pre, &i2);
  }
}
