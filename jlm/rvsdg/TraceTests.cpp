/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/bitstring/constant.hpp>
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
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin(), false);
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin(), false);
  const auto & tracedX2 = traceOutputIntraProcedurally(*x2.origin(), false);
  // Trace from within one of the context variables of the gamma
  const auto & traceGammaEntry = traceOutputIntraProcedurally(*entryVar1.branchArgument[0], false);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin(), false);

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
  const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin(), false);
  const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin(), false);
  const auto & traceLoopVar0Pre = traceOutputIntraProcedurally(*loopVar0.pre, false);
  const auto & traceLoopVar1Pre = traceOutputIntraProcedurally(*loopVar1.pre, false);
  const auto & tracedNodeInput = traceOutputIntraProcedurally(*node->input(0)->origin(), false);

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
    const auto & tracedX0 = traceOutputIntraProcedurally(*x0.origin(), false);
    const auto & tracedX1 = traceOutputIntraProcedurally(*x1.origin(), false);
    const auto & tracedX2 = traceOutputIntraProcedurally(*x2.origin(), false);
    const auto & traceExitVar0 = traceOutputIntraProcedurally(*exitVar0.output, false);
    const auto & traceExitVar1 = traceOutputIntraProcedurally(*exitVar1.output, false);
    const auto & traceBranchArgument0 =
        traceOutputIntraProcedurally(*entryVar0.branchArgument[0], false);
    const auto & traceBranchArgument1 =
        traceOutputIntraProcedurally(*entryVar1.branchArgument[1], false);
    const auto & traceLoopVar0Pre = traceOutputIntraProcedurally(*loopVar0.pre, false);
    const auto & traceLoopVar1Pre = traceOutputIntraProcedurally(*loopVar1.pre, false);
    const auto & traceLoopVar2Pre = traceOutputIntraProcedurally(*loopVar2.pre, false);

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
    shallowTracer.setStructuralNodePolicy(
        OutputTracer::StructuralNodePolicy::traceThroughTriviallyInvariant);

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

/**
 * Tests tracing through and out of theta nodes where a loop variable is not invariant,
 * but takes its post value from an invariant loop variable.
 * The two variables do not share input origin.
 */
TEST(TraceTests, TestIndirectLoopInvariantOutput)
{
  using namespace jlm::rvsdg;

  /**
   * Creates a graph with a single theta node that looks like
   *
   *      20 40
   *      |  |
   *      |  |
   *      V  V
   *  +------------+
   *  |   |  V     |
   *  |   | USER1  |
   *  |   |\       |
   *  |   | \      |
   *  |   V  V     |
   *  +------------+
   *         V
   *        USER2
   *
   * Tracing from USER1 should stop at the theta pre argument,
   * while tracing from USER2 should lead to the constant 20.
   */
  Graph rvsdg;
  auto & c20 = BitConstantOperation::create(rvsdg.GetRootRegion(), { 32, 20 });
  auto & c40 = BitConstantOperation::create(rvsdg.GetRootRegion(), { 32, 40 });

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto invariantLoopVar = thetaNode->AddLoopVar(&c20);
  auto indirectLoopVar = thetaNode->AddLoopVar(&c40);

  // USER1 uses the pre-iteration value of the non-invariant loop variable.
  auto user1 = TestOperation::createNode(thetaNode->subregion(), { indirectLoopVar.pre }, {});

  // Make the post value of indirectLoopVar come from the trivially invariant loop var.
  indirectLoopVar.post->divert_to(invariantLoopVar.pre);

  // USER2 uses the loop ouput value of the non-trivially invariant loop variable.
  auto user2 = TestOperation::createNode(&rvsdg.GetRootRegion(), { indirectLoopVar.output }, {});

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedUser1 = traceOutputIntraProcedurally(*user1->input(0)->origin(), false);
  const auto & tracedUser2 = traceOutputIntraProcedurally(*user2->input(0)->origin(), false);

  // Assert
  EXPECT_TRUE(ThetaLoopVarIsInvariant(invariantLoopVar));
  EXPECT_FALSE(ThetaLoopVarIsInvariant(indirectLoopVar));
  EXPECT_EQ(&tracedUser1, indirectLoopVar.pre);
  EXPECT_EQ(&tracedUser2, &c20);
}

/**
 * Tests tracing through and out of theta nodes where a loop variable is not invariant,
 * but takes its post value from an invariant loop variable, with whom it shares input.
 */
TEST(TraceTests, TestIndirectLoopInvariance)
{
  using namespace jlm::rvsdg;

  /**
   * Creates a graph with a single theta node that looks like
   *
   *      20
   *      |\
   *      | \
   *      V  V
   *  +------------+
   *  |   |  V     |
   *  |   | USER1  |
   *  |   |\       |
   *  |   | \      |
   *  |   V  V     |
   *  +------------+
   *         V
   *        USER2
   *
   * Tracing from either USER1 and USER2 should lead to the constant integer 20,
   * despite the loop variable not being directly invariant.
   */
  Graph rvsdg;
  auto & c20 = BitConstantOperation::create(rvsdg.GetRootRegion(), { 32, 20 });

  const auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto invariantLoopVar = thetaNode->AddLoopVar(&c20);
  auto indirectLoopVar = thetaNode->AddLoopVar(&c20);

  // USER1 uses the pre-iteration value of the non-trivially invariant loop variable.
  auto user1 = TestOperation::createNode(thetaNode->subregion(), { indirectLoopVar.pre }, {});

  // Make the post value of indirectLoopVar come from the trivially invariant loop var.
  indirectLoopVar.post->divert_to(invariantLoopVar.pre);

  // USER2 uses the loop ouput value of the indirectly invariant loop variable.
  auto user2 = TestOperation::createNode(&rvsdg.GetRootRegion(), { indirectLoopVar.output }, {});

  view(&rvsdg.GetRootRegion(), stdout);

  // Act
  const auto & tracedUser1 = traceOutputIntraProcedurally(*user1->input(0)->origin(), false);
  const auto & tracedUser2 = traceOutputIntraProcedurally(*user2->input(0)->origin(), false);

  // Assert
  EXPECT_TRUE(ThetaLoopVarIsInvariant(invariantLoopVar));
  EXPECT_FALSE(ThetaLoopVarIsInvariant(indirectLoopVar));
  EXPECT_EQ(&tracedUser1, &c20);
  EXPECT_EQ(&tracedUser2, &c20);
}

/**
 * Tests tracing into the subregion of a theta node from the theta's outputs,
 * when the loop variable is not invariant.
 */
TEST(TraceTests, TestEnterThetaSubregion)
{
  using namespace jlm::rvsdg;

  /**
   * Creates a graph with a single theta node that looks like
   *
   *      import(i1)
   *          |
   *          |
   *          v
   *  +-theta-x----+
   *  |            |
   *  |  TestNode  |
   *  |       v    |
   *  +-------x----+
   *          v
   *        export(x1)
   *
   * The loop variable is not invariant: its post value comes from a node
   * inside the subregion, not from its pre argument.
   * Tracing x1 from the root region:
   *  - with mayEnterSubregions=true continues into the theta subregion and
   *    stops at the node output inside it.
   *  - with mayEnterSubregions=false stops at the theta output itself.
   */

  // Arrange
  const auto valueType = TestType::createValueType();

  Graph rvsdg;
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar = thetaNode->AddLoopVar(&i1);

  auto node = TestOperation::createNode(thetaNode->subregion(), { loopVar.pre }, { valueType });
  loopVar.post->divert_to(node->output(0));

  auto & x1 = GraphExport::Create(*loopVar.output, "x1");

  // Act & Assert
  // When tracing may enter subregions, tracing goes into the theta subregion
  const auto & tracedIn = traceOutputIntraProcedurally(*x1.origin(), true);
  EXPECT_EQ(&tracedIn, node->output(0));
  EXPECT_EQ(tracedIn.region(), thetaNode->subregion());

  // When tracing may not enter subregions, tracing stops at the theta output
  const auto & tracedOut = traceOutputIntraProcedurally(*x1.origin(), false);
  EXPECT_EQ(&tracedOut, loopVar.output);
  EXPECT_EQ(tracedOut.region(), &rvsdg.GetRootRegion());
}

TEST(TraceTests, GammaCachingTest)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

  Graph rvsdg;

  auto & predicate = GraphImport::Create(rvsdg, controlType, "predicate");
  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  auto gammaNode = GammaNode::create(&predicate, 2);
  auto i1EntryVar = gammaNode->AddEntryVar(&i1);
  auto i2EntryVar = gammaNode->AddEntryVar(&i2);

  auto exitVar =
      gammaNode->AddExitVar({ i1EntryVar.branchArgument[0], i1EntryVar.branchArgument[1] });

  auto & graphExport = GraphExport::Create(*exitVar.output, "export");

  OutputTracer tracer;
  tracer.setInvarianceCaching(true);
  // predicate checking reduces caching of gamma nodes, so disable it for this test
  tracer.setRegionPredicateCheckingEnabled(false);

  // Act & Assert
  // This is the first time we are tracing this output. We expect it to arrive at i1.
  auto traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == &i1);

  // Divert the origins of the exit variable results to i2.
  exitVar.branchResult[0]->divert_to(i2EntryVar.branchArgument[0]);
  exitVar.branchResult[1]->divert_to(i2EntryVar.branchArgument[1]);

  // Since we traced graphExport already and had caching enabled in the tracer, we expect the tracer
  // to still return i1 even though we redirected the origins of the exit variable results to i2.
  traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == &i1);

  // Clear the tracing cache. We should now arrive at i2.
  tracer.clearInvarianceCache();
  traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == &i2);
}

TEST(TraceTests, ThetaCachingTest)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();

  Graph rvsdg;

  auto & i1 = GraphImport::Create(rvsdg, valueType, "i1");
  auto & i2 = GraphImport::Create(rvsdg, valueType, "i2");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar1 = thetaNode->AddLoopVar(&i1);
  auto loopVar2 = thetaNode->AddLoopVar(&i2);

  auto & graphExport = GraphExport::Create(*loopVar1.output, "export");

  OutputTracer tracer;
  tracer.setInvarianceCaching(true);

  // Act & Assert
  // This is the first time we are tracing this output. We expect it to arrive at i1.
  auto traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == &i1);

  // Divert the origins of the loop variables' post value
  loopVar1.post->divert_to(loopVar2.pre);
  loopVar2.post->divert_to(loopVar1.pre);

  // Since we traced graphExport already and had caching enabled in the tracer, we expect the tracer
  // to still return i1 even though we redirected loopVar1.
  traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == &i1);

  // Clear the tracing cache. We should now arrive at the output of loopVar1.
  tracer.clearInvarianceCache();
  traceResult = &tracer.trace(*graphExport.origin());
  assert(traceResult == loopVar1.output);
}

TEST(TraceTests, RegionPredicationThetaTest)
{
  using namespace jlm::rvsdg;

  /**
   * Creates an RVSDG that looks like
   *
   *             Int(1)  Int(2)  Int(3)
   *               v       v       v
   * +-theta-------x-------x-------x-------------------+
   * |             |       |       |                   |
   * |  TestOp     |       |       |                   |
   * |    v        v       v       v                   |
   * | +-gamma---x--x--x------+---------x--x------x-+  |
   * | |         |  |         |         |         | |  |
   * | | CTRL(0) |  | Int(4)  | CTRL(1) |  Int(5) | |  |
   * | |   v     v  v  v      |   v     v    v    v |  |
   * | +---x-----x--x--x------+---x-----x----x----x-+  |
   * |     |       |       |       |                   |
   * |     v       v       v       v                   |
   * +-----x-------x-------x-------x-------------------+
   *               |       |       |
   *               v       v       v
   *              exp(x)  exp(y)  exp(z)
   *
   * and checks that tracing from "x" leads all the way to Int(1),
   * tracing from "y" leads to to the pre of the second loop variable,
   * and tracing from "z" leads to Int(4)
   */
}
