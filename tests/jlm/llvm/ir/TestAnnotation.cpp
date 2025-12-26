/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/Annotation.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/TestType.hpp>

static void
TestBasicBlockAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto v0 = module.create_variable(vt, "v0");

    ThreeAddressCodeList bb;
    bb.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v0 }));
    auto v1 = bb.last()->result(0);

    bb.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v2 = bb.last()->result(0);

    auto root = BasicBlockAggregationNode::create(std::move(bb));

    return std::make_tuple(std::move(root), v0, v1, v2);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, v0, v1, v2] = SetupAggregationTree(module);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<BasicBlockAnnotationSet>(*aggregationTreeRoot)
      == BasicBlockAnnotationSet({ v0 }, { v1, v2 }, { v1, v2 }));
}

static void
TestLinearSubgraphAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule &, jlm::llvm::Argument & argument)
  {
    /*
     * Setup simple linear CFG: Entry -> B1 -> B2 -> Exit
     */
    auto vt = jlm::rvsdg::TestType::createValueType();

    ThreeAddressCodeList bb1, bb2;
    bb1.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { &argument }));
    auto v1 = bb1.last()->result(0);

    bb2.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v2 = bb2.last()->result(0);

    auto entryNode = EntryAggregationNode::create({ &argument });
    auto basicBlockNode1 = BasicBlockAggregationNode::create(std::move(bb1));
    auto basicBlockNode2 = BasicBlockAggregationNode::create(std::move(bb2));
    auto exitNode = ExitAggregationNode::create({ v2 });

    auto linearNode1 =
        LinearAggregationNode::create(std::move(entryNode), std::move(basicBlockNode1));
    auto linearNode2 =
        LinearAggregationNode::create(std::move(basicBlockNode2), std::move(exitNode));

    auto root = LinearAggregationNode::create(std::move(linearNode1), std::move(linearNode2));

    return std::make_tuple(std::move(root), v1, v2);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  jlm::llvm::Argument argument("argument", jlm::rvsdg::TestType::createValueType());
  auto [aggregationTreeRoot, v1, v2] = SetupAggregationTree(module, argument);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<LinearAnnotationSet>(*aggregationTreeRoot)
      == LinearAnnotationSet({}, { v1, &argument, v2 }, { v1, &argument, v2 }));
  {
    auto linearNode1 = aggregationTreeRoot->child(0);
    assert(
        demandMap->Lookup<LinearAnnotationSet>(*linearNode1)
        == LinearAnnotationSet({}, { v1, &argument }, { v1, &argument }));
    {
      auto entryNode = linearNode1->child(0);
      assert(
          demandMap->Lookup<EntryAnnotationSet>(*entryNode)
          == EntryAnnotationSet({}, { &argument }, { &argument }, {}));

      auto basicBlockNode1 = linearNode1->child(1);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode1)
          == BasicBlockAnnotationSet({ &argument }, { v1 }, { v1 }));
    }

    auto linearNode2 = aggregationTreeRoot->child(1);
    assert(
        demandMap->Lookup<LinearAnnotationSet>(*linearNode2)
        == LinearAnnotationSet({ v1 }, { v2 }, { v2 }));
    {
      auto basicBlockNode2 = linearNode2->child(0);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode2)
          == BasicBlockAnnotationSet({ v1 }, { v2 }, { v2 }));

      auto exitNode = linearNode2->child(1);
      assert(demandMap->Lookup<ExitAnnotationSet>(*exitNode) == ExitAnnotationSet({ v2 }, {}, {}));
    }
  }
}

static void
TestBranchAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    /*
     * Setup conditional CFG with nodes bbs, b1, b2, and edges bbs -> b1 and bbs -> b2.
     */
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto argument = module.create_variable(vt, "arg");
    auto v3 = module.create_variable(vt, "v3");

    ThreeAddressCodeList splitTacList, bb1, bb2;
    splitTacList.append_last(
        ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { argument }));
    auto v1 = splitTacList.last()->result(0);

    bb2.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v2 = bb2.last()->result(0);

    bb1.append_last(AssignmentOperation::create(v2, v3));
    bb2.append_last(AssignmentOperation::create(v1, v3));
    bb2.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v3 }));
    auto v4 = bb2.last()->result(0);

    auto basicBlockSplit = BasicBlockAggregationNode::create(std::move(splitTacList));
    auto basicBlock1 = BasicBlockAggregationNode::create(std::move(bb1));
    auto basicBlock2 = BasicBlockAggregationNode::create(std::move(bb2));

    auto branch = BranchAggregationNode::create();
    branch->add_child(std::move(basicBlock1));
    branch->add_child(std::move(basicBlock2));

    auto root = LinearAggregationNode::create(std::move(basicBlockSplit), std::move(branch));

    return std::make_tuple(std::move(root), argument, v1, v2, v3, v4);
  };

  auto vt = jlm::rvsdg::TestType::createValueType();
  jlm::tests::TestOperation op({ vt }, { vt });

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, argument, v1, v2, v3, v4] = SetupAggregationTree(module);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<LinearAnnotationSet>(*aggregationTreeRoot)
      == LinearAnnotationSet({ argument, v2 }, { v1, v2, v3, v4 }, { v1, v3 }));
  {
    auto splitNode = aggregationTreeRoot->child(0);
    assert(
        demandMap->Lookup<BasicBlockAnnotationSet>(*splitNode)
        == BasicBlockAnnotationSet({ argument }, { v1 }, { v1 }));

    auto branchNode = aggregationTreeRoot->child(1);
    assert(
        demandMap->Lookup<BranchAnnotationSet>(*branchNode)
        == BranchAnnotationSet({ v1, v2 }, { v2, v3, v4 }, { v3 }, { v1, v2 }, {}));
    {
      auto basicBlockNode1 = branchNode->child(0);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode1)
          == BasicBlockAnnotationSet({ v2 }, { v3 }, { v3 }));

      auto basicBlockNode2 = branchNode->child(1);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode2)
          == BasicBlockAnnotationSet({ v1 }, { v2, v4, v3 }, { v2, v4, v3 }));
    }
  }
}

static void
TestLoopAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto v1 = module.create_variable(vt, "v1");
    auto v4 = module.create_variable(vt, "v4");

    ThreeAddressCodeList bb;
    bb.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v2 = bb.last()->result(0);

    bb.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v2 }));
    auto v3 = bb.last()->result(0);

    auto exitNode = ExitAggregationNode::create({ v3, v4 });
    auto basicBlockNode = BasicBlockAggregationNode::create(std::move(bb));

    auto loopNode = LoopAggregationNode::create(std::move(basicBlockNode));
    auto root = LinearAggregationNode::create(std::move(loopNode), std::move(exitNode));

    return std::make_tuple(std::move(root), v1, v2, v3, v4);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, v1, v2, v3, v4] = SetupAggregationTree(module);

  /*
   * Assert
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Act
   */
  assert(
      demandMap->Lookup<LinearAnnotationSet>(*aggregationTreeRoot)
      == LinearAnnotationSet({ v1, v4 }, { v2, v3 }, { v2, v3 }));
  {
    auto loopNode = aggregationTreeRoot->child(0);
    assert(
        demandMap->Lookup<LoopAnnotationSet>(*loopNode)
        == LoopAnnotationSet({ v1 }, { v2, v3 }, { v2, v3 }, { v1, v3, v4 }));
    {
      auto basicBlockNode = loopNode->child(0);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode)
          == BasicBlockAnnotationSet({ v1 }, { v2, v3 }, { v2, v3 }));
    }

    auto exitNode = aggregationTreeRoot->child(1);
    assert(
        demandMap->Lookup<ExitAnnotationSet>(*exitNode) == ExitAnnotationSet({ v3, v4 }, {}, {}));
  }
}

static void
TestBranchInLoopAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto v1 = module.create_variable(vt, "v1");
    auto v3 = module.create_variable(vt, "v3");

    ThreeAddressCodeList tl_cb1, tl_cb2;
    tl_cb1.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v2 = tl_cb1.last()->result(0);

    tl_cb1.append_last(AssignmentOperation::create(v1, v3));
    tl_cb1.append_last(ThreeAddressCode::create(TestOperation::create({ vt }, { vt }), { v1 }));
    auto v4 = tl_cb1.last()->result(0);

    tl_cb2.append_last(AssignmentOperation::create(v1, v3));
    tl_cb2.append_last(AssignmentOperation::create(v4, v3));

    auto exitNode = ExitAggregationNode::create({ v2, v3 });

    auto basicBlock1 = BasicBlockAggregationNode::create(std::move(tl_cb1));
    auto basicBlock2 = BasicBlockAggregationNode::create(std::move(tl_cb2));

    auto branchNode = BranchAggregationNode::create();
    branchNode->add_child(std::move(basicBlock1));
    branchNode->add_child(std::move(basicBlock2));

    auto loopNode = LoopAggregationNode::create(std::move(branchNode));

    auto root = LinearAggregationNode::create(std::move(loopNode), std::move(exitNode));

    return std::make_tuple(std::move(root), v1, v2, v3, v4);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, v1, v2, v3, v4] = SetupAggregationTree(module);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<LinearAnnotationSet>(*aggregationTreeRoot)
      == LinearAnnotationSet({ v1, v2, v4 }, { v2, v3, v4 }, { v3 }));
  {
    auto loopNode = aggregationTreeRoot->child(0);
    assert(
        demandMap->Lookup<LoopAnnotationSet>(*loopNode)
        == LoopAnnotationSet({ v1, v4 }, { v2, v3, v4 }, { v3 }, { v1, v2, v3, v4 }));
    {
      auto branchNode = loopNode->child(0);
      assert(
          demandMap->Lookup<BranchAnnotationSet>(*branchNode)
          == BranchAnnotationSet(
              { v1, v4 },
              { v2, v3, v4 },
              { v3 },
              { v1, v2, v4 },
              { v2, v3, v4 }));
      {
        auto basicBlockNode1 = branchNode->child(0);
        assert(
            demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode1)
            == BasicBlockAnnotationSet({ v1 }, { v2, v3, v4 }, { v2, v3, v4 }));

        auto basicBlockNode2 = branchNode->child(1);
        assert(
            demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode2)
            == BasicBlockAnnotationSet({ v1, v4 }, { v3 }, { v3 }));
      }
    }

    auto exitNode = aggregationTreeRoot->child(1);
    assert(
        demandMap->Lookup<ExitAnnotationSet>(*exitNode) == ExitAnnotationSet({ v2, v3 }, {}, {}));
  }
}

static void
TestAssignmentAnnotation()
{
  using namespace jlm::llvm;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto v1 = module.create_variable(vt, "v1");
    auto v2 = module.create_variable(vt, "v2");

    ThreeAddressCodeList bb;
    bb.append_last(AssignmentOperation::create(v1, v2));

    auto root = BasicBlockAggregationNode::create(std::move(bb));

    return std::make_tuple(std::move(root), v1, v2);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, v1, v2] = SetupAggregationTree(module);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<BasicBlockAnnotationSet>(*aggregationTreeRoot)
      == BasicBlockAnnotationSet({ v1 }, { v2 }, { v2 }));
}

static void
TestBranchPassByAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto SetupAggregationTree = [](InterProceduralGraphModule & module)
  {
    auto vt = jlm::rvsdg::TestType::createValueType();

    auto v3 = module.create_variable(vt, "v3");

    ThreeAddressCodeList tlsplit, tlb1, tlb2;
    tlsplit.append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
    auto v1 = tlsplit.last()->result(0);

    tlsplit.append_last(ThreeAddressCode::create(TestOperation::create({}, { vt }), {}));
    auto v2 = tlsplit.last()->result(0);

    tlb1.append_last(AssignmentOperation::create(v1, v2));
    tlb1.append_last(AssignmentOperation::create(v1, v3));
    tlb2.append_last(AssignmentOperation::create(v1, v3));

    auto splitNode = BasicBlockAggregationNode::create(std::move(tlsplit));

    auto basicBlockNode1 = BasicBlockAggregationNode::create(std::move(tlb1));
    auto basicBlockNode2 = BasicBlockAggregationNode::create(std::move(tlb2));

    auto branchNode = BranchAggregationNode::create();
    branchNode->add_child(std::move(basicBlockNode1));
    branchNode->add_child(std::move(basicBlockNode2));

    auto joinNode = BasicBlockAggregationNode::create();

    auto exitNode = ExitAggregationNode::create({ v1, v2, v3 });

    auto root = LinearAggregationNode::create(std::move(splitNode), std::move(branchNode));
    root->add_child(std::move(joinNode));
    root->add_child(std::move(exitNode));

    return std::make_tuple(std::move(root), v1, v2, v3);
  };

  InterProceduralGraphModule module(jlm::util::FilePath(""), "", "");
  auto [aggregationTreeRoot, v1, v2, v3] = SetupAggregationTree(module);

  /*
   * Act
   */
  auto demandMap = Annotate(*aggregationTreeRoot);
  print(*aggregationTreeRoot, *demandMap, stdout);

  /*
   * Assert
   */
  assert(
      demandMap->Lookup<LinearAnnotationSet>(*aggregationTreeRoot)
      == LinearAnnotationSet({}, { v1, v2, v3 }, { v1, v2, v3 }));
  {
    auto splitNode = aggregationTreeRoot->child(0);
    assert(
        demandMap->Lookup<BasicBlockAnnotationSet>(*splitNode)
        == BasicBlockAnnotationSet({}, { v1, v2 }, { v1, v2 }));

    auto branchNode = aggregationTreeRoot->child(1);
    assert(
        demandMap->Lookup<BranchAnnotationSet>(*branchNode)
        == BranchAnnotationSet({ v1 }, { v2, v3 }, { v3 }, { v1, v2 }, { v2, v3 }));
    {
      auto basicBlockNode1 = branchNode->child(0);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode1)
          == BasicBlockAnnotationSet({ v1 }, { v2, v3 }, { v2, v3 }));

      auto basicBlockNode2 = branchNode->child(1);
      assert(
          demandMap->Lookup<BasicBlockAnnotationSet>(*basicBlockNode2)
          == BasicBlockAnnotationSet({ v1 }, { v3 }, { v3 }));
    }

    auto joinNode = aggregationTreeRoot->child(2);
    assert(
        demandMap->Lookup<BasicBlockAnnotationSet>(*joinNode)
        == BasicBlockAnnotationSet({}, {}, {}));

    auto exitNode = aggregationTreeRoot->child(3);
    assert(
        demandMap->Lookup<ExitAnnotationSet>(*exitNode)
        == ExitAnnotationSet({ v1, v2, v3 }, {}, {}));
  }
}

static void
TestAnnotation()
{
  TestBasicBlockAnnotation();
  TestLinearSubgraphAnnotation();
  TestBranchAnnotation();
  TestLoopAnnotation();
  TestBranchInLoopAnnotation();
  TestAssignmentAnnotation();
  TestBranchPassByAnnotation();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TestAnnotation", TestAnnotation)
