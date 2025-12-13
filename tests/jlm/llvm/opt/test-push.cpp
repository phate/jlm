/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/util/Statistics.hpp>

static void
simpleGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto entryVar = gammaNode->AddEntryVar(valueArgument);

  // gamma subregion 0
  auto constantNode = TestOperation::create(gammaNode->subregion(0), {}, { valueType });
  auto binaryNode = TestOperation::create(
      gammaNode->subregion(0),
      { entryVar.branchArgument[0], constantNode->output(0) },
      { valueType });

  // gamma subregion 1
  auto unaryNode =
      TestOperation::create(gammaNode->subregion(1), { entryVar.branchArgument[1] }, { valueType });

  auto exitVar = gammaNode->AddExitVar({ binaryNode->output(0), unaryNode->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ exitVar.output });

  GraphExport::Create(*lambdaOutput, "x");

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // All nodes from the gamma subregions should have been hoisted to the lambda subregion
  assert(lambdaNode->subregion()->numNodes() == 4);

  // The original nodes in the gamma subregions should have been removed
  assert(gammaNode->subregion(0)->numNodes() == 0);
  assert(gammaNode->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-simpleGamma", simpleGamma)

static void
nestedGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto gammaNode1 = GammaNode::create(controlArgument, 2);
  auto controlEntryVar = gammaNode1->AddEntryVar(controlArgument);
  auto valueEntryVar1 = gammaNode1->AddEntryVar(valueArgument);

  // gamma1 subregion 0
  auto constantNode1 = TestOperation::create(gammaNode1->subregion(0), {}, { valueType });

  auto gammaNode2 = GammaNode::create(controlEntryVar.branchArgument[0], 2);
  auto valueEntryVar2 = gammaNode2->AddEntryVar(valueEntryVar1.branchArgument[0]);
  auto valueEntryVar3 = gammaNode2->AddEntryVar(constantNode1->output(0));

  // gamma2 subregion 0
  auto binaryNode = TestOperation::create(
      gammaNode1->subregion(0),
      { valueEntryVar2.branchArgument[0], valueEntryVar3.branchArgument[0] },
      { valueType });

  // gamma2 subregion 1
  auto unaryNode = TestOperation::create(
      gammaNode1->subregion(1),
      { valueEntryVar2.branchArgument[1] },
      { valueType });

  auto exitVar1 = gammaNode2->AddExitVar({ binaryNode->output(0), unaryNode->output(0) });

  // gamma1 subregion 1
  auto constantNode2 = TestOperation::create(gammaNode1->subregion(1), {}, { valueType });

  auto exitVar2 = gammaNode1->AddExitVar({ exitVar1.output, constantNode2->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ exitVar2.output });

  GraphExport::Create(*lambdaOutput, "x");

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // All simple nodes from both gamma subregions should have been hoisted to the lambda subregion
  assert(lambdaNode->subregion()->numNodes() == 5);

  // Only gamma node 2 should be left in gamma node 1 subregion 0
  assert(gammaNode1->subregion(0)->numNodes() == 1);
  assert(gammaNode1->subregion(1)->numNodes() == 0);

  // All nodes should have been hoisted out
  assert(gammaNode2->subregion(0)->numNodes() == 0);
  assert(gammaNode2->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-nestedGamma", nestedGamma)

#if 0
static inline void
test_theta()
{
  using namespace jlm::llvm;

  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::tests::TestOperation nop({}, { vt });
  jlm::tests::TestOperation bop({ vt, vt }, { vt });
  jlm::tests::TestOperation sop({ vt, st }, { st });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto s = &jlm::rvsdg::GraphImport::Create(graph, st, "s");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv1 = theta->AddLoopVar(c);
  auto lv2 = theta->AddLoopVar(x);
  auto lv3 = theta->AddLoopVar(x);
  auto lv4 = theta->AddLoopVar(s);

  auto o1 = jlm::tests::TestOperation::create(theta->subregion(), {}, { vt })->output(0);
  auto o2 =
      jlm::tests::TestOperation::create(theta->subregion(), { o1, lv3.pre }, { vt })->output(0);
  auto o3 =
      jlm::tests::TestOperation::create(theta->subregion(), { lv2.pre, o2 }, { vt })->output(0);
  auto o4 = jlm::tests::TestOperation::create(theta->subregion(), { lv3.pre, lv4.pre }, { st })
                ->output(0);

  lv2.post->divert_to(o3);
  lv4.post->divert_to(o4);

  theta->set_predicate(lv1.pre);

  jlm::rvsdg::GraphExport::Create(*theta->output(0), "c");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::NodeHoisting pushout;
  pushout.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(graph.GetRootRegion().numNodes() == 3);
}


static inline void
test_push_theta_bottom()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pt = PointerType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::rvsdg::Graph graph;
  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto a = &jlm::rvsdg::GraphImport::Create(graph, pt, "a");
  auto v = &jlm::rvsdg::GraphImport::Create(graph, vt, "v");
  auto s = &jlm::rvsdg::GraphImport::Create(graph, mt, "s");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lvc = theta->AddLoopVar(c);
  auto lva = theta->AddLoopVar(a);
  auto lvv = theta->AddLoopVar(v);
  auto lvs = theta->AddLoopVar(s);

  auto s1 = StoreNonVolatileOperation::Create(lva.pre, lvv.pre, { lvs.pre }, 4)[0];

  lvs.post->divert_to(s1);
  theta->set_predicate(lvc.pre);

  auto & ex = jlm::rvsdg::GraphExport::Create(*lvs.output, "s");

  jlm::rvsdg::view(graph, stdout);
  jlm::llvm::push_bottom(theta);
  jlm::rvsdg::view(graph, stdout);

  auto storenode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*ex.origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(storenode));
  assert(storenode->input(0)->origin() == a);
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*storenode->input(1)->origin()));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*storenode->input(2)->origin()));
}


static void
ioBarrier()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto valueType = ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & controlImport = jlm::rvsdg::GraphImport::Create(rvsdg, controlType, "control");
  auto & addressImport = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "address");
  auto & ioStateImport = jlm::rvsdg::GraphImport::Create(rvsdg, ioStateType, "ioState");

  auto gammaNode = GammaNode::create(&controlImport, 2);

  auto addressEntryVar = gammaNode->AddEntryVar(&addressImport);
  auto ioStateEntryVar = gammaNode->AddEntryVar(&ioStateImport);

  auto & ioBarrierNode = IOBarrierOperation::createNode(
      *addressEntryVar.branchArgument[0],
      *ioStateEntryVar.branchArgument[0]);

  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, valueType, 4);

  auto undefValue = UndefValueOperation::Create(*gammaNode->subregion(1), valueType);

  auto exitVar = gammaNode->AddExitVar({ loadNode.output(0), undefValue });

  GraphExport::Create(*exitVar.output, "x");

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect that only the undef value was hoisted

  // simpleGamma node and undef value
  assert(rvsdg.GetRootRegion().numNodes() == 2);

  // IOBarrier and load
  assert(gammaNode->subregion(0)->numNodes() == 2);
  assert(gammaNode->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-ioBarrier", ioBarrier)
#endif
#if 0
static void
verify()
{
  test_gamma();
  test_theta();
  // test_push_theta_bottom();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push", verify)
#endif
