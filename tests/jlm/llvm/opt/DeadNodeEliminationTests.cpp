/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunDeadNodeElimination(jlm::llvm::RvsdgModule & rvsdgModule)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::DeadNodeElimination deadNodeElimination;
  deadNodeElimination.Run(rvsdgModule, statisticsCollector);
}

static void
RootRegion()
{
  using namespace jlm::llvm;

  // Arrange
  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  jlm::rvsdg::GraphImport::Create(graph, jlm::tests::ValueType::Create(), "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, jlm::tests::ValueType::Create(), "y");

  jlm::rvsdg::GraphExport::Create(*y, "z");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(graph.GetRootRegion().narguments() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-RootRegion", RootRegion)

static void
Gamma1()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto c = &jlm::rvsdg::GraphImport::Create(graph, controlType, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);
  auto ev1 = gamma->AddEntryVar(x);
  auto ev2 = gamma->AddEntryVar(y);
  auto ev3 = gamma->AddEntryVar(x);

  auto t =
      jlm::tests::create_testop(gamma->subregion(1), { ev2.branchArgument[1] }, { valueType })[0];

  gamma->AddExitVar(ev1.branchArgument);
  gamma->AddExitVar({ ev2.branchArgument[0], t });
  gamma->AddExitVar({ ev3.branchArgument[0], ev1.branchArgument[1] });

  jlm::rvsdg::GraphExport::Create(*gamma->output(0), "z");
  jlm::rvsdg::GraphExport::Create(*gamma->output(2), "w");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(gamma->noutputs() == 2);
  assert(gamma->subregion(1)->nnodes() == 0);
  assert(gamma->subregion(1)->narguments() == 3);
  assert(gamma->ninputs() == 3);
  assert(graph.GetRootRegion().narguments() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Gamma1", Gamma1)

static void
Gamma2()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto c = &jlm::rvsdg::GraphImport::Create(graph, controlType, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);
  gamma->AddEntryVar(x);

  auto n1 = jlm::tests::create_testop(gamma->subregion(0), {}, { valueType })[0];
  auto n2 = jlm::tests::create_testop(gamma->subregion(1), {}, { valueType })[0];

  gamma->AddExitVar({ n1, n2 });

  jlm::rvsdg::GraphExport::Create(*gamma->output(0), "x");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(graph.GetRootRegion().narguments() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Gamma2", Gamma2)

static void
Theta()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, valueType, "z");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv1 = theta->AddLoopVar(x);
  auto lv2 = theta->AddLoopVar(y);
  auto lv3 = theta->AddLoopVar(z);
  auto lv4 = theta->AddLoopVar(y);

  lv1.post->divert_to(lv2.pre);
  lv2.post->divert_to(lv1.pre);

  auto t = jlm::tests::create_testop(theta->subregion(), { lv3.pre }, { valueType })[0];
  lv3.post->divert_to(t);
  lv4.post->divert_to(lv2.pre);

  auto c = jlm::tests::create_testop(theta->subregion(), {}, { controlType })[0];
  theta->set_predicate(c);

  jlm::rvsdg::GraphExport::Create(*lv1.output, "a");
  jlm::rvsdg::GraphExport::Create(*lv4.output, "b");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(theta->noutputs() == 3);
  assert(theta->subregion()->nnodes() == 1);
  assert(graph.GetRootRegion().narguments() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Theta", Theta)

static void
NestedTheta()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto c = &jlm::rvsdg::GraphImport::Create(graph, controlType, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");

  auto outerTheta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lvo1 = outerTheta->AddLoopVar(c);
  auto lvo2 = outerTheta->AddLoopVar(x);
  auto lvo3 = outerTheta->AddLoopVar(y);

  auto innerTheta = jlm::rvsdg::ThetaNode::create(outerTheta->subregion());

  auto lvi1 = innerTheta->AddLoopVar(lvo1.pre);
  auto lvi2 = innerTheta->AddLoopVar(lvo2.pre);
  auto lvi3 = innerTheta->AddLoopVar(lvo3.pre);

  lvi2.post->divert_to(lvi3.pre);

  innerTheta->set_predicate(lvi1.pre);

  lvo2.post->divert_to(lvi2.output);
  lvo3.post->divert_to(lvi2.output);

  outerTheta->set_predicate(lvo1.pre);

  jlm::rvsdg::GraphExport::Create(*lvo3.output, "y");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(outerTheta->noutputs() == 3);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-NestedTheta", NestedTheta)

static void
EvolvingTheta()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto controlType = jlm::rvsdg::ControlType::Create(2);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto c = &jlm::rvsdg::GraphImport::Create(graph, controlType, "c");
  auto x1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x1");
  auto x2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x2");
  auto x3 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x3");
  auto x4 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x4");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv0 = theta->AddLoopVar(c);
  auto lv1 = theta->AddLoopVar(x1);
  auto lv2 = theta->AddLoopVar(x2);
  auto lv3 = theta->AddLoopVar(x3);
  auto lv4 = theta->AddLoopVar(x4);

  lv1.post->divert_to(lv2.pre);
  lv2.post->divert_to(lv3.pre);
  lv3.post->divert_to(lv4.pre);

  theta->set_predicate(lv0.pre);

  jlm::rvsdg::GraphExport::Create(*lv1.output, "x1");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(theta->noutputs() == 5);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-EvolvingTheta", EvolvingTheta)

static void
Lambda()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  auto x = &jlm::rvsdg::GraphImport::Create(graph, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType, valueType }),
          "f",
          linkage::external_linkage));

  auto cv1 = lambda->AddContextVar(*x).inner;
  auto cv2 = lambda->AddContextVar(*y).inner;
  jlm::tests::create_testop(
      lambda->subregion(),
      { lambda->GetFunctionArguments()[0], cv1 },
      { valueType });

  auto output = lambda->finalize({ lambda->GetFunctionArguments()[0], cv2 });

  jlm::rvsdg::GraphExport::Create(*output, "f");
  jlm::rvsdg::view(graph, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(lambda->subregion()->nnodes() == 0);
  assert(graph.GetRootRegion().narguments() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Lambda", Lambda)

static void
Phi()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();
  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "z");

  auto setupF1 = [&](Region & region, Output & rv2, Output & dx)
  {
    auto lambda1 = jlm::rvsdg::LambdaNode::Create(
        region,
        LlvmLambdaOperation::Create(functionType, "f1", linkage::external_linkage));
    auto f2Argument = lambda1->AddContextVar(rv2).inner;
    auto xArgument = lambda1->AddContextVar(dx).inner;

    auto result =
        jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
            { lambda1->GetFunctionArguments()[0], f2Argument, xArgument },
            std::vector<std::shared_ptr<const Type>>{ valueType, functionType, valueType },
            std::vector<std::shared_ptr<const Type>>{ valueType })
            .output(0);

    return lambda1->finalize({ result });
  };

  auto setupF2 = [&](Region & region, Output & rv1, Output & dy)
  {
    auto lambda2 = jlm::rvsdg::LambdaNode::Create(
        region,
        LlvmLambdaOperation::Create(functionType, "f2", linkage::external_linkage));
    auto f1Argument = lambda2->AddContextVar(rv1).inner;
    lambda2->AddContextVar(dy);

    auto result = jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
                      { lambda2->GetFunctionArguments()[0], f1Argument },
                      std::vector<std::shared_ptr<const Type>>{ valueType, functionType },
                      std::vector<std::shared_ptr<const Type>>{ valueType })
                      .output(0);

    return lambda2->finalize({ result });
  };

  auto setupF3 = [&](Region & region, Output & dz)
  {
    auto lambda3 = jlm::rvsdg::LambdaNode::Create(
        region,
        LlvmLambdaOperation::Create(functionType, "f3", linkage::external_linkage));
    auto zArgument = lambda3->AddContextVar(dz).inner;

    auto result = jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
                      { lambda3->GetFunctionArguments()[0], zArgument },
                      std::vector<std::shared_ptr<const Type>>{ valueType, valueType },
                      std::vector<std::shared_ptr<const Type>>{ valueType })
                      .output(0);

    return lambda3->finalize({ result });
  };

  auto setupF4 = [&](Region & region)
  {
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        region,
        LlvmLambdaOperation::Create(functionType, "f4", linkage::external_linkage));
    return lambda->finalize({ lambda->GetFunctionArguments()[0] });
  };

  PhiBuilder phiBuilder;
  phiBuilder.begin(&rvsdg.GetRootRegion());
  auto & phiSubregion = *phiBuilder.subregion();

  auto rv1 = phiBuilder.AddFixVar(functionType);
  auto rv2 = phiBuilder.AddFixVar(functionType);
  auto rv3 = phiBuilder.AddFixVar(functionType);
  auto rv4 = phiBuilder.AddFixVar(functionType);
  auto dx = phiBuilder.AddContextVar(*x);
  auto dy = phiBuilder.AddContextVar(*y);
  auto dz = phiBuilder.AddContextVar(*z);

  auto f1 = setupF1(phiSubregion, *rv2.recref, *dx.inner);
  auto f2 = setupF2(phiSubregion, *rv1.recref, *dy.inner);
  auto f3 = setupF3(phiSubregion, *dz.inner);
  auto f4 = setupF4(phiSubregion);

  rv1.result->divert_to(f1);
  rv2.result->divert_to(f2);
  rv3.result->divert_to(f3);
  rv4.result->divert_to(f4);
  auto phiNode = phiBuilder.end();

  jlm::rvsdg::GraphExport::Create(*phiNode->output(0), "f1");
  jlm::rvsdg::GraphExport::Create(*phiNode->output(3), "f4");
  view(rvsdg, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  view(rvsdg, stdout);

  // Assert
  assert(phiNode->noutputs() == 3); // f1, f2, and f4 are alive
  assert(phiNode->output(0) == rv1.output);
  assert(phiNode->output(1) == rv2.output);
  assert(phiNode->output(2) == rv4.output);
  assert(phiSubregion.nresults() == 3); // f1, f2, and f4 are alive
  assert(phiSubregion.result(0) == rv1.result);
  assert(phiSubregion.result(1) == rv2.result);
  assert(phiSubregion.result(2) == rv4.result);
  assert(phiSubregion.narguments() == 4); // f1, f2, f4, and dx are alive
  assert(phiSubregion.argument(0) == rv1.recref);
  assert(phiSubregion.argument(1) == rv2.recref);
  assert(phiSubregion.argument(2) == rv4.recref);
  assert(phiSubregion.argument(3) == dx.inner);
  assert(phiNode->ninputs() == 1); // dx is alive
  assert(phiNode->input(0) == dx.input);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Phi", Phi)

static void
Delta()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "z");

  auto deltaNode = jlm::rvsdg::DeltaNode::Create(
      &rvsdg.GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(valueType, "delta", linkage::external_linkage, "", false));

  auto xArgument = deltaNode->AddContextVar(*x).inner;
  deltaNode->AddContextVar(*y);
  auto zArgument = deltaNode->AddContextVar(*z).inner;

  auto result = jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
                    { xArgument },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
      { zArgument },
      std::vector<std::shared_ptr<const Type>>{ valueType },
      std::vector<std::shared_ptr<const Type>>{ valueType });

  auto deltaOutput = &deltaNode->finalize(result);
  jlm::rvsdg::GraphExport::Create(*deltaOutput, "");
  view(rvsdg, stdout);

  // Act
  RunDeadNodeElimination(rvsdgModule);
  view(rvsdg, stdout);

  // Assert
  assert(deltaNode->subregion()->nnodes() == 1);
  assert(deltaNode->ninputs() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/DeadNodeEliminationTests-Delta", Delta)
