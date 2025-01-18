/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::util::StatisticsCollector statisticsCollector;

static void
test1()
{
  using namespace jlm::llvm;

  // Arrange
  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto i = &jlm::tests::GraphImport::Create(graph, jlm::tests::valuetype::Create(), "i");

  auto SetupF1 = [&]()
  {
    auto vt = jlm::tests::valuetype::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { vt, iostatetype::Create(), MemoryStateType::Create() },
        { vt, iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(&graph.GetRootRegion(), functionType, "f1", linkage::external_linkage);
    lambda->AddContextVar(*i);

    auto t = jlm::tests::test_op::create(
        lambda->subregion(),
        { lambda->GetFunctionArguments()[0] },
        { vt });

    return lambda->finalize(
        { t->output(0), lambda->GetFunctionArguments()[1], lambda->GetFunctionArguments()[2] });
  };

  auto SetupF2 = [&](jlm::rvsdg::output * f1)
  {
    auto vt = jlm::tests::valuetype::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto ct = jlm::rvsdg::ControlType::Create(2);
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::ControlType::Create(2),
          vt,
          iostatetype::Create(),
          MemoryStateType::Create() },
        { vt, iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(&graph.GetRootRegion(), functionType, "f1", linkage::external_linkage);
    auto d = lambda->AddContextVar(*f1).inner;
    auto controlArgument = lambda->GetFunctionArguments()[0];
    auto valueArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto gamma = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    auto gammaInputF1 = gamma->AddEntryVar(d);
    auto gammaInputValue = gamma->AddEntryVar(valueArgument);
    auto gammaInputIoState = gamma->AddEntryVar(iOStateArgument);
    auto gammaInputMemoryState = gamma->AddEntryVar(memoryStateArgument);

    auto callResults = CallNode::Create(
        gammaInputF1.branchArgument[0],
        jlm::rvsdg::AssertGetOwnerNode<lambda::node>(*f1).Type(),
        { gammaInputValue.branchArgument[0],
          gammaInputIoState.branchArgument[0],
          gammaInputMemoryState.branchArgument[0] });

    auto gammaOutputValue =
        gamma->AddExitVar({ callResults[0], gammaInputValue.branchArgument[1] });
    auto gammaOutputIoState =
        gamma->AddExitVar({ callResults[1], gammaInputIoState.branchArgument[1] });
    auto gammaOutputMemoryState =
        gamma->AddExitVar({ callResults[2], gammaInputMemoryState.branchArgument[1] });

    return lambda->finalize(
        { gammaOutputValue.output, gammaOutputIoState.output, gammaOutputMemoryState.output });
  };

  auto f1 = SetupF1();
  auto f2 = SetupF2(f1);

  GraphExport::Create(*f2, "f2");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  jlm::llvm::fctinline fctinline;
  fctinline.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  assert(!jlm::rvsdg::Region::Contains<CallOperation>(graph.GetRootRegion(), true));
}

static void
test2()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto functionType1 = jlm::rvsdg::FunctionType::Create(
      { vt, iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto pt = PointerType::Create();

  auto functionType2 = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto i = &jlm::tests::GraphImport::Create(graph, functionType2, "i");

  auto SetupF1 = [&](const std::shared_ptr<const jlm::rvsdg::FunctionType> & functionType)
  {
    auto lambda =
        lambda::node::create(&graph.GetRootRegion(), functionType, "f1", linkage::external_linkage);
    return lambda->finalize(
        { lambda->GetFunctionArguments()[1], lambda->GetFunctionArguments()[2] });
  };

  auto SetupF2 = [&](jlm::rvsdg::output * f1)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(&graph.GetRootRegion(), functionType, "f2", linkage::external_linkage);
    auto cvi = lambda->AddContextVar(*i).inner;
    auto cvf1 = lambda->AddContextVar(*f1).inner;
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto callResults =
        CallNode::Create(cvi, functionType2, { cvf1, iOStateArgument, memoryStateArgument });

    return lambda->finalize(callResults);
  };

  auto f1 = SetupF1(functionType1);
  auto f2 = SetupF2(
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ f1 }, functionType1).output(0));

  GraphExport::Create(*f2, "f2");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  jlm::llvm::fctinline fctinline;
  fctinline.Run(rm, statisticsCollector);
  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  // Function f1 should not have been inlined.
  assert(is<CallOperation>(jlm::rvsdg::output::GetNode(
      *jlm::rvsdg::AssertGetOwnerNode<lambda::node>(*f2).GetFunctionResults()[0]->origin())));
}

static int
verify()
{
  test1();
  test2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining", verify)
