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
    auto functionType = FunctionType::Create(
        { vt, iostatetype::Create(), MemoryStateType::Create() },
        { vt, iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph.root(), functionType, "f1", linkage::external_linkage);
    lambda->add_ctxvar(i);

    auto t = jlm::tests::test_op::create(lambda->subregion(), { lambda->fctargument(0) }, { vt });

    return lambda->finalize({ t->output(0), lambda->fctargument(1), lambda->fctargument(2) });
  };

  auto SetupF2 = [&](lambda::output * f1)
  {
    auto vt = jlm::tests::valuetype::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto ct = jlm::rvsdg::ctltype::Create(2);
    auto functionType = FunctionType::Create(
        { jlm::rvsdg::ctltype::Create(2), vt, iostatetype::Create(), MemoryStateType::Create() },
        { vt, iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph.root(), functionType, "f1", linkage::external_linkage);
    auto d = lambda->add_ctxvar(f1);
    auto controlArgument = lambda->fctargument(0);
    auto valueArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

    auto gamma = jlm::rvsdg::gamma_node::create(controlArgument, 2);
    auto gammaInputF1 = gamma->add_entryvar(d);
    auto gammaInputValue = gamma->add_entryvar(valueArgument);
    auto gammaInputIoState = gamma->add_entryvar(iOStateArgument);
    auto gammaInputMemoryState = gamma->add_entryvar(memoryStateArgument);

    auto callResults = CallNode::Create(
        gammaInputF1->argument(0),
        f1->node()->Type(),
        { gammaInputValue->argument(0),
          gammaInputIoState->argument(0),
          gammaInputMemoryState->argument(0) });

    auto gammaOutputValue = gamma->add_exitvar({ callResults[0], gammaInputValue->argument(1) });
    auto gammaOutputIoState =
        gamma->add_exitvar({ callResults[1], gammaInputIoState->argument(1) });
    auto gammaOutputMemoryState =
        gamma->add_exitvar({ callResults[2], gammaInputMemoryState->argument(1) });

    return lambda->finalize({ gammaOutputValue, gammaOutputIoState, gammaOutputMemoryState });
  };

  auto f1 = SetupF1();
  auto f2 = SetupF2(f1);

  graph.add_export(f2, { f2->Type(), "f2" });

  //	jlm::rvsdg::view(graph.root(), stdout);

  // Act
  jlm::llvm::fctinline fctinline;
  fctinline.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  assert(!jlm::rvsdg::region::Contains<CallOperation>(*graph.root(), true));
}

static void
test2()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto functionType1 = FunctionType::Create(
      { vt, iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto pt = PointerType::Create();

  auto functionType2 = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto i = &jlm::tests::GraphImport::Create(graph, pt, "i");

  auto SetupF1 = [&](const std::shared_ptr<const FunctionType> & functionType)
  {
    auto lambda = lambda::node::create(graph.root(), functionType, "f1", linkage::external_linkage);
    return lambda->finalize({ lambda->fctargument(1), lambda->fctargument(2) });
  };

  auto SetupF2 = [&](lambda::output * f1)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph.root(), functionType, "f2", linkage::external_linkage);
    auto cvi = lambda->add_ctxvar(i);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto callResults =
        CallNode::Create(cvi, functionType2, { cvf1, iOStateArgument, memoryStateArgument });

    return lambda->finalize(callResults);
  };

  auto f1 = SetupF1(functionType1);
  auto f2 = SetupF2(f1);

  graph.add_export(f2, { f2->Type(), "f2" });

  jlm::rvsdg::view(graph.root(), stdout);

  // Act
  jlm::llvm::fctinline fctinline;
  fctinline.run(rm, statisticsCollector);
  jlm::rvsdg::view(graph.root(), stdout);

  // Assert
  // Function f1 should not have been inlined.
  assert(is<CallOperation>(jlm::rvsdg::node_output::node(f2->node()->fctresult(0)->origin())));
}

static int
verify()
{
  test1();
  test2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining", verify)
