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

static jlm::StatisticsCollector statisticsCollector;

static void
test1()
{
	using namespace jlm;

  /**
   * Arrange
   */
	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto i = graph.add_import({jlm::valuetype(), "i"});

  auto SetupF1 = [&]()
  {
    jlm::valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&vt, &iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph.root(),
      functionType,
      "f1",
      linkage::external_linkage);
    lambda->add_ctxvar(i);

    auto t = test_op::create(lambda->subregion(), {lambda->fctargument(0)}, {&vt});

    return lambda->finalize({t->output(0), lambda->fctargument(1), lambda->fctargument(2), lambda->fctargument(3)});
  };

  auto SetupF2 = [&](lambda::output * f1)
  {
    jlm::valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    jive::ctltype ct(2);
    FunctionType functionType(
      {&ct, &vt, &iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph.root(),
      functionType,
      "f1",
      linkage::external_linkage);
    auto d = lambda->add_ctxvar(f1);
    auto controlArgument = lambda->fctargument(0);
    auto valueArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);

    auto gamma = jive::gamma_node::create(controlArgument, 2);
    auto gammaInputF1 = gamma->add_entryvar(d);
    auto gammaInputValue = gamma->add_entryvar(valueArgument);
    auto gammaInputIoState = gamma->add_entryvar(iOStateArgument);
    auto gammaInputMemoryState = gamma->add_entryvar(memoryStateArgument);
    auto gammaInputLoopState = gamma->add_entryvar(loopStateArgument);

    auto callResults = CallNode::Create(
      gammaInputF1->argument(0),
      f1->node()->type(),
      {gammaInputValue->argument(0), gammaInputIoState->argument(0), gammaInputMemoryState->argument(0),
       gammaInputLoopState->argument(0)});

    auto gammaOutputValue = gamma->add_exitvar({callResults[0], gammaInputValue->argument(1)});
    auto gammaOutputIoState = gamma->add_exitvar({callResults[1], gammaInputIoState->argument(1)});
    auto gammaOutputMemoryState = gamma->add_exitvar({callResults[2], gammaInputMemoryState->argument(1)});
    auto gammaOutputLoopState = gamma->add_exitvar({callResults[3], gammaInputLoopState->argument(1)});

    return lambda->finalize({gammaOutputValue, gammaOutputIoState, gammaOutputMemoryState, gammaOutputLoopState});
  };

  auto f1 = SetupF1();
  auto f2 = SetupF2(f1);

	graph.add_export(f2, {f2->type(), "f2"});

//	jive::view(graph.root(), stdout);

  /*
   * Act
   */
	jlm::fctinline fctinline;
	fctinline.run(rm, statisticsCollector);
//	jive::view(graph.root(), stdout);

  /*
   * Assert
   */
	assert(!jive::region::Contains<jlm::CallOperation>(*graph.root(), true));
}

static void
test2()
{
  /*
   * Arrange
   */
	using namespace jlm;

	valuetype vt;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;

  FunctionType functionType1(
    {&vt, &iOStateType, &memoryStateType, &loopStateType},
    {&iOStateType, &memoryStateType, &loopStateType});
	PointerType pt;

	FunctionType functionType2(
    {&pt, &iOStateType, &memoryStateType, &loopStateType},
    {&iOStateType, &memoryStateType, &loopStateType});


	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();
	auto i = graph.add_import({pt, "i"});

  auto SetupF1 = [&](const FunctionType & functionType)
  {
    auto lambda = lambda::node::create(
      graph.root(),
      functionType,
      "f1",
      linkage::external_linkage);
    return lambda->finalize({lambda->fctargument(1), lambda->fctargument(2), lambda->fctargument(3)});
  };

  auto SetupF2 = [&](lambda::output * f1)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph.root(),
      functionType,
      "f2",
      linkage::external_linkage);
    auto cvi = lambda->add_ctxvar(i);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto callResults = CallNode::Create(
      cvi,
      functionType2,
      {cvf1, iOStateArgument, memoryStateArgument, loopStateArgument});

    return lambda->finalize(callResults);
  };

  auto f1 = SetupF1(functionType1);
  auto f2 = SetupF2(f1);

	graph.add_export(f2, {f2->type(), "f2"});

	jive::view(graph.root(), stdout);

  /*
   * Act
   */
	jlm::fctinline fctinline;
	fctinline.run(rm, statisticsCollector);
	jive::view(graph.root(), stdout);

  /*
   * Assert
   *
   * Function f1 should not have been inlined.
   */
	assert(is<CallOperation>(jive::node_output::node(f2->node()->fctresult(0)->origin())));
}

static int
verify()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining", verify)
