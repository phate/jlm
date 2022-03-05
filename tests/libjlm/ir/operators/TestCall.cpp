/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/operators.hpp>

#include <jive/rvsdg/theta.hpp>
#include <jive/view.hpp>

static void
TraceFunctionInputTest1()
{
	using namespace jlm;

  /*
   * Arrange
   */
	valuetype vt;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
	FunctionType fcttype1(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&vt, &iOStateType, &memoryStateType, &loopStateType});
	ptrtype pt(fcttype1);
	FunctionType fcttype2(
    {&pt, &iOStateType, &memoryStateType, &loopStateType},
    {&vt, &iOStateType, &memoryStateType, &loopStateType});

	auto module = RvsdgModule::Create(filepath(""), "", "");
	auto graph = &module->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

  auto SetupFunction = [&]()
  {
    auto lambda = lambda::node::create(graph->root(), fcttype2, "fct", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto one = jive::create_bitconstant(lambda->subregion(), 32, 1);

    auto alloca = alloca_op::create(ptrtype(fcttype1), one, 8);

    auto store = store_op::create(alloca[0], lambda->fctargument(0), {alloca[1]}, 8);

    auto load = LoadOperation::Create(alloca[0], store, 8);

    auto callResults = CallNode::Create(
      load[0],
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    lambda->finalize(callResults);

    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "f"});

    return std::make_tuple(
      AssertedCast<CallNode>(jive::node_output::node(callResults[0])),
      load[0]);
  };

  auto [callNode, loadOutput] = SetupFunction();

  /*
   * Act
   */
  auto tracedOutput = CallNode::TraceFunctionInput(*callNode);

  /*
   * Assert
   */
	assert(loadOutput == tracedOutput);
}

static void
TraceFunctionInputTest2()
{
  /*
   * Arrange
   */
	using namespace jlm;

	auto module = RvsdgModule::Create(filepath(""), "", "");
	auto graph = &module->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

  auto SetupFunctionG = [&]()
  {
    valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;

    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "g",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto constant = test_op::create(lambda->subregion(), {}, {&vt});

    auto lambdaOutput = lambda->finalize(
      {constant->output(0), iOStateArgument, memoryStateArgument, loopStateArgument});

    return lambdaOutput;
  };

  auto SetupFunctionF = [&](lambda::output * g)
  {
    auto SetupOuterTheta = [](jive::region * region, jive::argument * functionG)
    {
      auto outerTheta = jive::theta_node::create(region);
      auto otf = outerTheta->add_loopvar(functionG);

      auto innerTheta = jive::theta_node::create(outerTheta->subregion());
      auto itf = innerTheta->add_loopvar(otf->argument());

      auto predicate = jive_control_false(innerTheta->subregion());
      auto gamma = jive::gamma_node::create(predicate, 2);
      auto ev = gamma->add_entryvar(itf->argument());
      auto xv = gamma->add_exitvar({ev->argument(0), ev->argument(1)});

      itf->result()->divert_to(xv);
      otf->result()->divert_to(itf);

      return otf;
    };

    valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;

    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto functionGArgument = lambda->add_ctxvar(g);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto functionG = SetupOuterTheta(lambda->subregion(), functionGArgument);

    auto callResults = CallNode::Create(
      functionG,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    lambda->finalize(callResults);

    return std::make_tuple(
      lambda,
      AssertedCast<CallNode>(jive::node_output::node(callResults[0])));
  };

  auto g = SetupFunctionG();
  auto [f, callNode] = SetupFunctionF(g);

	graph->add_export(f->output(), {ptrtype(f->type()), "f"});

//	jive::view(graph->root(), stdout);

	// Act
	auto tracedOutput = CallNode::TraceFunctionInput(*callNode);

	// Assert
	assert(tracedOutput == g);
}

static void
TraceFunctionInputTest3()
{
	using namespace jlm;

	/*
	 * Arrange
	 */
	auto module = RvsdgModule::Create(filepath(""), "", "");
	auto graph = &module->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

  auto SetupFunctionG = [&]()
  {
    valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;

    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "g",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto c1 = test_op::create(lambda->subregion(), {}, {&vt});

    return lambda->finalize({c1->output(0), iOStateArgument, memoryStateArgument, loopStateArgument});
  };

  auto SetupFunctionF = [&](lambda::output * g)
  {
    auto SetupOuterTheta = [&](
      jive::region * region,
      jive::argument * g,
      jive::output * value,
      jive::output * iOState,
      jive::output * memoryState,
      jive::output * loopState)
    {
      auto SetupInnerTheta = [&](
        jive::region * region,
        jive::argument * g,
        jive::argument * loopState)
      {
        auto innerTheta = jive::theta_node::create(region);
        auto thetaOutputG = innerTheta->add_loopvar(g);
        auto thetaOutputLoopState = innerTheta->add_loopvar(loopState);

        return std::make_tuple(thetaOutputG, thetaOutputLoopState);
      };

      auto outerTheta = jive::theta_node::create(region);
      auto thetaOutputG = outerTheta->add_loopvar(g);
      auto thetaOutputValue = outerTheta->add_loopvar(value);
      auto thetaOutputIoState = outerTheta->add_loopvar(iOState);
      auto thetaOutputMemoryState = outerTheta->add_loopvar(memoryState);
      auto thetaOutputLoopState = outerTheta->add_loopvar(loopState);

      auto [functionG, innerLoopState] = SetupInnerTheta(
        outerTheta->subregion(),
        thetaOutputG->argument(),
        thetaOutputLoopState->argument());

      auto callResults = CallNode::Create(
        functionG,
        {thetaOutputIoState->argument(), thetaOutputMemoryState->argument(), innerLoopState});

      thetaOutputG->result()->divert_to(functionG);
      thetaOutputValue->result()->divert_to(callResults[0]);
      thetaOutputIoState->result()->divert_to(callResults[1]);
      thetaOutputMemoryState->result()->divert_to(callResults[2]);
      thetaOutputLoopState->result()->divert_to(callResults[3]);

      return std::make_tuple(
        thetaOutputValue,
        thetaOutputIoState,
        thetaOutputMemoryState,
        thetaOutputLoopState,
        AssertedCast<CallNode>(jive::node_output::node(callResults[0])));
    };

    valuetype vt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;

    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&vt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto functionG = lambda->add_ctxvar(g);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto value = test_op::create(lambda->subregion(), {}, {&vt})->output(0);

    auto [loopValue, iOState, memoryState, loopState, callNode] = SetupOuterTheta(
      lambda->subregion(),
      functionG,
      value,
      iOStateArgument,
      memoryStateArgument,
      loopStateArgument);

    auto lambdaOutput = lambda->finalize({loopValue, iOState, memoryState, loopState});

    return std::make_tuple(lambdaOutput, callNode);
  };

  auto g = SetupFunctionG();
  auto [f, callNode] = SetupFunctionF(g);
	graph->add_export(f, {ptrtype(f->node()->type()), "f"});

	jive::view(graph->root(), stdout);

	/*
	 * Act
	 */
	auto tracedOutput = CallNode::TraceFunctionInput(*callNode);

	/*
	 * Assert
	 */
	assert(tracedOutput == g);

}

static int
Test()
{
  TraceFunctionInputTest1();
  TraceFunctionInputTest2();
  TraceFunctionInputTest3();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/TestCall", Test)
