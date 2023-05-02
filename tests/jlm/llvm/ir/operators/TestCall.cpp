/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators.hpp>

#include <jlm/rvsdg/view.hpp>

static void
TestCallTypeClassifierIndirectCall()
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
	PointerType pt;
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

    auto alloca = alloca_op::create(pt, one, 8);

    auto store = StoreNode::Create(alloca[0], lambda->fctargument(0), {alloca[1]}, 8);

    auto load = LoadNode::Create(alloca[0], store, pt, 8);

    auto callResults = CallNode::Create(
      load[0],
      fcttype1,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    lambda->finalize(callResults);

    graph->add_export(lambda->output(), {pt, "f"});

    return std::make_tuple(
      AssertedCast<CallNode>(jive::node_output::node(callResults[0])),
      load[0]);
  };

  auto [callNode, loadOutput] = SetupFunction();

  /*
   * Act
   */
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

  /*
   * Assert
   */
  assert(callTypeClassifier->IsIndirectCall());
	assert(loadOutput == &callTypeClassifier->GetFunctionOrigin());
}

static void
TestCallTypeClassifierNonRecursiveDirectCall()
{
  /*
   * Arrange
   */
	using namespace jlm;

	auto module = RvsdgModule::Create(filepath(""), "", "");
	auto graph = &module->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

  valuetype vt;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;

  FunctionType functionTypeG(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&vt, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupFunctionG = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      functionTypeG,
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
      functionTypeG,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    lambda->finalize(callResults);

    return std::make_tuple(
      lambda,
      AssertedCast<CallNode>(jive::node_output::node(callResults[0])));
  };

  auto g = SetupFunctionG();
  auto [f, callNode] = SetupFunctionF(g);

	graph->add_export(f->output(), {PointerType(), "f"});

//	jive::view(graph->root(), stdout);

	// Act
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

	// Assert
  assert(callTypeClassifier->IsNonRecursiveDirectCall());
	assert(&callTypeClassifier->GetLambdaOutput() == g);
}

static void
TestCallTypeClassifierNonRecursiveDirectCallTheta()
{
	using namespace jlm;

	/*
	 * Arrange
	 */
	auto module = RvsdgModule::Create(filepath(""), "", "");
	auto graph = &module->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

  valuetype vt;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;

  FunctionType functionTypeG(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&vt, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupFunctionG = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      functionTypeG,
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
        functionTypeG,
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
	graph->add_export(f, {PointerType(), "f"});

	jive::view(graph->root(), stdout);

	/*
	 * Act
	 */
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

	/*
	 * Assert
	 */
  assert(callTypeClassifier->IsNonRecursiveDirectCall());
  assert(&callTypeClassifier->GetLambdaOutput() == g);
}

static void
TestCallTypeClassifierRecursiveDirectCall()
{
  /*
   * Arrange
   */
  using namespace jlm;

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupFib = [&]()
  {
    PointerType pbit64;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&jive::bit64, &pbit64, &iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});
    PointerType pt;

    jlm::phi::builder pb;
    pb.begin(graph->root());
    auto fibrv = pb.add_recvar(pt);

    auto lambda = lambda::node::create(
      pb.subregion(),
      functionType,
      "fib",
      linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto pointerArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);
    auto ctxVarFib = lambda->add_ctxvar(fibrv->argument());

    auto two = jive::create_bitconstant(lambda->subregion(), 64, 2);
    auto bitult = jive::bitult_op::create(64, valueArgument, two);
    auto predicate = jive::match(1, {{0, 1}}, 0, 2, bitult);

    auto gammaNode = jive::gamma_node::create(predicate, 2);
    auto nev = gammaNode->add_entryvar(valueArgument);
    auto resultev = gammaNode->add_entryvar(pointerArgument);
    auto fibev = gammaNode->add_entryvar(ctxVarFib);
    auto gIIoState = gammaNode->add_entryvar(iOStateArgument);
    auto gIMemoryState = gammaNode->add_entryvar(memoryStateArgument);
    auto gILoopState = gammaNode->add_entryvar(loopStateArgument);

    /* gamma subregion 0 */
    auto one = jive::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jive::bitsub_op::create(64, nev->argument(0), one);
    auto callfibm1Results = CallNode::Create(
      fibev->argument(0),
      functionType,
      {nm1, resultev->argument(0), gIIoState->argument(0), gIMemoryState->argument(0), gILoopState->argument(0)});

    two = jive::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jive::bitsub_op::create(64, nev->argument(0), two);
    auto callfibm2Results = CallNode::Create(
      fibev->argument(0),
      functionType,
      {nm2, resultev->argument(0), callfibm1Results[0], callfibm1Results[1], callfibm1Results[2]});

    auto gepnm1 = GetElementPtrOperation::Create(
      resultev->argument(0),
      {nm1},
      jive::bit64,
      pbit64);
    auto ldnm1 = LoadNode::Create(gepnm1, {callfibm2Results[1]}, jive::bit64, 8);

    auto gepnm2 = GetElementPtrOperation::Create(
      resultev->argument(0),
      {nm2},
      jive::bit64,
      pbit64);
    auto ldnm2 = LoadNode::Create(gepnm2, {ldnm1[1]}, jive::bit64, 8);

    auto sum = jive::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->add_exitvar({sum, nev->argument(1)});
    auto gOIoState = gammaNode->add_exitvar({callfibm2Results[0], gIIoState->argument(1)});
    auto gOMemoryState = gammaNode->add_exitvar({ldnm2[1], gIMemoryState->argument(1)});
    auto gOLoopState = gammaNode->add_exitvar({callfibm2Results[2], gILoopState->argument(1)});

    auto gepn = GetElementPtrOperation::Create(
      pointerArgument,
      {valueArgument},
      jive::bit64,
      pbit64);
    auto store = StoreNode::Create(gepn, sumex, {gOMemoryState}, 8);

    auto lambdaOutput = lambda->finalize({gOIoState, store[0], gOLoopState});

    fibrv->result()->divert_to(lambdaOutput);
    pb.end();

    graph->add_export(fibrv, {pt, "fib"});

    return std::make_tuple(
      lambdaOutput,
      AssertedCast<CallNode>(jive::node_output::node(callfibm1Results[0])),
      AssertedCast<CallNode>(jive::node_output::node(callfibm2Results[0])));
  };

  auto [fibfct, callFib1, callFib2] = SetupFib();

  /*
   * Act
   */
  auto callTypeClassifier1 = CallNode::ClassifyCall(*callFib1);
  auto callTypeClassifier2 = CallNode::ClassifyCall(*callFib2);

  /*
   * Assert
   */
  assert(callTypeClassifier1->IsRecursiveDirectCall());
  assert(&callTypeClassifier1->GetLambdaOutput() == fibfct);

  assert(callTypeClassifier2->IsRecursiveDirectCall());
  assert(&callTypeClassifier2->GetLambdaOutput() == fibfct);
}

static int
Test()
{
  TestCallTypeClassifierIndirectCall();
  TestCallTypeClassifierNonRecursiveDirectCall();
  TestCallTypeClassifierNonRecursiveDirectCallTheta();
  TestCallTypeClassifierRecursiveDirectCall();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestCall", Test)
