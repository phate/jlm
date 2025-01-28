/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestCopy()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { valueType, IOStateType::Create(), MemoryStateType::Create() },
      { valueType, IOStateType::Create(), MemoryStateType::Create() });

  jlm::rvsdg::Graph rvsdg;
  auto function1 = &jlm::tests::GraphImport::Create(rvsdg, functionType, "function1");
  auto value1 = &jlm::tests::GraphImport::Create(rvsdg, valueType, "value1");
  auto iOState1 = &jlm::tests::GraphImport::Create(rvsdg, iOStateType, "iOState1");
  auto memoryState1 = &jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "memoryState1");

  auto function2 = &jlm::tests::GraphImport::Create(rvsdg, functionType, "function2");
  auto value2 = &jlm::tests::GraphImport::Create(rvsdg, valueType, "value2");
  auto iOState2 = &jlm::tests::GraphImport::Create(rvsdg, iOStateType, "iOState2");
  auto memoryState2 = &jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "memoryState2");

  auto callResults = CallNode::Create(function1, functionType, { value1, iOState1, memoryState1 });

  // Act
  auto node = jlm::rvsdg::output::GetNode(*callResults[0]);
  auto callNode = jlm::util::AssertedCast<const CallNode>(node);
  auto copiedNode =
      callNode->copy(&rvsdg.GetRootRegion(), { function2, value2, iOState2, memoryState2 });

  // Assert
  auto copiedCallNode = dynamic_cast<const CallNode *>(copiedNode);
  assert(copiedNode != nullptr);
  assert(callNode->GetOperation() == copiedCallNode->GetOperation());
}

static void
TestCallNodeAccessors()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { valueType, IOStateType::Create(), MemoryStateType::Create() },
      { valueType, IOStateType::Create(), MemoryStateType::Create() });

  jlm::rvsdg::Graph rvsdg;
  auto f = &jlm::tests::GraphImport::Create(rvsdg, functionType, "function");
  auto v = &jlm::tests::GraphImport::Create(rvsdg, valueType, "value");
  auto i = &jlm::tests::GraphImport::Create(rvsdg, iOStateType, "IOState");
  auto m = &jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "memoryState");

  // Act
  auto results = CallNode::Create(f, functionType, { v, i, m });
  auto & callNode = *jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*results[0]));

  // Assert
  assert(callNode.NumArguments() == 3);
  assert(callNode.NumArguments() == callNode.ninputs() - 1);
  assert(callNode.Argument(0)->origin() == v);
  assert(callNode.Argument(1)->origin() == i);
  assert(callNode.Argument(2)->origin() == m);

  assert(callNode.NumResults() == 3);
  assert(callNode.Result(0)->type() == *valueType);
  assert(callNode.Result(1)->type() == *iOStateType);
  assert(callNode.Result(2)->type() == *memoryStateType);

  assert(callNode.GetFunctionInput()->origin() == f);
  assert(callNode.GetIoStateInput()->origin() == i);
  assert(callNode.GetMemoryStateInput()->origin() == m);

  assert(callNode.GetIoStateOutput()->type() == *iOStateType);
  assert(callNode.GetMemoryStateOutput()->type() == *memoryStateType);
}

static void
TestCallTypeClassifierIndirectCall()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto fcttype1 = jlm::rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { vt, IOStateType::Create(), MemoryStateType::Create() });
  auto fcttype2 = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { vt, IOStateType::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupFunction = [&]()
  {
    auto lambda =
        lambda::node::create(&graph->GetRootRegion(), fcttype2, "fct", linkage::external_linkage);
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto one = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 1);

    auto alloca = alloca_op::create(PointerType::Create(), one, 8);

    auto store = StoreNonVolatileNode::Create(
        alloca[0],
        lambda->GetFunctionArguments()[0],
        { alloca[1] },
        8);

    auto load = LoadNonVolatileNode::Create(alloca[0], store, PointerType::Create(), 8);
    auto fn = jlm::rvsdg::CreateOpNode<PointerToFunctionOperation>({ load[0] }, fcttype1).output(0);

    auto callResults = CallNode::Create(fn, fcttype1, { iOStateArgument, memoryStateArgument });

    lambda->finalize(callResults);

    GraphExport::Create(*lambda->output(), "f");

    return std::make_tuple(
        jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*callResults[0])),
        fn);
  };

  auto [callNode, loadOutput] = SetupFunction();

  // Act
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

  // Assert
  assert(callTypeClassifier->IsIndirectCall());
  assert(loadOutput == &callTypeClassifier->GetFunctionOrigin());
}

static void
TestCallTypeClassifierNonRecursiveDirectCall()
{
  // Arrange
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto vt = jlm::tests::valuetype::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto functionTypeG = jlm::rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { vt, IOStateType::Create(), MemoryStateType::Create() });

  auto SetupFunctionG = [&]()
  {
    auto lambda = lambda::node::create(
        &graph->GetRootRegion(),
        functionTypeG,
        "g",
        linkage::external_linkage);
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constant = jlm::tests::test_op::create(lambda->subregion(), {}, { vt });

    auto lambdaOutput =
        lambda->finalize({ constant->output(0), iOStateArgument, memoryStateArgument });

    return lambdaOutput;
  };

  auto SetupFunctionF = [&](jlm::rvsdg::output * g)
  {
    auto SetupOuterTheta = [](jlm::rvsdg::Region * region, jlm::rvsdg::output * functionG)
    {
      auto outerTheta = jlm::rvsdg::ThetaNode::create(region);
      auto otf = outerTheta->AddLoopVar(functionG);

      auto innerTheta = jlm::rvsdg::ThetaNode::create(outerTheta->subregion());
      auto itf = innerTheta->AddLoopVar(otf.pre);

      auto predicate = jlm::rvsdg::control_false(innerTheta->subregion());
      auto gamma = jlm::rvsdg::GammaNode::create(predicate, 2);
      auto ev = gamma->AddEntryVar(itf.pre);
      auto xv = gamma->AddExitVar(ev.branchArgument);

      itf.post->divert_to(xv.output);
      otf.post->divert_to(itf.output);

      return otf;
    };

    auto vt = jlm::tests::valuetype::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();

    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { vt, IOStateType::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(&graph->GetRootRegion(), functionType, "f", linkage::external_linkage);
    auto functionGArgument = lambda->AddContextVar(*g).inner;
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto functionG = SetupOuterTheta(lambda->subregion(), functionGArgument);

    auto callResults =
        CallNode::Create(functionG.output, functionTypeG, { iOStateArgument, memoryStateArgument });

    lambda->finalize(callResults);

    return std::make_tuple(
        lambda,
        jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*callResults[0])));
  };

  auto g = SetupFunctionG();
  auto [f, callNode] = SetupFunctionF(g);

  GraphExport::Create(*f->output(), "f");

  //	jlm::rvsdg::view(&graph->GetRootRegion(), stdout);

  // Act
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

  // Assert
  assert(callTypeClassifier->IsNonRecursiveDirectCall());
  assert(&callTypeClassifier->GetLambdaOutput() == g);
}

static void
TestCallTypeClassifierNonRecursiveDirectCallTheta()
{
  using namespace jlm::llvm;

  // Arrange
  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto vt = jlm::tests::valuetype::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto functionTypeG = jlm::rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { vt, IOStateType::Create(), MemoryStateType::Create() });

  auto SetupFunctionG = [&]()
  {
    auto lambda = lambda::node::create(
        &graph->GetRootRegion(),
        functionTypeG,
        "g",
        linkage::external_linkage);
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto c1 = jlm::tests::test_op::create(lambda->subregion(), {}, { vt });

    return lambda->finalize({ c1->output(0), iOStateArgument, memoryStateArgument });
  };

  auto SetupFunctionF = [&](jlm::rvsdg::output * g)
  {
    auto SetupOuterTheta = [&](jlm::rvsdg::Region * region,
                               jlm::rvsdg::output * g,
                               jlm::rvsdg::output * value,
                               jlm::rvsdg::output * iOState,
                               jlm::rvsdg::output * memoryState)
    {
      auto SetupInnerTheta = [&](jlm::rvsdg::Region * region, jlm::rvsdg::output * g)
      {
        auto innerTheta = jlm::rvsdg::ThetaNode::create(region);
        auto thetaOutputG = innerTheta->AddLoopVar(g);

        return thetaOutputG;
      };

      auto outerTheta = jlm::rvsdg::ThetaNode::create(region);
      auto thetaOutputG = outerTheta->AddLoopVar(g);
      auto thetaOutputValue = outerTheta->AddLoopVar(value);
      auto thetaOutputIoState = outerTheta->AddLoopVar(iOState);
      auto thetaOutputMemoryState = outerTheta->AddLoopVar(memoryState);

      auto functionG = SetupInnerTheta(outerTheta->subregion(), thetaOutputG.pre);

      auto callResults = CallNode::Create(
          functionG.output,
          functionTypeG,
          { thetaOutputIoState.pre, thetaOutputMemoryState.pre });

      thetaOutputG.post->divert_to(functionG.output);
      thetaOutputValue.post->divert_to(callResults[0]);
      thetaOutputIoState.post->divert_to(callResults[1]);
      thetaOutputMemoryState.post->divert_to(callResults[2]);

      return std::make_tuple(
          thetaOutputValue,
          thetaOutputIoState,
          thetaOutputMemoryState,
          jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*callResults[0])));
    };

    auto vt = jlm::tests::valuetype::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();

    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { vt, IOStateType::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(&graph->GetRootRegion(), functionType, "f", linkage::external_linkage);
    auto functionG = lambda->AddContextVar(*g).inner;
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto value = jlm::tests::test_op::create(lambda->subregion(), {}, { vt })->output(0);

    auto [loopValue, iOState, memoryState, callNode] = SetupOuterTheta(
        lambda->subregion(),
        functionG,
        value,
        iOStateArgument,
        memoryStateArgument);

    auto lambdaOutput = lambda->finalize({ loopValue.output, iOState.output, memoryState.output });

    return std::make_tuple(lambdaOutput, callNode);
  };

  auto g = SetupFunctionG();
  auto [f, callNode] = SetupFunctionF(g);
  GraphExport::Create(*f, "f");

  jlm::rvsdg::view(&graph->GetRootRegion(), stdout);

  // Act
  auto callTypeClassifier = CallNode::ClassifyCall(*callNode);

  // Assert
  assert(callTypeClassifier->IsNonRecursiveDirectCall());
  assert(&callTypeClassifier->GetLambdaOutput() == g);
}

static void
TestCallTypeClassifierRecursiveDirectCall()
{
  // Arrange
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupFib = [&]()
  {
    auto pbit64 = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });
    auto pt = PointerType::Create();

    jlm::llvm::phi::builder pb;
    pb.begin(&graph->GetRootRegion());
    auto fibrv = pb.add_recvar(functionType);

    auto lambda =
        lambda::node::create(pb.subregion(), functionType, "fib", linkage::external_linkage);
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto pointerArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];
    auto ctxVarFib = lambda->AddContextVar(*fibrv->argument()).inner;

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 2);
    auto bitult = jlm::rvsdg::bitult_op::create(64, valueArgument, two);
    auto predicate = jlm::rvsdg::match(1, { { 0, 1 } }, 0, 2, bitult);

    auto gammaNode = jlm::rvsdg::GammaNode::create(predicate, 2);
    auto nev = gammaNode->AddEntryVar(valueArgument);
    auto resultev = gammaNode->AddEntryVar(pointerArgument);
    auto fibev = gammaNode->AddEntryVar(ctxVarFib);
    auto gIIoState = gammaNode->AddEntryVar(iOStateArgument);
    auto gIMemoryState = gammaNode->AddEntryVar(memoryStateArgument);

    /* gamma subregion 0 */
    auto one = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jlm::rvsdg::bitsub_op::create(64, nev.branchArgument[0], one);
    auto callfibm1Results = CallNode::Create(
        fibev.branchArgument[0],
        functionType,
        { nm1,
          resultev.branchArgument[0],
          gIIoState.branchArgument[0],
          gIMemoryState.branchArgument[0] });

    two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jlm::rvsdg::bitsub_op::create(64, nev.branchArgument[0], two);
    auto callfibm2Results = CallNode::Create(
        fibev.branchArgument[0],
        functionType,
        { nm2, resultev.branchArgument[0], callfibm1Results[0], callfibm1Results[1] });

    auto gepnm1 = GetElementPtrOperation::Create(
        resultev.branchArgument[0],
        { nm1 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm1 = LoadNonVolatileNode::Create(
        gepnm1,
        { callfibm2Results[1] },
        jlm::rvsdg::bittype::Create(64),
        8);

    auto gepnm2 = GetElementPtrOperation::Create(
        resultev.branchArgument[0],
        { nm2 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm2 =
        LoadNonVolatileNode::Create(gepnm2, { ldnm1[1] }, jlm::rvsdg::bittype::Create(64), 8);

    auto sum = jlm::rvsdg::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->AddExitVar({ sum, nev.branchArgument[1] });
    auto gOIoState = gammaNode->AddExitVar({ callfibm2Results[0], gIIoState.branchArgument[1] });
    auto gOMemoryState = gammaNode->AddExitVar({ ldnm2[1], gIMemoryState.branchArgument[1] });

    auto gepn = GetElementPtrOperation::Create(
        pointerArgument,
        { valueArgument },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto store = StoreNonVolatileNode::Create(gepn, sumex.output, { gOMemoryState.output }, 8);

    auto lambdaOutput = lambda->finalize({ gOIoState.output, store[0] });

    fibrv->result()->divert_to(lambdaOutput);
    pb.end();

    GraphExport::Create(*fibrv, "fib");

    return std::make_tuple(
        lambdaOutput,
        jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*callfibm1Results[0])),
        jlm::util::AssertedCast<CallNode>(jlm::rvsdg::output::GetNode(*callfibm2Results[0])));
  };

  auto [fibfct, callFib1, callFib2] = SetupFib();

  // Act
  auto callTypeClassifier1 = CallNode::ClassifyCall(*callFib1);
  auto callTypeClassifier2 = CallNode::ClassifyCall(*callFib2);

  // Assert
  assert(callTypeClassifier1->IsRecursiveDirectCall());
  assert(&callTypeClassifier1->GetLambdaOutput() == fibfct);

  assert(callTypeClassifier2->IsRecursiveDirectCall());
  assert(&callTypeClassifier2->GetLambdaOutput() == fibfct);
}

static int
Test()
{
  TestCopy();

  TestCallNodeAccessors();
  TestCallTypeClassifierIndirectCall();
  TestCallTypeClassifierNonRecursiveDirectCall();
  TestCallTypeClassifierNonRecursiveDirectCallTheta();
  TestCallTypeClassifierRecursiveDirectCall();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestCall", Test)
