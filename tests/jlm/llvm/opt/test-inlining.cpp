/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

/**
 * Runs the inlining pass on the given module, and returns the pass statistics
 * @param rm the RVSDG module
 * @return the statistics instance produced by the inlining pass
 */
static std::unique_ptr<jlm::util::Statistics>
runInlining(jlm::llvm::RvsdgModule & rm)
{
  jlm::llvm::FunctionInlining fctinline;
  jlm::util::StatisticsCollectorSettings settings({ jlm::util::Statistics::Id::FunctionInlining });
  jlm::util::StatisticsCollector collector(settings);
  fctinline.Run(rm, collector);

  assert(collector.NumCollectedStatistics() == 1);
  return collector.releaseStatistic(jlm::util::Statistics::Id::FunctionInlining);
}

static void
testSimpleInlining()
{
  /**
   * Creates an RVSDG that looks like:
   *
   * import i : ValueType
   * lambda f1(val, io, mem) -> (ValueType, io, mem)
   *    context: i
   *    body:
   *       t = TestOperation(val)
   *       return (t, io, mem)
   *
   * lambda f2(ctrl, val, io, m) -> (ValueType, io, mem)
   *    context: f1
   *    body:
   *       gamma(ctrl)
   *          context: f1, val, io, mem
   *          branch 0:
   *             (val0, io0, m0) = call f1(val, io, mem)
   *
   *          branch 1:
   *             // nop
   *
   *         exits vars:
   *           v_out  = (val0, val)
   *           io_out = (io0,  io)
   *           m_out  = (mem0, mem)
   *
   *       return (v_out, io_out, m_out)
   *
   * export: f2
   *
   * After inlining, the call in f2 should be removed, and be replaced by a TestOperation
   */

  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto vt = TestType::createValueType();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto i = &jlm::rvsdg::GraphImport::Create(graph, vt, "i");

  Region * gammaRegion0 = nullptr;

  auto SetupF1 = [&]()
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { vt, IOStateType::Create(), MemoryStateType::Create() },
        { vt, IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f1", Linkage::externalLinkage));
    lambda->AddContextVar(*i);

    auto t = TestOperation::createNode(
        lambda->subregion(),
        { lambda->GetFunctionArguments()[0] },
        { vt });

    return lambda->finalize(
        { t->output(0), lambda->GetFunctionArguments()[1], lambda->GetFunctionArguments()[2] });
  };

  auto SetupF2 = [&](jlm::rvsdg::Output * f1)
  {
    auto ct = jlm::rvsdg::ControlType::Create(2);
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::ControlType::Create(2),
          vt,
          IOStateType::Create(),
          MemoryStateType::Create() },
        { vt, IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f2", Linkage::externalLinkage));
    auto d = lambda->AddContextVar(*f1).inner;
    auto controlArgument = lambda->GetFunctionArguments()[0];
    auto valueArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto gamma = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    gammaRegion0 = gamma->subregion(0);
    auto gammaInputF1 = gamma->AddEntryVar(d);
    auto gammaInputValue = gamma->AddEntryVar(valueArgument);
    auto gammaInputIoState = gamma->AddEntryVar(iOStateArgument);
    auto gammaInputMemoryState = gamma->AddEntryVar(memoryStateArgument);

    auto callResults = CallOperation::Create(
        gammaInputF1.branchArgument[0],
        jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f1).GetOperation().Type(),
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
  auto statistics = runInlining(rm);

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Assert
  // Check that the call has been replaced by the test operation inside f1
  assert(!Region::ContainsOperation<CallOperation>(graph.GetRootRegion(), true));
  assert(Region::ContainsOperation<TestOperation>(*gammaRegion0, true));

  // Check that the statistics match what we expect. f2 is technically inlineable
  assert(statistics->GetMeasurementValue<uint64_t>("#Functions") == 2);
  assert(statistics->GetMeasurementValue<uint64_t>("#InlineableFunctions") == 2);
  assert(statistics->GetMeasurementValue<uint64_t>("#FunctionCalls") == 1);
  assert(statistics->GetMeasurementValue<uint64_t>("#InlinableCalls") == 1);
  assert(statistics->GetMeasurementValue<uint64_t>("#CallsInlined") == 1);
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining-testSimpleInlining", testSimpleInlining)

static void
testInliningWithAlloca()
{
  /**
   * Creates an RVSDG that looks like:
   *
   * import i : ValueType
   * lambda f1(val, io, mem) -> (io, mem)
   *    context: i
   *    body:
   *       count = I32(1)
   *       ptr, aMem = AllocaOperation(count)
   *       mem2 = Store(ptr, val, mem)
   *       return (io, mem2)
   *
   * lambda f2(ctrl, val, io, m) -> (io, mem)
   *    context: f1
   *    body:
   *       gamma(ctrl)
   *          context: f1, val, io, mem
   *          branch 0:
   *             (io0, m0) = call f1(val, io, mem)
   *
   *          branch 1:
   *             // nop
   *
   *         exits vars:
   *           io_out = (io0,  io)
   *           m_out  = (mem0, mem)
   *
   *       return (io_out, m_out)
   *
   * export: f2
   *
   * After inlining, the call in f2 should be removed, and be replaced by the store.
   * The inlined alloca should however be hoisted to the root region of f2
   */

  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto vt = TestType::createValueType();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto i = &jlm::rvsdg::GraphImport::Create(graph, vt, "i");

  Region * gammaRegion0 = nullptr;
  Region * f2Region = nullptr;

  auto SetupF1 = [&]()
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { vt, IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f1", Linkage::externalLinkage));
    lambda->AddContextVar(*i);
    auto lambdaArgs = lambda->GetFunctionArguments();

    const auto & one = IntegerConstantOperation::Create(*lambda->subregion(), 32, 1);
    auto alloca = AllocaOperation::create(vt, one.output(0), 4);
    auto store = StoreNonVolatileOperation::Create(alloca[0], lambdaArgs[0], { lambdaArgs[2] }, 4);

    return lambda->finalize({ lambdaArgs[1], store[0] });
  };

  auto SetupF2 = [&](jlm::rvsdg::Output * f1)
  {
    auto ct = jlm::rvsdg::ControlType::Create(2);
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::ControlType::Create(2),
          vt,
          IOStateType::Create(),
          MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f2", Linkage::externalLinkage));
    auto d = lambda->AddContextVar(*f1).inner;
    auto controlArgument = lambda->GetFunctionArguments()[0];
    auto valueArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];
    f2Region = lambda->subregion();

    auto gamma = jlm::rvsdg::GammaNode::create(controlArgument, 2);
    gammaRegion0 = gamma->subregion(0);
    auto gammaInputF1 = gamma->AddEntryVar(d);
    auto gammaInputValue = gamma->AddEntryVar(valueArgument);
    auto gammaInputIoState = gamma->AddEntryVar(iOStateArgument);
    auto gammaInputMemoryState = gamma->AddEntryVar(memoryStateArgument);

    auto callResults = CallOperation::Create(
        gammaInputF1.branchArgument[0],
        jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::LambdaNode>(*f1).GetOperation().Type(),
        { gammaInputValue.branchArgument[0],
          gammaInputIoState.branchArgument[0],
          gammaInputMemoryState.branchArgument[0] });

    auto gammaOutputIoState =
        gamma->AddExitVar({ callResults[0], gammaInputIoState.branchArgument[1] });
    auto gammaOutputMemoryState =
        gamma->AddExitVar({ callResults[1], gammaInputMemoryState.branchArgument[1] });

    return lambda->finalize({ gammaOutputIoState.output, gammaOutputMemoryState.output });
  };

  auto f1 = SetupF1();
  auto f2 = SetupF2(f1);

  GraphExport::Create(*f2, "f2");

  // jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  // Act
  runInlining(rm);

  // jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  // Check that the call is gone
  assert(!Region::ContainsOperation<CallOperation>(graph.GetRootRegion(), true));
  // A store should have taken its place in the gamma subregion
  assert(Region::ContainsOperation<StoreNonVolatileOperation>(*gammaRegion0, true));
  // Check that the alloca operation is not inside the gamma subregion
  assert(!Region::ContainsOperation<AllocaOperation>(*gammaRegion0, true));
  // The alloca should have been moved to the top level of f2
  assert(Region::ContainsOperation<AllocaOperation>(*f2Region, false));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining-testInliningWithAlloca", testInliningWithAlloca)

static void
testIndirectCall()
{
  /**
   * Creates an RVSDG graph with two functions.
   * f1() is a simple no-op function
   * f2() calls f1(), but via an indirect call
   */

  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto functionType1 = jlm::rvsdg::FunctionType::Create(
      { vt, IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto pt = PointerType::Create();

  auto functionType2 = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto i = &jlm::rvsdg::GraphImport::Create(graph, functionType2, "i");

  auto SetupF1 = [&](const std::shared_ptr<const jlm::rvsdg::FunctionType> & functionType)
  {
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f1", Linkage::externalLinkage));
    return lambda->finalize(
        { lambda->GetFunctionArguments()[1], lambda->GetFunctionArguments()[2] });
  };

  auto SetupF2 = [&](jlm::rvsdg::Output * f1)
  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f2", Linkage::externalLinkage));
    auto cvi = lambda->AddContextVar(*i).inner;
    auto cvf1 = lambda->AddContextVar(*f1).inner;
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto callResults =
        CallOperation::Create(cvi, functionType2, { cvf1, iOStateArgument, memoryStateArgument });

    return lambda->finalize(callResults);
  };

  auto f1 = SetupF1(functionType1);
  auto f2 = SetupF2(
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ f1 }, functionType1).output(0));

  GraphExport::Create(*f2, "f2");

  // jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto statistics = runInlining(rm);

  // jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  // No inlining happens in this test, but both f1 and f2 are technically possible to inline
  assert(statistics->GetMeasurementValue<uint64_t>("#Functions") == 2);
  assert(statistics->GetMeasurementValue<uint64_t>("#InlineableFunctions") == 2);
  assert(statistics->GetMeasurementValue<uint64_t>("#FunctionCalls") == 1);
  assert(statistics->GetMeasurementValue<uint64_t>("#InlinableCalls") == 0);
  assert(statistics->GetMeasurementValue<uint64_t>("#CallsInlined") == 0);
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inlining-testIndirectCall", testIndirectCall)

/**
 * Creates an RVSDG graph with a single function f1.
 * The function contains an alloca inside a theta, which disqualifies it from being inlined
 */
static void
testFunctionWithDisqualifyingAlloca()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto SetupF1 = [&]()
  {
    auto functionType = FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph.GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "f1", Linkage::externalLinkage));
    auto theta = ThetaNode::create(lambda->subregion());

    const auto & one = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
    AllocaOperation::create(vt, one.output(0), 4);

    return lambda->finalize(
        { lambda->GetFunctionArguments()[0], lambda->GetFunctionArguments()[1] });
  };
  SetupF1();

  // Act
  auto statistics = runInlining(rm);

  // Assert
  assert(statistics->GetMeasurementValue<uint64_t>("#Functions") == 1);
  // f1 should not be considered inlinable, due to the alloca
  assert(statistics->GetMeasurementValue<uint64_t>("#InlineableFunctions") == 0);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/test-inlining-testFunctionWithDisqualifyingAlloca",
    testFunctionWithDisqualifyingAlloca)
