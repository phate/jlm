/*
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestPhiCreation()
{
  using namespace jlm::llvm;

  jlm::rvsdg::Graph graph;

  auto vtype = jlm::rvsdg::TestType::createValueType();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto f0type = jlm::rvsdg::FunctionType::Create(
      { vtype, IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto f1type = jlm::rvsdg::FunctionType::Create(
      { vtype, IOStateType::Create(), MemoryStateType::Create() },
      { vtype, IOStateType::Create(), MemoryStateType::Create() });

  auto SetupEmptyLambda = [&](jlm::rvsdg::Region * region, const std::string & name)
  {
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        *region,
        LlvmLambdaOperation::Create(f0type, name, Linkage::externalLinkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    return lambda->finalize({ iOStateArgument, memoryStateArgument });
  };

  auto SetupF2 = [&](jlm::rvsdg::Region * region, jlm::rvsdg::Output * f2)
  {
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        *region,
        LlvmLambdaOperation::Create(f1type, "f2", Linkage::externalLinkage));
    auto ctxVarF2 = lambda->AddContextVar(*f2).inner;
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto callResults = CallOperation::Create(
        ctxVarF2,
        f1type,
        { valueArgument, iOStateArgument, memoryStateArgument });

    return lambda->finalize(callResults);
  };

  jlm::rvsdg::PhiBuilder pb;
  pb.begin(&graph.GetRootRegion());
  auto rv1 = pb.AddFixVar(f0type);
  auto rv2 = pb.AddFixVar(f0type);
  auto rv3 = pb.AddFixVar(f1type);

  auto lambdaOutput0 = SetupEmptyLambda(pb.subregion(), "f0");
  auto lambdaOutput1 = SetupEmptyLambda(pb.subregion(), "f1");
  auto lambdaOutput2 = SetupF2(pb.subregion(), rv3.recref);

  rv1.result->divert_to(lambdaOutput0);
  rv2.result->divert_to(lambdaOutput1);
  rv3.result->divert_to(lambdaOutput2);

  auto phi = pb.end();
  jlm::rvsdg::GraphExport::Create(*phi->output(0), "dummy");

  graph.PruneNodes();

  view(&graph.GetRootRegion(), stderr);
}

static void
TestPhi()
{
  TestPhiCreation();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestPhi", TestPhi)
