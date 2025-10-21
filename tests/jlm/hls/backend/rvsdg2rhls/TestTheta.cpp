/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestUnknownBoundaries()
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  // Arrange
  auto b32 = jlm::rvsdg::BitType::Create(32);
  auto ft = jlm::rvsdg::FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto idv = theta->AddLoopVar(lambda->GetFunctionArguments()[0]);
  auto lvs = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  auto lve = theta->AddLoopVar(lambda->GetFunctionArguments()[2]);

  auto arm = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitadd_op>({ idv.pre, lvs.pre }, 32).output(0);
  auto cmp = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitult_op>({ arm, lve.pre }, 32).output(0);
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv.post->divert_to(arm);
  theta->set_predicate(match);

  auto f = lambda->finalize({ theta->output(0), theta->output(1), theta->output(2) });
  jlm::rvsdg::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  ThetaNodeConversion::CreateAndRun(rm, statisticsCollector);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  auto lambdaRegion = lambda->subregion();
  assert(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::ContainsOperation<PredicateBufferOperation>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::ContainsOperation<jlm::hls::BranchOperation>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::ContainsOperation<MuxOperation>(*lambdaRegion, true));
  // Check that two constant buffers are created for the loop invariant variables
  assert(jlm::rvsdg::Region::ContainsOperation<LoopConstantBufferOperation>(*lambdaRegion, true));
  assert(lambdaRegion->argument(0)->nusers() == 1);
  auto & loopNode =
      jlm::rvsdg::AssertGetOwnerNode<LoopNode>(lambdaRegion->argument(0)->SingleUser());
  {
    assert(jlm::rvsdg::IsOwnerNodeOperation<LoopConstantBufferOperation>(
        loopNode.subregion()->argument(3)->SingleUser()));
  }
  {
    assert(jlm::rvsdg::IsOwnerNodeOperation<LoopConstantBufferOperation>(
        loopNode.subregion()->argument(4)->SingleUser()));
  }
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestTheta", TestUnknownBoundaries)
