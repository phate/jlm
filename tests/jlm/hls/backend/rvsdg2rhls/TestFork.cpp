/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

static void
ForkInsertion()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto bit32Type = rvsdg::bittype::Create(32);
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { bit32Type, bit32Type, bit32Type },
      { bit32Type, bit32Type, bit32Type });

  RvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  auto loop = hls::LoopNode::create(lambda->subregion());
  rvsdg::Output * idvBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  rvsdg::Output * lvsBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[1], &lvsBuffer);
  rvsdg::Output * lveBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[2], &lveBuffer);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, lvsBuffer }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, lveBuffer }, 32).output(0);
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto lambdaOutput = lambda->finalize({ loop->output(0), loop->output(1), loop->output(2) });
  rvsdg::GraphExport::Create(*lambdaOutput, "");

  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  util::StatisticsCollector statisticsCollector;
  hls::ForkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  {
    assert(rootRegion.nnodes() == 1);
    auto lambda = util::AssertedCast<jlm::rvsdg::LambdaNode>(rootRegion.Nodes().begin().ptr());
    assert(dynamic_cast<const jlm::rvsdg::LambdaNode *>(lambda));

    auto lambdaSubregion = lambda->subregion();
    assert(lambdaSubregion->nnodes() == 1);
    auto loop = util::AssertedCast<hls::LoopNode>(lambdaSubregion->Nodes().begin().ptr());
    assert(dynamic_cast<const hls::LoopNode *>(loop));

    auto [forkNode, forkOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(
        *loop->subregion()->result(0)->origin());
    assert(forkNode && forkOperation);
    assert(forkNode->ninputs() == 1);
    assert(forkNode->noutputs() == 4);
    assert(forkOperation->IsConstant() == false);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestFork-ForkInsertion", ForkInsertion)

static void
ConstantForkInsertion()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto bit32Type = rvsdg::bittype::Create(32);
  const auto functionType = rvsdg::FunctionType::Create({ bit32Type }, { bit32Type });

  RvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  auto lambda = rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  auto loop = hls::LoopNode::create(lambda->subregion());
  auto subregion = loop->subregion();
  rvsdg::Output * idvBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  auto bitConstant1 = rvsdg::create_bitconstant(subregion, 32, 1);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, bitConstant1 }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, bitConstant1 }, 32).output(0);
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto lambdaOutput = lambda->finalize({ loop->output(0) });
  rvsdg::GraphExport::Create(*lambdaOutput, "");

  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  util::StatisticsCollector statisticsCollector;
  hls::ForkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  {
    assert(rootRegion.nnodes() == 1);
    auto lambda = util::AssertedCast<jlm::rvsdg::LambdaNode>(rootRegion.Nodes().begin().ptr());
    assert(rvsdg::is<jlm::rvsdg::LambdaOperation>(lambda));

    auto lambdaRegion = lambda->subregion();
    assert(lambdaRegion->nnodes() == 1);

    const rvsdg::node_output * loopOutput = nullptr;
    assert(loopOutput = dynamic_cast<jlm::rvsdg::node_output *>(lambdaRegion->result(0)->origin()));
    auto loopNode = loopOutput->node();
    assert(rvsdg::is<hls::LoopOperation>(loopNode));
    auto loop = util::AssertedCast<hls::LoopNode>(loopNode);

    auto [forkNode, forkOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(
        *loop->subregion()->result(0)->origin());
    assert(forkNode && forkOperation);
    assert(forkNode->ninputs() == 1);
    assert(forkNode->noutputs() == 2);
    assert(forkOperation->IsConstant() == false);

    auto matchNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*forkNode->input(0)->origin());
    auto bitsUltNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*matchNode->input(0)->origin());
    auto [cForkNode, cForkOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(*bitsUltNode->input(1)->origin());
    assert(cForkNode->ninputs() == 1);
    assert(cForkNode->noutputs() == 2);
    assert(cForkOperation->IsConstant() == true);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/TestFork-ConstantForkInsertion",
    ConstantForkInsertion)
