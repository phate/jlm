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

static inline void
TestFork()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto b32 = rvsdg::bittype::Create(32);
  auto ft = jlm::rvsdg::FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  RvsdgModule rm(util::filepath(""), "", "");

  auto lambda =
      lambda::node::create(&rm.Rvsdg().GetRootRegion(), ft, "f", linkage::external_linkage);

  auto loop = hls::loop_node::create(lambda->subregion());
  rvsdg::output * idvBuffer;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  rvsdg::output * lvsBuffer;
  loop->AddLoopVar(lambda->GetFunctionArguments()[1], &lvsBuffer);
  rvsdg::output * lveBuffer;
  loop->AddLoopVar(lambda->GetFunctionArguments()[2], &lveBuffer);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, lvsBuffer }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, lveBuffer }, 32).output(0);
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto f = lambda->finalize({ loop->output(0), loop->output(1), loop->output(2) });
  jlm::llvm::GraphExport::Create(*f, "");

  rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  hls::add_forks(rm);
  rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  {
    auto omegaRegion = &rm.Rvsdg().GetRootRegion();
    assert(omegaRegion->nnodes() == 1);
    auto lambda = util::AssertedCast<lambda::node>(omegaRegion->Nodes().begin().ptr());
    assert(is<lambda::operation>(lambda));

    auto lambdaRegion = lambda->subregion();
    assert(lambdaRegion->nnodes() == 1);
    auto loop = util::AssertedCast<hls::loop_node>(lambdaRegion->Nodes().begin().ptr());
    assert(is<hls::loop_op>(loop));

    // Traverse the rvsgd graph upwards to check connections
    rvsdg::node_output * forkNodeOutput;
    assert(
        forkNodeOutput =
            dynamic_cast<rvsdg::node_output *>(loop->subregion()->result(0)->origin()));
    auto forkNode = forkNodeOutput->node();
    auto forkOp = util::AssertedCast<const hls::fork_op>(&forkNode->GetOperation());
    assert(forkNode->ninputs() == 1);
    assert(forkNode->noutputs() == 4);
    assert(forkOp->IsConstant() == false);
  }
}

static inline void
TestConstantFork()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto b32 = rvsdg::bittype::Create(32);
  auto ft = jlm::rvsdg::FunctionType::Create({ b32 }, { b32 });

  RvsdgModule rm(util::filepath(""), "", "");

  auto lambda =
      lambda::node::create(&rm.Rvsdg().GetRootRegion(), ft, "f", linkage::external_linkage);
  auto lambdaRegion = lambda->subregion();

  auto loop = hls::loop_node::create(lambdaRegion);
  auto subregion = loop->subregion();
  rvsdg::output * idvBuffer;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  auto bitConstant1 = rvsdg::create_bitconstant(subregion, 32, 1);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, bitConstant1 }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, bitConstant1 }, 32).output(0);
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto f = lambda->finalize({ loop->output(0) });
  jlm::llvm::GraphExport::Create(*f, "");

  rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  hls::add_forks(rm);
  rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  {
    auto omegaRegion = &rm.Rvsdg().GetRootRegion();
    assert(omegaRegion->nnodes() == 1);
    auto lambda = util::AssertedCast<lambda::node>(omegaRegion->Nodes().begin().ptr());
    assert(is<lambda::operation>(lambda));

    auto lambdaRegion = lambda->subregion();
    assert(lambdaRegion->nnodes() == 1);

    rvsdg::node_output * loopOutput;
    assert(loopOutput = dynamic_cast<jlm::rvsdg::node_output *>(lambdaRegion->result(0)->origin()));
    auto loopNode = loopOutput->node();
    assert(is<hls::loop_op>(loopNode));
    auto loop = util::AssertedCast<hls::loop_node>(loopNode);

    // Traverse the rvsgd graph upwards to check connections
    rvsdg::node_output * forkNodeOutput;
    assert(
        forkNodeOutput =
            dynamic_cast<rvsdg::node_output *>(loop->subregion()->result(0)->origin()));
    auto forkNode = forkNodeOutput->node();
    auto forkOp = util::AssertedCast<const hls::fork_op>(&forkNode->GetOperation());
    assert(forkNode->ninputs() == 1);
    assert(forkNode->noutputs() == 2);
    assert(forkOp->IsConstant() == false);
    auto matchNodeOutput = dynamic_cast<rvsdg::node_output *>(forkNode->input(0)->origin());
    auto matchNode = matchNodeOutput->node();
    auto bitsUltNodeOutput = dynamic_cast<rvsdg::node_output *>(matchNode->input(0)->origin());
    auto bitsUltNode = bitsUltNodeOutput->node();
    auto cforkNodeOutput = dynamic_cast<rvsdg::node_output *>(bitsUltNode->input(1)->origin());
    auto cforkNode = cforkNodeOutput->node();
    auto cforkOp = util::AssertedCast<const hls::fork_op>(&cforkNode->GetOperation());
    assert(cforkNode->ninputs() == 1);
    assert(cforkNode->noutputs() == 2);
    assert(cforkOp->IsConstant() == true);
  }
}

static int
Test()
{
  std::cout << std::endl << "### Test fork ###" << std::endl << std::endl;
  TestFork();
  std::cout << std::endl << "### Test constant ###" << std::endl << std::endl;
  TestConstantFork();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestFork", Test)
