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
  auto ft = FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  RvsdgModule rm(util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(rvsdg::operation));
  nf->set_mutable(false);

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

  rvsdg::bitult_op ult(32);
  rvsdg::bitadd_op add(32);

  auto loop = hls::loop_node::create(lambda->subregion());
  auto subregion = loop->subregion();
  rvsdg::output * idvBuffer;
  loop->add_loopvar(lambda->fctargument(0), &idvBuffer);
  rvsdg::output * lvsBuffer;
  loop->add_loopvar(lambda->fctargument(1), &lvsBuffer);
  rvsdg::output * lveBuffer;
  loop->add_loopvar(lambda->fctargument(2), &lveBuffer);

  auto arm = rvsdg::simple_node::create_normalized(subregion, add, { idvBuffer, lvsBuffer })[0];
  auto cmp = rvsdg::simple_node::create_normalized(subregion, ult, { arm, lveBuffer })[0];
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto f = lambda->finalize({ loop->output(0), loop->output(1), loop->output(2) });
  rm.Rvsdg().add_export(f, { f->Type(), "" });

  rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  hls::add_forks(rm);
  rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  {
    auto omegaRegion = rm.Rvsdg().root();
    assert(omegaRegion->nnodes() == 1);
    auto lambda = util::AssertedCast<lambda::node>(omegaRegion->nodes.first());
    assert(is<lambda::operation>(lambda));

    auto lambdaRegion = lambda->subregion();
    assert(lambdaRegion->nnodes() == 1);
    auto loop = util::AssertedCast<hls::loop_node>(lambdaRegion->nodes.first());
    assert(is<hls::loop_op>(loop));

    // Traverse the rvsgd graph upwards to check connections
    rvsdg::node_output * forkNodeOutput;
    assert(
        forkNodeOutput =
            dynamic_cast<rvsdg::node_output *>(loop->subregion()->result(0)->origin()));
    auto forkNode = forkNodeOutput->node();
    auto forkOp = util::AssertedCast<const hls::fork_op>(&forkNode->operation());
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
  auto ft = FunctionType::Create({}, { b32, b32, b32 });

  RvsdgModule rm(util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(rvsdg::operation));
  nf->set_mutable(false);

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);
  auto lambdaRegion = lambda->subregion();

  auto bitConstant0 = rvsdg::create_bitconstant(lambdaRegion, 32, 0);
  auto bitConstant1 = rvsdg::create_bitconstant(lambdaRegion, 32, 1);
  auto bitConstant5 = rvsdg::create_bitconstant(lambdaRegion, 32, 5);
  rvsdg::bitult_op ult(32);
  rvsdg::bitadd_op add(32);

  auto loop = hls::loop_node::create(lambdaRegion);
  auto subregion = loop->subregion();
  rvsdg::output * idvBuffer;
  loop->add_loopvar(bitConstant0, &idvBuffer);
  rvsdg::output * lvsBuffer;
  loop->add_loopvar(bitConstant1, &lvsBuffer);
  rvsdg::output * lveBuffer;
  loop->add_loopvar(bitConstant5, &lveBuffer);

  auto arm = rvsdg::simple_node::create_normalized(subregion, add, { idvBuffer, lvsBuffer })[0];
  auto cmp = rvsdg::simple_node::create_normalized(subregion, ult, { arm, lveBuffer })[0];
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  loop->set_predicate(match);

  auto f = lambda->finalize({ loop->output(0), loop->output(1), loop->output(2) });
  rm.Rvsdg().add_export(f, { f->Type(), "" });

  rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  hls::add_forks(rm);
  rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  {
    auto omegaRegion = rm.Rvsdg().root();
    assert(omegaRegion->nnodes() == 1);
    auto lambda = util::AssertedCast<lambda::node>(omegaRegion->nodes.first());
    assert(is<lambda::operation>(lambda));

    auto lambdaRegion = lambda->subregion();
    assert(lambdaRegion->nnodes() == 4);

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
    auto forkOp = util::AssertedCast<const hls::fork_op>(&forkNode->operation());
    assert(forkNode->ninputs() == 1);
    assert(forkNode->noutputs() == 4);
    assert(forkOp->IsConstant() == false);
  }
}

static inline void
TestRvsdgFork()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto b32 = rvsdg::bittype::Create(32);
  auto ft = FunctionType::Create({}, { b32, b32, b32 });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(rvsdg::operation));
  nf->set_mutable(false);

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);
  auto lambdaRegion = lambda->subregion();

  rvsdg::bitult_op ult(32);
  rvsdg::bitadd_op add(32);
  auto bitConstant0 = rvsdg::create_bitconstant(lambdaRegion, 32, 0);
  auto bitConstant1 = rvsdg::create_bitconstant(lambdaRegion, 32, 1);
  auto bitConstant5 = rvsdg::create_bitconstant(lambdaRegion, 32, 5);

  auto theta = rvsdg::theta_node::create(lambdaRegion);
  auto subregion = theta->subregion();
  auto idv = theta->add_loopvar(bitConstant0);
  auto lvs = theta->add_loopvar(bitConstant1);
  auto lve = theta->add_loopvar(bitConstant5);

  auto arm = rvsdg::simple_node::create_normalized(
      subregion,
      add,
      { idv->argument(), lvs->argument() })[0];
  auto cmp = rvsdg::simple_node::create_normalized(subregion, ult, { arm, lve->argument() })[0];
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv->result()->divert_to(arm);
  theta->set_predicate(match);

  auto f = lambda->finalize({ theta->output(0), theta->output(1), theta->output(2) });
  rm.Rvsdg().add_export(f, { f->Type(), "" });

  rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  hls::ConvertThetaNodes(rm);
  rvsdg::view(rm.Rvsdg(), stdout);
  hls::add_forks(rm);
  rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  assert(rvsdg::region::Contains<hls::loop_op>(*lambda->subregion(), true));
  assert(rvsdg::region::Contains<hls::predicate_buffer_op>(*lambda->subregion(), true));
  assert(rvsdg::region::Contains<hls::branch_op>(*lambda->subregion(), true));
  assert(rvsdg::region::Contains<hls::mux_op>(*lambda->subregion(), true));
}

static int
Test()
{
  std::cout << std::endl << "### Test fork ###" << std::endl << std::endl;
  TestFork();
  std::cout << std::endl << "### Test constant ###" << std::endl << std::endl;
  TestConstantFork();
  std::cout << std::endl
            << "### Test starting from conventional RVSDG ###" << std::endl
            << std::endl;
  TestRvsdgFork();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestFork", Test)
