/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

static inline void
TestUnknownBoundaries()
{
  using namespace jlm::llvm;

  // Arrange
  auto b32 = jlm::rvsdg::bittype::Create(32);
  auto ft = FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

  jlm::rvsdg::bitult_op ult(32);
  jlm::rvsdg::bitsgt_op sgt(32);
  jlm::rvsdg::bitadd_op add(32);
  jlm::rvsdg::bitsub_op sub(32);

  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto subregion = theta->subregion();
  auto idv = theta->add_loopvar(lambda->fctargument(0));
  auto lvs = theta->add_loopvar(lambda->fctargument(1));
  auto lve = theta->add_loopvar(lambda->fctargument(2));

  auto arm = jlm::rvsdg::simple_node::create_normalized(
      subregion,
      add,
      { idv->argument(), lvs->argument() })[0];
  auto cmp =
      jlm::rvsdg::simple_node::create_normalized(subregion, ult, { arm, lve->argument() })[0];
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv->result()->divert_to(arm);
  theta->set_predicate(match);

  auto f = lambda->finalize({ theta->output(0), theta->output(1), theta->output(2) });
  jlm::llvm::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  jlm::hls::ConvertThetaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  assert(jlm::rvsdg::region::Contains<jlm::hls::loop_op>(*lambda->subregion(), true));
  assert(jlm::rvsdg::region::Contains<jlm::hls::predicate_buffer_op>(*lambda->subregion(), true));
  assert(jlm::rvsdg::region::Contains<jlm::hls::branch_op>(*lambda->subregion(), true));
  assert(jlm::rvsdg::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static int
Test()
{
  TestUnknownBoundaries();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestTheta", Test)
