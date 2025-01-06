/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

static int
TestUnknownBoundaries()
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  // Arrange
  auto b32 = jlm::rvsdg::bittype::Create(32);
  auto ft = FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto nf = rm.Rvsdg().GetNodeNormalForm(typeid(jlm::rvsdg::Operation));
  nf->set_mutable(false);

  auto lambda =
      lambda::node::create(&rm.Rvsdg().GetRootRegion(), ft, "f", linkage::external_linkage);

  jlm::rvsdg::bitult_op ult(32);
  jlm::rvsdg::bitsgt_op sgt(32);
  jlm::rvsdg::bitadd_op add(32);
  jlm::rvsdg::bitsub_op sub(32);

  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto subregion = theta->subregion();
  auto idv = theta->AddLoopVar(lambda->GetFunctionArguments()[0]);
  auto lvs = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  auto lve = theta->AddLoopVar(lambda->GetFunctionArguments()[2]);

  auto arm = jlm::rvsdg::SimpleNode::create_normalized(subregion, add, { idv.pre, lvs.pre })[0];
  auto cmp = jlm::rvsdg::SimpleNode::create_normalized(subregion, ult, { arm, lve.pre })[0];
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv.post->divert_to(arm);
  theta->set_predicate(match);

  auto f = lambda->finalize({ theta->output(0), theta->output(1), theta->output(2) });
  GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  ConvertThetaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  auto lambdaRegion = lambda->subregion();
  assert(jlm::rvsdg::Region::Contains<loop_op>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::Contains<predicate_buffer_op>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::Contains<jlm::hls::branch_op>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::Contains<mux_op>(*lambdaRegion, true));
  // Check that two constant buffers are created for the loop invariant variables
  assert(jlm::rvsdg::Region::Contains<loop_constant_buffer_op>(*lambdaRegion, true));
  assert(lambdaRegion->argument(0)->nusers() == 1);
  auto loopInput =
      jlm::util::AssertedCast<jlm::rvsdg::StructuralInput>(*lambdaRegion->argument(0)->begin());
  auto loopNode = jlm::util::AssertedCast<loop_node>(loopInput->node());
  auto loopConstInput = jlm::util::AssertedCast<jlm::rvsdg::simple_input>(
      *loopNode->subregion()->argument(3)->begin());
  jlm::util::AssertedCast<const loop_constant_buffer_op>(&loopConstInput->node()->GetOperation());
  loopConstInput = jlm::util::AssertedCast<jlm::rvsdg::simple_input>(
      *loopNode->subregion()->argument(4)->begin());
  jlm::util::AssertedCast<const loop_constant_buffer_op>(&loopConstInput->node()->GetOperation());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestTheta", TestUnknownBoundaries)
