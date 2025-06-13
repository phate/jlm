/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 *                Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/hls/backend/rvsdg2rhls/GammaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestWithMatch()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ft = jlm::rvsdg::FunctionType::Create({ jlm::rvsdg::bittype::Create(1), vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");

  /* Setup graph */

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambda->GetFunctionArguments()[0]);
  auto gamma = jlm::rvsdg::GammaNode::create(match, 2);
  auto ev1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto ev2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto ex = gamma->AddExitVar({ ev1.branchArgument[0], ev2.branchArgument[1] });

  auto f = lambda->finalize({ ex.output });
  jlm::llvm::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Convert graph to RHLS */

  jlm::hls::ConvertGammaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Verify output */

  assert(jlm::rvsdg::Region::ContainsOperation<jlm::hls::MuxOperation>(*lambda->subregion(), true));
}

static void
TestWithoutMatch()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ft =
      jlm::rvsdg::FunctionType::Create({ jlm::rvsdg::ControlType::Create(2), vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");

  /* Setup graph */

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));

  auto gamma = jlm::rvsdg::GammaNode::create(lambda->GetFunctionArguments()[0], 2);
  auto ev1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto ev2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto ex = gamma->AddExitVar({ ev1.branchArgument[0], ev2.branchArgument[1] });

  auto f = lambda->finalize({ ex.output });
  jlm::llvm::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Convert graph to RHLS */

  jlm::hls::ConvertGammaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Verify output */

  assert(jlm::rvsdg::Region::ContainsOperation<jlm::hls::MuxOperation>(*lambda->subregion(), true));
}

static int
Test()
{
  TestWithMatch();
  TestWithoutMatch();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestGamma", Test)
