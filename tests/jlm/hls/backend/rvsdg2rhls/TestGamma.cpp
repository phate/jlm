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

  jlm::tests::valuetype vt;
  FunctionType ft({ &*jlm::rvsdg::bittype::Create(1), &vt, &vt }, { &vt });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  /* Setup graph */

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambda->fctargument(0));
  auto gamma = jlm::rvsdg::gamma_node::create(match, 2);
  auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
  auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
  auto ex = gamma->add_exitvar({ ev1->argument(0), ev2->argument(1) });

  auto f = lambda->finalize({ ex });
  rm.Rvsdg().add_export(f, { f->type(), "" });

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Convert graph to RHLS */

  jlm::hls::ConvertGammaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Verify output */

  assert(jlm::rvsdg::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static void
TestWithoutMatch()
{
  using namespace jlm::llvm;

  jlm::tests::valuetype vt;
  FunctionType ft({ &*jlm::rvsdg::ctltype::Create(2), &vt, &vt }, { &vt });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto nf = rm.Rvsdg().node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  /* Setup graph */

  auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

  auto gamma = jlm::rvsdg::gamma_node::create(lambda->fctargument(0), 2);
  auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
  auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
  auto ex = gamma->add_exitvar({ ev1->argument(0), ev2->argument(1) });

  auto f = lambda->finalize({ ex });
  rm.Rvsdg().add_export(f, { f->type(), "" });

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Convert graph to RHLS */

  jlm::hls::ConvertGammaNodes(rm);
  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  /* Verify output */

  assert(jlm::rvsdg::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static int
Test()
{
  TestWithMatch();
  TestWithoutMatch();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/TestGamma", Test)
