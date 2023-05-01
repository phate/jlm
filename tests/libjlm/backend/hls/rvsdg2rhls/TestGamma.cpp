/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 *                Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/backend/hls/rvsdg2rhls/gamma-conv.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/ir/hls/hls.hpp>

static void
TestWithMatch()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::bittype bt1(1);
	FunctionType ft({&bt1, &vt, &vt}, {&vt});

	RvsdgModule rm(filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* Setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto match = jive::match(1, {{0, 0}}, 1, 2, lambda->fctargument(0));
	auto gamma = jive::gamma_node::create(match, 2);
	auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
	auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto f = lambda->finalize({ex});
	rm.Rvsdg().add_export(f, {f->type(), ""});

	jive::view(rm.Rvsdg(), stdout);

	/* Convert graph to RHLS */

	hls::gamma_conv(rm);
	jive::view(rm.Rvsdg(), stdout);

	/* Verify output */

	assert(jive::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static void
TestWithoutMatch()
{
	using namespace jlm;

	jlm::valuetype vt;
	jive::ctltype ctl2(2);
	jive::bittype bt1(1);
	FunctionType ft({&ctl2, &vt, &vt}, {&vt});

	RvsdgModule rm(filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* Setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	auto gamma = jive::gamma_node::create(lambda->fctargument(0), 2);
	auto ev1 = gamma->add_entryvar(lambda->fctargument(1));
	auto ev2 = gamma->add_entryvar(lambda->fctargument(2));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});

	auto f = lambda->finalize({ex});
	rm.Rvsdg().add_export(f, {f->type(), ""});

	jive::view(rm.Rvsdg(), stdout);

	/* Convert graph to RHLS */

	hls::gamma_conv(rm);
	jive::view(rm.Rvsdg(), stdout);

	/* Verify output */

	assert(jive::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static int
Test()
{
	TestWithMatch();
	TestWithoutMatch();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/backend/hls/rvsdg2rhls/TestGamma", Test)
