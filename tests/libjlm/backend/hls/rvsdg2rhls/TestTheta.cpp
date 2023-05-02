/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/backend/hls/rvsdg2rhls/theta-conv.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/ir/hls/hls.hpp>

static inline void
TestUnknownBoundaries()
{
	using namespace jlm;

	auto b32 = jive::bit32;
	FunctionType ft({&b32, &b32, &b32}, {&b32, &b32, &b32});

	RvsdgModule rm(filepath(""), "", "");
	auto nf = rm.Rvsdg().node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* Setup graph */

	auto lambda = lambda::node::create(rm.Rvsdg().root(), ft, "f", linkage::external_linkage);

	jive::bitult_op ult(32);
	jive::bitsgt_op sgt(32);
	jive::bitadd_op add(32);
	jive::bitsub_op sub(32);

	auto theta = jive::theta_node::create(lambda->subregion());
	auto subregion = theta->subregion();
	auto idv = theta->add_loopvar(lambda->fctargument(0));
	auto lvs = theta->add_loopvar(lambda->fctargument(1));
	auto lve = theta->add_loopvar(lambda->fctargument(2));

	auto arm = jive::simple_node::create_normalized(subregion, add, {idv->argument(), lvs->argument()})[0];
	auto cmp = jive::simple_node::create_normalized(subregion, ult, {arm, lve->argument()})[0];
	auto match = jive::match(1, {{1, 1}}, 0, 2, cmp);

	idv->result()->divert_to(arm);
	theta->set_predicate(match);

	auto f = lambda->finalize({theta->output(0), theta->output(1), theta->output(2)});
	rm.Rvsdg().add_export(f, {f->type(), ""});

	jive::view(rm.Rvsdg(), stdout);

	/* Convert graph to RHLS */

	hls::theta_conv(theta);
	jive::view(rm.Rvsdg(), stdout);


	/* Verify graph */

	assert(jive::region::Contains<jlm::hls::loop_op>(*lambda->subregion(), true));
	assert(jive::region::Contains<jlm::hls::predicate_buffer_op>(*lambda->subregion(), true));
	assert(jive::region::Contains<jlm::hls::branch_op>(*lambda->subregion(), true));
	assert(jive::region::Contains<jlm::hls::mux_op>(*lambda->subregion(), true));
}

static int
Test()
{
	TestUnknownBoundaries();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/backend/hls/rvsdg2rhls/TestTheta", Test)
