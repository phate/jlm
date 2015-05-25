/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/address-transform.h>
#include <jive/arch/memlayout-simple.h>
#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <jive/view.h>

#include <assert.h>

static int
verify_bitcast(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_fpext(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_fptosi(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_fptoui(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_fptrunc(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_inttoptr(const jive_graph * graph)
{
	/* FIXME: remove when the evaluator understands the address type */
	setlocale(LC_ALL, "");

	jive::memlayout_mapper_simple mapper(8);
	jive_graph_address_transform(const_cast<jive_graph*>(graph), &mapper);

	jive_graph_normalize(const_cast<jive_graph*>(graph));
	jive_graph_prune(const_cast<jive_graph*>(graph));
	jive_view(graph, stdout);

	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(jive::bits::value_repr(64, 3)));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_inttoptr", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 3);

	return 0;
}

static int
verify_ptrtoint(const jive_graph * graph)
{
	/* FIXME: remove when the evaluator understands the address type */
	setlocale(LC_ALL, "");

	jive::memlayout_mapper_simple mapper(8);
	jive_graph_address_transform(const_cast<jive_graph*>(graph), &mapper);

	jive_graph_normalize(const_cast<jive_graph*>(graph));
	jive_graph_prune(const_cast<jive_graph*>(graph));
	jive_view(graph, stdout);

	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, 3));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_ptrtoint", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 3);

	return 0;
}

static int
verify_sext(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(4, 1));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_sext", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 1);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr().nbits() == 8);


	xl = jive::bits::value_repr(4, 0x8);
	result = std::move(eval(graph, "test_sext", {&xl, &state})->copy());

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 0xF8);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr().nbits() == 8);

	return 0;
}

static int
verify_sitofp(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_trunc(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, 0xFFFFFFFF0000000F));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_trunc", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 0x0000000F);

	return 0;
}

static int
verify_uitofp(const jive_graph * graph)
{
	/* FIXME: insert checks */
	return 0;
}

static int
verify_zext(const jive_graph * graph)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(16, 13));

	std::unique_ptr<const literal> result;
	result = std::move(eval(graph, "test_zext", {&xl, &state})->copy());

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 13);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr().nbits() == 64);

	return 0;
}

static int
verify(const jive_graph * graph)
{
	verify_bitcast(graph);
	verify_fpext(graph);
	verify_fptosi(graph);
	verify_fptoui(graph);
	verify_fptrunc(graph);
	verify_inttoptr(graph);
	verify_ptrtoint(graph);
	verify_sext(graph);
	verify_sitofp(graph);
	verify_trunc(graph);
	verify_uitofp(graph);
	verify_zext(graph);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-casts", nullptr, verify);
