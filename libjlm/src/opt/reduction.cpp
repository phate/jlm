/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/binary.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/reduction.hpp>

namespace jlm {

void
reduce(rvsdg_module & rm)
{
	auto & graph = *rm.graph();

	/* alloca operation */
	{
		auto nf = graph.node_normal_form(typeid(jlm::alloca_op));
		auto allocanf = static_cast<jlm::alloca_normal_form*>(nf);
		allocanf->set_mutable(true);
		allocanf->set_alloca_alloca_reducible(true);
		allocanf->set_alloca_mux_reducible(true);
	}

	/* mux operation */
	{
		auto nf = graph.node_normal_form(typeid(jive::mux_op));
		auto mnf = static_cast<jive::mux_normal_form*>(nf);
		mnf->set_mutable(true);
		mnf->set_mux_mux_reducible(true);
		mnf->set_multiple_origin_reducible(true);
	}

	/* store operation */
	{
		auto nf = jlm::store_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_store_mux_reducible(true);
		nf->set_store_store_reducible(true);
		nf->set_store_alloca_reducible(true);
		nf->set_multiple_origin_reducible(true);
	}

	/* load operation */
	{
		auto nf = jlm::load_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_load_mux_reducible(true);
		nf->set_load_store_reducible(true);
		nf->set_load_alloca_reducible(true);
		nf->set_multiple_origin_reducible(true);
		nf->set_load_store_state_reducible(true);
		nf->set_load_store_alloca_reducible(true);
		nf->set_load_load_state_reducible(true);
	}

	/* gamma operation */
	{
		auto nf = jive::gamma_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_predicate_reduction(true);
		nf->set_control_constant_reduction(true);
	}

	/* unary operation */
	{
		auto nf = jive::unary_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_reducible(true);
	}

	/* binary operation */
	{
		auto nf = jive::binary_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_reducible(true);
	}

	graph.normalize();
}

}
