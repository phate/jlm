a*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.h"
#include "testtypes.h"

#include <assert.hpp>
#include <stdio.hpp>

#include <jive/view.hpp>
#include <jive/vsdg.hpp>

#include "testnodes.h"

typedef struct graph_desc {
	std::unique_ptr<jive::graph> graph;
	jive::node * a1, * a2;
	jive::node * b1, * b2;
} graph_desc;

static graph_desc
prepare_graph()
{
	graph_desc g;
	g.graph = std::move(std::unique_ptr<jive::graph>(new jive::graph()));
	
	jive::region * region = g.graph->root();
	jive::test::valuetype type;
	g.a1 = jive::test::simple_node_create(region, {}, {}, {type});
	g.a2 = jive::test::simple_node_create(region, {type}, {g.a1->output(0)}, {type});
	g.b1 = jive::test::simple_node_create(region, {}, {}, {type});
	g.b2 = jive::test::simple_node_create(region, {type}, {g.b1->output(0)}, {type});
	
	return g;
}

static void
test_simple_upward_cone()
{
	graph_desc g = prepare_graph();
	
	{
		jive::upward_cone_traverser trav(g.a2);
		
		assert( trav.next() == g.a2 );
		assert( trav.next() == g.a1 );
		assert( trav.next() == NULL );
	}
}

static void
test_mutable_upward_cone_1()
{
	graph_desc g = prepare_graph();
	
	{
		jive::upward_cone_traverser trav(g.a2);
	
		assert( trav.next() == g.a2 );
		delete g.b2;
		assert( trav.next() == g.a1 );
		assert( trav.next() == nullptr );
	}
}

static void
test_mutable_upward_cone_2()
{
	graph_desc g = prepare_graph();
	
	{
		jive::upward_cone_traverser trav(g.a2);

		delete g.a2;
		assert( trav.next() == g.a1 );
		assert( trav.next() == nullptr );
	}
}

static void
test_mutable_upward_cone_3()
{
	graph_desc g = prepare_graph();
	
	{
		jive::upward_cone_traverser trav(g.a2);
	
		g.a2->input(0)->divert_origin(g.b1->output(0));
		assert( trav.next() == g.a2 );
		assert( trav.next() == g.b1 );
	}
}

static int test_main(void)
{
	test_simple_upward_cone();
	test_mutable_upward_cone_1();
	test_mutable_upward_cone_2();
	test_mutable_upward_cone_3();
	
	return 0;
}

JIVE_UNIT_TEST_REGISTER("vsdg/traverser/test-cone-traverser", test_main);
