/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <assert.h>
#include <stdio.h>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/view.hpp>

static int types_bitstring_arithmetic_test_bitand(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 5);

	auto and0 = bitand_op::create(32, s0, s1);
	auto and1 = bitand_op::create(32, c0, c1);

	graph.add_export(and0, {and0->type(), "dummy"});
	graph.add_export(and1, {and1->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(and0)->operation() == bitand_op(32));
	assert(node_output::node(and1)->operation() == int_constant_op(32, +1));

	return 0;
}

static int types_bitstring_arithmetic_test_bitashr(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 16);
	auto c1 = create_bitconstant(graph.root(), 32, -16);
	auto c2 = create_bitconstant(graph.root(), 32, 2);
	auto c3 = create_bitconstant(graph.root(), 32, 32);

	auto ashr0 = bitashr_op::create(32, s0, s1);
	auto ashr1 = bitashr_op::create(32, c0, c2);
	auto ashr2 = bitashr_op::create(32, c0, c3);
	auto ashr3 = bitashr_op::create(32, c1, c2);
	auto ashr4 = bitashr_op::create(32, c1, c3);

	graph.add_export(ashr0, {ashr0->type(), "dummy"});
	graph.add_export(ashr1, {ashr1->type(), "dummy"});
	graph.add_export(ashr2, {ashr2->type(), "dummy"});
	graph.add_export(ashr3, {ashr3->type(), "dummy"});
	graph.add_export(ashr4, {ashr4->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(ashr0)->operation() == bitashr_op(32));
	assert(node_output::node(ashr1)->operation() == int_constant_op(32, 4));
	assert(node_output::node(ashr2)->operation() == int_constant_op(32, 0));
	assert(node_output::node(ashr3)->operation() == int_constant_op(32, -4));
	assert(node_output::node(ashr4)->operation() == int_constant_op(32, -1));

	return 0;
}

static int types_bitstring_arithmetic_test_bitdifference(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto diff = bitsub_op::create(32, s0, s1);

	graph.add_export(diff, {diff->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(diff)->operation() == bitsub_op(32));

	return 0;
}

static int types_bitstring_arithmetic_test_bitnegate(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto c0 = create_bitconstant(graph.root(), 32, 3);

	auto neg0 = bitneg_op::create(32, s0);
	auto neg1 = bitneg_op::create(32, c0);
	auto neg2 = bitneg_op::create(32, neg1);

	graph.add_export(neg0, {neg0->type(), "dummy"});
	graph.add_export(neg1, {neg1->type(), "dummy"});
	graph.add_export(neg2, {neg2->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(neg0)->operation() == bitneg_op(32));
	assert(node_output::node(neg1)->operation() == int_constant_op(32, -3));
	assert(node_output::node(neg2)->operation() == int_constant_op(32, 3));

	return 0;
}

static int types_bitstring_arithmetic_test_bitnot(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto c0 = create_bitconstant(graph.root(), 32, 3);

	auto not0 = bitnot_op::create(32, s0);
	auto not1 = bitnot_op::create(32, c0);
	auto not2 = bitnot_op::create(32, not1);

	graph.add_export(not0, {not0->type(), "dummy"});
	graph.add_export(not1, {not1->type(), "dummy"});
	graph.add_export(not2, {not2->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(not0)->operation() == bitnot_op(32));
	assert(node_output::node(not1)->operation() == int_constant_op(32, -4));
	assert(node_output::node(not2)->operation() == int_constant_op(32, 3));

	return 0;
}

static int types_bitstring_arithmetic_test_bitor(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 5);

	auto or0 = bitor_op::create(32, s0, s1);
	auto or1 = bitor_op::create(32, c0, c1);

	graph.add_export(or0, {or0->type(), "dummy"});
	graph.add_export(or1, {or1->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(or0)->operation() == bitor_op(32));
	assert(node_output::node(or1)->operation() == uint_constant_op(32, 7));

	return 0;
}

static int types_bitstring_arithmetic_test_bitproduct(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 5);

	auto product0 = bitmul_op::create(32, s0, s1);
	auto product1 = bitmul_op::create(32, c0, c1);

	graph.add_export(product0, {product0->type(), "dummy"});
	graph.add_export(product1, {product1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(product0)->operation() == bitmul_op(32));
	assert(node_output::node(product1)->operation() == uint_constant_op(32, 15));

	return 0;
}

static int types_bitstring_arithmetic_test_bitshiproduct(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto shiproduct = bitsmulh_op::create(32, s0, s1);

	graph.add_export(shiproduct, {shiproduct->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(shiproduct)->operation() == bitsmulh_op(32));

	return 0;
}

static int types_bitstring_arithmetic_test_bitshl(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 16);
	auto c1 = create_bitconstant(graph.root(), 32, 2);
	auto c2 = create_bitconstant(graph.root(), 32, 32);

	auto shl0 = bitshl_op::create(32, s0, s1);
	auto shl1 = bitshl_op::create(32, c0, c1);
	auto shl2 = bitshl_op::create(32, c0, c2);

	graph.add_export(shl0, {shl0->type(), "dummy"});
	graph.add_export(shl1, {shl1->type(), "dummy"});
	graph.add_export(shl2, {shl2->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(shl0)->operation() == bitshl_op(32));
	assert(node_output::node(shl1)->operation() == uint_constant_op(32, 64));
	assert(node_output::node(shl2)->operation() == uint_constant_op(32, 0));

	return 0;
}

static int types_bitstring_arithmetic_test_bitshr(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 16);
	auto c1 = create_bitconstant(graph.root(), 32, 2);
	auto c2 = create_bitconstant(graph.root(), 32, 32);

	auto shr0 = bitshr_op::create(32, s0, s1);
	auto shr1 = bitshr_op::create(32, c0, c1);
	auto shr2 = bitshr_op::create(32, c0, c2);

	graph.add_export(shr0, {shr0->type(), "dummy"});
	graph.add_export(shr1, {shr1->type(), "dummy"});
	graph.add_export(shr2, {shr2->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);
	
	assert(node_output::node(shr0)->operation() == bitshr_op(32));
	assert(node_output::node(shr1)->operation() == uint_constant_op(32, 4));
	assert(node_output::node(shr2)->operation() == uint_constant_op(32, 0));

	return 0;
}

static int types_bitstring_arithmetic_test_bitsmod(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, -7);
	auto c1 = create_bitconstant(graph.root(), 32, 3);

	auto smod0 = bitsmod_op::create(32, s0, s1);
	auto smod1 = bitsmod_op::create(32, c0, c1);

	graph.add_export(smod0, {smod0->type(), "dummy"});
	graph.add_export(smod1, {smod1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(smod0)->operation() == bitsmod_op(32));
	assert(node_output::node(smod1)->operation() == int_constant_op(32, -1));

	return 0;
}

static int types_bitstring_arithmetic_test_bitsquotient(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 7);
	auto c1 = create_bitconstant(graph.root(), 32, -3);

	auto squot0 = bitsdiv_op::create(32, s0, s1);
	auto squot1 = bitsdiv_op::create(32, c0, c1);

	graph.add_export(squot0, {squot0->type(), "dummy"});
	graph.add_export(squot1, {squot1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(squot0)->operation() == bitsdiv_op(32));
	assert(node_output::node(squot1)->operation() == int_constant_op(32, -2));

	return 0;
}

static int types_bitstring_arithmetic_test_bitsum(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 5);

	auto sum0 = bitadd_op::create(32, s0, s1);
	auto sum1 = bitadd_op::create(32, c0, c1);

	graph.add_export(sum0, {sum0->type(), "dummy"});
	graph.add_export(sum1, {sum1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(sum0)->operation() == bitadd_op(32));
	assert(node_output::node(sum1)->operation() == int_constant_op(32, 8));

	return 0;
}

static int types_bitstring_arithmetic_test_bituhiproduct(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto uhiproduct = bitumulh_op::create(32, s0, s1);

	graph.add_export(uhiproduct, {uhiproduct->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(uhiproduct)->operation() == bitumulh_op(32));

	return 0;
}

static int types_bitstring_arithmetic_test_bitumod(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 7);
	auto c1 = create_bitconstant(graph.root(), 32, 3);

	auto umod0 = bitumod_op::create(32, s0, s1);
	auto umod1 = bitumod_op::create(32, c0, c1);

	graph.add_export(umod0, {umod0->type(), "dummy"});
	graph.add_export(umod1, {umod1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(umod0)->operation() == bitumod_op(32));
	assert(node_output::node(umod1)->operation() == int_constant_op(32, 1));

	return 0;
}

static int types_bitstring_arithmetic_test_bituquotient(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 7);
	auto c1 = create_bitconstant(graph.root(), 32, 3);

	auto uquot0 = bitudiv_op::create(32, s0, s1);
	auto uquot1 = bitudiv_op::create(32, c0, c1);

	graph.add_export(uquot0, {uquot0->type(), "dummy"});
	graph.add_export(uquot1, {uquot1->type(), "dummy"});

	graph.normalize();
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(uquot0)->operation() == bitudiv_op(32));
	assert(node_output::node(uquot1)->operation() == int_constant_op(32, 2));

	return 0;
}

static int types_bitstring_arithmetic_test_bitxor(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 5);

	auto xor0 = bitxor_op::create(32, s0, s1);
	auto xor1 = bitxor_op::create(32, c0, c1);

	graph.add_export(xor0, {xor0->type(), "dummy"});
	graph.add_export(xor1, {xor1->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(xor0)->operation() == bitxor_op(32));
	assert(node_output::node(xor1)->operation() == int_constant_op(32, 6));

	return 0;
}

static inline void
expect_static_true(jive::output * port)
{
	auto node = jive::node_output::node(port);
	auto op = dynamic_cast<const jive::bitconstant_op*>(&node->operation());
	assert(op && op->value().nbits() == 1 && op->value().str() == "1");
}

static inline void
expect_static_false(jive::output * port)
{
	auto node = jive::node_output::node(port);
	auto op = dynamic_cast<const jive::bitconstant_op*>(&node->operation());
	assert(op && op->value().nbits() == 1 && op->value().str() == "0");
}

static int types_bitstring_comparison_test_bitequal(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant_undefined(graph.root(), 32);

	auto equal0 = biteq_op::create(32, s0, s1);
	auto equal1 = biteq_op::create(32, c0, c0);
	auto equal2 = biteq_op::create(32, c0, c1);
	auto equal3 = biteq_op::create(32, c0, c2);

	graph.add_export(equal0, {equal0->type(), "dummy"});
	graph.add_export(equal1, {equal1->type(), "dummy"});
	graph.add_export(equal2, {equal2->type(), "dummy"});
	graph.add_export(equal3, {equal3->type(), "dummy"});
	
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(equal0)->operation() == biteq_op(32));
	expect_static_true(equal1);
	expect_static_false(equal2);
	assert(node_output::node(equal3)->operation() == biteq_op(32));

	return 0;
}

static int types_bitstring_comparison_test_bitnotequal(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant_undefined(graph.root(), 32);

	auto nequal0 = bitne_op::create(32, s0, s1);
	auto nequal1 = bitne_op::create(32, c0, c0);
	auto nequal2 = bitne_op::create(32, c0, c1);
	auto nequal3 = bitne_op::create(32, c0, c2);

	graph.add_export(nequal0, {nequal0->type(), "dummy"});
	graph.add_export(nequal1, {nequal1->type(), "dummy"});
	graph.add_export(nequal2, {nequal2->type(), "dummy"});
	graph.add_export(nequal3, {nequal3->type(), "dummy"});
	
	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(nequal0)->operation() == bitne_op(32));
	expect_static_false(nequal1);
	expect_static_true(nequal2);
	assert(node_output::node(nequal3)->operation() == bitne_op(32));

	return 0;
}

static int types_bitstring_comparison_test_bitsgreater(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, 0x7fffffffL);
	auto c3 = create_bitconstant(graph.root(), 32, (-0x7fffffffL-1));

	auto sgreater0 = bitsgt_op::create(32, s0, s1);
	auto sgreater1 = bitsgt_op::create(32, c0, c1);
	auto sgreater2 = bitsgt_op::create(32, c1, c0);
	auto sgreater3 = bitsgt_op::create(32, s0, c2);
	auto sgreater4 = bitsgt_op::create(32, c3, s1);

	graph.add_export(sgreater0, {sgreater0->type(), "dummy"});
	graph.add_export(sgreater1, {sgreater1->type(), "dummy"});
	graph.add_export(sgreater2, {sgreater2->type(), "dummy"});
	graph.add_export(sgreater3, {sgreater3->type(), "dummy"});
	graph.add_export(sgreater4, {sgreater4->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(sgreater0)->operation() == bitsgt_op(32));
	expect_static_false(sgreater1);
	expect_static_true(sgreater2);
	expect_static_false(sgreater3);
	expect_static_false(sgreater4);

	return 0;
}

static int types_bitstring_comparison_test_bitsgreatereq(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, 0x7fffffffL);
	auto c3 = create_bitconstant(graph.root(), 32, (-0x7fffffffL-1));

	auto sgreatereq0 = bitsge_op::create(32, s0, s1);
	auto sgreatereq1 = bitsge_op::create(32, c0, c1);
	auto sgreatereq2 = bitsge_op::create(32, c1, c0);
	auto sgreatereq3 = bitsge_op::create(32, c0, c0);
	auto sgreatereq4 = bitsge_op::create(32, c2, s0);
	auto sgreatereq5 = bitsge_op::create(32, s1, c3);

	graph.add_export(sgreatereq0, {sgreatereq0->type(), "dummy"});
	graph.add_export(sgreatereq1, {sgreatereq1->type(), "dummy"});
	graph.add_export(sgreatereq2, {sgreatereq2->type(), "dummy"});
	graph.add_export(sgreatereq3, {sgreatereq3->type(), "dummy"});
	graph.add_export(sgreatereq4, {sgreatereq4->type(), "dummy"});
	graph.add_export(sgreatereq5, {sgreatereq5->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(sgreatereq0)->operation() == bitsge_op(32));
	expect_static_false(sgreatereq1);
	expect_static_true(sgreatereq2);
	expect_static_true(sgreatereq3);
	expect_static_true(sgreatereq4);
	expect_static_true(sgreatereq5);

	return 0;
}

static int types_bitstring_comparison_test_bitsless(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, 0x7fffffffL);
	auto c3 = create_bitconstant(graph.root(), 32, (-0x7fffffffL-1));

	auto sless0 = bitslt_op::create(32, s0, s1);
	auto sless1 = bitslt_op::create(32, c0, c1);
	auto sless2 = bitslt_op::create(32, c1, c0);
	auto sless3 = bitslt_op::create(32, c2, s0);
	auto sless4 = bitslt_op::create(32, s1, c3);

	graph.add_export(sless0, {sless0->type(), "dummy"});
	graph.add_export(sless1, {sless1->type(), "dummy"});
	graph.add_export(sless2, {sless2->type(), "dummy"});
	graph.add_export(sless3, {sless3->type(), "dummy"});
	graph.add_export(sless4, {sless4->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(sless0)->operation() == bitslt_op(32));
	expect_static_true(sless1);
	expect_static_false(sless2);
	expect_static_false(sless3);
	expect_static_false(sless4);

	return 0;
}

static int types_bitstring_comparison_test_bitslesseq(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, 0x7fffffffL);
	auto c3 = create_bitconstant(graph.root(), 32, (-0x7fffffffL-1));

	auto slesseq0 = bitsle_op::create(32, s0, s1);
	auto slesseq1 = bitsle_op::create(32, c0, c1);
	auto slesseq2 = bitsle_op::create(32, c0, c0);
	auto slesseq3 = bitsle_op::create(32, c1, c0);
	auto slesseq4 = bitsle_op::create(32, s0, c2);
	auto slesseq5 = bitsle_op::create(32, c3, s1);

	graph.add_export(slesseq0, {slesseq0->type(), "dummy"});
	graph.add_export(slesseq1, {slesseq1->type(), "dummy"});
	graph.add_export(slesseq2, {slesseq2->type(), "dummy"});
	graph.add_export(slesseq3, {slesseq3->type(), "dummy"});
	graph.add_export(slesseq4, {slesseq4->type(), "dummy"});
	graph.add_export(slesseq5, {slesseq5->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(slesseq0)->operation() == bitsle_op(32));
	expect_static_true(slesseq1);
	expect_static_true(slesseq2);
	expect_static_false(slesseq3);
	expect_static_true(slesseq4);
	expect_static_true(slesseq5);

	return 0;
}

static int types_bitstring_comparison_test_bitugreater(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, (0xffffffffUL));
	auto c3 = create_bitconstant(graph.root(), 32, 0);

	auto ugreater0 = bitugt_op::create(32, s0, s1);
	auto ugreater1 = bitugt_op::create(32, c0, c1);
	auto ugreater2 = bitugt_op::create(32, c1, c0);
	auto ugreater3 = bitugt_op::create(32, s0, c2);
	auto ugreater4 = bitugt_op::create(32, c3, s1);

	graph.add_export(ugreater0, {ugreater0->type(), "dummy"});
	graph.add_export(ugreater1, {ugreater1->type(), "dummy"});
	graph.add_export(ugreater2, {ugreater2->type(), "dummy"});
	graph.add_export(ugreater3, {ugreater3->type(), "dummy"});
	graph.add_export(ugreater4, {ugreater4->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(ugreater0)->operation() == bitugt_op(32));
	expect_static_false(ugreater1);
	expect_static_true(ugreater2);
	expect_static_false(ugreater3);
	expect_static_false(ugreater4);

	return 0;
}

static int types_bitstring_comparison_test_bitugreatereq(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, (0xffffffffUL));
	auto c3 = create_bitconstant(graph.root(), 32, 0);

	auto ugreatereq0 = bituge_op::create(32, s0, s1);
	auto ugreatereq1 = bituge_op::create(32, c0, c1);
	auto ugreatereq2 = bituge_op::create(32, c1, c0);
	auto ugreatereq3 = bituge_op::create(32, c0, c0);
	auto ugreatereq4 = bituge_op::create(32, c2, s0);
	auto ugreatereq5 = bituge_op::create(32, s1, c3);

	graph.add_export(ugreatereq0, {ugreatereq0->type(), "dummy"});
	graph.add_export(ugreatereq1, {ugreatereq1->type(), "dummy"});
	graph.add_export(ugreatereq2, {ugreatereq2->type(), "dummy"});
	graph.add_export(ugreatereq3, {ugreatereq3->type(), "dummy"});
	graph.add_export(ugreatereq4, {ugreatereq4->type(), "dummy"});
	graph.add_export(ugreatereq5, {ugreatereq5->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(ugreatereq0)->operation() == bituge_op(32));
	expect_static_false(ugreatereq1);
	expect_static_true(ugreatereq2);
	expect_static_true(ugreatereq3);
	expect_static_true(ugreatereq4);
	expect_static_true(ugreatereq5);

	return 0;
}

static int types_bitstring_comparison_test_bituless(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, (0xffffffffUL));
	auto c3 = create_bitconstant(graph.root(), 32, 0);

	auto uless0 = bitult_op::create(32, s0, s1);
	auto uless1 = bitult_op::create(32, c0, c1);
	auto uless2 = bitult_op::create(32, c1, c0);
	auto uless3 = bitult_op::create(32, c2, s0);
	auto uless4 = bitult_op::create(32, s1, c3);

	graph.add_export(uless0, {uless0->type(), "dummy"});
	graph.add_export(uless1, {uless1->type(), "dummy"});
	graph.add_export(uless2, {uless2->type(), "dummy"});
	graph.add_export(uless3, {uless3->type(), "dummy"});
	graph.add_export(uless4, {uless4->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(uless0)->operation() == bitult_op(32));
	expect_static_true(uless1);
	expect_static_false(uless2);
	expect_static_false(uless3);
	expect_static_false(uless4);

	return 0;
}

static int types_bitstring_comparison_test_bitulesseq(void)
{
	using namespace jive;

	jive::graph graph;

	auto s0 = graph.add_import({bittype(32), "s0"});
	auto s1 = graph.add_import({bittype(32), "s1"});
	auto c0 = create_bitconstant(graph.root(), 32, 4);
	auto c1 = create_bitconstant(graph.root(), 32, 5);
	auto c2 = create_bitconstant(graph.root(), 32, (0xffffffffUL));
	auto c3 = create_bitconstant(graph.root(), 32, 0);

	auto ulesseq0 = bitule_op::create(32, s0, s1);
	auto ulesseq1 = bitule_op::create(32, c0, c1);
	auto ulesseq2 = bitule_op::create(32, c0, c0);
	auto ulesseq3 = bitule_op::create(32, c1, c0);
	auto ulesseq4 = bitule_op::create(32, s0, c2);
	auto ulesseq5 = bitule_op::create(32, c3, s1);

	graph.add_export(ulesseq0, {ulesseq0->type(), "dummy"});
	graph.add_export(ulesseq1, {ulesseq1->type(), "dummy"});
	graph.add_export(ulesseq2, {ulesseq2->type(), "dummy"});
	graph.add_export(ulesseq3, {ulesseq3->type(), "dummy"});
	graph.add_export(ulesseq4, {ulesseq4->type(), "dummy"});
	graph.add_export(ulesseq5, {ulesseq5->type(), "dummy"});

	graph.prune();
	jive::view(graph.root(), stdout);

	assert(node_output::node(ulesseq0)->operation() == bitule_op(32));
	expect_static_true(ulesseq1);
	expect_static_true(ulesseq2);
	expect_static_false(ulesseq3);
	expect_static_true(ulesseq4);
	expect_static_true(ulesseq5);

	return 0;
}

#define ZERO_64 \
	"00000000" "00000000" "00000000" "00000000" \
	"00000000" "00000000" "00000000" "00000000"
#define ONE_64 \
	"10000000" "00000000" "00000000" "00000000" \
	"00000000" "00000000" "00000000" "00000000"
#define MONE_64 \
	"11111111" "11111111" "11111111" "11111111" \
	"11111111" "11111111" "11111111" "11111111"

static int types_bitstring_test_constant(void)
{
	using namespace jive;

	jive::graph graph;

	auto b1 = node_output::node(create_bitconstant(graph.root(), "00110011"));
	auto b2 = node_output::node(create_bitconstant(graph.root(), 8, 204));
	auto b3 = node_output::node(create_bitconstant(graph.root(), 8, 204));
	auto b4 = node_output::node(create_bitconstant(graph.root(), "001100110"));
	
	assert(b1->operation() == uint_constant_op(8, 204));
	assert(b1->operation() == int_constant_op(8, -52));

	assert(b1 == b2);
	assert(b1 == b3);
	
	assert(b1->operation() == uint_constant_op(8, 204));
	assert(b1->operation() == int_constant_op(8, -52));

	assert(b4->operation() == uint_constant_op(9, 204));
	assert(b4->operation() == int_constant_op(9, 204));

	auto plus_one_128 = node_output::node(create_bitconstant(graph.root(), ONE_64 ZERO_64));
	assert(plus_one_128->operation() == uint_constant_op(128, 1));
	assert(plus_one_128->operation() == int_constant_op(128, 1));

	auto minus_one_128 = node_output::node(create_bitconstant(graph.root(), MONE_64 MONE_64));
	assert(minus_one_128->operation() == int_constant_op(128, -1));

	jive::view(graph.root(), stdout);

	return 0;
}

static int types_bitstring_test_normalize(void)
{
	using namespace jive;

	jive::graph graph;

	bittype bits32(32);
	auto imp = graph.add_import({bits32, "imp"});

	auto c0 = create_bitconstant(graph.root(), 32, 3);
	auto c1 = create_bitconstant(graph.root(), 32, 4);
	
	auto  sum_nf = graph.node_normal_form(typeid(bitadd_op));
	assert(sum_nf);
	sum_nf->set_mutable(false);

	auto sum0 = node_output::node(bitadd_op::create(32, imp, c0));
	assert(sum0->operation() == bitadd_op(32));
	assert(sum0->ninputs() == 2);

	auto sum1 = node_output::node(bitadd_op::create(32, sum0->output(0), c1));
	assert(sum1->operation() == bitadd_op(32));
	assert(sum1->ninputs() == 2);

	auto exp = graph.add_export(sum1->output(0), {sum1->output(0)->type(), "dummy"});

	sum_nf->set_mutable(true);
	graph.normalize();
	graph.prune();

	auto origin = dynamic_cast<node_output*>(exp->origin());
	assert(origin->node()->operation() == bitadd_op(32));
	assert(origin->node()->ninputs() == 2);
	auto op1 = origin->node()->input(0)->origin();
	auto op2 = origin->node()->input(1)->origin();
	if (!is<node_output>(op1)) {
		auto tmp = op1; op1 = op2; op2 = tmp;
	}
	/* FIXME: the graph traversers are currently broken, that is why it won't normalize */
	assert(node_output::node(op1)->operation() == int_constant_op(32, 3 + 4));
	assert(op2 == imp);

	jive::view(graph.root(), stdout);

	return 0;
}

static void
assert_constant(jive::output * bitstr, size_t nbits, const char bits[])
{
	auto node = jive::node_output::node(bitstr);
	auto op = dynamic_cast<const jive::bitconstant_op &>(node->operation());
	assert(op.value() == jive::bitvalue_repr(std::string(bits, nbits).c_str()));
}

static int types_bitstring_test_reduction(void)
{
	using namespace jive;

	jive::graph graph;

	auto a = create_bitconstant(graph.root(), "1100");
	auto b = create_bitconstant(graph.root(), "1010");

	assert_constant(bitand_op::create(4, a, b), 4, "1000");
	assert_constant(bitor_op::create(4, a, b), 4, "1110");
	assert_constant(bitxor_op::create(4, a, b), 4, "0110");
	assert_constant(bitadd_op::create(4, a, b), 4, "0001");
	assert_constant(bitmul_op::create(4, a, b), 4, "1111");
	assert_constant(jive_bitconcat({a, b}), 8, "11001010");
	assert_constant(bitneg_op::create(4, a), 4, "1011");
	assert_constant(bitneg_op::create(4, b), 4, "1101");
	
	graph.prune();
	
	auto x = graph.add_import({bittype(16), "x"});
	auto y = graph.add_import({bittype(16), "y"});
	
	{
		auto concat = jive_bitconcat({x, y});
		auto node = node_output::node(jive_bitslice(concat, 8, 24));
		auto o0 = dynamic_cast<node_output*>(node->input(0)->origin());
		auto o1 = dynamic_cast<node_output*>(node->input(1)->origin());
		assert(dynamic_cast<const bitconcat_op*>(&node->operation()));
		assert(node->ninputs() == 2);
		assert(dynamic_cast<const bitslice_op*>(&o0->node()->operation()));
		assert(dynamic_cast<const bitslice_op*>(&o1->node()->operation()));
		
		const bitslice_op * attrs;
		attrs = dynamic_cast<const bitslice_op*>(&o0->node()->operation());
		assert( (attrs->low() == 8) && (attrs->high() == 16) );
		attrs = dynamic_cast<const bitslice_op*>(&o1->node()->operation());
		assert( (attrs->low() == 0) && (attrs->high() == 8) );
		
		assert(o0->node()->input(0)->origin() == x);
		assert(o1->node()->input(0)->origin() == y);
	}
	
	{
		auto slice1 = jive_bitslice(x, 0, 8);
		auto slice2 = jive_bitslice(x, 8, 16);
		auto concat = jive_bitconcat({slice1, slice2});
		assert(concat == x);
	}

	return 0;
}

static int types_bitstring_test_slice_concat(void)
{
	using namespace jive;

	jive::graph graph;
	
	auto base_const1 = create_bitconstant(graph.root(), "00110111");
	auto base_const2 = create_bitconstant(graph.root(), "11001000");
	
	auto base_x = graph.add_import({bittype(8), "x"});
	auto base_y = graph.add_import({bittype(8), "y"});
	auto base_z = graph.add_import({bittype(8), "z"});
	
	{
		/* slice of constant */
		auto a = node_output::node(jive_bitslice(base_const1, 2, 6));
		
		auto & op = dynamic_cast<const bitconstant_op&>(a->operation());
		assert(op.value() == bitvalue_repr("1101"));
	}
	
	{
		/* slice of slice */
		auto a = jive_bitslice(base_x, 2, 6);
		auto b = node_output::node(jive_bitslice(a, 1, 3));

		assert(dynamic_cast<const bitslice_op*>(&b->operation()));
		const bitslice_op * attrs;
		attrs = dynamic_cast<const bitslice_op*>(&b->operation());
		assert(attrs->low() == 3 && attrs->high() == 5);
	}
	
	{
		/* slice of full node */
		auto a = jive_bitslice(base_x, 0, 8);
		
		assert(a == base_x);
	}
	
	{
		/* slice of concat */
		auto a = jive_bitconcat({base_x, base_y});
		auto b = jive_bitslice(a, 0, 8);
		
		assert(static_cast<const bittype*>(&b->type())->nbits() == 8);
		
		assert(b == base_x);
	}
	
	{
		/* concat flattening */
		auto a = jive_bitconcat({base_x, base_y});
		auto b = node_output::node(jive_bitconcat({a, base_z}));
		
		assert(dynamic_cast<const bitconcat_op*>(&b->operation()));
		assert(b->ninputs() == 3);
		assert(b->input(0)->origin() == base_x);
		assert(b->input(1)->origin() == base_y);
		assert(b->input(2)->origin() == base_z);
	}
	
	{
		/* concat of single node */
		auto a = jive_bitconcat({base_x});
		
		assert(a==base_x);
	}
	
	{
		/* concat of slices */
		auto a = jive_bitslice(base_x, 0, 4);
		auto b = jive_bitslice(base_x, 4, 8);
		auto c = jive_bitconcat({a, b});
		
		assert(c==base_x);
	}
	
	{
		/* concat of constants */
		auto a = node_output::node(jive_bitconcat({base_const1, base_const2}));
		
		auto & op = dynamic_cast<const bitconstant_op&>(a->operation());
		assert(op.value() == bitvalue_repr("0011011111001000"));
	}
	
	{
		/* CSE */
		auto b = create_bitconstant(graph.root(), "00110111");
		assert(b == base_const1);
		
		auto c = jive_bitslice(base_x, 2, 6);
		auto d = jive_bitslice(base_x, 2, 6);
		assert(c == d);
		
		auto e = jive_bitconcat({base_x, base_y});
		auto f = jive_bitconcat({base_x, base_y});
		assert(e == f);
	}
	
	return 0;
}

static const char * bs[] = {
	"00000000",
	"11111111",
	"10000000",
	"01111111",
	"00001111",
	"XXXX0011",
	"XD001100",
	"XXXXDDDD",
	"10XDDX01",
	"0DDDDDD1"};

static std::string bitstring_not[] = {
	"11111111",
	"00000000",
	"01111111",
	"10000000",
	"11110000",
	"XXXX1100",
	"XD110011",
	"XXXXDDDD",
	"01XDDX10",
	"1DDDDDD0"};

static std::string bitstring_xor[10][10] = {
	{"00000000", "11111111", "10000000", "01111111", "00001111", "XXXX0011", "XD001100", "XXXXDDDD",
		"10XDDX01", "0DDDDDD1"},
	{"11111111", "00000000", "01111111", "10000000", "11110000", "XXXX1100", "XD110011", "XXXXDDDD",
		"01XDDX10", "1DDDDDD0"},
	{"10000000", "01111111", "00000000", "11111111", "10001111", "XXXX0011", "XD001100", "XXXXDDDD",
		"00XDDX01", "1DDDDDD1"},
	{"01111111", "10000000", "11111111", "00000000", "01110000", "XXXX1100", "XD110011", "XXXXDDDD",
		"11XDDX10", "0DDDDDD0"},
	{"00001111", "11110000", "10001111", "01110000", "00000000", "XXXX1100", "XD000011", "XXXXDDDD",
		"10XDDX10", "0DDDDDD0"},
	{"XXXX0011", "XXXX1100", "XXXX0011", "XXXX1100", "XXXX1100", "XXXX0000", "XXXX1111", "XXXXDDDD",
		"XXXXDX10", "XXXXDDD0"},
	{"XD001100", "XD110011", "XD001100", "XD110011", "XD000011", "XXXX1111", "XD000000", "XXXXDDDD",
		"XDXDDX01", "XDDDDDD1"},
	{"XXXXDDDD", "XXXXDDDD", "XXXXDDDD", "XXXXDDDD", "XXXXDDDD", "XXXXDDDD", "XXXXDDDD", "XXXXDDDD",
		"XXXXDXDD", "XXXXDDDD"},
	{"10XDDX01", "01XDDX10", "00XDDX01", "11XDDX10", "10XDDX10", "XXXXDX10", "XDXDDX01", "XXXXDXDD",
		"00XDDX00", "1DXDDXD0"},
	{"0DDDDDD1", "1DDDDDD0", "1DDDDDD1", "0DDDDDD0", "0DDDDDD0", "XXXXDDD0", "XDDDDDD1", "XXXXDDDD",
		"1DXDDXD0", "0DDDDDD0"}};

static std::string bitstring_or[10][10] = {
	{"00000000", "11111111", "10000000", "01111111", "00001111", "XXXX0011", "XD001100", "XXXXDDDD",
		"10XDDX01", "0DDDDDD1"},
	{"11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111",
		"11111111", "11111111"},
	{"10000000", "11111111", "10000000", "11111111", "10001111", "1XXX0011", "1D001100", "1XXXDDDD",
		"10XDDX01", "1DDDDDD1"},
	{"01111111", "11111111", "11111111", "01111111", "01111111", "X1111111", "X1111111", "X1111111",
		"11111111", "01111111"},
	{"00001111", "11111111", "10001111", "01111111", "00001111", "XXXX1111", "XD001111", "XXXX1111",
		"10XD1111", "0DDD1111"},
	{"XXXX0011", "11111111", "1XXX0011", "X1111111", "XXXX1111", "XXXX0011", "XXXX1111", "XXXXDD11",
		"1XXXDX11", "XXXXDD11"},
	{"XD001100", "11111111", "1D001100", "X1111111", "XD001111", "XXXX1111", "XD001100", "XXXX11DD",
		"1DXD1101", "XDDD11D1"},
	{"XXXXDDDD", "11111111", "1XXXDDDD", "X1111111", "XXXX1111", "XXXXDD11", "XXXX11DD", "XXXXDDDD",
		"1XXXDXD1", "XXXXDDD1"},
	{"10XDDX01", "11111111", "10XDDX01", "11111111", "10XD1111", "1XXXDX11", "1DXD1101", "1XXXDXD1",
		"10XDDX01", "1DXDDXD1"},
	{"0DDDDDD1", "11111111", "1DDDDDD1", "01111111", "0DDD1111", "XXXXDD11", "XDDD11D1", "XXXXDDD1",
		"1DXDDXD1", "0DDDDDD1"}};

static std::string bitstring_and[10][10] = {
	{"00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000",
		"00000000", "00000000"},
	{"00000000", "11111111", "10000000", "01111111", "00001111", "XXXX0011", "XD001100", "XXXXDDDD",
		"10XDDX01", "0DDDDDD1"},
	{"00000000", "10000000", "10000000", "00000000", "00000000", "X0000000", "X0000000", "X0000000",
		"10000000", "00000000"},
	{"00000000", "01111111", "00000000", "01111111", "00001111", "0XXX0011", "0D001100", "0XXXDDDD",
		"00XDDX01", "0DDDDDD1"},
	{"00000000", "00001111", "00000000", "00001111", "00001111", "00000011", "00001100", "0000DDDD",
		"0000DX01", "0000DDD1"},
	{"00000000", "XXXX0011", "X0000000", "0XXX0011", "00000011", "XXXX0011", "XX000000", "XXXX00DD",
		"X0XX0001", "0XXX00D1"},
	{"00000000", "XD001100", "X0000000", "0D001100", "00001100", "XX000000", "XD001100", "XX00DD00",
		"X000DX00", "0D00DD00"},
	{"00000000", "XXXXDDDD", "X0000000", "0XXXDDDD", "0000DDDD", "XXXX00DD", "XX00DD00", "XXXXDDDD",
		"X0XXDX0D", "0XXXDDDD"},
	{"00000000", "10XDDX01", "10000000", "00XDDX01", "0000DX01", "X0XX0001", "X000DX00", "X0XXDX0D",
		"10XDDX01", "00XDDX01"},
	{"00000000", "0DDDDDD1", "00000000", "0DDDDDD1", "0000DDD1", "0XXX00D1", "0D00DD00", "0XXXDDDD",
		"00XDDX01", "0DDDDDD1"}};

static char equal[10][10] = {
	{'1', '0', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'0', '1', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'0', '0', '1', '0', '0', '0', '0', 'X', '0', '0'},
	{'0', '0', '0', '1', '0', '0', '0', 'X', '0', 'D'},
	{'0', '0', '0', '0', '1', '0', '0', 'X', '0', 'D'},
	{'0', '0', '0', '0', '0', 'X', '0', 'X', '0', 'X'},
	{'0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0'},
	{'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'0', '0', '0', '0', '0', '0', '0', 'X', 'X', '0'},
	{'0', '0', '0', 'D', 'D', 'X', '0', 'X', '0', 'D'}};

static char notequal[10][10] = {
	{'0', '1', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'1', '0', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'1', '1', '0', '1', '1', '1', '1', 'X', '1', '1'},
	{'1', '1', '1', '0', '1', '1', '1', 'X', '1', 'D'},
	{'1', '1', '1', '1', '0', '1', '1', 'X', '1', 'D'},
	{'1', '1', '1', '1', '1', 'X', '1', 'X', '1', 'X'},
	{'1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1'},
	{'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'1', '1', '1', '1', '1', '1', '1', 'X', 'X', '1'},
	{'1', '1', '1', 'D', 'D', 'X', '1', 'X', '1', 'D'}};

static char sgreatereq[10][10] = {
	{'1', '1', '0', '1', '1', '1', '0', 'X', '1', '1'},
	{'0', '1', '0', '1', '1', '1', '0', 'D', '1', '1'},
	{'1', '1', '1', '1', '1', '1', '0', 'X', '1', '1'},
	{'0', '0', '0', '1', '1', '1', '0', 'X', '1', '1'},
	{'0', '0', '0', '0', '1', '1', '0', 'X', '1', 'D'},
	{'0', '0', '0', '0', '0', 'X', '0', 'X', '1', 'X'},
	{'1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1'},
	{'D', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X'},
	{'0', '0', '0', '0', '0', '0', '0', 'X', 'X', 'X'},
	{'0', '0', '0', 'D', 'D', 'X', '0', 'X', 'X', 'D'}};

static char sgreater[10][10] = {
	{'0', '1', '0', '1', '1', '1', '0', 'D', '1', '1'},
	{'0', '0', '0', '1', '1', '1', '0', 'X', '1', '1'},
	{'1', '1', '0', '1', '1', '1', '0', 'X', '1', '1'},
	{'0', '0', '0', '0', '1', '1', '0', 'X', '1', 'D'},
	{'0', '0', '0', '0', '0', '1', '0', 'D', '1', 'D'},
	{'0', '0', '0', '0', '0', 'X', '0', 'X', '1', 'X'},
	{'1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1'},
	{'X', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'0', '0', '0', '0', '0', '0', '0', 'X', 'X', 'X'},
	{'0', '0', '0', '0', 'D', 'X', '0', 'X', 'X', 'D'}};

static char slesseq[10][10] = {
	{'1', '0', '1', '0', '0', '0', '1', 'D', '0', '0'},
	{'1', '1', '1', '0', '0', '0', '1', 'X', '0', '0'},
	{'0', '0', '1', '0', '0', '0', '1', 'X', '0', '0'},
	{'1', '1', '1', '1', '0', '0', '1', 'X', '0', 'D'},
	{'1', '1', '1', '1', '1', '0', '1', 'D', '0', 'D'},
	{'1', '1', '1', '1', '1', 'X', '1', 'X', '0', 'X'},
	{'0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0'},
	{'X', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'1', '1', '1', '1', '1', '1', '1', 'X', 'X', 'X'},
	{'1', '1', '1', '1', 'D', 'X', '1', 'X', 'X', 'D'}};

static char sless[10][10] = {
	{'0', '0', '1', '0', '0', '0', '1', 'X', '0', '0'},
	{'1', '0', '1', '0', '0', '0', '1', 'D', '0', '0'},
	{'0', '0', '0', '0', '0', '0', '1', 'X', '0', '0'},
	{'1', '1', '1', '0', '0', '0', '1', 'X', '0', '0'},
	{'1', '1', '1', '1', '0', '0', '1', 'X', '0', 'D'},
	{'1', '1', '1', '1', '1', 'X', '1', 'X', '0', 'X'},
	{'0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0'},
	{'D', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X'},
	{'1', '1', '1', '1', '1', '1', '1', 'X', 'X', 'X'},
	{'1', '1', '1', 'D', 'D', 'X', '1', 'X', 'X', 'D'}};

static char ugreatereq[10][10] = {
	{'1', '0', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'1', '1', '1', '1', '1', '1', '1', '1', '1', '1'},
	{'1', '0', '1', '0', '0', '0', '0', 'X', '0', '0'},
	{'1', '0', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'1', '0', '1', '0', '1', '1', '1', 'X', '1', 'D'},
	{'1', '0', '1', '0', '0', 'X', '1', 'X', '1', 'X'},
	{'1', '0', '1', '0', '0', '0', 'X', 'X', '0', '0'},
	{'1', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X'},
	{'1', '0', '1', '0', '0', '0', '1', 'X', 'X', 'X'},
	{'1', '0', '1', 'D', 'D', 'X', '1', 'X', 'X', 'D'}};

static char ugreater[10][10] = {
	{'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'},
	{'1', '0', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'1', '0', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'1', '0', '1', '0', '1', '1', '1', 'X', '1', 'D'},
	{'1', '0', '1', '0', '0', '1', '1', 'D', '1', 'D'},
	{'1', '0', '1', '0', '0', 'X', '1', 'X', '1', 'X'},
	{'1', '0', '1', '0', '0', '0', 'X', 'X', '0', '0'},
	{'X', '0', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'1', '0', '1', '0', '0', '0', '1', 'X', 'X', 'X'},
	{'1', '0', '1', '0', 'D', 'X', '1', 'X', 'X', 'D'}};

static char ulesseq[10][10] = {
	{'1', '1', '1', '1', '1', '1', '1', '1', '1', '1'},
	{'0', '1', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'0', '1', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'0', '1', '0', '1', '0', '0', '0', 'X', '0', 'D'},
	{'0', '1', '0', '1', '1', '0', '0', 'D', '0', 'D'},
	{'0', '1', '0', '1', '1', 'X', '0', 'X', '0', 'X'},
	{'0', '1', '0', '1', '1', '1', 'X', 'X', '1', '1'},
	{'X', '1', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'},
	{'0', '1', '0', '1', '1', '1', '0', 'X', 'X', 'X'},
	{'0', '1', '0', '1', 'D', 'X', '0', 'X', 'X', 'D'}};

static char uless[10][10] = {
	{'0', '1', '1', '1', '1', '1', '1', 'X', '1', '1'},
	{'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'},
	{'0', '1', '0', '1', '1', '1', '1', 'X', '1', '1'},
	{'0', '1', '0', '0', '0', '0', '0', 'X', '0', '0'},
	{'0', '1', '0', '1', '0', '0', '0', 'X', '0', 'D'},
	{'0', '1', '0', '1', '1', 'X', '0', 'X', '0', 'X'},
	{'0', '1', '0', '1', '1', '1', 'X', 'X', '1', '1'},
	{'0', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X'},
	{'0', '1', '0', '1', '1', '1', '0', 'X', 'X', 'X'},
	{'0', '1', '0', 'D', 'D', 'X', '0', 'X', 'X', 'D'}};

static int
types_bitstring_test_value_representation()
{
	using namespace jive;

	for (size_t r = 0; r < 10; r++) {
		assert(bitvalue_repr(bs[r]).lnot() == bitstring_not[r]);
		for (size_t c = 0; c < 10; c++) {
			assert(bitvalue_repr(bs[r]).land(bs[c]) == bitstring_and[r][c]);
			assert(bitvalue_repr(bs[r]).lor(bs[c]) == bitstring_or[r][c]);
			assert(bitvalue_repr(bs[r]).lxor(bs[c]) == bitstring_xor[r][c]);

			assert(bitvalue_repr(bs[r]).ult(bs[c]) == uless[r][c]);
			assert(bitvalue_repr(bs[r]).slt(bs[c]) == sless[r][c]);

			assert(bitvalue_repr(bs[r]).ule(bs[c]) == ulesseq[r][c]);
			assert(bitvalue_repr(bs[r]).sle(bs[c]) == slesseq[r][c]);

			assert(bitvalue_repr(bs[r]).eq(bs[c]) == equal[r][c]);
			assert(bitvalue_repr(bs[r]).ne(bs[c]) == notequal[r][c]);

			assert(bitvalue_repr(bs[r]).uge(bs[c]) == ugreatereq[r][c]);
			assert(bitvalue_repr(bs[r]).sge(bs[c]) == sgreatereq[r][c]);

			assert(bitvalue_repr(bs[r]).ugt(bs[c]) == ugreater[r][c]);
			assert(bitvalue_repr(bs[r]).sgt(bs[c]) == sgreater[r][c]);
		}
	}

	assert(bitvalue_repr("000110").to_uint() == 24);
	assert(bitvalue_repr("00011").to_int() == -8);

	for(ssize_t r = -4; r < 5; r++){
		bitvalue_repr rbits(32, r);

		assert(rbits.neg() == -r);
		assert(rbits.shl(1) == r << 1);
		assert(rbits.shl(32) == 0);
		assert(rbits.ashr(1) == r >> 1);
		assert(rbits.ashr(34) == (r < 0 ? -1 : 0));

		if (r >= 0) {
			assert(rbits.shr(1) == r >> 1);
			assert(rbits.shr(34) == 0);
		}

		for (ssize_t c = -4; c < 5; c++) {
			bitvalue_repr cbits(32, c);

			assert(rbits.add(cbits) == r+c);
			assert(rbits.sub(cbits) == r-c);
			assert(rbits.mul(cbits) == r*c);

			if (r >= 0 && c > 0) {
				assert(rbits.udiv(cbits) == r/c);
				assert(rbits.umod(cbits) == r%c);
			}

			if (c != 0) {
				assert(rbits.sdiv(cbits) == r/c);
				assert(rbits.smod(cbits) == r%c);
			}
		}
	}

	return 0;
}

static int
RunTests()
{
	types_bitstring_arithmetic_test_bitand();
	types_bitstring_arithmetic_test_bitashr();
	types_bitstring_arithmetic_test_bitdifference();
	types_bitstring_arithmetic_test_bitnegate();
	types_bitstring_arithmetic_test_bitnot();
	types_bitstring_arithmetic_test_bitor();
	types_bitstring_arithmetic_test_bitproduct();
	types_bitstring_arithmetic_test_bitshiproduct();
	types_bitstring_arithmetic_test_bitshl();
	types_bitstring_arithmetic_test_bitshr();
	types_bitstring_arithmetic_test_bitsmod();
	types_bitstring_arithmetic_test_bitsquotient();
	types_bitstring_arithmetic_test_bitsum();
	types_bitstring_arithmetic_test_bituhiproduct();
	types_bitstring_arithmetic_test_bitumod();
	types_bitstring_arithmetic_test_bituquotient();
	types_bitstring_arithmetic_test_bitxor();
	types_bitstring_comparison_test_bitequal();
	types_bitstring_comparison_test_bitnotequal();
	types_bitstring_comparison_test_bitsgreater();
	types_bitstring_comparison_test_bitsgreatereq();
	types_bitstring_comparison_test_bitsless();
	types_bitstring_comparison_test_bitslesseq();
	types_bitstring_comparison_test_bitugreater();
	types_bitstring_comparison_test_bitugreatereq();
	types_bitstring_comparison_test_bituless();
	types_bitstring_comparison_test_bitulesseq();
	types_bitstring_test_constant();
	types_bitstring_test_normalize();
	types_bitstring_test_reduction();
	types_bitstring_test_slice_concat();
	types_bitstring_test_value_representation();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring", RunTests);
