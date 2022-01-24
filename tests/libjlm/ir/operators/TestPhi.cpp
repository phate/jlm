/*
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.hpp>

#include <jlm/ir/operators/call.hpp>
#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/operators/Phi.hpp>
#include <jlm/ir/types.hpp>


static int test_main()
{
  using namespace jlm;

  jive::graph graph;

  jlm::valuetype vtype;
  FunctionType f0type({&vtype}, {});
  FunctionType f1type({&vtype}, {&vtype});

  phi::builder pb;
  pb.begin(graph.root());
  auto rv1 = pb.add_recvar(ptrtype(f0type));
  auto rv2 = pb.add_recvar(ptrtype(f0type));
  auto rv3 = pb.add_recvar(ptrtype(f1type));

  auto lambdaF0 = lambda::node::create(
    pb.subregion(),
    f0type,
    "f0",
    linkage::external_linkage);
  auto lambda0 = lambdaF0->finalize({});

  auto lambdaF1 = lambda::node::create(
    pb.subregion(),
    f0type,
    "f1",
    linkage::external_linkage);
  auto lambda1 = lambdaF1->finalize({});

  auto lambdaF2 = lambda::node::create(
    pb.subregion(),
    f1type,
    "f2",
    linkage::external_linkage);
  auto dep = lambdaF2->add_ctxvar(rv3->argument());
  auto ret = CallNode::Create(dep, {lambdaF2->fctargument(0)})[0];
  auto lambda2 = lambdaF2->finalize({ret});

  rv1->set_rvorigin(lambda0);
  rv2->set_rvorigin(lambda1);
  rv3->set_rvorigin(lambda2);

  auto phi = pb.end();
  graph.add_export(phi->output(0), {phi->output(0)->type(), "dummy"});

  graph.normalize();
  graph.prune();

  jive::view(graph.root(), stderr);

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/TestPhi", test_main)
