/*
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/call.hpp>

static int test_main()
{
  using namespace jlm;

  jive::graph graph;

  jlm::valuetype vtype;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType f0type(
    {&vtype, &iOStateType, &memoryStateType, &loopStateType},
    {&iOStateType, &memoryStateType, &loopStateType});
  FunctionType f1type(
    {&vtype, &iOStateType, &memoryStateType, &loopStateType},
    {&vtype, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupEmptyLambda = [&](jive::region * region, const std::string & name)
  {
    auto lambda = lambda::node::create(
      region,
      f0type,
      name,
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    return lambda->finalize({iOStateArgument, memoryStateArgument, loopStateArgument});
  };

  auto SetupF2 = [&](jive::region * region, jive::argument * f2)
  {
    auto lambda = lambda::node::create(
      region,
      f1type,
      "f2",
      linkage::external_linkage);
    auto ctxVarF2 = lambda->add_ctxvar(f2);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto callResults = CallNode::Create(
      ctxVarF2,
      f1type,
      {valueArgument, iOStateArgument, memoryStateArgument, loopStateArgument});

    return lambda->finalize(callResults);
  };

  phi::builder pb;
  pb.begin(graph.root());
  auto rv1 = pb.add_recvar(PointerType());
  auto rv2 = pb.add_recvar(PointerType());
  auto rv3 = pb.add_recvar(PointerType());

  auto lambdaOutput0 = SetupEmptyLambda(pb.subregion(), "f0");
  auto lambdaOutput1 = SetupEmptyLambda(pb.subregion(), "f1");
  auto lambdaOutput2 = SetupF2(pb.subregion(), rv3->argument());

  rv1->set_rvorigin(lambdaOutput0);
  rv2->set_rvorigin(lambdaOutput1);
  rv3->set_rvorigin(lambdaOutput2);

  auto phi = pb.end();
  graph.add_export(phi->output(0), {phi->output(0)->type(), "dummy"});

  graph.normalize();
  graph.prune();

  jive::view(graph.root(), stderr);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestPhi", test_main)
