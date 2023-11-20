/*
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestPhiCreation()
{
  using namespace jlm::llvm;

  jlm::rvsdg::graph graph;

  jlm::tests::valuetype vtype;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType f0type(
      { &vtype, &iOStateType, &memoryStateType, &loopStateType },
      { &iOStateType, &memoryStateType, &loopStateType });
  FunctionType f1type(
      { &vtype, &iOStateType, &memoryStateType, &loopStateType },
      { &vtype, &iOStateType, &memoryStateType, &loopStateType });

  auto SetupEmptyLambda = [&](jlm::rvsdg::region * region, const std::string & name)
  {
    auto lambda = lambda::node::create(region, f0type, name, linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    return lambda->finalize({ iOStateArgument, memoryStateArgument, loopStateArgument });
  };

  auto SetupF2 = [&](jlm::rvsdg::region * region, jlm::rvsdg::argument * f2)
  {
    auto lambda = lambda::node::create(region, f1type, "f2", linkage::external_linkage);
    auto ctxVarF2 = lambda->add_ctxvar(f2);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto callResults = CallNode::Create(
        ctxVarF2,
        f1type,
        { valueArgument, iOStateArgument, memoryStateArgument, loopStateArgument });

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
  graph.add_export(phi->output(0), { phi->output(0)->type(), "dummy" });

  graph.normalize();
  graph.prune();

  jlm::rvsdg::view(graph.root(), stderr);
}

static void
TestRemovePhiArgumentsWhere()
{
  using namespace jlm::llvm;

  // Arrange
  // The phi setup is nonsense, but it is sufficient for this test
  jlm::tests::valuetype valueType;
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = rvsdgModule.Rvsdg().add_import({valueType, ""});

  phi::builder phiBuilder;
  phiBuilder.begin(rvsdgModule.Rvsdg().root());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);
  auto phiArgument3 = phiBuilder.add_ctxvar(x);
  auto phiArgument4 = phiBuilder.add_ctxvar(x);

  auto result = jlm::tests::SimpleNode::Create(
    *phiBuilder.subregion(),
    {phiOutput0->argument(), phiOutput2->argument(), phiArgument4},
    {&valueType})
    .output(0);

  phiOutput0->set_rvorigin(result);
  phiOutput1->set_rvorigin(result);
  phiOutput2->set_rvorigin(result);

  auto & phiNode = *phiBuilder.end();

  // Act & Assert
  // Try to remove phiArgument0 even though it is used
  auto numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
    [&](const jlm::rvsdg::argument& argument){ return argument.index() == phiOutput0->argument()->index(); });
  assert(numRemovedArguments == 0);
  assert(phiNode.subregion()->narguments() == 5);
  assert(phiNode.ninputs() == 2);

  // Remove phiArgument1
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
    [&](const jlm::rvsdg::argument& argument){ return argument.index() == phiOutput1->argument()->index(); });
  assert(numRemovedArguments == 1);
  assert(phiNode.subregion()->narguments() == 4);
  assert(phiNode.ninputs() == 2);
  assert(phiOutput0->argument()->index() == 0);
  assert(phiOutput2->argument()->index() == 1);
  assert(phiArgument3->index() == 2);
  assert(phiArgument4->index() == 3);

  // Try to remove anything else, but the only dead argument, i.e, phiArgument3
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
    [&](const jlm::rvsdg::argument& argument){ return argument.index() != phiArgument3->index(); });
  assert(numRemovedArguments == 0);
  assert(phiNode.subregion()->narguments() == 4);
  assert(phiNode.ninputs() == 2);

  // Remove everything that is dead, i.e., phiArgument3
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere([&](const jlm::rvsdg::argument& argument){ return true; });
  assert(numRemovedArguments == 1);
  assert(phiNode.subregion()->narguments() == 3);
  assert(phiNode.ninputs() == 1);
  assert(phiOutput0->argument()->index() == 0);
  assert(phiOutput2->argument()->index() == 1);
  assert(phiArgument4->index() == 2);
  assert(phiArgument4->input()->index() == 0);
}

static void
TestPrunePhiArguments()
{
  using namespace jlm::llvm;

  // Arrange
  // The phi setup is nonsense, but it is sufficient for this test
  jlm::tests::valuetype valueType;
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = rvsdgModule.Rvsdg().add_import({ valueType, "" });

  phi::builder phiBuilder;
  phiBuilder.begin(rvsdgModule.Rvsdg().root());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);
  phiBuilder.add_ctxvar(x);
  auto phiArgument4 = phiBuilder.add_ctxvar(x);

  auto result = jlm::tests::SimpleNode::Create(
                    *phiBuilder.subregion(),
                    { phiOutput0->argument(), phiOutput2->argument(), phiArgument4 },
                    { &valueType })
                    .output(0);

  phiOutput0->set_rvorigin(result);
  phiOutput1->set_rvorigin(result);
  phiOutput2->set_rvorigin(result);

  auto & phiNode = *phiBuilder.end();

  // Act
  auto numRemovedArguments = phiNode.PrunePhiArguments();

  // Assert
  assert(numRemovedArguments == 2);
  assert(phiNode.subregion()->narguments() == 3);
  assert(phiNode.ninputs() == 1);
  assert(phiOutput0->argument()->index() == 0);
  assert(phiOutput2->argument()->index() == 1);
  assert(phiArgument4->index() == 2);
  assert(phiArgument4->input()->index() == 0);
}

static int
TestPhi()
{
  TestPhiCreation();
  TestRemovePhiArgumentsWhere();
  TestPrunePhiArguments();

  return 0;
}

JLM_UNIT_TEST_REGISTER(
  "jlm/llvm/ir/operators/TestPhi",
  TestPhi)
