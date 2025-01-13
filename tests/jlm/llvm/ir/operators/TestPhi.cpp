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

  jlm::rvsdg::Graph graph;

  auto vtype = jlm::tests::valuetype::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto f0type = jlm::rvsdg::FunctionType::Create(
      { vtype, iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto f1type = jlm::rvsdg::FunctionType::Create(
      { vtype, iostatetype::Create(), MemoryStateType::Create() },
      { vtype, iostatetype::Create(), MemoryStateType::Create() });

  auto SetupEmptyLambda = [&](jlm::rvsdg::Region * region, const std::string & name)
  {
    auto lambda = lambda::node::create(region, f0type, name, linkage::external_linkage);
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    return lambda->finalize({ iOStateArgument, memoryStateArgument });
  };

  auto SetupF2 = [&](jlm::rvsdg::Region * region, jlm::rvsdg::RegionArgument * f2)
  {
    auto lambda = lambda::node::create(region, f1type, "f2", linkage::external_linkage);
    auto ctxVarF2 = lambda->AddContextVar(*f2).inner;
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto callResults =
        CallNode::Create(ctxVarF2, f1type, { valueArgument, iOStateArgument, memoryStateArgument });

    return lambda->finalize(callResults);
  };

  phi::builder pb;
  pb.begin(&graph.GetRootRegion());
  auto rv1 = pb.add_recvar(f0type);
  auto rv2 = pb.add_recvar(f0type);
  auto rv3 = pb.add_recvar(f1type);

  auto lambdaOutput0 = SetupEmptyLambda(pb.subregion(), "f0");
  auto lambdaOutput1 = SetupEmptyLambda(pb.subregion(), "f1");
  auto lambdaOutput2 = SetupF2(pb.subregion(), rv3->argument());

  rv1->set_rvorigin(lambdaOutput0);
  rv2->set_rvorigin(lambdaOutput1);
  rv3->set_rvorigin(lambdaOutput2);

  auto phi = pb.end();
  GraphExport::Create(*phi->output(0), "dummy");

  graph.Normalize();
  graph.PruneNodes();

  jlm::rvsdg::view(&graph.GetRootRegion(), stderr);
}

static void
TestRemovePhiArgumentsWhere()
{
  using namespace jlm::llvm;

  // Arrange
  // The phi setup is nonsense, but it is sufficient for this test
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  phi::builder phiBuilder;
  phiBuilder.begin(&rvsdgModule.Rvsdg().GetRootRegion());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);
  auto phiArgument3 = phiBuilder.add_ctxvar(x);
  auto phiArgument4 = phiBuilder.add_ctxvar(x);

  auto result = jlm::tests::SimpleNode::Create(
                    *phiBuilder.subregion(),
                    { phiOutput0->argument(), phiOutput2->argument(), phiArgument4 },
                    { valueType })
                    .output(0);

  phiOutput0->set_rvorigin(result);
  phiOutput1->set_rvorigin(result);
  phiOutput2->set_rvorigin(result);

  auto & phiNode = *phiBuilder.end();

  // Act & Assert
  // Try to remove phiArgument0 even though it is used
  auto numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
      [&](const jlm::rvsdg::RegionArgument & argument)
      {
        return argument.index() == phiOutput0->argument()->index();
      });
  assert(numRemovedArguments == 0);
  assert(phiNode.subregion()->narguments() == 5);
  assert(phiNode.ninputs() == 2);

  // Remove phiArgument1
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
      [&](const jlm::rvsdg::RegionArgument & argument)
      {
        return argument.index() == 1;
      });
  assert(numRemovedArguments == 1);
  assert(phiNode.subregion()->narguments() == 4);
  assert(phiNode.ninputs() == 2);
  assert(phiOutput0->argument()->index() == 0);
  assert(phiOutput2->argument()->index() == 1);
  assert(phiArgument3->index() == 2);
  assert(phiArgument4->index() == 3);

  // Try to remove anything else, but the only dead argument, i.e, phiArgument3
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
      [&](const jlm::rvsdg::RegionArgument & argument)
      {
        return argument.index() != phiArgument3->index();
      });
  assert(numRemovedArguments == 0);
  assert(phiNode.subregion()->narguments() == 4);
  assert(phiNode.ninputs() == 2);

  // Remove everything that is dead, i.e., phiArgument3
  numRemovedArguments = phiNode.RemovePhiArgumentsWhere(
      [&](const jlm::rvsdg::RegionArgument &)
      {
        return true;
      });
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
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  phi::builder phiBuilder;
  phiBuilder.begin(&rvsdgModule.Rvsdg().GetRootRegion());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);
  phiBuilder.add_ctxvar(x);
  auto phiArgument4 = phiBuilder.add_ctxvar(x);

  auto result = jlm::tests::SimpleNode::Create(
                    *phiBuilder.subregion(),
                    { phiOutput0->argument(), phiOutput2->argument(), phiArgument4 },
                    { valueType })
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

static void
TestRemovePhiOutputsWhere()
{
  using namespace jlm::llvm;

  // Arrange
  // The phi setup is nonsense, but it is sufficient for this test
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  phi::builder phiBuilder;
  phiBuilder.begin(&rvsdgModule.Rvsdg().GetRootRegion());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);

  auto result = jlm::tests::SimpleNode::Create(
                    *phiBuilder.subregion(),
                    { phiOutput0->argument(), phiOutput2->argument() },
                    { valueType })
                    .output(0);

  phiOutput0->set_rvorigin(result);
  phiOutput1->set_rvorigin(result);
  phiOutput2->set_rvorigin(result);

  auto & phiNode = *phiBuilder.end();

  // Act & Assert
  auto numRemovedOutputs = phiNode.RemovePhiOutputsWhere(
      [&](const phi::rvoutput & output)
      {
        return output.index() == 1;
      });
  assert(numRemovedOutputs == 1);
  assert(phiNode.noutputs() == 2);
  assert(phiOutput0->index() == 0);
  assert(phiOutput2->index() == 1);

  numRemovedOutputs = phiNode.RemovePhiOutputsWhere(
      [&](const phi::rvoutput &)
      {
        return true;
      });
  assert(numRemovedOutputs == 2);
  assert(phiNode.noutputs() == 0);
}

static void
TestPrunePhiOutputs()
{
  using namespace jlm::llvm;

  // Arrange
  // The phi setup is nonsense, but it is sufficient for this test
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  phi::builder phiBuilder;
  phiBuilder.begin(&rvsdgModule.Rvsdg().GetRootRegion());

  auto phiOutput0 = phiBuilder.add_recvar(valueType);
  auto phiOutput1 = phiBuilder.add_recvar(valueType);
  auto phiOutput2 = phiBuilder.add_recvar(valueType);

  auto result = jlm::tests::SimpleNode::Create(
                    *phiBuilder.subregion(),
                    { phiOutput0->argument(), phiOutput2->argument() },
                    { valueType })
                    .output(0);

  phiOutput0->set_rvorigin(result);
  phiOutput1->set_rvorigin(result);
  phiOutput2->set_rvorigin(result);

  auto & phiNode = *phiBuilder.end();

  // Act
  auto numRemovedOutputs = phiNode.PrunePhiOutputs();

  // Assert
  assert(numRemovedOutputs == 3);
  assert(phiNode.noutputs() == 0);
}

static int
TestPhi()
{
  TestPhiCreation();
  TestRemovePhiArgumentsWhere();
  TestPrunePhiArguments();
  TestRemovePhiOutputsWhere();
  TestPrunePhiOutputs();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestPhi", TestPhi)
