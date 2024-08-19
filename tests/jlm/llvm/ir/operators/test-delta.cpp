/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

static void
TestDeltaCreation()
{
  using namespace jlm::llvm;

  // Arrange & Act
  auto valueType = jlm::tests::valuetype::Create();
  auto pointerType = PointerType::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto imp = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto delta1 = delta::node::Create(
      rvsdgModule.Rvsdg().root(),
      valueType,
      "test-delta1",
      linkage::external_linkage,
      "",
      true);
  auto dep = delta1->add_ctxvar(imp);
  auto d1 =
      delta1->finalize(jlm::tests::create_testop(delta1->subregion(), { dep }, { valueType })[0]);

  auto delta2 = delta::node::Create(
      rvsdgModule.Rvsdg().root(),
      valueType,
      "test-delta2",
      linkage::internal_linkage,
      "",
      false);
  auto d2 = delta2->finalize(jlm::tests::create_testop(delta2->subregion(), {}, { valueType })[0]);

  GraphExport::Create(*d1, "");
  GraphExport::Create(*d2, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  assert(rvsdgModule.Rvsdg().root()->nnodes() == 2);

  assert(delta1->linkage() == linkage::external_linkage);
  assert(delta1->constant() == true);
  assert(delta1->type() == *valueType);

  assert(delta2->linkage() == linkage::internal_linkage);
  assert(delta2->constant() == false);
  assert(delta2->type() == *valueType);
}

static void
TestRemoveDeltaInputsWhere()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = delta::node::Create(
      rvsdgModule.Rvsdg().root(),
      valueType,
      "delta",
      linkage::external_linkage,
      "",
      true);
  auto deltaInput0 = deltaNode->add_ctxvar(x)->input();
  auto deltaInput1 = deltaNode->add_ctxvar(x)->input();
  deltaNode->add_ctxvar(x)->input();

  auto result = jlm::tests::SimpleNode::Create(
                    *deltaNode->subregion(),
                    { deltaInput1->argument() },
                    { valueType })
                    .output(0);

  deltaNode->finalize(result);

  // Act & Assert
  // Try to remove deltaInput1 even though it is used
  auto numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const delta::cvinput & input)
      {
        return input.index() == deltaInput1->index();
      });
  assert(numRemovedInputs == 0);
  assert(deltaNode->ninputs() == 3);
  assert(deltaNode->ncvarguments() == 3);

  // Remove deltaInput2
  numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const delta::cvinput & input)
      {
        return input.index() == 2;
      });
  assert(numRemovedInputs == 1);
  assert(deltaNode->ninputs() == 2);
  assert(deltaNode->ncvarguments() == 2);
  assert(deltaNode->input(0) == deltaInput0);
  assert(deltaNode->input(1) == deltaInput1);

  // Remove deltaInput0
  numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const delta::cvinput & input)
      {
        return input.index() == 0;
      });
  assert(numRemovedInputs == 1);
  assert(deltaNode->ninputs() == 1);
  assert(deltaNode->ncvarguments() == 1);
  assert(deltaNode->input(0) == deltaInput1);
  assert(deltaInput1->index() == 0);
  assert(deltaInput1->argument()->index() == 0);
}

static void
TestPruneDeltaInputs()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = delta::node::Create(
      rvsdgModule.Rvsdg().root(),
      valueType,
      "delta",
      linkage::external_linkage,
      "",
      true);

  deltaNode->add_ctxvar(x);
  auto deltaInput1 = deltaNode->add_ctxvar(x)->input();
  deltaNode->add_ctxvar(x);

  auto result = jlm::tests::SimpleNode::Create(
                    *deltaNode->subregion(),
                    { deltaInput1->argument() },
                    { valueType })
                    .output(0);

  deltaNode->finalize(result);

  // Act
  auto numRemovedInputs = deltaNode->PruneDeltaInputs();

  // Assert
  assert(numRemovedInputs == 2);
  assert(deltaNode->ninputs() == 1);
  assert(deltaNode->ncvarguments() == 1);
  assert(deltaNode->input(0) == deltaInput1);
  assert(deltaNode->cvargument(0) == deltaInput1->argument());
  assert(deltaInput1->index() == 0);
  assert(deltaInput1->argument()->index() == 0);
}

static int
TestDelta()
{
  TestDeltaCreation();
  TestRemoveDeltaInputsWhere();
  TestPruneDeltaInputs();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-delta", TestDelta)
