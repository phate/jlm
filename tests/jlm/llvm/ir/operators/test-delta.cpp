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
  auto valueType = jlm::tests::ValueType::Create();
  auto pointerType = PointerType::Create();
  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto imp = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto delta1 = DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      valueType,
      "test-delta1",
      linkage::external_linkage,
      "",
      true);
  auto dep = delta1->AddContextVar(*imp).inner;
  auto d1 =
      &delta1->finalize(jlm::tests::create_testop(delta1->subregion(), { dep }, { valueType })[0]);

  auto delta2 = DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      valueType,
      "test-delta2",
      linkage::internal_linkage,
      "",
      false);
  auto d2 = &delta2->finalize(jlm::tests::create_testop(delta2->subregion(), {}, { valueType })[0]);

  GraphExport::Create(*d1, "");
  GraphExport::Create(*d2, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  assert(rvsdgModule.Rvsdg().GetRootRegion().nnodes() == 2);

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
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      valueType,
      "delta",
      linkage::external_linkage,
      "",
      true);
  auto deltaInput0 = deltaNode->AddContextVar(*x).input;
  auto deltaInput1 = deltaNode->AddContextVar(*x).input;
  deltaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
                    { deltaNode->MapInputContextVar(*deltaInput1).inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
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
  assert(deltaNode->MapInputContextVar(*deltaInput1).inner->index() == 0);
}

static void
TestPruneDeltaInputs()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto x = &jlm::tests::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      valueType,
      "delta",
      linkage::external_linkage,
      "",
      true);

  deltaNode->AddContextVar(*x);
  auto deltaInput1 = deltaNode->AddContextVar(*x).input;
  deltaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<jlm::tests::TestOperation>(
                    { deltaNode->MapInputContextVar(*deltaInput1).inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  deltaNode->finalize(result);

  // Act
  auto numRemovedInputs = deltaNode->PruneDeltaInputs();

  // Assert
  assert(numRemovedInputs == 2);
  assert(deltaNode->ninputs() == 1);
  assert(deltaNode->ncvarguments() == 1);
  assert(deltaNode->input(0) == deltaInput1);
  assert(deltaNode->subregion()->argument(0) == deltaNode->MapInputContextVar(*deltaInput1).inner);
  assert(deltaInput1->index() == 0);
  assert(deltaNode->MapInputContextVar(*deltaInput1).inner->index() == 0);
}

static void
TestDelta()
{
  TestDeltaCreation();
  TestRemoveDeltaInputsWhere();
  TestPruneDeltaInputs();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-delta", TestDelta)
