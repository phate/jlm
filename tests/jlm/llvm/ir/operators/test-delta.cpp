/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestDeltaCreation()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange & Act
  auto valueType = TestType::createValueType();
  auto pointerType = PointerType::Create();
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto imp = &jlm::rvsdg::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto delta1 = jlm::rvsdg::DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(
          valueType,
          "test-delta1",
          Linkage::externalLinkage,
          "",
          true));
  auto dep = delta1->AddContextVar(*imp).inner;
  auto d1 = &delta1->finalize(
      TestOperation::createNode(delta1->subregion(), { dep }, { valueType })->output(0));

  auto delta2 = jlm::rvsdg::DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(
          valueType,
          "test-delta2",
          Linkage::internalLinkage,
          "",
          false));
  auto d2 = &delta2->finalize(
      TestOperation::createNode(delta2->subregion(), {}, { valueType })->output(0));

  GraphExport::Create(*d1, "");
  GraphExport::Create(*d2, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  assert(rvsdgModule.Rvsdg().GetRootRegion().numNodes() == 2);

  assert(delta1->constant() == true);
  assert(*delta1->Type() == *valueType);

  assert(delta2->constant() == false);
  assert(*delta2->Type() == *valueType);
}

static void
TestRemoveDeltaInputsWhere()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = jlm::rvsdg::DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(valueType, "delta", Linkage::externalLinkage, "", true));
  auto deltaInput0 = deltaNode->AddContextVar(*x).input;
  auto deltaInput1 = deltaNode->AddContextVar(*x).input;
  deltaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<TestOperation>(
                    { deltaNode->MapInputContextVar(*deltaInput1).inner },
                    std::vector<std::shared_ptr<const Type>>{ valueType },
                    std::vector<std::shared_ptr<const Type>>{ valueType })
                    .output(0);

  deltaNode->finalize(result);

  // Act & Assert
  // Try to remove deltaInput1 even though it is used
  auto numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == deltaInput1->index();
      });
  assert(numRemovedInputs == 0);
  assert(deltaNode->ninputs() == 3);
  assert(deltaNode->GetContextVars().size() == 3);

  // Remove deltaInput2
  numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == 2;
      });
  assert(numRemovedInputs == 1);
  assert(deltaNode->ninputs() == 2);
  assert(deltaNode->GetContextVars().size() == 2);
  assert(deltaNode->input(0) == deltaInput0);
  assert(deltaNode->input(1) == deltaInput1);

  // Remove deltaInput0
  numRemovedInputs = deltaNode->RemoveDeltaInputsWhere(
      [&](const jlm::rvsdg::Input & input)
      {
        return input.index() == 0;
      });
  assert(numRemovedInputs == 1);
  assert(deltaNode->ninputs() == 1);
  assert(deltaNode->GetContextVars().size() == 1);
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
  auto valueType = TestType::createValueType();
  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdgModule.Rvsdg(), valueType, "");

  auto deltaNode = jlm::rvsdg::DeltaNode::Create(
      &rvsdgModule.Rvsdg().GetRootRegion(),
      jlm::llvm::DeltaOperation::Create(valueType, "delta", Linkage::externalLinkage, "", true));

  deltaNode->AddContextVar(*x);
  auto deltaInput1 = deltaNode->AddContextVar(*x).input;
  deltaNode->AddContextVar(*x);

  auto result = jlm::rvsdg::CreateOpNode<TestOperation>(
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
  assert(deltaNode->GetContextVars().size() == 1);
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
