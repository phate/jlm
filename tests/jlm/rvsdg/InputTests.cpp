/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

static void
TestInputIterator()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i = &jlm::tests::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  jlm::tests::GraphExport::Create(*node.output(0), "x0");
  jlm::tests::GraphExport::Create(*node.output(0), "x1");
  jlm::tests::GraphExport::Create(*node.output(0), "x2");

  // Act & Assert
  auto nodeIt = Input::Iterator(node.input(0));
  assert(nodeIt.GetInput() == node.input(0));
  assert(nodeIt->index() == node.input(0)->index());
  assert((*nodeIt).index() == node.input(0)->index());
  assert(nodeIt == Input::Iterator(node.input(0)));
  assert(nodeIt != Input::Iterator(node.input(1)));

  nodeIt++;
  assert(nodeIt.GetInput() == node.input(1));

  ++nodeIt;
  assert(nodeIt.GetInput() == node.input(2));

  ++nodeIt;
  ++nodeIt;
  assert(nodeIt.GetInput() == node.input(4));

  ++nodeIt;
  assert(nodeIt.GetInput() == nullptr);

  auto regionIt = Input::Iterator(rootRegion.result(0));
  assert(regionIt.GetInput() == rootRegion.result(0));
  assert(regionIt->index() == rootRegion.result(0)->index());
  assert((*regionIt).index() == rootRegion.result(0)->index());
  assert(regionIt == Input::Iterator(rootRegion.result(0)));
  assert(regionIt != Input::Iterator(rootRegion.result(1)));

  regionIt++;
  regionIt++;
  assert(regionIt.GetInput() == rootRegion.result(2));

  regionIt++;
  assert(regionIt.GetInput() == nullptr);

  auto it = Input::Iterator(nullptr);
  it++;
  ++it;
  assert(it.GetInput() == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/InputTests-TestInputIterator", TestInputIterator)

static void
TestInputConstIterator()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i = &jlm::tests::GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      { i, i, i, i, i },
      std::vector<std::shared_ptr<const Type>>(5, valueType),
      std::vector<std::shared_ptr<const Type>>{ valueType });

  jlm::tests::GraphExport::Create(*node.output(0), "x0");
  jlm::tests::GraphExport::Create(*node.output(0), "x1");
  jlm::tests::GraphExport::Create(*node.output(0), "x2");

  // Act & Assert
  auto nodeIt = Input::ConstIterator(node.input(0));
  assert(nodeIt.GetInput() == node.input(0));
  assert(nodeIt->index() == node.input(0)->index());
  assert((*nodeIt).index() == node.input(0)->index());
  assert(nodeIt == Input::ConstIterator(node.input(0)));
  assert(nodeIt != Input::ConstIterator(node.input(1)));

  nodeIt++;
  assert(nodeIt.GetInput() == node.input(1));

  ++nodeIt;
  assert(nodeIt.GetInput() == node.input(2));

  ++nodeIt;
  ++nodeIt;
  assert(nodeIt.GetInput() == node.input(4));

  ++nodeIt;
  assert(nodeIt.GetInput() == nullptr);

  auto regionIt = Input::ConstIterator(rootRegion.result(0));
  assert(regionIt.GetInput() == rootRegion.result(0));
  assert(regionIt->index() == rootRegion.result(0)->index());
  assert((*regionIt).index() == rootRegion.result(0)->index());
  assert(regionIt == Input::ConstIterator(rootRegion.result(0)));
  assert(regionIt != Input::ConstIterator(rootRegion.result(1)));

  regionIt++;
  regionIt++;
  assert(regionIt.GetInput() == rootRegion.result(2));

  regionIt++;
  assert(regionIt.GetInput() == nullptr);

  auto it = Input::ConstIterator(nullptr);
  it++;
  ++it;
  assert(it.GetInput() == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/InputTests-TestInputConstIterator", TestInputConstIterator)
