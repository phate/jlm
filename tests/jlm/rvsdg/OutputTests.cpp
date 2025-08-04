/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

static void
TestOutputIterator()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto i0 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i1 = &GraphImport::Create(rvsdg, valueType, "i");
  auto i2 = &GraphImport::Create(rvsdg, valueType, "i");

  auto & node = CreateOpNode<jlm::tests::TestOperation>(
      rootRegion,
      std::vector<std::shared_ptr<const Type>>(),
      std::vector<std::shared_ptr<const Type>>(5, valueType));

  GraphExport::Create(*node.output(0), "x0");

  // Act & Assert
  auto nodeIt = Output::Iterator(node.output(0));
  assert(nodeIt.GetOutput() == node.output(0));
  assert(nodeIt->index() == node.output(0)->index());
  assert((*nodeIt).index() == node.output(0)->index());
  assert(nodeIt == Output::Iterator(node.output(0)));
  assert(nodeIt != Output::Iterator(node.output(1)));

  nodeIt++;
  assert(nodeIt.GetOutput() == node.output(1));

  ++nodeIt;
  assert(nodeIt.GetOutput() == node.output(2));

  ++nodeIt;
  ++nodeIt;
  assert(nodeIt.GetOutput() == node.output(4));

  ++nodeIt;
  assert(nodeIt.GetOutput() == nullptr);

  auto regionIt = Output::Iterator(rootRegion.argument(0));
  assert(regionIt.GetOutput() == i0);
  assert(regionIt->index() == i0->index());
  assert((*regionIt).index() == i0->index());
  assert(regionIt == Output::Iterator(i0));
  assert(regionIt != Output::Iterator(i1));

  regionIt++;
  regionIt++;
  assert(regionIt.GetOutput() == i2);

  regionIt++;
  assert(regionIt.GetOutput() == nullptr);

  auto it = Input::Iterator(nullptr);
  it++;
  ++it;
  assert(it.GetInput() == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/OutputTests-TestOutputIterator", TestOutputIterator)
