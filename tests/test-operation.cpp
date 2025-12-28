/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"

#include <jlm/rvsdg/substitution.hpp>

namespace jlm::tests
{

TestStructuralOperation::~TestStructuralOperation() noexcept = default;

std::string
TestStructuralOperation::debug_string() const
{
  return "TestStructuralOperation";
}

std::unique_ptr<rvsdg::Operation>
TestStructuralOperation::copy() const
{
  return std::make_unique<TestStructuralOperation>(*this);
}

TestStructuralNode::~TestStructuralNode() noexcept = default;

[[nodiscard]] const TestStructuralOperation &
TestStructuralNode::GetOperation() const noexcept
{
  static TestStructuralOperation singleton;
  return singleton;
}

TestStructuralNode *
TestStructuralNode::copy(rvsdg::Region * parent, rvsdg::SubstitutionMap & smap) const
{
  auto node = create(parent, nsubregions());

  /* copy inputs */
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = smap.lookup(input(n)->origin());
    auto neworigin = origin ? origin : input(n)->origin();
    auto new_input = new rvsdg::StructuralInput(node, neworigin, input(n)->Type());
    node->addInput(std::unique_ptr<rvsdg::StructuralInput>(new_input), true);
    smap.insert(input(n), new_input);
  }

  /* copy outputs */
  for (size_t n = 0; n < noutputs(); n++)
  {
    auto new_output = new rvsdg::StructuralOutput(node, output(n)->Type());
    node->addOutput(std::unique_ptr<rvsdg::StructuralOutput>(new_output));
    smap.insert(output(n), new_output);
  }

  /* copy regions */
  for (size_t n = 0; n < nsubregions(); n++)
    subregion(n)->copy(node->subregion(n), smap, true, true);

  return node;
}

rvsdg::StructuralInput &
TestStructuralNode::addInputOnly(rvsdg::Output & origin)
{
  const auto input =
      addInput(std::make_unique<rvsdg::StructuralInput>(this, &origin, origin.Type()), true);
  return *input;
}

TestStructuralNode::InputVar
TestStructuralNode::addInputWithArguments(rvsdg::Output & origin)
{
  InputVar inputVar{ &addInputOnly(origin), {} };

  for (auto & subregion : Subregions())
  {
    const auto argument = &rvsdg::RegionArgument::Create(
        subregion,
        util::assertedCast<rvsdg::StructuralInput>(inputVar.input),
        inputVar.input->Type());
    inputVar.argument.push_back(argument);
  }

  return inputVar;
}

void
TestStructuralNode::removeInputAndArguments(size_t index)
{
  if (index >= ninputs())
    throw std::out_of_range("Invalid input index.");

  auto in = input(index);
  for (auto & argument : in->arguments)
  {
    argument.region()->RemoveArgument(argument.index());
  }

  removeInput(index, true);
}

TestStructuralNode::InputVar
TestStructuralNode::addArguments(const std::shared_ptr<const rvsdg::Type> & type)
{
  std::vector<rvsdg::RegionArgument *> arguments;
  for (auto & subregion : Subregions())
  {
    const auto argument = &rvsdg::RegionArgument::Create(subregion, nullptr, type);
    arguments.push_back(argument);
  }

  return { nullptr, std::move(arguments) };
}

rvsdg::StructuralOutput &
TestStructuralNode::addOutputOnly(std::shared_ptr<const rvsdg::Type> type)
{
  return *addOutput(std::make_unique<rvsdg::StructuralOutput>(this, std::move(type)));
}

TestStructuralNode::OutputVar
TestStructuralNode::addOutputWithResults(const std::vector<rvsdg::Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  OutputVar outputVar{ &addOutputOnly(origins[0]->Type()), {} };
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &rvsdg::RegionResult::Create(
        subregion,
        *origin,
        util::assertedCast<rvsdg::StructuralOutput>(outputVar.output),
        origin->Type());
    outputVar.result.push_back(result);
  }

  return outputVar;
}

void
TestStructuralNode::removeOutputAndResults(size_t index)
{
  if (index >= noutputs())
    throw std::out_of_range("Invalid output index.");

  auto out = output(index);
  for (auto & result : out->results)
  {
    result.region()->RemoveResult(result.index());
  }

  removeOutput(index);
}

TestStructuralNode::OutputVar
TestStructuralNode::addResults(const std::vector<rvsdg::Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  std::vector<rvsdg::RegionResult *> results;
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &rvsdg::RegionResult::Create(subregion, *origin, nullptr, origin->Type());
    results.push_back(result);
  }

  return { nullptr, std::move(results) };
}

}
