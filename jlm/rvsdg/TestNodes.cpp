/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/TestNodes.hpp>

namespace jlm::rvsdg
{

TestStructuralOperation::~TestStructuralOperation() noexcept = default;

std::string
TestStructuralOperation::debug_string() const
{
  return "TestStructuralOperation";
}

std::unique_ptr<Operation>
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
TestStructuralNode::copy(Region * parent, SubstitutionMap & smap) const
{
  auto node = create(parent, nsubregions());

  // copy inputs
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = smap.lookup(input(n)->origin());
    auto neworigin = origin ? origin : input(n)->origin();
    auto new_input = new StructuralInput(node, neworigin, input(n)->Type());
    node->addInput(std::unique_ptr<StructuralInput>(new_input), true);
    smap.insert(input(n), new_input);
  }

  // copy outputs
  for (size_t n = 0; n < noutputs(); n++)
  {
    auto new_output = new StructuralOutput(node, output(n)->Type());
    node->addOutput(std::unique_ptr<StructuralOutput>(new_output));
    smap.insert(output(n), new_output);
  }

  // copy regions
  for (size_t n = 0; n < nsubregions(); n++)
    subregion(n)->copy(node->subregion(n), smap, true, true);

  return node;
}

StructuralInput &
TestStructuralNode::addInputOnly(Output & origin)
{
  const auto input =
      addInput(std::make_unique<StructuralInput>(this, &origin, origin.Type()), true);
  return *input;
}

TestStructuralNode::InputVar
TestStructuralNode::addInputWithArguments(Output & origin)
{
  InputVar inputVar{ &addInputOnly(origin), {} };

  for (auto & subregion : Subregions())
  {
    const auto argument = &RegionArgument::Create(
        subregion,
        util::assertedCast<StructuralInput>(inputVar.input),
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
    argument.region()->RemoveArguments({ argument.index() });
  }

  RemoveInputs({ index }, true);
}

TestStructuralNode::InputVar
TestStructuralNode::addArguments(const std::shared_ptr<const Type> & type)
{
  std::vector<RegionArgument *> arguments;
  for (auto & subregion : Subregions())
  {
    const auto argument = &RegionArgument::Create(subregion, nullptr, type);
    arguments.push_back(argument);
  }

  return { nullptr, std::move(arguments) };
}

StructuralOutput &
TestStructuralNode::addOutputOnly(std::shared_ptr<const Type> type)
{
  return *addOutput(std::make_unique<StructuralOutput>(this, std::move(type)));
}

TestStructuralNode::OutputVar
TestStructuralNode::addOutputWithResults(const std::vector<Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  OutputVar outputVar{ &addOutputOnly(origins[0]->Type()), {} };
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &RegionResult::Create(
        subregion,
        *origin,
        util::assertedCast<StructuralOutput>(outputVar.output),
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
    result.region()->RemoveResults({ result.index() });
  }

  RemoveOutputs({ index });
}

TestStructuralNode::OutputVar
TestStructuralNode::addResults(const std::vector<Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  std::vector<RegionResult *> results;
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &RegionResult::Create(subregion, *origin, nullptr, origin->Type());
    results.push_back(result);
  }

  return { nullptr, std::move(results) };
}

}
