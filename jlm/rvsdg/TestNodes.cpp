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

  // copy inputs and arguments
  for (auto & argument : subregion(0)->Arguments())
  {
    if (const auto input = argument->input())
    {
      auto oldInputVar = mapInput(*input);
      auto & newOrigin = smap.lookup(*input->origin());
      auto newInputVar = node->addInputWithArguments(newOrigin);
      for (size_t n = 0; n < oldInputVar.argument.size(); n++)
      {
        auto oldArgument = oldInputVar.argument[n];
        auto newArgument = newInputVar.argument[n];
        smap.insert(oldArgument, newArgument);
      }
    }
    else
    {
      auto newInputVar = node->addArguments(argument->Type());
      for (auto & subregion : Subregions())
      {
        auto oldArgument = subregion.argument(argument->index());
        JLM_ASSERT(oldArgument->input() == nullptr);
        smap.insert(oldArgument, newInputVar.argument[subregion.index()]);
      }
    }
  }

  JLM_ASSERT(ninputs() == node->ninputs());
  for (auto & subregion : Subregions())
  {
    JLM_ASSERT(subregion.narguments() == node->subregion(subregion.index())->narguments());
  }

  // copy subregions
  for (auto & subregion : Subregions())
  {
    subregion.copy(node->subregion(subregion.index()), smap, false, false);
  }

  // copy results and outputs
  for (auto & result : subregion(0)->Results())
  {
    if (const auto output = result->output())
    {
      auto oldOutputVar = mapOutput(*output);

      std::vector<Output *> newOrigins;
      for (auto oldOutputVarResult : oldOutputVar.result)
      {
        auto & newOrigin = smap.lookup(*oldOutputVarResult->origin());
        newOrigins.push_back(&newOrigin);
      }
      auto newOutputVar = node->addOutputWithResults(newOrigins);
      smap.insert(oldOutputVar.output, newOutputVar.output);
    }
    else
    {
      std::vector<Output *> newOrigins;
      for (auto & subregion : Subregions())
      {
        auto subregionResult = subregion.result(result->index());
        JLM_ASSERT(subregionResult->output() == nullptr);
        auto & newOrigin = smap.lookup(*subregionResult->origin());
        newOrigins.push_back(&newOrigin);
      }
      node->addResults(newOrigins);
    }
  }

  JLM_ASSERT(noutputs() == node->noutputs());
  for (auto & subregion : Subregions())
  {
    JLM_ASSERT(subregion.nresults() == node->subregion(subregion.index())->nresults());
  }

  return node;
}

TestStructuralNode::InputVar
TestStructuralNode::mapInput(const Input & input) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<TestStructuralNode>(input) == this);

  InputVar inputVar;
  inputVar.input = this->input(input.index());
  for (auto & subregion : Subregions())
  {
    for (auto & argument : subregion.Arguments())
    {
      if (argument->input() == inputVar.input)
      {
        inputVar.argument.push_back(argument);
      }
    }
  }

  JLM_ASSERT(inputVar.argument.size() == nsubregions());
  return inputVar;
}

TestStructuralNode::OutputVar
TestStructuralNode::mapOutput(const Output & output) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<TestStructuralNode>(output) == this);

  OutputVar outputVar;
  outputVar.output = this->output(output.index());
  for (auto & subregion : Subregions())
  {
    for (auto & result : subregion.Results())
    {
      if (result->output() == outputVar.output)
      {
        outputVar.result.push_back(result);
      }
    }
  }

  JLM_ASSERT(outputVar.result.size() == nsubregions());
  return outputVar;
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

  RemoveInputs({ index });
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

StructuralInput &
TestStructuralNode::addInputOnly(Output & origin)
{
  return *addInput(std::make_unique<StructuralInput>(this, &origin, origin.Type()), true);
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
