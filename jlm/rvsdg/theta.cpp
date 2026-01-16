/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <algorithm>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <algorithm>

namespace jlm::rvsdg
{

ThetaOperation::~ThetaOperation() noexcept = default;

std::string
ThetaOperation::debug_string() const
{
  return "THETA";
}

std::unique_ptr<Operation>
ThetaOperation::copy() const
{
  return std::make_unique<ThetaOperation>(*this);
}

ThetaNode::~ThetaNode() noexcept = default;

[[nodiscard]] const ThetaOperation &
ThetaNode::GetOperation() const noexcept
{
  // Theta presently has no parametrization, so we can indeed
  // just return a singleton here.
  static const ThetaOperation singleton;
  return singleton;
}

ThetaNode::ThetaNode(rvsdg::Region & parent)
    : StructuralNode(&parent, 1)
{
  auto predicate = &ControlConstantOperation::createFalse(*subregion());
  RegionResult::Create(*subregion(), *predicate, nullptr, ControlType::Create(2));
}

ThetaNode::LoopVar
ThetaNode::AddLoopVar(rvsdg::Output * origin)
{
  Node::addInput(std::make_unique<StructuralInput>(this, origin, origin->Type()), true);
  Node::addOutput(std::make_unique<StructuralOutput>(this, origin->Type()));

  auto input = ThetaNode::input(ninputs() - 1);
  auto output = ThetaNode::output(noutputs() - 1);
  auto & thetaArgument = RegionArgument::Create(*subregion(), input, input->Type());
  auto & thetaResult = RegionResult::Create(*subregion(), thetaArgument, output, output->Type());

  return LoopVar{ input, &thetaArgument, &thetaResult, output };
}

void
ThetaNode::RemoveLoopVars(std::vector<LoopVar> loopVars)
{
  util::HashSet<size_t> inputIndices;
  util::HashSet<size_t> argumentIndices;
  util::HashSet<size_t> resultIndices;
  util::HashSet<size_t> outputIndices;
  for (const auto & [input, pre, post, output] : loopVars)
  {
    JLM_ASSERT(output->IsDead());

    // If the pre argument has a user, it can only be the corresponding post result
    JLM_ASSERT(pre->nusers() <= 1);
    if (pre->nusers() == 1)
    {
      JLM_ASSERT(post->origin() == pre);
    }

    inputIndices.insert(input->index());
    argumentIndices.insert(pre->index());
    resultIndices.insert(post->index());
    outputIndices.insert(output->index());
  }

  [[maybe_unused]] const auto numRemovedResults = subregion()->RemoveResults(resultIndices);
  JLM_ASSERT(numRemovedResults == resultIndices.Size());

  [[maybe_unused]] const auto numRemovedArguments = subregion()->RemoveArguments(argumentIndices);
  JLM_ASSERT(numRemovedArguments == argumentIndices.Size());

  [[maybe_unused]] const auto numRemovedOutputs = RemoveOutputs(outputIndices);
  JLM_ASSERT(numRemovedOutputs == outputIndices.Size());

  [[maybe_unused]] const auto numRemovedInputs = RemoveInputs(inputIndices);
  JLM_ASSERT(numRemovedInputs == inputIndices.Size());
}

ThetaNode *
ThetaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  SubstitutionMap rmap;
  auto theta = create(region);

  /* add loop variables */
  std::vector<LoopVar> oldLoopVars = GetLoopVars();
  std::vector<LoopVar> newLoopVars;
  for (auto olv : oldLoopVars)
  {
    auto nlv = theta->AddLoopVar(smap.lookup(olv.input->origin()));
    newLoopVars.push_back(nlv);
    rmap.insert(olv.pre, nlv.pre);
  }

  /* copy subregion */
  subregion()->copy(theta->subregion(), rmap);
  theta->set_predicate(rmap.lookup(predicate()->origin()));

  /* redirect loop variables */
  for (size_t i = 0; i < oldLoopVars.size(); ++i)
  {
    newLoopVars[i].post->divert_to(rmap.lookup(oldLoopVars[i].post->origin()));
    smap.insert(oldLoopVars[i].output, newLoopVars[i].output);
  }

  return theta;
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapInputLoopVar(const rvsdg::Input & input) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<ThetaNode>(input) == this);
  return LoopVar{ const_cast<rvsdg::Input *>(&input),
                  subregion()->argument(input.index()),
                  subregion()->result(input.index() + 1),
                  output(input.index()) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPreLoopVar(const rvsdg::Output & argument) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<ThetaNode>(argument) == this);
  return LoopVar{ input(argument.index()),
                  const_cast<rvsdg::Output *>(&argument),
                  subregion()->result(argument.index() + 1),
                  output(argument.index()) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPostLoopVar(const rvsdg::Input & result) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<ThetaNode>(result) == this);
  if (result.index() == 0)
  {
    // This is the loop continuation predicate.
    // There is nothing sensible to return here.
    throw std::logic_error("cannot map loop continuation predicate to loop variable");
  }
  return LoopVar{ input(result.index() - 1),
                  subregion()->argument(result.index() - 1),
                  const_cast<rvsdg::Input *>(&result),
                  output(result.index() - 1) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapOutputLoopVar(const rvsdg::Output & output) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<ThetaNode>(output) == this);
  return LoopVar{ input(output.index()),
                  subregion()->argument(output.index()),
                  subregion()->result(output.index() + 1),
                  const_cast<rvsdg::Output *>(&output) };
}

[[nodiscard]] std::vector<ThetaNode::LoopVar>
ThetaNode::GetLoopVars() const
{
  std::vector<LoopVar> loopvars;
  for (size_t index = 0; index < ninputs(); ++index)
  {
    loopvars.push_back(LoopVar{ input(index),
                                subregion()->argument(index),
                                subregion()->result(index + 1),
                                output(index) });
  }
  return loopvars;
}

}
