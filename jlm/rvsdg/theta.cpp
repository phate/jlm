/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

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

ThetaNode::ThetaNode(rvsdg::Region & parent)
    : StructuralNode(ThetaOperation(), &parent, 1)
{
  auto predicate = control_false(subregion());
  RegionResult::Create(*subregion(), *predicate, nullptr, ControlType::Create(2));
}

ThetaNode::LoopVar
ThetaNode::AddLoopVar(rvsdg::output * origin)
{
  Node::add_input(std::make_unique<StructuralInput>(this, origin, origin->Type()));
  Node::add_output(std::make_unique<StructuralOutput>(this, origin->Type()));

  auto input = ThetaNode::input(ninputs() - 1);
  auto output = ThetaNode::output(noutputs() - 1);
  auto & thetaArgument = RegionArgument::Create(*subregion(), input, input->Type());
  auto & thetaResult = RegionResult::Create(*subregion(), thetaArgument, output, output->Type());

  return LoopVar{ input, &thetaArgument, &thetaResult, output };
}

ThetaNode *
ThetaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  auto nf = graph()->GetNodeNormalForm(typeid(Operation));
  nf->set_mutable(false);

  rvsdg::SubstitutionMap rmap;
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
  subregion()->copy(theta->subregion(), rmap, false, false);
  theta->set_predicate(rmap.lookup(predicate()->origin()));

  /* redirect loop variables */
  for (size_t i = 0; i < oldLoopVars.size(); ++i)
  {
    newLoopVars[i].post->divert_to(rmap.lookup(oldLoopVars[i].post->origin()));
    smap.insert(oldLoopVars[i].output, newLoopVars[i].output);
  }

  nf->set_mutable(true);
  return theta;
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapInputLoopVar(const rvsdg::input & input) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<ThetaNode>(input) == this);
  auto peer = MapInputToOutputIndex(input.index());
  return LoopVar{ const_cast<rvsdg::input *>(&input),
                  subregion()->argument(input.index()),
                  peer ? subregion()->result(*peer + 1) : nullptr,
                  peer ? output(*peer) : nullptr };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPreLoopVar(const rvsdg::output & argument) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<ThetaNode>(argument) == this);
  auto peer = MapInputToOutputIndex(argument.index());
  return LoopVar{ input(argument.index()),
                  const_cast<rvsdg::output *>(&argument),
                  peer ? subregion()->result(*peer + 1) : nullptr,
                  peer ? output(*peer) : nullptr };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPostLoopVar(const rvsdg::input & result) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<ThetaNode>(result) == this);
  if (result.index() == 0)
  {
    // This is the loop continuation predicate.
    // There is nothing sensible to return here.
    throw std::logic_error("cannot map loop continuation predicate to loop variable");
  }
  auto peer = MapOutputToInputIndex(result.index() - 1);
  return LoopVar{ peer ? input(*peer) : nullptr,
                  peer ? subregion()->argument(*peer) : nullptr,
                  const_cast<rvsdg::input *>(&result),
                  output(result.index() - 1) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapOutputLoopVar(const rvsdg::output & output) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<ThetaNode>(output) == this);
  auto peer = MapOutputToInputIndex(output.index());
  return LoopVar{ peer ? input(*peer) : nullptr,
                  peer ? subregion()->argument(*peer) : nullptr,
                  subregion()->result(output.index() + 1),
                  const_cast<rvsdg::output *>(&output) };
}

[[nodiscard]] std::vector<ThetaNode::LoopVar>
ThetaNode::GetLoopVars() const
{
  std::vector<LoopVar> loopvars;
  for (size_t input_index = 0; input_index < ninputs(); ++input_index)
  {
    // Check if there is a matching input/output -- if we are in
    // the process of deleting a loop variable, inputs and outputs
    // might be unmatched.
    auto output_index = MapInputToOutputIndex(input_index);
    if (output_index)
    {
      loopvars.push_back(LoopVar{ input(input_index),
                                  subregion()->argument(input_index),
                                  subregion()->result(*output_index + 1),
                                  output(*output_index) });
    }
  }
  return loopvars;
}

std::optional<std::size_t>
ThetaNode::MapInputToOutputIndex(std::size_t index) const noexcept
{
  std::size_t offset = 0;
  for (std::size_t unmatched : unmatchedInputs)
  {
    if (unmatched == index)
    {
      return std::nullopt;
    }
    if (unmatched < index)
    {
      ++offset;
    }
  }

  index -= offset;
  offset = 0;
  for (std::size_t unmatched : unmatchedOutputs)
  {
    if (unmatched <= index)
    {
      ++offset;
    }
  }
  return index + offset;
}

std::optional<std::size_t>
ThetaNode::MapOutputToInputIndex(std::size_t index) const noexcept
{
  std::size_t offset = 0;
  for (std::size_t unmatched : unmatchedOutputs)
  {
    if (unmatched == index)
    {
      return std::nullopt;
    }
    if (unmatched < index)
    {
      ++offset;
    }
  }

  index -= offset;
  offset = 0;
  for (std::size_t unmatched : unmatchedInputs)
  {
    if (unmatched <= index)
    {
      ++offset;
    }
  }
  return index + offset;
}

void
ThetaNode::MarkInputIndexErased(std::size_t index) noexcept
{
  if (auto peer = MapInputToOutputIndex(index))
  {
    unmatchedOutputs.push_back(*peer);
  }
  else
  {
    auto i = std::remove(unmatchedInputs.begin(), unmatchedInputs.end(), index);
    unmatchedInputs.erase(i, unmatchedInputs.end());
  }
  for (auto & unmatched : unmatchedInputs)
  {
    if (unmatched > index)
    {
      unmatched -= 1;
    }
  }
}

void
ThetaNode::MarkOutputIndexErased(std::size_t index) noexcept
{
  if (auto peer = MapOutputToInputIndex(index))
  {
    unmatchedInputs.push_back(*peer);
  }
  else
  {
    auto i = std::remove(unmatchedOutputs.begin(), unmatchedOutputs.end(), index);
    unmatchedOutputs.erase(i, unmatchedOutputs.end());
  }
  for (auto & unmatched : unmatchedOutputs)
  {
    if (unmatched > index)
    {
      unmatched -= 1;
    }
  }
}

}
