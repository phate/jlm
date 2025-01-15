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

void
ThetaNode::RemoveLoopVars(std::vector<LoopVar> loopvars)
{
  // Sort them by descending index to avoid renumbering issues.
  std::sort(
      loopvars.begin(),
      loopvars.end(),
      [](const LoopVar & x, const LoopVar & y)
      {
        return x.input->index() > y.input->index();
      });

  for (const auto & loopvar : loopvars)
  {
    JLM_ASSERT(loopvar.output->IsDead());
    JLM_ASSERT(loopvar.post->origin() == loopvar.pre);
    JLM_ASSERT(loopvar.pre->nusers() == 1);

    subregion()->RemoveResult(loopvar.post->index());
    subregion()->RemoveArgument(loopvar.pre->index());
    RemoveOutput(loopvar.output->index());
    RemoveInput(loopvar.input->index());
  }
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
  return LoopVar{ const_cast<rvsdg::input *>(&input),
                  subregion()->argument(input.index()),
                  subregion()->result(input.index() + 1),
                  output(input.index()) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPreLoopVar(const rvsdg::output & argument) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<ThetaNode>(argument) == this);
  return LoopVar{ input(argument.index()),
                  const_cast<rvsdg::output *>(&argument),
                  subregion()->result(argument.index() + 1),
                  output(argument.index()) };
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
  return LoopVar{ input(result.index() - 1),
                  subregion()->argument(result.index() - 1),
                  const_cast<rvsdg::input *>(&result),
                  output(result.index() - 1) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapOutputLoopVar(const rvsdg::output & output) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<ThetaNode>(output) == this);
  return LoopVar{ input(output.index()),
                  subregion()->argument(output.index()),
                  subregion()->result(output.index() + 1),
                  const_cast<rvsdg::output *>(&output) };
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
