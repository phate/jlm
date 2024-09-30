/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::rvsdg
{

/* theta operation */

ThetaOperation::~ThetaOperation() noexcept = default;

std::string
ThetaOperation::debug_string() const
{
  return "THETA";
}

std::unique_ptr<jlm::rvsdg::operation>
ThetaOperation::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new ThetaOperation(*this));
}

ThetaNode::ThetaNode(rvsdg::Region & parent)
    : structural_node(ThetaOperation(), &parent, 1)
{
  auto predicate = control_false(subregion());
  ThetaPredicateResult::Create(*predicate);
}

ThetaInput::~ThetaInput() noexcept
{
  if (output_)
    output_->input_ = nullptr;
}

/* theta output */

ThetaOutput::~ThetaOutput() noexcept
{
  if (input_)
    input_->output_ = nullptr;
}

ThetaArgument::~ThetaArgument() noexcept = default;

ThetaArgument &
ThetaArgument::Copy(rvsdg::Region & region, structural_input * input)
{
  auto thetaInput = util::AssertedCast<ThetaInput>(input);
  return ThetaArgument::Create(region, *thetaInput);
}

ThetaResult::~ThetaResult() noexcept = default;

ThetaResult &
ThetaResult::Copy(rvsdg::output & origin, structural_output * output)
{
  auto thetaOutput = util::AssertedCast<ThetaOutput>(output);
  return ThetaResult::Create(origin, *thetaOutput);
}

ThetaPredicateResult::~ThetaPredicateResult() noexcept = default;

ThetaPredicateResult &
ThetaPredicateResult::Copy(rvsdg::output & origin, structural_output * output)
{
  JLM_ASSERT(output == nullptr);
  return ThetaPredicateResult::Create(origin);
}

/* theta node */

ThetaNode::~ThetaNode() noexcept = default;

const ThetaNode::loopvar_iterator &
ThetaNode::loopvar_iterator::operator++() noexcept
{
  if (output_ == nullptr)
    return *this;

  auto node = output_->node();
  auto index = output_->index();
  if (index == node->noutputs() - 1)
  {
    output_ = nullptr;
    return *this;
  }

  index++;
  output_ = node->output(index);
  return *this;
}

ThetaOutput *
ThetaNode::add_loopvar(jlm::rvsdg::output * origin)
{
  return jlm::util::AssertedCast<ThetaOutput>(AddLoopVar(origin).exit);
}

ThetaNode::LoopVar
ThetaNode::AddLoopVar(rvsdg::output * origin)
{
  node::add_input(std::make_unique<ThetaInput>(this, origin, origin->Type()));
  node::add_output(std::make_unique<ThetaOutput>(this, origin->Type()));

  auto input = ThetaNode::input(ninputs() - 1);
  auto output = ThetaNode::output(noutputs() - 1);
  jlm::util::AssertedCast<ThetaInput>(input)->output_ = output;
  output->input_ = jlm::util::AssertedCast<ThetaInput>(input);

  auto & thetaArgument =
      ThetaArgument::Create(*subregion(), *jlm::util::AssertedCast<ThetaInput>(input));
  auto & thetaResult = ThetaResult::Create(thetaArgument, *output);
  return LoopVar{ input, &thetaArgument, &thetaResult, output };
}

ThetaNode *
ThetaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  auto nf = graph()->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  rvsdg::SubstitutionMap rmap;
  auto theta = create(region);

  /* add loop variables */
  for (auto olv : *this)
  {
    auto nlv = theta->add_loopvar(smap.lookup(olv->input()->origin()));
    rmap.insert(olv->argument(), nlv->argument());
  }

  /* copy subregion */
  subregion()->copy(theta->subregion(), rmap, false, false);
  theta->set_predicate(rmap.lookup(predicate()->origin()));

  /* redirect loop variables */
  for (auto olv = begin(), nlv = theta->begin(); olv != end(); olv++, nlv++)
  {
    (*nlv)->result()->divert_to(rmap.lookup((*olv)->result()->origin()));
    smap.insert(olv.output(), nlv.output());
  }

  nf->set_mutable(true);
  return theta;
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapEntryLoopVar(const rvsdg::input & input) const
{
  JLM_ASSERT(rvsdg::input::GetNode(input) == this);
  auto peer = MapInputToOutputIndex(input.index());
  return LoopVar{ const_cast<rvsdg::input *>(&input),
                  subregion()->argument(input.index()),
                  peer ? subregion()->result(*peer + 1) : nullptr,
                  peer ? output(*peer) : nullptr };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPreLoopVar(const RegionArgument & argument) const
{
  JLM_ASSERT(argument.region() == subregion());
  auto peer = MapInputToOutputIndex(argument.index());
  return LoopVar{ input(argument.index()),
                  const_cast<RegionArgument *>(&argument),
                  peer ? subregion()->result(*peer + 1) : nullptr,
                  peer ? output(*peer) : nullptr };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapPostLoopVar(const RegionResult & result) const
{
  JLM_ASSERT(result.region() == subregion());
  if (result.index() == 0)
  {
    // This is the loop continuation predicate.
    // There is nothing sensible to return here.
    throw std::logic_error("cannot map loop continuation predicate to loop variable");
  }
  auto peer = MapOutputToInputIndex(result.index() - 1);
  return LoopVar{ peer ? input(*peer) : nullptr,
                  peer ? subregion()->argument(*peer) : nullptr,
                  const_cast<RegionResult *>(&result),
                  output(result.index() - 1) };
}

[[nodiscard]] ThetaNode::LoopVar
ThetaNode::MapExitLoopVar(const rvsdg::output & output) const
{
  JLM_ASSERT(rvsdg::output::GetNode(output) == this);
  auto peer = MapOutputToInputIndex(output.index());
  return LoopVar{ peer ? input(*peer) : nullptr,
                  peer ? subregion()->argument(*peer) : nullptr,
                  subregion()->result(output.index() + 1),
                  const_cast<rvsdg::output *>(&output) };
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
