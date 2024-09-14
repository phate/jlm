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

theta_node::theta_node(rvsdg::region & parent)
    : structural_node(ThetaOperation(), &parent, 1)
{
  auto predicate = control_false(subregion());
  ThetaPredicateResult::Create(*predicate);
}

/* theta input */

theta_input::~theta_input() noexcept
{
  if (output_)
    output_->input_ = nullptr;
}

/* theta output */

theta_output::~theta_output() noexcept
{
  if (input_)
    input_->output_ = nullptr;
}

ThetaArgument::~ThetaArgument() noexcept = default;

ThetaArgument &
ThetaArgument::Copy(rvsdg::region & region, structural_input * input)
{
  auto thetaInput = util::AssertedCast<theta_input>(input);
  return ThetaArgument::Create(region, *thetaInput);
}

ThetaResult::~ThetaResult() noexcept = default;

ThetaResult &
ThetaResult::Copy(rvsdg::output & origin, structural_output * output)
{
  auto thetaOutput = util::AssertedCast<theta_output>(output);
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

theta_node::~theta_node()
{}

const theta_node::loopvar_iterator &
theta_node::loopvar_iterator::operator++() noexcept
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

jlm::rvsdg::theta_output *
theta_node::add_loopvar(jlm::rvsdg::output * origin)
{
  node::add_input(std::make_unique<theta_input>(this, origin, origin->Type()));
  node::add_output(std::make_unique<theta_output>(this, origin->Type()));

  auto input = theta_node::input(ninputs() - 1);
  auto output = theta_node::output(noutputs() - 1);
  input->output_ = output;
  output->input_ = input;

  auto & thetaArgument = ThetaArgument::Create(*subregion(), *input);
  ThetaResult::Create(thetaArgument, *output);
  return output;
}

jlm::rvsdg::theta_node *
theta_node::copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const
{
  auto nf = graph()->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  jlm::rvsdg::substitution_map rmap;
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

}
