/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* gamma normal form */

static bool
is_predicate_reducible(const GammaNode * gamma)
{
  auto constant = output::GetNode(*gamma->predicate()->origin());
  return constant && is_ctlconstant_op(constant->GetOperation());
}

static void
perform_predicate_reduction(GammaNode * gamma)
{
  auto origin = gamma->predicate()->origin();
  auto constant = static_cast<node_output *>(origin)->node();
  auto cop = static_cast<const ctlconstant_op *>(&constant->GetOperation());
  auto alternative = cop->value().alternative();

  rvsdg::SubstitutionMap smap;
  for (auto it = gamma->begin_entryvar(); it != gamma->end_entryvar(); it++)
    smap.insert(it->argument(alternative), it->origin());

  gamma->subregion(alternative)->copy(gamma->region(), smap, false, false);

  for (auto it = gamma->begin_exitvar(); it != gamma->end_exitvar(); it++)
    it->divert_users(smap.lookup(it->result(alternative)->origin()));

  remove(gamma);
}

static bool
perform_invariant_reduction(GammaNode * gamma)
{
  bool was_normalized = true;
  for (auto it = gamma->begin_exitvar(); it != gamma->end_exitvar(); it++)
  {
    auto argument = dynamic_cast<const rvsdg::RegionArgument *>(it->result(0)->origin());
    if (!argument)
      continue;

    size_t n;
    auto input = argument->input();
    for (n = 1; n < it->nresults(); n++)
    {
      auto argument = dynamic_cast<const rvsdg::RegionArgument *>(it->result(n)->origin());
      if (!argument && argument->input() != input)
        break;
    }

    if (n == it->nresults())
    {
      it->divert_users(argument->input()->origin());
      was_normalized = false;
    }
  }

  return was_normalized;
}

static std::unordered_set<StructuralOutput *>
is_control_constant_reducible(GammaNode * gamma)
{
  /* check gamma predicate */
  auto match = output::GetNode(*gamma->predicate()->origin());
  if (!is<match_op>(match))
    return {};

  /* check number of alternatives */
  auto match_op = static_cast<const jlm::rvsdg::match_op *>(&match->GetOperation());
  std::unordered_set<uint64_t> set({ match_op->default_alternative() });
  for (const auto & pair : *match_op)
    set.insert(pair.second);

  if (set.size() != gamma->nsubregions())
    return {};

  /* check for constants */
  std::unordered_set<StructuralOutput *> outputs;
  for (auto it = gamma->begin_exitvar(); it != gamma->end_exitvar(); it++)
  {
    if (!is_ctltype(it->type()))
      continue;

    size_t n;
    for (n = 0; n < it->nresults(); n++)
    {
      auto node = output::GetNode(*it->result(n)->origin());
      if (!is<ctlconstant_op>(node))
        break;

      auto op = static_cast<const jlm::rvsdg::ctlconstant_op *>(&node->GetOperation());
      if (op->value().nalternatives() != 2)
        break;
    }
    if (n == it->nresults())
      outputs.insert(it.output());
  }

  return outputs;
}

static void
perform_control_constant_reduction(std::unordered_set<StructuralOutput *> & outputs)
{
  auto gamma = static_cast<GammaNode *>((*outputs.begin())->node());
  auto origin = static_cast<node_output *>(gamma->predicate()->origin());
  auto match = origin->node();
  auto & match_op = to_match_op(match->GetOperation());

  std::unordered_map<uint64_t, uint64_t> map;
  for (const auto & pair : match_op)
    map[pair.second] = pair.first;

  for (auto xv = gamma->begin_exitvar(); xv != gamma->end_exitvar(); xv++)
  {
    if (outputs.find(xv.output()) == outputs.end())
      continue;

    size_t defalt = 0;
    size_t nalternatives = 0;
    std::unordered_map<uint64_t, uint64_t> new_mapping;
    for (size_t n = 0; n < xv->nresults(); n++)
    {
      auto origin = static_cast<node_output *>(xv->result(n)->origin());
      auto & value = to_ctlconstant_op(origin->node()->GetOperation()).value();
      nalternatives = value.nalternatives();
      if (map.find(n) != map.end())
        new_mapping[map[n]] = value.alternative();
      else
        defalt = value.alternative();
    }

    auto origin = match->input(0)->origin();
    auto m = jlm::rvsdg::match(match_op.nbits(), new_mapping, defalt, nalternatives, origin);
    xv->divert_users(m);
  }
}

gamma_normal_form::~gamma_normal_form() noexcept
{}

gamma_normal_form::gamma_normal_form(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    Graph * graph) noexcept
    : structural_normal_form(operator_class, parent, graph),
      enable_predicate_reduction_(false),
      enable_invariant_reduction_(false),
      enable_control_constant_reduction_(false)
{
  if (auto p = dynamic_cast<gamma_normal_form *>(parent))
  {
    enable_predicate_reduction_ = p->enable_predicate_reduction_;
    enable_invariant_reduction_ = p->enable_invariant_reduction_;
    enable_control_constant_reduction_ = p->enable_control_constant_reduction_;
  }
}

bool
gamma_normal_form::normalize_node(Node * node_) const
{
  auto node = util::AssertedCast<GammaNode>(node_);

  if (!get_mutable())
    return true;

  if (get_predicate_reduction() && is_predicate_reducible(node))
  {
    perform_predicate_reduction(node);
    return false;
  }

  bool was_normalized = true;
  if (get_invariant_reduction())
    was_normalized |= perform_invariant_reduction(node);

  auto outputs = is_control_constant_reducible(node);
  if (get_control_constant_reduction() && !outputs.empty())
  {
    perform_control_constant_reduction(outputs);
    was_normalized = false;
  }

  return was_normalized;
}

void
gamma_normal_form::set_predicate_reduction(bool enable)
{
  if (enable_predicate_reduction_ == enable)
  {
    return;
  }

  children_set<gamma_normal_form, &gamma_normal_form::set_predicate_reduction>(enable);

  enable_predicate_reduction_ = enable;

  if (enable && get_mutable())
    graph()->mark_denormalized();
}

void
gamma_normal_form::set_invariant_reduction(bool enable)
{
  if (enable_invariant_reduction_ == enable)
  {
    return;
  }

  children_set<gamma_normal_form, &gamma_normal_form::set_invariant_reduction>(enable);

  enable_invariant_reduction_ = enable;

  if (enable && get_mutable())
    graph()->mark_denormalized();
}

void
gamma_normal_form::set_control_constant_reduction(bool enable)
{
  if (enable_control_constant_reduction_ == enable)
    return;

  children_set<gamma_normal_form, &gamma_normal_form::set_control_constant_reduction>(enable);

  enable_control_constant_reduction_ = enable;
  if (enable && get_mutable())
    graph()->mark_denormalized();
}

/* gamma operation */

GammaOperation::~GammaOperation() noexcept
{}

std::string
GammaOperation::debug_string() const
{
  return "GAMMA";
}

std::unique_ptr<jlm::rvsdg::operation>
GammaOperation::copy() const
{
  return std::unique_ptr<operation>(new GammaOperation(*this));
}

bool
GammaOperation::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const GammaOperation *>(&other);
  return op && op->nalternatives_ == nalternatives_;
}

/* gamma input */

GammaInput::~GammaInput() noexcept = default;

/* gamma output */

GammaOutput::~GammaOutput() noexcept = default;

bool
GammaOutput::IsInvariant(rvsdg::output ** invariantOrigin) const noexcept
{
  auto argument = dynamic_cast<const rvsdg::RegionArgument *>(result(0)->origin());
  if (!argument)
  {
    return false;
  }

  size_t n;
  auto origin = argument->input()->origin();
  for (n = 1; n < nresults(); n++)
  {
    argument = dynamic_cast<const rvsdg::RegionArgument *>(result(n)->origin());
    if (argument == nullptr || argument->input()->origin() != origin)
      break;
  }

  auto isInvariant = (n == nresults());
  if (isInvariant && invariantOrigin != nullptr)
  {
    *invariantOrigin = origin;
  }

  return isInvariant;
}

/* gamma node */

GammaNode::~GammaNode() noexcept = default;

const GammaNode::entryvar_iterator &
GammaNode::entryvar_iterator::operator++() noexcept
{
  if (input_ == nullptr)
    return *this;

  auto node = input_->node();
  auto index = input_->index();
  if (index == node->ninputs() - 1)
  {
    input_ = nullptr;
    return *this;
  }

  input_ = static_cast<GammaInput *>(node->input(++index));
  return *this;
}

const GammaNode::exitvar_iterator &
GammaNode::exitvar_iterator::operator++() noexcept
{
  if (output_ == nullptr)
    return *this;

  auto node = output_->node();
  auto index = output_->index();
  if (index == node->nexitvars() - 1)
  {
    output_ = nullptr;
    return *this;
  }

  output_ = node->exitvar(++index);
  return *this;
}

GammaNode *
GammaNode::copy(rvsdg::Region * region, SubstitutionMap & smap) const
{
  auto gamma = create(smap.lookup(predicate()->origin()), nsubregions());

  /* add entry variables to new gamma */
  std::vector<SubstitutionMap> rmap(nsubregions());
  for (auto oev = begin_entryvar(); oev != end_entryvar(); oev++)
  {
    auto nev = gamma->add_entryvar(smap.lookup(oev->origin()));
    for (size_t n = 0; n < nev->narguments(); n++)
      rmap[n].insert(oev->argument(n), nev->argument(n));
  }

  /* copy subregions */
  for (size_t r = 0; r < nsubregions(); r++)
    subregion(r)->copy(gamma->subregion(r), rmap[r], false, false);

  /* add exit variables to new gamma */
  for (auto oex = begin_exitvar(); oex != end_exitvar(); oex++)
  {
    std::vector<jlm::rvsdg::output *> operands;
    for (size_t n = 0; n < oex->nresults(); n++)
      operands.push_back(rmap[n].lookup(oex->result(n)->origin()));
    auto nex = gamma->add_exitvar(operands);
    smap.insert(oex.output(), nex);
  }

  return gamma;
}

GammaArgument::~GammaArgument() noexcept = default;

GammaArgument &
GammaArgument::Copy(rvsdg::Region & region, StructuralInput * input)
{
  auto gammaInput = util::AssertedCast<GammaInput>(input);
  return Create(region, *gammaInput);
}

GammaResult::~GammaResult() noexcept = default;

GammaResult &
GammaResult::Copy(rvsdg::output & origin, StructuralOutput * output)
{
  auto gammaOutput = util::AssertedCast<GammaOutput>(output);
  return GammaResult::Create(*origin.region(), origin, *gammaOutput);
}

}

jlm::rvsdg::node_normal_form *
gamma_node_get_default_normal_form_(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::Graph * graph)
{
  return new jlm::rvsdg::gamma_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::GammaOperation),
      gamma_node_get_default_normal_form_);
}
