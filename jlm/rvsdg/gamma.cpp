/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <algorithm>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/UnitType.hpp>

namespace jlm::rvsdg
{

/* gamma normal form */

static bool
is_predicate_reducible(const GammaNode * gamma)
{
  auto constant = rvsdg::TryGetOwnerNode<SimpleNode>(*gamma->predicate()->origin());
  return constant && is_ctlconstant_op(constant->GetOperation());
}

static void
perform_predicate_reduction(GammaNode * gamma)
{
  auto origin = gamma->predicate()->origin();
  auto & constant = AssertGetOwnerNode<SimpleNode>(*origin);
  auto cop = static_cast<const ControlConstantOperation *>(&constant.GetOperation());
  auto alternative = cop->value().alternative();

  rvsdg::SubstitutionMap smap;
  for (const auto & ev : gamma->GetEntryVars())
    smap.insert(ev.branchArgument[alternative], ev.input->origin());

  gamma->subregion(alternative)->copy(gamma->region(), smap, false, false);

  for (auto exitvar : gamma->GetExitVars())
    exitvar.output->divert_users(smap.lookup(exitvar.branchResult[alternative]->origin()));

  remove(gamma);
}

static bool
perform_invariant_reduction(GammaNode * gamma)
{
  bool was_normalized = true;
  for (auto exitvar : gamma->GetExitVars())
  {
    auto argument = dynamic_cast<const rvsdg::RegionArgument *>(exitvar.branchResult[0]->origin());
    if (!argument)
      continue;

    size_t n = 0;
    auto input = argument->input();
    for (n = 1; n < exitvar.branchResult.size(); n++)
    {
      auto argument =
          dynamic_cast<const rvsdg::RegionArgument *>(exitvar.branchResult[n]->origin());
      if (!argument && argument->input() != input)
        break;
    }

    if (n == exitvar.branchResult.size())
    {
      exitvar.output->divert_users(argument->input()->origin());
      was_normalized = false;
    }
  }

  return was_normalized;
}

static std::unordered_set<jlm::rvsdg::Output *>
is_control_constant_reducible(GammaNode * gamma)
{
  // check gamma predicate
  auto match = rvsdg::TryGetOwnerNode<SimpleNode>(*gamma->predicate()->origin());
  if (!is<MatchOperation>(match))
    return {};

  /* check number of alternatives */
  auto match_op = static_cast<const MatchOperation *>(&match->GetOperation());
  std::unordered_set<uint64_t> set({ match_op->default_alternative() });
  for (const auto & pair : *match_op)
    set.insert(pair.second);

  if (set.size() != gamma->nsubregions())
    return {};

  /* check for constants */
  std::unordered_set<jlm::rvsdg::Output *> outputs;
  for (const auto & exitvar : gamma->GetExitVars())
  {
    if (!is_ctltype(*exitvar.output->Type()))
      continue;

    size_t n = 0;
    for (n = 0; n < exitvar.branchResult.size(); n++)
    {
      auto node = rvsdg::TryGetOwnerNode<SimpleNode>(*exitvar.branchResult[n]->origin());
      if (!is<ControlConstantOperation>(node))
        break;

      auto op = static_cast<const ControlConstantOperation *>(&node->GetOperation());
      if (op->value().nalternatives() != 2)
        break;
    }
    if (n == exitvar.branchResult.size())
      outputs.insert(exitvar.output);
  }

  return outputs;
}

static void
perform_control_constant_reduction(std::unordered_set<jlm::rvsdg::Output *> & outputs)
{
  auto & gamma = rvsdg::AssertGetOwnerNode<GammaNode>(**outputs.begin());
  auto origin = gamma.predicate()->origin();
  auto [matchNode, matchOperation] = TryGetSimpleNodeAndOptionalOp<MatchOperation>(*origin);
  JLM_ASSERT(matchNode && matchOperation);

  std::unordered_map<uint64_t, uint64_t> map;
  for (const auto & pair : *matchOperation)
    map[pair.second] = pair.first;

  for (const auto & xv : gamma.GetExitVars())
  {
    if (outputs.find(xv.output) == outputs.end())
      continue;

    size_t defalt = 0;
    size_t nalternatives = 0;
    std::unordered_map<uint64_t, uint64_t> new_mapping;
    for (size_t n = 0; n < xv.branchResult.size(); n++)
    {
      auto origin = xv.branchResult[n]->origin();
      auto & value =
          to_ctlconstant_op(AssertGetOwnerNode<SimpleNode>(*origin).GetOperation()).value();
      nalternatives = value.nalternatives();
      if (map.find(n) != map.end())
        new_mapping[map[n]] = value.alternative();
      else
        defalt = value.alternative();
    }

    auto origin = matchNode->input(0)->origin();
    auto m = match(matchOperation->nbits(), new_mapping, defalt, nalternatives, origin);
    xv.output->divert_users(m);
  }
}

bool
ReduceGammaWithStaticallyKnownPredicate(Node & node)
{
  auto gammaNode = dynamic_cast<GammaNode *>(&node);
  if (gammaNode && is_predicate_reducible(gammaNode))
  {
    perform_predicate_reduction(gammaNode);
    return true;
  }

  return false;
}

bool
ReduceGammaControlConstant(Node & node)
{
  auto gammaNode = dynamic_cast<GammaNode *>(&node);
  if (gammaNode == nullptr)
    return false;

  auto outputs = is_control_constant_reducible(gammaNode);
  if (outputs.empty())
    return false;

  perform_control_constant_reduction(outputs);
  return true;
}

bool
ReduceGammaInvariantVariables(Node & node)
{
  const auto gammaNode = dynamic_cast<GammaNode *>(&node);
  if (gammaNode == nullptr)
    return false;

  return !perform_invariant_reduction(gammaNode);
}

GammaOperation::~GammaOperation() noexcept
{}

std::string
GammaOperation::debug_string() const
{
  return "GAMMA";
}

std::unique_ptr<Operation>
GammaOperation::copy() const
{
  return std::make_unique<GammaOperation>(*this);
}

bool
GammaOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const GammaOperation *>(&other);
  return op && op->numAlternatives_ == numAlternatives_;
}

std::shared_ptr<const Type>
GammaOperation::GetMatchContentType(std::size_t alternative) const
{
  return alternative < MatchContentTypes_.size() ? MatchContentTypes_[alternative]
                                                 : UnitType::Create();
}

/* gamma node */

GammaNode::~GammaNode() noexcept = default;

GammaNode::GammaNode(rvsdg::Output & predicate, std::unique_ptr<GammaOperation> op)
    : StructuralNode(predicate.region(), op->nalternatives()),
      Operation_(std::move(op))
{
  addInput(
      std::make_unique<StructuralInput>(
          this,
          &predicate,
          ControlType::Create(Operation_->nalternatives())),
      false);
  for (std::size_t n = 0; n < Operation_->nalternatives(); ++n)
  {
    RegionArgument::Create(*subregion(n), nullptr, Operation_->GetMatchContentType(n));
  }
}

GammaNode::GammaNode(
    rvsdg::Output & predicate,
    size_t nalternatives,
    std::vector<std::shared_ptr<const Type>> match_content_types)
    : GammaNode(
          predicate,
          std::make_unique<GammaOperation>(nalternatives, std::move(match_content_types)))
{}

[[nodiscard]] const GammaOperation &
GammaNode::GetOperation() const noexcept
{
  return *Operation_;
}

GammaNode::EntryVar
GammaNode::AddEntryVar(rvsdg::Output * origin)
{
  auto gammaInput = new StructuralInput(this, origin, origin->Type());
  addInput(std::unique_ptr<StructuralInput>(gammaInput), true);

  EntryVar ev;
  ev.input = gammaInput;

  for (size_t n = 0; n < nsubregions(); n++)
  {
    ev.branchArgument.push_back(
        &RegionArgument::Create(*subregion(n), gammaInput, gammaInput->Type()));
  }

  return ev;
}

GammaNode::EntryVar
GammaNode::GetEntryVar(std::size_t index) const
{
  JLM_ASSERT(index <= ninputs() - 1);
  EntryVar ev;
  ev.input = input(index + 1);
  for (size_t n = 0; n < nsubregions(); ++n)
  {
    ev.branchArgument.push_back(subregion(n)->argument(index + 1));
  }
  return ev;
}

GammaNode::MatchVar
GammaNode::GetMatchVar() const
{
  MatchVar mv;
  mv.input = input(0);
  for (size_t n = 0; n < nsubregions(); ++n)
  {
    mv.matchContent.push_back(subregion(n)->argument(0));
  }
  return mv;
}

std::vector<GammaNode::EntryVar>
GammaNode::GetEntryVars() const
{
  std::vector<GammaNode::EntryVar> vars;
  for (size_t n = 0; n < ninputs() - 1; ++n)
  {
    vars.push_back(GetEntryVar(n));
  }
  return vars;
}

std::variant<GammaNode::MatchVar, GammaNode::EntryVar>
GammaNode::MapInput(const rvsdg::Input & input) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<GammaNode>(input) == this);
  if (input.index() == 0)
  {
    return GetMatchVar();
  }
  else
  {
    return GetEntryVar(input.index() - 1);
  }
}

std::variant<GammaNode::MatchVar, GammaNode::EntryVar>
GammaNode::MapBranchArgument(const rvsdg::Output & output) const
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<GammaNode>(output) == this);
  if (output.index() == 0)
  {
    return GetMatchVar();
  }
  else
  {
    return GetEntryVar(output.index() - 1);
  }
}

GammaNode::ExitVar
GammaNode::AddExitVar(std::vector<jlm::rvsdg::Output *> values)
{
  if (values.size() != nsubregions())
    throw util::Error("Incorrect number of values.");

  const auto & type = values[0]->Type();
  auto output = addOutput(std::make_unique<StructuralOutput>(this, type));

  std::vector<rvsdg::Input *> branchResults;
  for (size_t n = 0; n < nsubregions(); n++)
  {
    branchResults.push_back(
        &rvsdg::RegionResult::Create(*subregion(n), *values[n], output, output->Type()));
  }

  return ExitVar{ std::move(branchResults), std::move(output) };
}

std::vector<GammaNode::ExitVar>
GammaNode::GetExitVars() const
{
  std::vector<GammaNode::ExitVar> vars;
  for (size_t n = 0; n < noutputs(); ++n)
  {
    std::vector<rvsdg::Input *> branchResults;
    for (size_t k = 0; k < nsubregions(); ++k)
    {
      branchResults.push_back(subregion(k)->result(n));
    }
    vars.push_back(ExitVar{ std::move(branchResults), output(n) });
  }
  return vars;
}

GammaNode::ExitVar
GammaNode::MapOutputExitVar(const rvsdg::Output & output) const
{
  JLM_ASSERT(TryGetOwnerNode<GammaNode>(output) == this);
  std::vector<rvsdg::Input *> branchResults;
  for (size_t k = 0; k < nsubregions(); ++k)
  {
    branchResults.push_back(subregion(k)->result(output.index()));
  }
  return ExitVar{ std::move(branchResults), Node::output(output.index()) };
}

GammaNode::ExitVar
GammaNode::MapBranchResultExitVar(const rvsdg::Input & input) const
{
  JLM_ASSERT(TryGetRegionParentNode<GammaNode>(input) == this);
  std::vector<rvsdg::Input *> branchResults;
  for (size_t k = 0; k < nsubregions(); ++k)
  {
    branchResults.push_back(subregion(k)->result(input.index()));
  }
  return ExitVar{ std::move(branchResults), Node::output(input.index()) };
}

void
GammaNode::RemoveExitVars(const std::vector<ExitVar> & exitvars)
{
  std::vector<std::size_t> indices;
  for (const auto & exitvar : exitvars)
  {
    JLM_ASSERT(TryGetOwnerNode<GammaNode>(*exitvar.output) == this);
    indices.push_back(exitvar.output->index());
  }
  std::sort(
      indices.begin(),
      indices.end(),
      [](std::size_t x, std::size_t y)
      {
        return x > y;
      });
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  for (std::size_t index : indices)
  {
    for (std::size_t r = 0; r < nsubregions(); ++r)
    {
      subregion(r)->RemoveResult(index);
    }
    removeOutput(index);
  }
}

void
GammaNode::RemoveEntryVars(const std::vector<EntryVar> & entryvars)
{
  std::vector<std::size_t> indices;
  for (const auto & entryvar : entryvars)
  {
    JLM_ASSERT(TryGetOwnerNode<GammaNode>(*entryvar.input) == this);
    indices.push_back(entryvar.input->index());
  }
  // Sort indices descending
  std::sort(indices.rbegin(), indices.rend());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  for (auto index : indices)
  {
    for (auto & subregion : Subregions())
    {
      subregion.RemoveArgument(index);
    }
    removeInput(index, true);
  }
}

GammaNode *
GammaNode::copy(rvsdg::Region *, SubstitutionMap & smap) const
{
  auto gamma = create(smap.lookup(predicate()->origin()), nsubregions());

  /* add entry variables to new gamma */
  std::vector<SubstitutionMap> rmap(nsubregions());
  for (const auto & oev : GetEntryVars())
  {
    auto nev = gamma->AddEntryVar(smap.lookup(oev.input->origin()));
    for (size_t n = 0; n < nsubregions(); n++)
      rmap[n].insert(oev.branchArgument[n], nev.branchArgument[n]);
  }

  /* copy subregions */
  for (size_t r = 0; r < nsubregions(); r++)
    subregion(r)->copy(gamma->subregion(r), rmap[r], false, false);

  /* add exit variables to new gamma */
  for (const auto & oex : GetExitVars())
  {
    std::vector<jlm::rvsdg::Output *> operands;
    for (size_t n = 0; n < oex.branchResult.size(); n++)
      operands.push_back(rmap[n].lookup(oex.branchResult[n]->origin()));
    auto nex = gamma->AddExitVar(std::move(operands));
    smap.insert(oex.output, nex.output);
  }

  return gamma;
}

std::optional<rvsdg::Output *>
GetGammaInvariantOrigin(const GammaNode & gamma, const GammaNode::ExitVar & exitvar)
{
  // For any region result, check if it directly maps to a
  // gamma entry variable, and returns the origin of its
  // corresponding value (the def site preceding the gamma node).
  auto GetExternalOriginOf = [&gamma](rvsdg::Input * use) -> std::optional<rvsdg::Output *>
  {
    // Test whether origin of this is a region entry argument of
    // this gamma node.
    auto def = use->origin();
    if (rvsdg::TryGetRegionParentNode<GammaNode>(*def) != &gamma)
    {
      return std::nullopt;
    }
    auto rolevar = gamma.MapBranchArgument(*def);
    if (auto entryvar = std::get_if<GammaNode::EntryVar>(&rolevar))
    {
      return entryvar->input->origin();
    }
    else
    {
      return std::nullopt;
    }
  };

  auto firstOrigin = GetExternalOriginOf(exitvar.branchResult[0]);
  if (!firstOrigin)
  {
    return std::nullopt;
  }

  for (size_t n = 1; n < exitvar.branchResult.size(); ++n)
  {
    auto currentOrigin = GetExternalOriginOf(exitvar.branchResult[n]);
    if (!currentOrigin || *firstOrigin != *currentOrigin)
    {
      return std::nullopt;
    }
  }

  return firstOrigin;
}

}
