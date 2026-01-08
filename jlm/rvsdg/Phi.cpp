/*
 * Copyright 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <algorithm>
#include <functional>

#include <jlm/rvsdg/Phi.hpp>

namespace jlm::rvsdg
{

PhiOperation::~PhiOperation() = default;

std::string
PhiOperation::debug_string() const
{
  return "Phi";
}

std::unique_ptr<Operation>
PhiOperation::copy() const
{
  return std::make_unique<PhiOperation>(*this);
}

PhiNode::~PhiNode() = default;

[[nodiscard]] const PhiOperation &
PhiNode::GetOperation() const noexcept
{
  // Phi nodes are not parameterized, so we can return operation singleton.
  static const PhiOperation singleton;
  return singleton;
}

PhiNode::ContextVar
PhiNode::AddContextVar(rvsdg::Output & origin)
{
  auto input = addInput(std::make_unique<StructuralInput>(this, &origin, origin.Type()), true);
  auto & argument =
      subregion()->addArgument(std::make_unique<RegionArgument>(subregion(), input, origin.Type()));

  return ContextVar{ input, &argument };
}

[[nodiscard]] std::vector<PhiNode::ContextVar>
PhiNode::GetContextVars() const noexcept
{
  std::vector<PhiNode::ContextVar> vars;

  for (size_t n = 0; n < ninputs(); ++n)
  {
    vars.push_back(ContextVar{ input(n), subregion()->argument(n + subregion()->nresults()) });
  }

  return vars;
}

[[nodiscard]] std::vector<PhiNode::FixVar>
PhiNode::GetFixVars() const noexcept
{
  std::vector<PhiNode::FixVar> vars;

  for (std::size_t n = 0; n < noutputs(); ++n)
  {
    vars.push_back(FixVar{ subregion()->argument(n), subregion()->result(n), output(n) });
  }

  return vars;
}

[[nodiscard]] std::optional<PhiNode::FixVar>
PhiNode::MapArgumentFixVar(const rvsdg::Output & argument) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<PhiNode>(argument) == this);
  if (argument.index() < subregion()->nresults())
  {
    size_t n = argument.index();
    return FixVar{ subregion()->argument(n), subregion()->result(n), output(n) };
  }
  else
  {
    return std::nullopt;
  }
}

[[nodiscard]] PhiNode::FixVar
PhiNode::MapResultFixVar(const rvsdg::Input & result) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<PhiNode>(result) == this);
  return FixVar{ subregion()->argument(result.index()),
                 subregion()->result(result.index()),
                 output(result.index()) };
}

[[nodiscard]] PhiNode::FixVar
PhiNode::MapOutputFixVar(const rvsdg::Output & output) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<PhiNode>(output) == this);
  return FixVar{ subregion()->argument(output.index()),
                 subregion()->result(output.index()),
                 PhiNode::output(output.index()) };
}

[[nodiscard]] PhiNode::ContextVar
PhiNode::MapInputContextVar(const rvsdg::Input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<PhiNode>(input) == this);
  return ContextVar{ PhiNode::input(input.index()),
                     subregion()->argument(input.index() + subregion()->nresults()) };
}

[[nodiscard]] std::optional<PhiNode::ContextVar>
PhiNode::MapArgumentContextVar(const rvsdg::Output & argument) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<PhiNode>(argument) == this);
  if (argument.index() >= subregion()->nresults())
  {
    size_t n = argument.index();
    return ContextVar{ input(n - subregion()->nresults()), subregion()->argument(n) };
  }
  else
  {
    return std::nullopt;
  }
}

[[nodiscard]] std::variant<PhiNode::FixVar, PhiNode::ContextVar>
PhiNode::MapArgument(const rvsdg::Output & argument) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetRegionParentNode<PhiNode>(argument) == this);
  if (auto ctxvar = MapArgumentContextVar(argument))
  {
    return *ctxvar;
  }
  else if (auto fixvar = MapArgumentFixVar(argument))
  {
    return *fixvar;
  }
  else
  {
    JLM_UNREACHABLE("phi binder is neither context nor fixpoint variable");
  }
}

void
PhiNode::RemoveContextVars(std::vector<ContextVar> vars)
{
  util::HashSet<size_t> inputIndices;
  util::HashSet<size_t> argumentIndices;
  for (const auto [input, argument] : vars)
  {
    argumentIndices.insert(argument->index());
    inputIndices.insert(input->index());
  }

  [[maybe_unused]] const auto numRemovedArguments = subregion()->RemoveArguments(argumentIndices);
  JLM_ASSERT(numRemovedArguments == argumentIndices.Size());

  [[maybe_unused]] const auto numRemovedInputs = RemoveInputs(inputIndices);
  JLM_ASSERT(numRemovedInputs == inputIndices.Size());
}

void
PhiNode::RemoveFixVars(std::vector<FixVar> vars)
{
  util::HashSet<size_t> resultIndices;
  util::HashSet<size_t> argumentIndices;
  util::HashSet<size_t> outputIndices;
  for (const auto & [argument, result, output] : vars)
  {
    resultIndices.insert(result->index());
    argumentIndices.insert(argument->index());
    outputIndices.insert(output->index());
  }

  [[maybe_unused]] const auto numRemovedResults = subregion()->RemoveResults(resultIndices);
  JLM_ASSERT(numRemovedResults == resultIndices.Size());

  [[maybe_unused]] const auto numRemovedArguments = subregion()->RemoveArguments(argumentIndices);
  JLM_ASSERT(numRemovedArguments == argumentIndices.Size());

  [[maybe_unused]] const auto numRemovedOutputs = RemoveOutputs(outputIndices);
  JLM_ASSERT(numRemovedOutputs == outputIndices.Size());
}

PhiNode *
PhiNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  PhiBuilder pb;
  pb.begin(region);

  // add context variables
  rvsdg::SubstitutionMap subregionmap;
  for (const auto & var : GetContextVars())
  {
    auto origin = smap.lookup(var.input->origin());
    if (!origin)
      throw util::Error("Operand not provided by susbtitution map.");

    auto newcv = pb.AddContextVar(*origin);
    subregionmap.insert(var.inner, newcv.inner);
  }

  // add recursion variables
  for (auto var : GetFixVars())
  {
    auto newrv = pb.AddFixVar(var.recref->Type());
    subregionmap.insert(var.recref, newrv.recref);
  }

  // copy subregion
  subregion()->copy(pb.subregion(), subregionmap, false, false);

  // finalize phi
  for (auto var : GetFixVars())
  {
    auto neworigin = subregionmap.lookup(var.result->origin());
    var.result->divert_to(neworigin);
  }

  return pb.end();
}

std::vector<rvsdg::LambdaNode *>
PhiNode::ExtractLambdaNodes(const PhiNode & phiNode)
{
  std::function<void(const PhiNode &, std::vector<rvsdg::LambdaNode *> &)> extractLambdaNodes =
      [&](auto & phiNode, auto & lambdaNodes)
  {
    for (auto & node : phiNode.subregion()->Nodes())
    {
      if (auto lambdaNode = dynamic_cast<rvsdg::LambdaNode *>(&node))
      {
        lambdaNodes.push_back(lambdaNode);
      }
      else if (auto innerPhiNode = dynamic_cast<const PhiNode *>(&node))
      {
        extractLambdaNodes(*innerPhiNode, lambdaNodes);
      }
    }
  };

  std::vector<rvsdg::LambdaNode *> lambdaNodes;
  extractLambdaNodes(phiNode, lambdaNodes);

  return lambdaNodes;
}

PhiNode::ContextVar
PhiBuilder::AddContextVar(rvsdg::Output & origin)
{
  return node_->AddContextVar(origin);
}

PhiNode::FixVar
PhiBuilder::AddFixVar(std::shared_ptr<const jlm::rvsdg::Type> type)
{
  auto output = node_->addOutput(std::make_unique<StructuralOutput>(node_, type));
  auto & argument = subregion()->insertArgument(
      subregion()->nresults(),
      std::make_unique<RegionArgument>(subregion(), nullptr, type));
  auto & result =
      subregion()->addResult(std::make_unique<RegionResult>(subregion(), &argument, output, type));

  return PhiNode::FixVar{ &argument, &result, output };
}

PhiNode *
PhiBuilder::end()
{
  if (!node_)
    return nullptr;

  for (auto var : node_->GetFixVars())
  {
    if (var.result->origin() == var.recref)
      throw util::Error("Recursion variable not properly set.");
  }

  auto node = node_;
  node_ = nullptr;

  return node;
}

}
