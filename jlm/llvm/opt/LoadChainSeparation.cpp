/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemCpy.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm
{

class LoadChainSeparation::Context
{
public:
  bool
  hasModRefChainLinkType(const rvsdg::Output & output) const noexcept
  {
    return Types_.find(&output) != Types_.end();
  }

  void
  add(const rvsdg::Output & output, const ModRefChainLink::Type & type)
  {
    JLM_ASSERT(is<MemoryStateType>(output.Type()));
    JLM_ASSERT(!hasModRefChainLinkType(output));
    Types_[&output] = type;
  }

  ModRefChainLink::Type
  getModRefChainLinkType(const rvsdg::Output & output) const
  {
    JLM_ASSERT(hasModRefChainLinkType(output));
    return Types_.find(&output)->second;
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  std::unordered_map<const rvsdg::Output *, ModRefChainLink::Type> Types_{};
};

LoadChainSeparation::~LoadChainSeparation() noexcept = default;

LoadChainSeparation::LoadChainSeparation()
    : Transformation("LoadChainSeparation")
{}

void
LoadChainSeparation::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  Context_ = Context::create();

  separateReferenceChainsInRegion(module.Rvsdg().GetRootRegion());
}

void
LoadChainSeparation::separateReferenceChainsInRegion(rvsdg::Region & region)
{
  util::HashSet<rvsdg::Output *> visitedOutputs;

  // We require a top-down traverser to ensure that lambda nodes are handled before call nodes
  for (const auto & node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          separateReferenceChainsInLambda(lambdaNode);
        },
        [&](rvsdg::PhiNode & phiNode)
        {
          separateReferenceChainsInRegion(*phiNode.subregion());
        },
        [&](rvsdg::GammaNode & gammaNode)
        {
          separateRefenceChainsInGamma(gammaNode);
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          separateRefenceChainsInTheta(thetaNode, visitedOutputs);
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [&](rvsdg::SimpleNode & simpleNode)
        {
          for (auto & output : simpleNode.Outputs())
          {
            if (output.IsDead() && is<MemoryStateType>(output.Type()))
            {
              // Dead memory state outputs will never be reachable from structural node results.
              // Thus, we need to handle them here in order to separate all reference chains.
              separateReferenceChains(output, visitedOutputs);
            }
          }
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node->DebugString()));
        });
  }
}

void
LoadChainSeparation::separateReferenceChainsInLambda(rvsdg::LambdaNode & lambdaNode)
{
  // Handle innermost regions first
  separateReferenceChainsInRegion(*lambdaNode.subregion());

  util::HashSet<rvsdg::Output *> visitedOutputs;
  separateReferenceChains(*GetMemoryStateRegionResult(lambdaNode).origin(), visitedOutputs);
}

void
LoadChainSeparation::separateRefenceChainsInGamma(rvsdg::GammaNode & gammaNode)
{
  // Handle innermost regions first
  for (auto & subregion : gammaNode.Subregions())
  {
    separateReferenceChainsInRegion(subregion);
  }

  std::vector<util::HashSet<rvsdg::Output *>> visitedOutputs(gammaNode.nsubregions());
  for (auto & [branchResults, output] : gammaNode.GetExitVars())
  {
    if (is<MemoryStateType>(output->Type()))
    {
      for (const auto branchResult : branchResults)
      {
        const auto regionIndex = branchResult->region()->index();
        JLM_ASSERT(regionIndex < visitedOutputs.size());
        separateReferenceChains(*branchResult->origin(), visitedOutputs[regionIndex]);
      }
    }
  }
}

void
LoadChainSeparation::separateRefenceChainsInTheta(
    rvsdg::ThetaNode & thetaNode,
    util::HashSet<rvsdg::Output *> & visitedOutputs)
{
  // Handle innermost region first
  separateReferenceChainsInRegion(*thetaNode.subregion());

  util::HashSet<rvsdg::Output *> visitedOutputsSubregion;
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (!is<MemoryStateType>(loopVar.output->Type()))
      continue;

    // Separate reference chains in theta subregion
    auto hasModificationChainLink =
        separateReferenceChains(*loopVar.post->origin(), visitedOutputsSubregion);
    Context_->add(
        *loopVar.output,
        hasModificationChainLink ? ModRefChainLink::Type::Modification
                                 : ModRefChainLink::Type::Reference);

    // Handle dead theta outputs
    if (loopVar.output->IsDead())
    {
      separateReferenceChains(*loopVar.output, visitedOutputs);
    }
  }
}

bool
LoadChainSeparation::separateReferenceChains(
    rvsdg::Output & startOutput,
    util::HashSet<rvsdg::Output *> & visitedOutputs)
{
  JLM_ASSERT(is<MemoryStateType>(startOutput.Type()));

  ModRefChainSummary summary;
  traceModRefChains(startOutput, visitedOutputs, summary);
  for (auto & modRefChain : summary.modRefChains)
  {
    const auto refSubchains = extractReferenceSubchains(modRefChain);
    for (const auto & [_, links] : refSubchains)
    {
      // Divert the operands of the respective inputs for each encountered reference node and
      // collect join operands
      std::vector<rvsdg::Output *> joinOperands;
      const auto newMemoryStateOperand = mapMemoryStateOutputToInput(*links.back().output).origin();
      for (auto [linkOutput, linkModRefType] : links)
      {
        JLM_ASSERT(linkModRefType == ModRefChainLink::Type::Reference);
        auto & modRefChainInput = mapMemoryStateOutputToInput(*linkOutput);
        modRefChainInput.divert_to(newMemoryStateOperand);
        joinOperands.push_back(linkOutput);
      }

      // Create join node and divert the current memory state output
      if (!links.front().output->IsDead())
      {
        auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
        links.front().output->divertUsersWhere(
            *joinNode.output(0),
            [&joinNode](const rvsdg::Input & user)
            {
              return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &joinNode;
            });
      }
    }
  }

  return summary.hasModificationChainLink;
}

rvsdg::Input &
LoadChainSeparation::mapMemoryStateOutputToInput(const rvsdg::Output & output)
{
  if (auto [loadNode, loadOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(output);
      loadOperation)
  {
    return LoadOperation::MapMemoryStateOutputToInput(output);
  }

  if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    return *thetaNode->MapOutputLoopVar(output).input;
  }

  throw std::logic_error("Unhandled node type!");
}

std::vector<LoadChainSeparation::ModRefChain>
LoadChainSeparation::extractReferenceSubchains(const ModRefChain & modRefChain)
{
  std::vector<ModRefChain> refSubchains;
  for (auto linkIt = modRefChain.links.begin(); linkIt != modRefChain.links.end();)
  {
    if (linkIt->type != ModRefChainLink::Type::Reference)
    {
      // The current link is not a reference. Let's continue with the next one.
      ++linkIt;
      continue;
    }

    auto nextLinkIt = std::next(linkIt);
    if (nextLinkIt == modRefChain.links.end()
        || nextLinkIt->type != ModRefChainLink::Type::Reference)
    {
      // We only want to separate reference chains with at least two links
      ++linkIt;
      continue;
    }

    // We found a new reference subchain. Let's grab all the links
    refSubchains.push_back({});
    while (linkIt != modRefChain.links.end() && linkIt->type == ModRefChainLink::Type::Reference)
    {
      refSubchains.back().links.push_back(*linkIt);
      ++linkIt;
    }
  }

  return refSubchains;
}

void
LoadChainSeparation::traceModRefChains(
    rvsdg::Output & startOutput,
    util::HashSet<rvsdg::Output *> & visitedOutputs,
    ModRefChainSummary & summary)
{
  JLM_ASSERT(is<MemoryStateType>(startOutput.Type()));

  if (!visitedOutputs.insert(&startOutput))
  {
    return;
  }

  ModRefChain currentModRefChain;
  rvsdg::Output * currentOutput = &startOutput;
  bool doneTracing = false;
  do
  {
    if (rvsdg::TryGetOwnerRegion(*currentOutput))
    {
      // We have a region argument. Stop tracing.
      break;
    }

    auto & node = rvsdg::AssertGetOwnerNode<rvsdg::Node>(*currentOutput);
    rvsdg::MatchTypeWithDefault(
        node,
        [&](const rvsdg::GammaNode & gammaNode)
        {
          // FIXME: I really would like that state edges through gammas would be recognized as
          // either modifying or just referencing. However, we would need to know what the
          // operations in the gamma on all branches are and which memory state exit variable maps
          // to which memory state entry variable. We need some more machinery for it first before
          // we can do that.
          currentModRefChain.add({ currentOutput, ModRefChainLink::Type::Modification });
          for (auto [entryVarInput, _] : gammaNode.GetEntryVars())
          {
            if (is<MemoryStateType>(entryVarInput->Type()))
            {
              traceModRefChains(*entryVarInput->origin(), visitedOutputs, summary);
            }
          }
          doneTracing = true;
        },
        [&](const rvsdg::ThetaNode &)
        {
          const auto modRefChainLinkType = Context_->getModRefChainLinkType(*currentOutput);
          currentModRefChain.add({ currentOutput, modRefChainLinkType });
          currentOutput = mapMemoryStateOutputToInput(*currentOutput).origin();
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          auto & operation = simpleNode.GetOperation();
          rvsdg::MatchTypeWithDefault(
              operation,
              [&](const LoadOperation &)
              {
                currentModRefChain.add({ currentOutput, ModRefChainLink::Type::Reference });
                currentOutput = LoadOperation::MapMemoryStateOutputToInput(*currentOutput).origin();
              },
              [&](const StoreOperation &)
              {
                currentModRefChain.add({ currentOutput, ModRefChainLink::Type::Modification });
                currentOutput =
                    StoreOperation::MapMemoryStateOutputToInput(*currentOutput).origin();
              },
              [&](const FreeOperation &)
              {
                currentModRefChain.add({ currentOutput, ModRefChainLink::Type::Modification });
                currentOutput = FreeOperation::mapMemoryStateOutputToInput(*currentOutput).origin();
              },
              [&](const MemCpyOperation &)
              {
                // FIXME: We really would like to know here which memory state belongs to the source
                // and which to the dst address. This would allow us to be more precise in the
                // separation.
                currentModRefChain.add({ currentOutput, ModRefChainLink::Type::Modification });
                currentOutput =
                    MemCpyOperation::mapMemoryStateOutputToInput(*currentOutput).origin();
              },
              [&](const CallOperation &)
              {
                // FIXME: I really would like that state edges through calls would be recognized as
                // either modifying or just referencing.
                traceModRefChains(
                    *CallOperation::GetMemoryStateInput(node).origin(),
                    visitedOutputs,
                    summary);
                doneTracing = true;
              },
              [&](const LambdaExitMemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  traceModRefChains(*nodeInput.origin(), visitedOutputs, summary);
                }
                doneTracing = true;
              },
              [&](const LambdaEntryMemoryStateSplitOperation &)
              {
                // LambdaEntryMemoryStateSplitOperation nodes should always be connected to a lambda
                // argument. In other words, this is as far as we can trace in the graph. Just
                // return what we found so far.
                doneTracing = true;
              },
              [&](const CallExitMemoryStateSplitOperation &)
              {
                // FIXME: I really would like that state edges through calls would be recognized as
                // either modifying or just referencing.
                traceModRefChains(*node.input(0)->origin(), visitedOutputs, summary);
                doneTracing = true;
              },
              [&](const CallEntryMemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  traceModRefChains(*nodeInput.origin(), visitedOutputs, summary);
                }
                doneTracing = true;
              },
              [&](const MemoryStateJoinOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  traceModRefChains(*nodeInput.origin(), visitedOutputs, summary);
                }
                doneTracing = true;
              },
              [&](const MemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  traceModRefChains(*nodeInput.origin(), visitedOutputs, summary);
                }
                doneTracing = true;
              },
              [&](const AllocaOperation &)
              {
                doneTracing = true;
              },
              [&](const MallocOperation &)
              {
                doneTracing = true;
              },
              [&](const UndefValueOperation &)
              {
                doneTracing = true;
              },
              [&]()
              {
                throw std::logic_error(
                    util::strfmt("Unhandled operation type: ", operation.debug_string()));
              });
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node.DebugString()));
        });
  } while (!doneTracing);

  summary.add(std::move(currentModRefChain));
}

}
