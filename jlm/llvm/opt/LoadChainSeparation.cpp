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

LoadChainSeparation::~LoadChainSeparation() noexcept = default;

LoadChainSeparation::LoadChainSeparation()
    : Transformation("LoadChainSeparation")
{}

void
LoadChainSeparation::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  separateModRefChainsInRegion(module.Rvsdg().GetRootRegion());
}

void
LoadChainSeparation::separateModRefChainsInRegion(rvsdg::Region & region)
{
  // FIXME: We currently do not recognize mod/ref chains that do not start at a result. For example,
  // the state output of a lod node that is dead would not be recognized.

  // We require a top-down traverser to ensure that lambda nodes are handled before call nodes
  for (const auto & node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          // Handle innermost regions first
          separateModRefChainsInRegion(*lambdaNode.subregion());
          separateModRefChains(GetMemoryStateRegionResult(lambdaNode));
        },
        [&](rvsdg::PhiNode & phiNode)
        {
          separateModRefChainsInRegion(*phiNode.subregion());
        },
        [&](rvsdg::GammaNode & gammaNode)
        {
          // Handle innermost regions first
          for (auto & subregion : gammaNode.Subregions())
          {
            separateModRefChainsInRegion(subregion);
          }

          for (auto & [branchResults, output] : gammaNode.GetExitVars())
          {
            if (is<MemoryStateType>(output->Type()))
            {
              for (const auto branchResult : branchResults)
              {
                separateModRefChains(*branchResult);
              }
            }
          }
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          // Handle innermost region first
          separateModRefChainsInRegion(*thetaNode.subregion());

          for (const auto loopVar : thetaNode.GetLoopVars())
          {
            if (is<MemoryStateType>(loopVar.output->Type()))
            {
              separateModRefChains(*loopVar.post);
            }
          }
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [](rvsdg::SimpleNode &)
        {
          // Nothing needs to be done
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node->DebugString()));
        });
  }
}

void
LoadChainSeparation::separateModRefChains(rvsdg::Input & startInput)
{
  JLM_ASSERT(is<MemoryStateType>(startInput.Type()));

  const auto modRefChains = traceModRefChains(startInput);
  for (auto & modRefChain : modRefChains)
  {
    const auto refSubchains = extractReferenceSubchains(modRefChain);
    for (auto [links] : refSubchains)
    {
      // Divert the operands of the respective inputs for each encountered reference node and
      // collect join operands
      std::vector<rvsdg::Output *> joinOperands;
      const auto newMemoryStateOperand = links.back().input->origin();
      for (auto [linkInput, linkModRefType] : links)
      {
        JLM_ASSERT(linkModRefType == ModRefChainLink::Type::Reference);
        const auto modRefChainInput = linkInput;
        modRefChainInput->divert_to(newMemoryStateOperand);
        joinOperands.push_back(&mapMemoryStateInputToOutput(*modRefChainInput));
      }

      // Create join node and divert the current memory state output
      auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
      mapMemoryStateInputToOutput(*links.front().input)
          .divertUsersWhere(
              *joinNode.output(0),
              [&joinNode](const rvsdg::Input & user)
              {
                return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &joinNode;
              });
    }
  }
}

rvsdg::Output &
LoadChainSeparation::mapMemoryStateInputToOutput(const rvsdg::Input & input)
{
  if (auto [loadNode, loadOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(input);
      loadOperation)
  {
    return LoadOperation::mapMemoryStateInputToOutput(input);
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

std::vector<LoadChainSeparation::ModRefChain>
LoadChainSeparation::traceModRefChains(rvsdg::Input & startInput)
{
  std::vector<ModRefChain> modRefChains;
  modRefChains.push_back(ModRefChain());

  rvsdg::Input * currentInput = &startInput;
  bool doneTracing = false;
  do
  {
    if (rvsdg::TryGetOwnerRegion(*currentInput->origin()))
    {
      // We have a region argument. Return the chains we found so far.
      return modRefChains;
    }

    auto & node = rvsdg::AssertGetOwnerNode<rvsdg::Node>(*currentInput->origin());
    rvsdg::MatchTypeWithDefault(
        node,
        [&](const rvsdg::GammaNode & gammaNode)
        {
          // FIXME: I really would like that state edges through gammas would be recognized as
          // either modifying or just referencing. However, we would need to know what the
          // operations in the gamma on all branches are and which memory state exit variable maps
          // to which memory state entry variable. We need some more machinery for it first before
          // we can do that.
          for (auto [entryVarInput, _] : gammaNode.GetEntryVars())
          {
            if (is<MemoryStateType>(entryVarInput->Type()))
            {
              auto tmpChains = traceModRefChains(*entryVarInput);
              modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
            }
          }
          doneTracing = true;
        },
        [&](const rvsdg::ThetaNode & thetaNode)
        {
          // FIXME: I really would like that state edges through thetas would be recognized as
          // either modifying or just referencing.
          for (const auto loopVar : thetaNode.GetLoopVars())
          {
            if (is<MemoryStateType>(loopVar.input->Type()))
            {
              auto tmpChains = traceModRefChains(*loopVar.input);
              modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
            }
          }
          doneTracing = true;
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          auto & operation = simpleNode.GetOperation();
          rvsdg::MatchTypeWithDefault(
              operation,
              [&](const LoadOperation &)
              {
                currentInput = &LoadOperation::MapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLink::Type::Reference });
              },
              [&](const StoreOperation &)
              {
                currentInput =
                    &StoreOperation::MapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLink::Type::Modification });
              },
              [&](const FreeOperation &)
              {
                currentInput = &FreeOperation::mapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLink::Type::Modification });
              },
              [&](const MemCpyOperation &)
              {
                // FIXME: We really would like to know here which memory state belongs to the source
                // and which to the dst address. This would allow us to be more precise in the
                // separation.
                currentInput =
                    &MemCpyOperation::mapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLink::Type::Modification });
              },
              [&](const CallOperation &)
              {
                // FIXME: I really would like that state edges through calls would be recognized as
                // either modifying or just referencing.
                doneTracing = true;
              },
              [&](const LambdaExitMemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  auto tmpChains = traceModRefChains(nodeInput);
                  modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
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
                doneTracing = true;
              },
              [&](const CallEntryMemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  auto tmpChains = traceModRefChains(nodeInput);
                  modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
                }
                doneTracing = true;
              },
              [&](const MemoryStateJoinOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  auto tmpChains = traceModRefChains(nodeInput);
                  modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
                }
                doneTracing = true;
              },
              [&](const MemoryStateMergeOperation &)
              {
                for (auto & nodeInput : node.Inputs())
                {
                  auto tmpChains = traceModRefChains(nodeInput);
                  modRefChains.insert(modRefChains.end(), tmpChains.begin(), tmpChains.end());
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

  return modRefChains;
}

}
