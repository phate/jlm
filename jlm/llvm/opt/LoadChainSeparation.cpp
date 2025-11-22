/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemCpy.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
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
LoadChainSeparation::separateModRefChains(rvsdg::Input & input)
{
  JLM_ASSERT(is<MemoryStateType>(input.Type()));

  const auto modRefChains = traceModRefChains(input);
  for (auto & modRefChain : modRefChains)
  {
    const auto subChain = computeReferenceSubchains(modRefChain);

    for (auto [start, end] : subChain)
    {
      // Divert the operands of the respective inputs for each encountered memory reference node and
      // collect join operands
      size_t n = start;
      std::vector<rvsdg::Output *> joinOperands;
      const auto newMemoryStateOperand = modRefChain.links[end - 1].input->origin();
      while (n != end)
      {
        JLM_ASSERT(modRefChain.links[n].modRefType == ModRefChainLinkType::Reference);
        const auto modRefChainInput = modRefChain.links[n].input;
        modRefChainInput->divert_to(newMemoryStateOperand);
        joinOperands.push_back(&mapMemoryStateInputToOutput(*modRefChainInput));
        n++;
      }

      // Create join node and divert the current memory state output
      auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
      mapMemoryStateInputToOutput(*modRefChain.links[start].input)
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

std::vector<std::pair<size_t, size_t>>
LoadChainSeparation::computeReferenceSubchains(const ModRefChain & modRefChain)
{
  std::vector<std::pair<size_t, size_t>> refSubchains;
  for (size_t i = 0; i < modRefChain.links.size();)
  {
    if (modRefChain.links[i].modRefType != ModRefChainLinkType::Reference)
    {
      i++;
      continue;
    }

    size_t start = i;
    size_t end = modRefChain.links.size();
    for (size_t j = i + 1; j < modRefChain.links.size(); ++j)
    {
      if (modRefChain.links[j].modRefType != ModRefChainLinkType::Reference)
      {
        end = j;
        break;
      }
    }
    i = end;

    if (end - start > 1)
    {
      // We only care about reference subchains that are longer than a single element
      refSubchains.push_back({ start, end });
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
                    { currentInput, ModRefChainLinkType::Reference });
              },
              [&](const StoreOperation &)
              {
                currentInput =
                    &StoreOperation::MapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLinkType::Modification });
              },
              [&](const MemCpyOperation &)
              {
                // FIXME: We really would like to know here which memory state belongs to the source
                // and which to the dst address. This would allow us to be more precise in the
                // separation.
                currentInput =
                    &MemCpyOperation::mapMemoryStateOutputToInput(*currentInput->origin());
                modRefChains.back().links.push_back(
                    { currentInput, ModRefChainLinkType::Modification });
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
