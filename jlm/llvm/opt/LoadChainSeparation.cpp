/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm
{
#if 0
class LoadChainSeparation::Context final
{
public:
  void
  addModRefType(const rvsdg::Input & input, const ModRefType modRefType)
  {
    JLM_ASSERT(!hasModRefType(input));
    inputModRefType_[&input] = modRefType;
  }

  void
  addModRefType(const rvsdg::Output & output, const ModRefType modRefType)
  {
    JLM_ASSERT(!hasModRefType(output));
    outputModRefType_[&output] = modRefType;
  }

  bool
  hasModRefType(const rvsdg::Input & input) const noexcept
  {
    return inputModRefType_.find(&input) != inputModRefType_.end();
  }

  bool
  hasModRefType(const rvsdg::Output & output) const noexcept
  {
    return outputModRefType_.find(&output) != outputModRefType_.end();
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  std::unordered_map<const rvsdg::Input *, ModRefType> inputModRefType_;
  std::unordered_map<const rvsdg::Output *, ModRefType> outputModRefType_;
};
#endif
LoadChainSeparation::~LoadChainSeparation() noexcept = default;

LoadChainSeparation::LoadChainSeparation()
    : Transformation("LoadChainSeparation")
{}

void
LoadChainSeparation::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  // context_ = Context::create();

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
  if (auto [_, loadOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(input);
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
      }
    }
    i = end;

    if (end - start > 1)
    {
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
  modRefChains.back().links.push_back({ &startInput, ModRefChainLinkType::Other });

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
              [&]()
              {
                throw std::logic_error(
                    util::strfmt("Unhandled node type: ", operation.debug_string()));
              });
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node.DebugString()));
        });
  } while (!doneTracing);

  return modRefChains;
}

void
LoadChainSeparation::separateLoadChain(rvsdg::Output & memoryStateOutput)
{
  JLM_ASSERT(rvsdg::is<MemoryStateType>(memoryStateOutput.Type()));
  JLM_ASSERT(rvsdg::IsOwnerNodeOperation<LoadOperation>(memoryStateOutput));

  std::vector<rvsdg::Output *> joinOperands;
  auto & newMemoryStateOperand = traceLoadNodeMemoryState(memoryStateOutput, joinOperands);
  JLM_ASSERT(joinOperands.size() > 1);

  // Divert the operands of the respective inputs for each encountered memory state output
  for (const auto output : joinOperands)
  {
    auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(*output);
    memoryStateInput.divert_to(&newMemoryStateOperand);
  }

  // Create join node and divert the current memory state output
  auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
  memoryStateOutput.divertUsersWhere(
      *joinNode.output(0),
      [&joinNode](const rvsdg::Input & user)
      {
        return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &joinNode;
      });
}

rvsdg::Output &
LoadChainSeparation::traceLoadNodeMemoryState(
    rvsdg::Output & output,
    std::vector<rvsdg::Output *> & joinOperands)
{
  JLM_ASSERT(rvsdg::is<MemoryStateType>(output.Type()));

  if (!is<LoadOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output)))
    return output;

  joinOperands.push_back(&output);
  return traceLoadNodeMemoryState(
      *LoadOperation::MapMemoryStateOutputToInput(output).origin(),
      joinOperands);
}

util::HashSet<rvsdg::Output *>
LoadChainSeparation::findLoadChainBottoms(rvsdg::Region & region)
{
  util::HashSet<rvsdg::Output *> loadChainBottoms;
  for (auto & node : region.Nodes())
  {
    if (!rvsdg::is<LoadOperation>(&node))
    {
      continue;
    }

    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      if (hasLoadNodeAsUserOwner(memoryStateOutput))
      {
        continue;
      }

      auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput);
      if (hasLoadNodeAsOperandOwner(memoryStateInput))
      {
        loadChainBottoms.insert(&memoryStateOutput);
      }
    }
  }

  return loadChainBottoms;
}

bool
LoadChainSeparation::hasLoadNodeAsOperandOwner(const rvsdg::Input & input)
{
  return rvsdg::IsOwnerNodeOperation<LoadOperation>(*input.origin());
}

bool
LoadChainSeparation::hasLoadNodeAsUserOwner(const rvsdg::Output & output)
{
  for (auto & user : output.Users())
  {
    if (rvsdg::IsOwnerNodeOperation<LoadOperation>(user))
      return true;
  }

  return false;
}

}
