/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>

namespace jlm::llvm::aa
{

std::vector<MemoryStateSummary>
collectMemoryStateDistribution(const rvsdg::Graph & rvsdg, const ModRefSummary & modRefSummary)
{
  std::function<
      void(const rvsdg::Region &, const rvsdg::LambdaNode *, std::vector<MemoryStateSummary> &)>
      collect = [&](const rvsdg::Region & region,
                    const rvsdg::LambdaNode * lambdaNode,
                    std::vector<MemoryStateSummary> & summaries)
  {
    for (auto & node : region.Nodes())
    {
      rvsdg::MatchTypeOrFail(
          node,
          [&](const rvsdg::PhiNode & phiNode)
          {
            JLM_ASSERT(lambdaNode == nullptr);
            collect(*phiNode.subregion(), lambdaNode, summaries);
          },
          [&](const rvsdg::DeltaNode &)
          {
            JLM_ASSERT(lambdaNode == nullptr);
            // Nothing needs to be done
          },
          [&](const rvsdg::LambdaNode & n)
          {
            JLM_ASSERT(lambdaNode == nullptr);
            auto entrySize = modRefSummary.GetLambdaEntryModRef(n).getModRefNodes().size();
            auto exitSize = modRefSummary.GetLambdaExitModRef(n).getModRefNodes().size();
            summaries.push_back({ &n, &n, entrySize, exitSize });

            collect(*n.subregion(), &n, summaries);
          },
          [&](const rvsdg::ThetaNode & thetaNode)
          {
            JLM_ASSERT(lambdaNode != nullptr);
            auto size = modRefSummary.GetThetaModRef(thetaNode).getModRefNodes().size();
            summaries.push_back({ lambdaNode, &thetaNode, size, size });

            collect(*thetaNode.subregion(), lambdaNode, summaries);
          },
          [&](const rvsdg::GammaNode & gammaNode)
          {
            JLM_ASSERT(lambdaNode != nullptr);
            auto entrySize = modRefSummary.GetGammaEntryModRef(gammaNode).getModRefNodes().size();
            auto exitSize = modRefSummary.GetGammaExitModRef(gammaNode).getModRefNodes().size();
            summaries.push_back({ lambdaNode, &gammaNode, entrySize, exitSize });

            for (auto & subregion : gammaNode.Subregions())
              collect(subregion, lambdaNode, summaries);
          },
          [&](const rvsdg::SimpleNode & simpleNode)
          {
            MatchTypeWithDefault(
                simpleNode.GetOperation(),
                [&](const StoreOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const LoadOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const MemCpyOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const MemMoveOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const MemSetOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const FreeOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const AllocaOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const MallocOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const CallOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  auto size = modRefSummary.GetSimpleNodeModRef(simpleNode).getModRefNodes().size();
                  summaries.push_back({ lambdaNode, &simpleNode, size, size });
                },
                [&](const MemoryStateOperation &)
                {
                  JLM_ASSERT(lambdaNode != nullptr);
                  // Nothing needs to be done
                },
                [&]()
                {
                  // Any remaining type of node should not involve any memory states
                  JLM_ASSERT(!hasMemoryState(node));
                });
          });
    }
  };

  std::vector<MemoryStateSummary> summaries;
  collect(rvsdg.GetRootRegion(), nullptr, summaries);
  return summaries;
}

std::string
toString(const std::vector<MemoryStateSummary> & memoryStateDistribution)
{
  auto toString = [](const MemoryStateSummary & memoryStateSummary)
  {
    constexpr char separator = '-';
    return util::strfmt(
        memoryStateSummary.lambdaNode->DebugString(),
        separator,
        memoryStateSummary.node->DebugString(),
        separator,
        "(",
        memoryStateSummary.node->region()->getRegionId(),
        ":",
        memoryStateSummary.node->GetNodeId(),
        ")",
        separator,
        memoryStateSummary.numMemoryInputStates,
        separator,
        memoryStateSummary.numMemoryOutputStates);
  };

  size_t n = 0;
  std::string summaryStr;
  for (auto & summary : memoryStateDistribution)
  {
    summaryStr += toString(summary);
    if (n != memoryStateDistribution.size() - 1)
      summaryStr += ",";
    n++;
  }

  return summaryStr;
}

}
