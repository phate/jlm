/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSetFolding.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/file.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace jlm::llvm::aa
{

/** \brief class returned by the Agnostic ModRefSummarizer
 *
 */
class AgnosticModRefSet final : public ModRefSet
{
public:
  void
  addMemoryNode(PointsToGraph::NodeIndex memoryNode, ModRefEffect modRefEffect)
  {
    JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);
    modRefNodes_[memoryNode] |= modRefEffect;
  }

  /**
   * Removes all memory effects on nodes not included in the given \p filter
   * @param filter the set of memory locations that can be kept
   */
  void
  filterMemoryNodes(util::HashSet<PointsToGraph::NodeIndex> & filter)
  {
    auto it = modRefNodes_.begin();
    while (it != modRefNodes_.end())
    {
      if (filter.Contains(it->first))
        ++it;
      else
        it = modRefNodes_.erase(it);
    }
  }
};

namespace
{

std::string
escapeDotString(const std::string & s)
{
  std::string result;
  for (char c : s)
  {
    if (c == '"')
      result += "\\\"";
    else if (c == '\\')
      result += "\\\\";
    else if (c == '\n')
      result += "\\n";
    else
      result += c;
  }
  return result;
}

std::string
formatMembers(const ModRefSet & modRefSet)
{
  std::vector<PointsToGraph::NodeIndex> indices;
  indices.reserve(modRefSet.getModRefNodes().size());
  for (const auto & [idx, _] : modRefSet.getModRefNodes())
  {
    indices.push_back(idx);
  }
  std::sort(indices.begin(), indices.end());

  std::string result;
  for (size_t i = 0; i < indices.size(); ++i)
  {
    if (i > 0)
      result += ",";
    result += std::to_string(indices[i]);
  }
  return result;
}

class VennDAGGraphBuilder
{
public:
  VennDAGGraphBuilder(const rvsdg::LambdaNode & topLambda, const ModRefSummary & summary)
      : topLambda_(topLambda),
        summary_(summary)
  {}

  void
  Build()
  {
    auto lambdaEntryId = "lambda_entry";
    auto lambdaExitId = "lambda_exit";

    std::string lambdaEntryMembers = formatMembers(summary_.GetLambdaEntryModRef(topLambda_));
    std::string lambdaExitMembers = formatMembers(summary_.GetLambdaExitModRef(topLambda_));

    addNode(lambdaEntryId, lambdaEntryMembers, "LAMBDA ENTRY " + topLambda_.DebugString());
    addNode(lambdaExitId, lambdaExitMembers, "LAMBDA EXIT " + topLambda_.DebugString());

    processRegion(*topLambda_.subregion(), lambdaEntryId, lambdaExitId);
  }

  void
  Output(std::ostream & out) const
  {
    std::string graphName = topLambda_.DebugString();
    out << "digraph \"" << escapeDotString(graphName) << "\" {\n";
    for (const auto & nodeDecl : nodeDeclarations_)
    {
      out << "  " << nodeDecl << "\n";
    }
    for (const auto & [src, dst] : edges_)
    {
      out << "  \"" << src << "\" -> \"" << dst << "\";\n";
    }
    out << "}\n";
  }

private:
  void
  addNode(const std::string & id, const std::string & members, const std::string & label)
  {
    std::string escapedLabel = escapeDotString(label);
    std::string decl = "\"" + id + "\" [members=\"" + members + "\", label=\"" + escapedLabel
                     + "\\n{" + members + "}\"];";
    nodeDeclarations_.push_back(decl);
  }

  void
  addEdge(const std::string & src, const std::string & dst)
  {
    edges_.insert({ src, dst });
  }

  std::string
  getSourceId(const rvsdg::Output & output, const std::string & regionEntryId)
  {
    const auto * node = rvsdg::TryGetOwnerNode<rvsdg::Node>(output);
    if (!node)
    {
      return regionEntryId;
    }
    std::string result;
    rvsdg::MatchType(
        *node,
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          result = "node_r" + std::to_string(simpleNode.region()->getRegionId()) + "_n"
                 + std::to_string(simpleNode.GetNodeId());
        },
        [&](const rvsdg::GammaNode & gammaNode)
        {
          result = "gamma_r" + std::to_string(gammaNode.region()->getRegionId()) + "_n"
                 + std::to_string(gammaNode.GetNodeId()) + "_exit";
        },
        [&](const rvsdg::ThetaNode & thetaNode)
        {
          result = "theta_r" + std::to_string(thetaNode.region()->getRegionId()) + "_n"
                 + std::to_string(thetaNode.GetNodeId()) + "_exit";
        },
        [&](const rvsdg::LambdaNode & lambdaNode)
        {
          result = "lambda_r" + std::to_string(lambdaNode.region()->getRegionId()) + "_n"
                 + std::to_string(lambdaNode.GetNodeId()) + "_exit";
        },
        [&](const rvsdg::Node & otherNode)
        {
          result = "node_r" + std::to_string(otherNode.region()->getRegionId()) + "_n"
                 + std::to_string(otherNode.GetNodeId());
        });
    return result;
  }

  void
  processRegion(
      const rvsdg::Region & region,
      const std::string & regionEntryId,
      const std::string & regionExitId)
  {
    for (const auto & node : region.Nodes())
    {
      rvsdg::MatchType(
          node,
          [&](const rvsdg::SimpleNode & simpleNode)
          {
            std::string id = "node_r" + std::to_string(region.getRegionId()) + "_n"
                           + std::to_string(simpleNode.GetNodeId());

            std::string members;
            try
            {
              members = formatMembers(summary_.GetSimpleNodeModRef(simpleNode));
            }
            catch (const std::exception &)
            {
              members = "";
            }

            std::string label = simpleNode.DebugString();
            addNode(id, members, label);

            for (size_t i = 0; i < simpleNode.ninputs(); ++i)
            {
              const auto & origin = *simpleNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          },
          [&](const rvsdg::GammaNode & gammaNode)
          {
            std::string gammaEntryId = "gamma_r" + std::to_string(region.getRegionId()) + "_n"
                                     + std::to_string(gammaNode.GetNodeId()) + "_entry";
            std::string gammaExitId = "gamma_r" + std::to_string(region.getRegionId()) + "_n"
                                    + std::to_string(gammaNode.GetNodeId()) + "_exit";

            addNode(gammaEntryId, "", "GAMMA ENTRY " + gammaNode.DebugString());
            addNode(gammaExitId, "", "GAMMA EXIT " + gammaNode.DebugString());

            for (size_t i = 0; i < gammaNode.ninputs(); ++i)
            {
              const auto & origin = *gammaNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, gammaEntryId);
            }

            for (const auto & subregion : gammaNode.Subregions())
            {
              processRegion(subregion, gammaEntryId, gammaExitId);
            }
          },
          [&](const rvsdg::ThetaNode & thetaNode)
          {
            std::string thetaEntryId = "theta_r" + std::to_string(region.getRegionId()) + "_n"
                                     + std::to_string(thetaNode.GetNodeId()) + "_entry";
            std::string thetaExitId = "theta_r" + std::to_string(region.getRegionId()) + "_n"
                                    + std::to_string(thetaNode.GetNodeId()) + "_exit";

            addNode(thetaEntryId, "", "THETA ENTRY " + thetaNode.DebugString());
            addNode(thetaExitId, "", "THETA EXIT " + thetaNode.DebugString());

            for (size_t i = 0; i < thetaNode.ninputs(); ++i)
            {
              const auto & origin = *thetaNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, thetaEntryId);
            }

            processRegion(*thetaNode.subregion(), thetaEntryId, thetaExitId);
          },
          [&](const rvsdg::LambdaNode & nestedLambda)
          {
            std::string lambdaEntryId = "lambda_r" + std::to_string(region.getRegionId()) + "_n"
                                      + std::to_string(nestedLambda.GetNodeId()) + "_entry";
            std::string lambdaExitId = "lambda_r" + std::to_string(region.getRegionId()) + "_n"
                                     + std::to_string(nestedLambda.GetNodeId()) + "_exit";

            std::string entryMembers = formatMembers(summary_.GetLambdaEntryModRef(nestedLambda));
            std::string exitMembers = formatMembers(summary_.GetLambdaExitModRef(nestedLambda));

            addNode(lambdaEntryId, entryMembers, "LAMBDA ENTRY " + nestedLambda.DebugString());
            addNode(lambdaExitId, exitMembers, "LAMBDA EXIT " + nestedLambda.DebugString());

            for (size_t i = 0; i < nestedLambda.ninputs(); ++i)
            {
              const auto & origin = *nestedLambda.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, lambdaEntryId);
            }

            processRegion(*nestedLambda.subregion(), lambdaEntryId, lambdaExitId);
          },
          [&](const rvsdg::Node & otherNode)
          {
            if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&otherNode))
            {
              for (const auto & subregion : structuralNode->Subregions())
              {
                processRegion(subregion, regionEntryId, regionExitId);
              }
            }
          });
    }

    for (size_t i = 0; i < region.nresults(); ++i)
    {
      const auto & origin = *region.result(i)->origin();
      std::string srcId = getSourceId(origin, regionEntryId);
      addEdge(srcId, regionExitId);
    }
  }

  const rvsdg::LambdaNode & topLambda_;
  const ModRefSummary & summary_;
  std::vector<std::string> nodeDeclarations_;
  std::set<std::pair<std::string, std::string>> edges_;
};

void
collectLambdas(const rvsdg::Region & region, std::vector<const rvsdg::LambdaNode *> & lambdas)
{
  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchType(
        node,
        [&](const rvsdg::LambdaNode & lambda)
        {
          lambdas.push_back(&lambda);
        },
        [&](const rvsdg::PhiNode & phiNode)
        {
          collectLambdas(*phiNode.subregion(), lambdas);
        });
  }
}

void
collectIntraproceduralRegions(
    const rvsdg::Region & region,
    std::vector<const rvsdg::Region *> & regions)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto * structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (const auto & subregion : structuralNode->Subregions())
      {
        regions.push_back(&subregion);
        collectIntraproceduralRegions(subregion, regions);
      }
    }
  }
}

void
collectAllRegions(const rvsdg::Region & region, std::vector<const rvsdg::Region *> & regions)
{
  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchType(
        node,
        [&](const rvsdg::LambdaNode & lambda)
        {
          regions.push_back(lambda.subregion());
          collectIntraproceduralRegions(*lambda.subregion(), regions);
        },
        [&](const rvsdg::PhiNode & phiNode)
        {
          collectAllRegions(*phiNode.subregion(), regions);
        });
  }
}

class VennDAGPerRegionGraphBuilder
{
public:
  VennDAGPerRegionGraphBuilder(const rvsdg::Region & region, const ModRefSummary & summary)
      : region_(region),
        summary_(summary)
  {}

  void
  Build()
  {
    auto regionEntryId = "region_entry";
    auto regionExitId = "region_exit";

    std::string regionMembers = formatMembers(getRegionModRef(region_, summary_));
    std::string graphName = getRegionGraphName(region_);

    addNode(regionEntryId, regionMembers, "REGION ENTRY " + graphName);
    addNode(regionExitId, regionMembers, "REGION EXIT " + graphName);

    processRegion(regionEntryId, regionExitId);
  }

  void
  Output(std::ostream & out) const
  {
    std::string graphName = getRegionGraphName(region_);
    out << "digraph \"" << escapeDotString(graphName) << "\" {\n";
    for (const auto & nodeDecl : nodeDeclarations_)
    {
      out << "  " << nodeDecl << "\n";
    }
    for (const auto & [src, dst] : edges_)
    {
      out << "  \"" << src << "\" -> \"" << dst << "\";\n";
    }
    out << "}\n";
  }

private:
  static const ModRefSet &
  getRegionModRef(const rvsdg::Region & region, const ModRefSummary & summary)
  {
    static const AgnosticModRefSet emptySet;
    if (!region.node())
    {
      return emptySet;
    }
    const ModRefSet * result = &emptySet;
    rvsdg::MatchType(
        *region.node(),
        [&](const rvsdg::LambdaNode & lambda)
        {
          result = &summary.GetLambdaEntryModRef(lambda);
        },
        [&](const rvsdg::GammaNode & gamma)
        {
          result = &summary.GetGammaEntryModRef(gamma);
        },
        [&](const rvsdg::ThetaNode & theta)
        {
          result = &summary.GetThetaModRef(theta);
        },
        [&](const rvsdg::StructuralNode &)
        {
          result = &emptySet;
        });
    return *result;
  }

  static std::string
  getRegionGraphName(const rvsdg::Region & region)
  {
    if (region.node())
    {
      std::string name;
      rvsdg::MatchType(
          *region.node(),
          [&](const rvsdg::LambdaNode & lambda)
          {
            name = lambda.DebugString();
          },
          [&](const rvsdg::GammaNode & gamma)
          {
            name = gamma.DebugString() + "_r" + std::to_string(region.getRegionId()) + "_b"
                 + std::to_string(region.index());
          },
          [&](const rvsdg::ThetaNode & theta)
          {
            name = theta.DebugString() + "_r" + std::to_string(region.getRegionId());
          },
          [&](const rvsdg::StructuralNode & structuralNode)
          {
            name = structuralNode.DebugString() + "_r" + std::to_string(region.getRegionId());
          });
      return name;
    }
    return "REGION_r" + std::to_string(region.getRegionId());
  }

  void
  addNode(const std::string & id, const std::string & members, const std::string & label)
  {
    std::string escapedLabel = escapeDotString(label);
    std::string decl = "\"" + id + "\" [members=\"" + members + "\", label=\"" + escapedLabel
                     + "\\n{" + members + "}\"];";
    nodeDeclarations_.push_back(decl);
  }

  void
  addEdge(const std::string & src, const std::string & dst)
  {
    edges_.insert({ src, dst });
  }

  std::string
  getSourceId(const rvsdg::Output & output, const std::string & regionEntryId) const
  {
    const auto * node = rvsdg::TryGetOwnerNode<rvsdg::Node>(output);
    if (!node || node->region() != &region_)
    {
      return regionEntryId;
    }
    std::string result;
    rvsdg::MatchType(
        *node,
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          result = "node_r" + std::to_string(region_.getRegionId()) + "_n"
                 + std::to_string(simpleNode.GetNodeId());
        },
        [&](const rvsdg::GammaNode & gammaNode)
        {
          result = "gamma_r" + std::to_string(region_.getRegionId()) + "_n"
                 + std::to_string(gammaNode.GetNodeId());
        },
        [&](const rvsdg::ThetaNode & thetaNode)
        {
          result = "theta_r" + std::to_string(region_.getRegionId()) + "_n"
                 + std::to_string(thetaNode.GetNodeId());
        },
        [&](const rvsdg::LambdaNode & lambdaNode)
        {
          result = "lambda_r" + std::to_string(region_.getRegionId()) + "_n"
                 + std::to_string(lambdaNode.GetNodeId());
        },
        [&](const rvsdg::Node & otherNode)
        {
          result = "node_r" + std::to_string(region_.getRegionId()) + "_n"
                 + std::to_string(otherNode.GetNodeId());
        });
    return result;
  }

  void
  processRegion(const std::string & regionEntryId, const std::string & regionExitId)
  {
    for (const auto & node : region_.Nodes())
    {
      rvsdg::MatchType(
          node,
          [&](const rvsdg::SimpleNode & simpleNode)
          {
            std::string id = "node_r" + std::to_string(region_.getRegionId()) + "_n"
                           + std::to_string(simpleNode.GetNodeId());

            std::string members;
            try
            {
              members = formatMembers(summary_.GetSimpleNodeModRef(simpleNode));
            }
            catch (const std::exception &)
            {
              members = "";
            }

            std::string label = simpleNode.DebugString();
            addNode(id, members, label);

            for (size_t i = 0; i < simpleNode.ninputs(); ++i)
            {
              const auto & origin = *simpleNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          },
          [&](const rvsdg::GammaNode & gammaNode)
          {
            std::string id = "gamma_r" + std::to_string(region_.getRegionId()) + "_n"
                           + std::to_string(gammaNode.GetNodeId());
            std::string members = formatMembers(summary_.GetGammaEntryModRef(gammaNode));
            std::string label = gammaNode.DebugString();
            addNode(id, members, label);

            for (size_t i = 0; i < gammaNode.ninputs(); ++i)
            {
              const auto & origin = *gammaNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          },
          [&](const rvsdg::ThetaNode & thetaNode)
          {
            std::string id = "theta_r" + std::to_string(region_.getRegionId()) + "_n"
                           + std::to_string(thetaNode.GetNodeId());
            std::string members = formatMembers(summary_.GetThetaModRef(thetaNode));
            std::string label = thetaNode.DebugString();
            addNode(id, members, label);

            for (size_t i = 0; i < thetaNode.ninputs(); ++i)
            {
              const auto & origin = *thetaNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          },
          [&](const rvsdg::LambdaNode & nestedLambda)
          {
            std::string id = "lambda_r" + std::to_string(region_.getRegionId()) + "_n"
                           + std::to_string(nestedLambda.GetNodeId());
            std::string members = formatMembers(summary_.GetLambdaEntryModRef(nestedLambda));
            std::string label = nestedLambda.DebugString();
            addNode(id, members, label);

            for (size_t i = 0; i < nestedLambda.ninputs(); ++i)
            {
              const auto & origin = *nestedLambda.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          },
          [&](const rvsdg::Node & otherNode)
          {
            std::string id = "node_r" + std::to_string(region_.getRegionId()) + "_n"
                           + std::to_string(otherNode.GetNodeId());
            std::string label = otherNode.DebugString();
            addNode(id, "", label);

            for (size_t i = 0; i < otherNode.ninputs(); ++i)
            {
              const auto & origin = *otherNode.input(i)->origin();
              std::string srcId = getSourceId(origin, regionEntryId);
              addEdge(srcId, id);
            }
          });
    }

    for (size_t i = 0; i < region_.nresults(); ++i)
    {
      const auto & origin = *region_.result(i)->origin();
      std::string srcId = getSourceId(origin, regionEntryId);
      addEdge(srcId, regionExitId);
    }
  }

  const rvsdg::Region & region_;
  const ModRefSummary & summary_;
  std::vector<std::string> nodeDeclarations_;
  std::set<std::pair<std::string, std::string>> edges_;
};

void
dumpVennDAGGraphs(const rvsdg::RvsdgModule & rvsdgModule, const ModRefSummary & summary)
{
  std::filesystem::create_directories("/tmp/VennDAGDump");

  std::string base = "module";
  if (rvsdgModule.SourceFilePath().has_value() && !rvsdgModule.SourceFilePath()->base().empty())
  {
    base = rvsdgModule.SourceFilePath()->base();
  }

  auto outputFilePath =
      util::FilePath::createUniqueFileName(util::FilePath("/tmp/VennDAGDump"), base + "-", ".dot");

  std::ofstream out(outputFilePath.to_str());
  if (!out.is_open())
  {
    return;
  }

  std::vector<const rvsdg::LambdaNode *> lambdas;
  collectLambdas(rvsdgModule.Rvsdg().GetRootRegion(), lambdas);

  for (const auto * lambda : lambdas)
  {
    VennDAGGraphBuilder builder(*lambda, summary);
    builder.Build();
    builder.Output(out);
  }
}

void
dumpVennDAGGraphsPerRegion(const rvsdg::RvsdgModule & rvsdgModule, const ModRefSummary & summary)
{
  std::filesystem::create_directories("/tmp/VennDAGDump");

  std::string base = "module";
  if (rvsdgModule.SourceFilePath().has_value() && !rvsdgModule.SourceFilePath()->base().empty())
  {
    base = rvsdgModule.SourceFilePath()->base();
  }

  std::vector<const rvsdg::Region *> regions;
  collectAllRegions(rvsdgModule.Rvsdg().GetRootRegion(), regions);

  for (const auto * region : regions)
  {
    auto outputFilePath =
        util::FilePath::createUniqueFileName(util::FilePath("/tmp/VennDAGDump"), base + "-", ".dot");

    std::ofstream out(outputFilePath.to_str());
    if (!out.is_open())
    {
      continue;
    }

    auto memoryStateGraph = constructMemoryStateGraph(*region, summary);
    out << toDot(*memoryStateGraph);
  }
}

}

/**
 * Operations like loads and stores have a size.
 * If the size is larger than the size of a memory represented by a memory node X,
 * X can be excluded from the Mod/Ref summary of the operation.
 */
static const bool ENABLE_OPERATION_SIZE_BLOCKING =
    !std::getenv("JLM_DISABLE_OPERATION_SIZE_BLOCKING");

/**
 * Constant memory, such as functions, constant globals and constant import, can never change.
 * We therefore never need to route their memory states through anything.
 */
static const bool ENABLE_CONSTANT_MEMORY_BLOCKING =
    !std::getenv("JLM_DISABLE_CONSTANT_MEMORY_BLOCKING");

/**
 * When true, ModRefSets can remove (or skip materializing) memory locations that
 * always imply Omega, in every ModRefSet within the same function.
 * In practice this yields the same result as if the folded memory location was
 * always represented by Omega.
 */
static const bool ENABLE_OMEGA_FOLDING = !std::getenv("JLM_DISABLE_OMEGA_FOLDING");

/**
 * When true, and statistics are demanded, a string describing every ModRefSet is created
 * and added to the statistics object. It can be very long.
 */
static const bool ENABLE_MEMORY_STATE_DISTRIBUTION_STATISTICS =
    !std::getenv("JLM_DISABLE_MEMORY_STATE_DISTRIBUTION_STATISTICS");

/** \brief Mod/Ref summary of agnostic mod/ref summarizer
 *
 */
class AgnosticModRefSummary final : public ModRefSummary
{
public:
  using ModRefSetIndex = size_t;

  ~AgnosticModRefSummary() noexcept override = default;

private:
  AgnosticModRefSummary(const PointsToGraph & pointsToGraph)
      : PointsToGraph_(pointsToGraph)
  {}

public:
  AgnosticModRefSummary(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary(AgnosticModRefSummary &&) = delete;

  AgnosticModRefSummary &
  operator=(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary &
  operator=(AgnosticModRefSummary &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  /**
   * Creates an empty ModRefSet and returns its index
   * @return the index of the created ModRefSet
   */
  ModRefSetIndex
  CreateModRefSet()
  {
    auto index = modRefSets_.size();
    modRefSets_.emplace_back();
    return index;
  }

  /**
   * Adds a ModRefSet to the summary, returning its index.
   * @param modRefSet the existing ModRefSet
   * @return the index assigned to the new set
   */
  ModRefSetIndex
  AddModRefSet(AgnosticModRefSet modRefSet)
  {
    auto index = modRefSets_.size();
    modRefSets_.emplace_back(std::move(modRefSet));
    return index;
  }

  /**
   * Gets a ModRefSet that is designated to represent operations that do not affect memory.
   * @return the index of the empty ModRefSet
   */
  ModRefSetIndex
  getEmptyModRefSet()
  {
    if (!emptyModRefSet_.has_value())
      emptyModRefSet_ = CreateModRefSet();

    return *emptyModRefSet_;
  }

  [[nodiscard]] AgnosticModRefSet &
  GetModRefSet(ModRefSetIndex modRefSetIndex)
  {
    JLM_ASSERT(modRefSetIndex < modRefSets_.size());
    return modRefSets_[modRefSetIndex];
  }

  /**
   * Associated the given \p node with the ModRefSet with the given index.
   * @param node the RVSDG node
   * @param modRefSet the index of the ModRefSet representing the node
   */
  void
  AttachNodeToModRefSet(const rvsdg::Node & node, ModRefSetIndex modRefSet)
  {
    JLM_ASSERT(modRefSet < modRefSets_.size());
    auto [_, inserted] = nodeModRefSets_.insert({ &node, modRefSet });
    JLM_ASSERT(inserted);
  }

  [[nodiscard]] const ModRefSet &
  GetModRefSetForNode(const rvsdg::Node & node) const
  {
    if (const auto it = nodeModRefSets_.find(&node); it != nodeModRefSets_.end())
    {
      return modRefSets_[it->second];
    }
    throw std::logic_error("No AgnosticModRefSet found for node");
  }

  [[nodiscard]] const ModRefSet &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    return GetModRefSetForNode(node);
  }

  [[nodiscard]] const ModRefSet &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const override
  {
    return GetModRefSetForNode(gamma);
  }

  [[nodiscard]] const ModRefSet &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const override
  {
    // Gamma exit uses the same set as gamma entry
    return GetGammaEntryModRef(gamma);
  }

  [[nodiscard]] const ModRefSet &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const override
  {
    return GetModRefSetForNode(theta);
  }

  [[nodiscard]] const ModRefSet &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return GetModRefSetForNode(lambda);
  }

  [[nodiscard]] const ModRefSet &
  GetLambdaExitModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    // Lambda exit uses the same set as lambda entry
    return GetLambdaEntryModRef(lambda);
  }

  static std::unique_ptr<AgnosticModRefSummary>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::unique_ptr<AgnosticModRefSummary>(new AgnosticModRefSummary(pointsToGraph));
  }

private:
  const PointsToGraph & PointsToGraph_;

  /**
   * List containing all ModRefSets in the summary
   */
  std::vector<AgnosticModRefSet> modRefSets_;

  /**
   * Mapping from Node to the index of their AgnosticModRefSet.
   * Some nodes may share ModRefSet, e.g., all structural nodes and calls in a function.
   */
  std::unordered_map<const rvsdg::Node *, ModRefSetIndex> nodeModRefSets_;

  /**
   * This ModRefSet is created once and re-used whenever a node needs an empty ModRefSet.
   * @see getEmptyModRefSet()
   */
  std::optional<ModRefSetIndex> emptyModRefSet_;
};

/**
 * When Omega folding is disabled, the ModRefSummary contains a set containing all
 * memory locations from the entire module.
 * The ModRefSet always has this index in the summary.
 */
static constexpr size_t AllMemoryNodesSet = 0;

AgnosticModRefSummarizer::AgnosticModRefSummarizer() = default;

AgnosticModRefSummarizer::~AgnosticModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics =
      Statistics::Create(rvsdgModule.SourceFilePath().value(), statisticsCollector, pointsToGraph);
  statistics->StartCollecting();

  ModRefSummary_ = AgnosticModRefSummary::Create(pointsToGraph);

  if (!ENABLE_OMEGA_FOLDING)
  {
    // When Omega folding is disabled, create a ModRefSet containing every memory location
    auto allMemoryNodes = GetAllMemoryNodes(pointsToGraph);
    auto allMemoryNodesSet = ModRefSummary_->AddModRefSet(std::move(allMemoryNodes));
    JLM_ASSERT(allMemoryNodesSet == AllMemoryNodesSet);
  }

  // Create ModRefSets for all structural nodes and simple nodes that affect memory
  annotateInterproceduralRegion(rvsdgModule.Rvsdg().GetRootRegion());

  statistics->StopCollecting();

  // Perform the collection of the memory state distribution AFTER we invoked StopCollecting() such
  // that it does not count into the timing measurements
  if (ENABLE_MEMORY_STATE_DISTRIBUTION_STATISTICS
      && statisticsCollector.IsDemanded(statistics->GetId()))
  {
    const auto summaries = collectMemoryStateDistribution(rvsdgModule.Rvsdg(), *ModRefSummary_);
    statistics->addMemoryStateDistribution(summaries);
  }

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  if (std::getenv(ENV_DUMP_VENNDAG))
  {
    dumpVennDAGGraphs(rvsdgModule, *ModRefSummary_);
  }

  if (std::getenv(ENV_DUMP_VENNDAG_PER_REGION))
  {
    dumpVennDAGGraphsPerRegion(rvsdgModule, *ModRefSummary_);
  }

  return std::move(ModRefSummary_);
}

AgnosticModRefSet
AgnosticModRefSummarizer::GetAllMemoryNodes(const PointsToGraph & pointsToGraph)
{
  AgnosticModRefSet modRefSet;

  auto addMemoryNode = [&pointsToGraph, &modRefSet](MemoryNodeId memoryNode)
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(memoryNode))
      return;
    modRefSet.addMemoryNode(memoryNode, ModRefEffect::ModRef);
  };

  for (const auto allocaNode : pointsToGraph.allocaNodes())
    addMemoryNode(allocaNode);

  for (const auto deltaNode : pointsToGraph.deltaNodes())
    addMemoryNode(deltaNode);

  for (const auto lambdaNode : pointsToGraph.lambdaNodes())
    addMemoryNode(lambdaNode);

  for (const auto mallocNode : pointsToGraph.mallocNodes())
    addMemoryNode(mallocNode);

  for (const auto importNode : pointsToGraph.importNodes())
    addMemoryNode(importNode);

  addMemoryNode(pointsToGraph.getExternalMemoryNode());

  return modRefSet;
}

void
AgnosticModRefSummarizer::annotateInterproceduralRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchType(
        node,
        [&](const rvsdg::LambdaNode & lambda)
        {
          annotateFunction(lambda);
        },
        [&](const rvsdg::PhiNode & phiNode)
        {
          annotateInterproceduralRegion(*phiNode.subregion());
        });
  }
}

void
AgnosticModRefSummarizer::annotateFunction(const rvsdg::LambdaNode & lambda)
{
  // The ModRefSet used for the function itself, and all structural nodes and calls in the function
  auto functionModRefSetIndex = AllMemoryNodesSet;
  if (ENABLE_OMEGA_FOLDING)
  {
    // With Omega folding, each function gets its own ModRefSet
    functionModRefSetIndex = ModRefSummary_->CreateModRefSet();
  }

  ModRefSummary_->AttachNodeToModRefSet(lambda, functionModRefSetIndex);

  std::vector<size_t> modRefSetsInFunction;
  annotateIntraproceduralRegion(*lambda.subregion(), functionModRefSetIndex, modRefSetsInFunction);

  if (ENABLE_OMEGA_FOLDING)
  {
    // Go over all created ModRefSets in the function,
    // and find all memory locations that are not implied by Omega.
    // These will be the set of memory locations in the function
    auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();
    auto omegaNode = pointsToGraph.getExternalMemoryNode();

    util::HashSet<PointsToGraph::NodeIndex> keepMemoryLocations;
    // Always keep the Omega node, since it cannot be folded into itself
    keepMemoryLocations.insert(omegaNode);

    for (auto modRefSetIndex : modRefSetsInFunction)
    {
      auto & modRefSet = ModRefSummary_->GetModRefSet(modRefSetIndex);
      auto & modRefSetEffects = modRefSet.getModRefNodes();

      // Find what effect the set has on Omega
      ModRefEffect omegaEffect = ModRefEffect::NoEffect;
      if (auto it = modRefSetEffects.find(omegaNode); it != modRefSetEffects.end())
      {
        omegaEffect = it->second;
        JLM_ASSERT(omegaEffect != ModRefEffect::NoEffect);
      }

      // ModRefSets where Omega is Mod/Ref cannot disqualify any memory location from folding
      if (omegaEffect == ModRefEffect::ModRef)
        continue;

      // Go over all memory effects and see if they are not implying Omega
      for (auto [target, effect] : modRefSetEffects)
      {
        if (isEffectSubset(effect, omegaEffect))
          continue;

        // this target is not implying Omega, so it must be kept
        keepMemoryLocations.insert(target);
      }
    }

    // Next, go back across all the ModRefSets inside the function,
    // and remove all memory effects on memory locations not marked for keeping
    for (auto modRefSetIndex : modRefSetsInFunction)
    {
      auto & modRefSet = ModRefSummary_->GetModRefSet(modRefSetIndex);
      modRefSet.filterMemoryNodes(keepMemoryLocations);
    }

    // Lastly add all kept memory locations to the ModRefSet representing the function,
    // and all structural nodes and calls inside the function
    auto & functionModRefSet = ModRefSummary_->GetModRefSet(functionModRefSetIndex);
    for (auto keptMemoryLocation : keepMemoryLocations.Items())
    {
      functionModRefSet.addMemoryNode(keptMemoryLocation, ModRefEffect::ModRef);
    }
  }
}

void
AgnosticModRefSummarizer::annotateIntraproceduralRegion(
    const rvsdg::Region & region,
    AgnosticModRefSummary::ModRefSetIndex functionModRefSetIndex,
    std::vector<size_t> & modRefSetsInFunction)
{
  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          annotateSimpleNode(simpleNode, functionModRefSetIndex, modRefSetsInFunction);
        },
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          ModRefSummary_->AttachNodeToModRefSet(structuralNode, functionModRefSetIndex);
          for (const auto & subregion : structuralNode.Subregions())
          {
            annotateIntraproceduralRegion(subregion, functionModRefSetIndex, modRefSetsInFunction);
          }
        });
  }
}

void
AgnosticModRefSummarizer::addPointerTargetsToModRefSet(
    AgnosticModRefSet & modRefSet,
    const rvsdg::Output & output,
    std::optional<size_t> minSize,
    ModRefEffect modRefEffect) const
{
  if (!ENABLE_OPERATION_SIZE_BLOCKING)
    minSize = std::nullopt;
  size_t minSizeOrZero = minSize.value_or(0);

  const auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();

  // Helper function for adding targets to the modRefSet, while skipping filtered targets
  const auto addTarget = [&](PointsToGraph::NodeIndex target)
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(target))
      return;

    if (minSizeOrZero > 0)
    {
      auto targetSize = pointsToGraph.tryGetNodeSize(target);
      if (targetSize.has_value() && *targetSize < minSizeOrZero)
        return;
    }

    modRefSet.addMemoryNode(target, modRefEffect);
  };

  JLM_ASSERT(IsPointerCompatible(output));
  const auto & addressReg = pointsToGraph.getNodeForRegister(output);
  for (const auto target : pointsToGraph.getExplicitTargets(addressReg).Items())
  {
    addTarget(target);
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(addressReg))
  {
    // Add all externally available memory nodes
    for (const auto implicitTarget : pointsToGraph.getExternallyAvailableNodes())
    {
      addTarget(implicitTarget);
    }
  }
}

void
AgnosticModRefSummarizer::annotateSimpleNode(
    const rvsdg::SimpleNode & node,
    AgnosticModRefSummary::ModRefSetIndex functionModRefSet,
    std::vector<size_t> & modRefSetsInFunction)
{
  const auto createModRefSetForNode =
      [this, &modRefSetsInFunction](const rvsdg::SimpleNode & node) -> AgnosticModRefSet &
  {
    auto modRefSetIndex = ModRefSummary_->CreateModRefSet();
    ModRefSummary_->AttachNodeToModRefSet(node, modRefSetIndex);
    modRefSetsInFunction.push_back(modRefSetIndex);
    return ModRefSummary_->GetModRefSet(modRefSetIndex);
  };

  MatchTypeWithDefault(
      node.GetOperation(),
      [&](const StoreOperation & storeOperation)
      {
        const auto & address = *StoreOperation::AddressInput(node).origin();
        const auto storeSize = GetTypeStoreSize(storeOperation.GetStoredType());
        auto & modRefSet = createModRefSetForNode(node);
        addPointerTargetsToModRefSet(modRefSet, address, storeSize, ModRefEffect::ModOnly);
      },
      [&](const LoadOperation & loadOperation)
      {
        const auto & address = *LoadOperation::AddressInput(node).origin();
        const auto loadSize = GetTypeStoreSize(*loadOperation.GetLoadedType());
        auto & modRefSet = createModRefSetForNode(node);
        addPointerTargetsToModRefSet(modRefSet, address, loadSize, ModRefEffect::RefOnly);
      },
      [&](const MemCpyOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto & srcAddress = *MemCpyOperation::sourceInput(node).origin();
        const auto & dstAddress = *MemCpyOperation::destinationInput(node).origin();
        const auto & countOrigin = *MemCpyOperation::countInput(node).origin();
        const auto count = jlm::llvm::tryGetConstantSignedInteger(countOrigin);
        addPointerTargetsToModRefSet(modRefSet, srcAddress, count, ModRefEffect::RefOnly);
        addPointerTargetsToModRefSet(modRefSet, dstAddress, count, ModRefEffect::ModOnly);
      },
      [&](const MemMoveOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto & srcAddress = *MemMoveOperation::sourceInput(node).origin();
        const auto & dstAddress = *MemMoveOperation::destinationInput(node).origin();
        const auto & countOrigin = *MemMoveOperation::lengthInput(node).origin();
        const auto count = jlm::llvm::tryGetConstantSignedInteger(countOrigin);
        addPointerTargetsToModRefSet(modRefSet, srcAddress, count, ModRefEffect::RefOnly);
        addPointerTargetsToModRefSet(modRefSet, dstAddress, count, ModRefEffect::ModOnly);
      },
      [&](const MemSetOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto & dstAddress = *MemSetOperation::destinationInput(node).origin();
        const auto & countOrigin = *MemSetOperation::lengthInput(node).origin();
        const auto count = jlm::llvm::tryGetConstantSignedInteger(countOrigin);
        addPointerTargetsToModRefSet(modRefSet, dstAddress, count, ModRefEffect::ModOnly);
      },
      [&](const FreeOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto & freeAddress = *FreeOperation::getAddressInput(node).origin();
        addPointerTargetsToModRefSet(modRefSet, freeAddress, std::nullopt, ModRefEffect::ModOnly);
      },
      [&](const AllocaOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto allocaMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForAlloca(node);
        // The alloca operation does not assign to the allocated memory, so Ref is more precise
        modRefSet.addMemoryNode(allocaMemoryNode, ModRefEffect::RefOnly);
      },
      [&](const MallocOperation &)
      {
        auto & modRefSet = createModRefSetForNode(node);
        const auto mallocMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForMalloc(node);
        // The malloc operation does not assign to the allocated memory, so Ref is more precise
        modRefSet.addMemoryNode(mallocMemoryNode, ModRefEffect::RefOnly);
      },
      [&](const CallOperation &)
      {
        auto & memStateOutput = CallOperation::GetMemoryStateOutput(node);
        if (memStateOutput.IsDead())
        {
          // If a previous pass, like InvariantValueRedirection, found out that this call does
          // not read from or write to memory, it can be detached from the memory state chain.
          // In that case, prevent memory state encoding from re-attaching it.
          ModRefSummary_->AttachNodeToModRefSet(node, ModRefSummary_->getEmptyModRefSet());
        }
        else
        {
          // Calls use the same ModRefSet as all structural nodes in the the function
          ModRefSummary_->AttachNodeToModRefSet(node, functionModRefSet);
        }
      },
      [&](const MemoryStateOperation &)
      {
        // Memory state operations are only used to route memory state edges
      },
      [&]()
      {
        // Any remaining type of node should not involve any memory states
        JLM_ASSERT(!hasMemoryState(node));
      });
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  AgnosticModRefSummarizer summarizer;
  return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}
