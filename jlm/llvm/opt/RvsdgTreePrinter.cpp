/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

#include <algorithm>

namespace jlm::llvm
{

/**
 * Struct used for tracking the total counts of selected types of nodes.
 */
struct NodeCounts
{
  // The total number of regions
  size_t numRegions = 0;

  // Counts of structural nodes
  size_t numPhiNodes = 0;
  size_t numDeltaNodes = 0;
  size_t numLambdaNodes = 0;
  size_t numGammaNodes = 0;
  size_t numThetaNodes = 0;

  // Count of all simple nodes
  size_t numSimpleNodes = 0;

  // Includes volatile loads
  size_t numLoadNodes = 0;
  // Includes volatile stores
  size_t numStoreNodes = 0;
  size_t numAllocaNodes = 0;
  size_t numCallNodes = 0;
};

/**
 * Recursively traverses the given \p region to count the number of nodes of selected kinds.
 * @see NodeCounts struct for the types of nodes being counted.
 *
 * @param region the region to traverse
 * @param nodeCount the struct containing the counter values being incremented.
 */
static void
countNodes(const rvsdg::Region & region, NodeCounts & nodeCounts);

class RvsdgTreePrinter::Statistics final : public util::Statistics
{
  static constexpr auto OutputFileNameLabel_ = "OutputFileName";

  static constexpr auto NumRegions_ = "#Regions";
  static constexpr auto NumPhiNodes_ = "#PhiNodes";
  static constexpr auto NumLambdaNodes_ = "#LambdaNodes";
  static constexpr auto NumDeltaNodes_ = "#DeltaNodes";
  static constexpr auto NumGammaNodes_ = "#GammaNodes";
  static constexpr auto NumThetaNodes_ = "#ThetaNodes";

  static constexpr auto NumSimpleNodes_ = "#SimpleNodes";
  static constexpr auto NumLoadNodes_ = "#LoadNodes";
  static constexpr auto NumStoreNodes_ = "#StoreNodes";
  static constexpr auto NumAllocaNodes_ = "#AllocaNodes";
  static constexpr auto NumCallNodes_ = "#CallNodes";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(util::Statistics::Id::RvsdgTreePrinter, sourceFile)
  {}

  void
  Start() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  addOutputFileName(std::string name)
  {
    AddMeasurement(OutputFileNameLabel_, std::move(name));
  }

  void
  addNodeCounts(const NodeCounts & nodeCounts)
  {
    AddMeasurement(NumRegions_, nodeCounts.numRegions);
    AddMeasurement(NumPhiNodes_, nodeCounts.numPhiNodes);
    AddMeasurement(NumLambdaNodes_, nodeCounts.numLambdaNodes);
    AddMeasurement(NumDeltaNodes_, nodeCounts.numDeltaNodes);
    AddMeasurement(NumGammaNodes_, nodeCounts.numGammaNodes);
    AddMeasurement(NumThetaNodes_, nodeCounts.numThetaNodes);

    AddMeasurement(NumSimpleNodes_, nodeCounts.numSimpleNodes);
    AddMeasurement(NumLoadNodes_, nodeCounts.numLoadNodes);
    AddMeasurement(NumStoreNodes_, nodeCounts.numStoreNodes);
    AddMeasurement(NumAllocaNodes_, nodeCounts.numAllocaNodes);
    AddMeasurement(NumCallNodes_, nodeCounts.numCallNodes);
  }

  void
  Stop() noexcept
  {
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

RvsdgTreePrinter::~RvsdgTreePrinter() noexcept = default;

void
RvsdgTreePrinter::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  // Only create the tree if there are any required annotations
  if (!Configuration_.RequiredAnnotations().IsEmpty())
  {
    auto annotationMap = ComputeAnnotationMap(rvsdgModule.Rvsdg());
    const auto json = rvsdg::Region::toJson(rvsdgModule.Rvsdg().GetRootRegion(), annotationMap);

    auto file = statisticsCollector.createOutputFile("rvsdgTree.txt", true);
    file.open("w");
    fprintf(file.fd(), "%s\n", json.c_str());
    file.close();
    statistics->addOutputFileName(file.path().to_str());
  }

  if (statisticsCollector.IsDemanded(*statistics))
  {
    // Create additional statistics containing the node counts of selected node types
    NodeCounts nodeCounts;
    countNodes(rvsdgModule.Rvsdg().GetRootRegion(), nodeCounts);
    statistics->addNodeCounts(nodeCounts);
  }

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

util::AnnotationMap
RvsdgTreePrinter::ComputeAnnotationMap(const rvsdg::Graph & rvsdg) const
{
  util::AnnotationMap annotationMap;
  for (const auto annotation : Configuration_.RequiredAnnotations().Items())
  {
    switch (annotation)
    {
    case Configuration::Annotation::DebugIds:
    {
      AnnotateDebugIds(rvsdg, annotationMap);
      break;
    }
    case Configuration::Annotation::NumRvsdgNodes:
    {
      const auto matchNode = [](const rvsdg::Node &)
      {
        return true;
      };
      AnnotateNumNodes(rvsdg, matchNode, "NumRvsdgNodes", annotationMap);
      break;
    }
    case Configuration::Annotation::NumAggregateAllocaNodes:
    {
      const auto matchAggregateAlloca = [](const rvsdg::Node & node)
      {
        const auto allocaOperation = dynamic_cast<const AllocaOperation *>(&node.GetOperation());
        return allocaOperation && IsAggregateType(*allocaOperation->allocatedType());
      };
      AnnotateNumNodes(rvsdg, matchAggregateAlloca, "NumAggregateAllocaNodes", annotationMap);
      break;
    }
    case Configuration::Annotation::NumAllocaNodes:
    {
      const auto matchAlloca = [](const rvsdg::Node & node)
      {
        return rvsdg::is<AllocaOperation>(&node);
      };
      AnnotateNumNodes(rvsdg, matchAlloca, "NumAllocaNodes", annotationMap);
      break;
    }
    case Configuration::Annotation::NumLoadNodes:
    {
      const auto matchLoad = [](const rvsdg::Node & node)
      {
        return rvsdg::is<LoadOperation>(&node);
      };
      AnnotateNumNodes(rvsdg, matchLoad, "NumLoadNodes", annotationMap);
      break;
    }
    case Configuration::Annotation::NumStoreNodes:
    {
      const auto matchStore = [](const rvsdg::Node & node)
      {
        return rvsdg::is<StoreOperation>(&node);
      };
      AnnotateNumNodes(rvsdg, matchStore, "NumStoreNodes", annotationMap);
      break;
    }
    case Configuration::Annotation::NumMemoryStateInputsOutputs:
    {
      AnnotateNumMemoryStateInputsOutputs(rvsdg, annotationMap);
      break;
    }
    default:
      JLM_UNREACHABLE("Unhandled RVSDG tree annotation.");
    }
  }

  return annotationMap;
}

void
RvsdgTreePrinter::AnnotateNumNodes(
    const rvsdg::Graph & rvsdg,
    const std::function<bool(const rvsdg::Node &)> & match,
    const std::string_view & label,
    util::AnnotationMap & annotationMap)
{
  std::function<size_t(const rvsdg::Region &)> annotateRegion = [&](const rvsdg::Region & region)
  {
    size_t numNodes = 0;
    for (auto & node : region.Nodes())
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        size_t numSubregionNodes = 0;
        for (size_t n = 0; n < structuralNode->nsubregions(); n++)
        {
          auto subregion = structuralNode->subregion(n);
          numSubregionNodes += annotateRegion(*subregion);
        }

        annotationMap.AddAnnotation(
            structuralNode,
            { label, static_cast<uint64_t>(numSubregionNodes) });
      }

      if (match(node))
      {
        numNodes++;
      }
    }

    annotationMap.AddAnnotation(&region, { label, static_cast<uint64_t>(numNodes) });

    return numNodes;
  };

  annotateRegion(rvsdg.GetRootRegion());
}

void
RvsdgTreePrinter::AnnotateNumMemoryStateInputsOutputs(
    const rvsdg::Graph & rvsdg,
    util::AnnotationMap & annotationMap)
{
  std::string_view argumentLabel("NumMemoryStateTypeArguments");
  std::string_view resultLabel("NumMemoryStateTypeResults");
  std::string_view inputLabel("NumMemoryStateTypeInputs");
  std::string_view outputLabel("NumMemoryStateTypeOutputs");

  std::function<void(const rvsdg::Region &)> annotateRegion = [&](const rvsdg::Region & region)
  {
    auto argumentRange = region.Arguments();
    auto numMemoryStateArguments =
        std::count_if(argumentRange.begin(), argumentRange.end(), IsMemoryStateOutput);
    annotationMap.AddAnnotation(
        &region,
        { argumentLabel, static_cast<uint64_t>(numMemoryStateArguments) });

    auto resultRange = region.Results();
    auto numMemoryStateResults =
        std::count_if(resultRange.begin(), resultRange.end(), IsMemoryStateInput);
    annotationMap.AddAnnotation(
        &region,
        { resultLabel, static_cast<uint64_t>(numMemoryStateResults) });

    for (auto & node : region.Nodes())
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        size_t numMemoryStateInputs = 0;
        for (size_t n = 0; n < structuralNode->ninputs(); n++)
        {
          auto input = structuralNode->input(n);
          if (rvsdg::is<MemoryStateType>(input->Type()))
          {
            numMemoryStateInputs++;
          }
        }
        annotationMap.AddAnnotation(
            structuralNode,
            { inputLabel, static_cast<uint64_t>(numMemoryStateInputs) });

        size_t numMemoryStateOutputs = 0;
        for (size_t n = 0; n < structuralNode->noutputs(); n++)
        {
          auto output = structuralNode->output(n);
          if (rvsdg::is<MemoryStateType>(output->Type()))
          {
            numMemoryStateOutputs++;
          }
        }
        annotationMap.AddAnnotation(
            structuralNode,
            { outputLabel, static_cast<uint64_t>(numMemoryStateOutputs) });

        for (size_t n = 0; n < structuralNode->nsubregions(); n++)
        {
          auto subregion = structuralNode->subregion(n);
          annotateRegion(*subregion);
        }
      }
    }
  };

  annotateRegion(rvsdg.GetRootRegion());
}

void
RvsdgTreePrinter::AnnotateDebugIds(const rvsdg::Graph & rvsdg, util::AnnotationMap & annotationMap)
{
  std::string_view nodeIdLabel("NodeId");
  std::string_view regionIdLabel("RegionId");

  std::function<void(const rvsdg::Region &)> annotateRegion = [&](const rvsdg::Region & region)
  {
    for (auto & node : region.Nodes())
    {
      if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        for (auto & subregion : structuralNode->Subregions())
          annotateRegion(subregion);

        annotationMap.AddAnnotation(
            structuralNode,
            { nodeIdLabel, static_cast<uint64_t>(structuralNode->GetNodeId()) });
      }
    }

    annotationMap.AddAnnotation(
        &region,
        { regionIdLabel, static_cast<uint64_t>(region.getRegionId()) });
  };

  annotateRegion(rvsdg.GetRootRegion());
}

bool
RvsdgTreePrinter::IsMemoryStateInput(const rvsdg::Input * input) noexcept
{
  return rvsdg::is<MemoryStateType>(input->Type());
}

bool
RvsdgTreePrinter::IsMemoryStateOutput(const rvsdg::Output * output) noexcept
{
  return rvsdg::is<MemoryStateType>(output->Type());
}

static void
countNodes(const rvsdg::Region & region, NodeCounts & nodeCounts)
{
  nodeCounts.numRegions++;

  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&nodeCounts](const rvsdg::PhiNode & phiNode)
        {
          nodeCounts.numPhiNodes++;
          countNodes(*phiNode.subregion(), nodeCounts);
        },
        [&nodeCounts](const rvsdg::DeltaNode & deltaNode)
        {
          nodeCounts.numDeltaNodes++;
          countNodes(*deltaNode.subregion(), nodeCounts);
        },
        [&nodeCounts](const rvsdg::LambdaNode & lambdaNode)
        {
          nodeCounts.numLambdaNodes++;
          countNodes(*lambdaNode.subregion(), nodeCounts);
        },
        [&nodeCounts](const rvsdg::GammaNode & gammaNode)
        {
          nodeCounts.numGammaNodes++;
          for (auto & subregion : gammaNode.Subregions())
          {
            countNodes(subregion, nodeCounts);
          }
        },
        [&nodeCounts](const rvsdg::ThetaNode & thetaNode)
        {
          nodeCounts.numThetaNodes++;
          countNodes(*thetaNode.subregion(), nodeCounts);
        },
        [&nodeCounts](const rvsdg::SimpleNode & simpleNode)
        {
          nodeCounts.numSimpleNodes++;

          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [&nodeCounts]([[maybe_unused]] const llvm::LoadOperation & _)
              {
                nodeCounts.numLoadNodes++;
              },
              [&nodeCounts]([[maybe_unused]] const llvm::StoreOperation & _)
              {
                nodeCounts.numStoreNodes++;
              },
              [&nodeCounts]([[maybe_unused]] const llvm::AllocaOperation & _)
              {
                nodeCounts.numAllocaNodes++;
              },
              [&nodeCounts]([[maybe_unused]] const llvm::CallOperation & _)
              {
                nodeCounts.numCallNodes++;
              });
        });
  }
}

}
