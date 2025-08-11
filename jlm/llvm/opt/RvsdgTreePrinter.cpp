/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/Statistics.hpp>

#include <algorithm>
#include <fstream>

namespace jlm::llvm
{

class RvsdgTreePrinter::Statistics final : public util::Statistics
{
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

  auto annotationMap = ComputeAnnotationMap(rvsdgModule.Rvsdg());
  auto tree = rvsdg::Region::ToTree(rvsdgModule.Rvsdg().GetRootRegion(), annotationMap);

  auto file = statisticsCollector.CreateOutputFile("rvsdgTree.txt", true);
  file.open("w");
  fprintf(file.fd(), "%s\n", tree.c_str());
  file.close();

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

util::AnnotationMap
RvsdgTreePrinter::ComputeAnnotationMap(const rvsdg::Graph & rvsdg) const
{
  util::AnnotationMap annotationMap;
  for (auto annotation : Configuration_.RequiredAnnotations().Items())
  {
    switch (annotation)
    {
    case Configuration::Annotation::NumRvsdgNodes:
    {
      const auto matchNode = [](const rvsdg::Node &)
      {
        return true;
      };
      AnnotateNumNodes(rvsdg, matchNode, "NumRvsdgNodes", annotationMap);
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

}
