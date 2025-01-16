/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

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

  explicit Statistics(const util::filepath & sourceFile)
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
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

RvsdgTreePrinter::~RvsdgTreePrinter() noexcept = default;

void
RvsdgTreePrinter::run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
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
      AnnotateNumRvsdgNodes(rvsdg, annotationMap);
      break;
    case Configuration::Annotation::NumMemoryStateInputsOutputs:
      AnnotateNumMemoryStateInputsOutputs(rvsdg, annotationMap);
      break;
    default:
      JLM_UNREACHABLE("Unhandled RVSDG tree annotation.");
    }
  }

  return annotationMap;
}

void
RvsdgTreePrinter::AnnotateNumRvsdgNodes(
    const rvsdg::Graph & rvsdg,
    util::AnnotationMap & annotationMap)
{
  static std::string_view label("NumRvsdgNodes");

  std::function<size_t(const rvsdg::Region &)> annotateRegion = [&](const rvsdg::Region & region)
  {
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

        annotationMap.AddAnnotation(structuralNode, { label, numSubregionNodes });
      }
    }

    auto numNodes = region.nnodes();
    annotationMap.AddAnnotation(&region, { label, numNodes });

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
    annotationMap.AddAnnotation(&region, { argumentLabel, numMemoryStateArguments });

    auto resultRange = region.Results();
    auto numMemoryStateResults =
        std::count_if(resultRange.begin(), resultRange.end(), IsMemoryStateInput);
    annotationMap.AddAnnotation(&region, { resultLabel, numMemoryStateResults });

    for (auto & node : region.Nodes())
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        size_t numMemoryStateInputs = 0;
        for (size_t n = 0; n < structuralNode->ninputs(); n++)
        {
          auto input = structuralNode->input(n);
          if (rvsdg::is<MemoryStateType>(input->type()))
          {
            numMemoryStateInputs++;
          }
        }
        annotationMap.AddAnnotation(structuralNode, { inputLabel, numMemoryStateInputs });

        size_t numMemoryStateOutputs = 0;
        for (size_t n = 0; n < structuralNode->noutputs(); n++)
        {
          auto output = structuralNode->output(n);
          if (rvsdg::is<MemoryStateType>(output->type()))
          {
            numMemoryStateOutputs++;
          }
        }
        annotationMap.AddAnnotation(structuralNode, { outputLabel, numMemoryStateOutputs });

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
RvsdgTreePrinter::IsMemoryStateInput(const rvsdg::input * input) noexcept
{
  return rvsdg::is<MemoryStateType>(input->Type());
}

bool
RvsdgTreePrinter::IsMemoryStateOutput(const rvsdg::output * output) noexcept
{
  return rvsdg::is<MemoryStateType>(output->Type());
}

}
