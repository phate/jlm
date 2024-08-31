/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/Statistics.hpp>

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
  auto tree = rvsdg::region::ToTree(*rvsdgModule.Rvsdg().root(), annotationMap);
  WriteTreeToFile(rvsdgModule, tree);

  statistics->Stop();
}

void
RvsdgTreePrinter::run(RvsdgModule & rvsdgModule)
{
  util::StatisticsCollector collector;
  run(rvsdgModule, collector);
}

util::AnnotationMap
RvsdgTreePrinter::ComputeAnnotationMap(const rvsdg::graph & rvsdg) const
{
  util::AnnotationMap annotationMap;
  for (auto annotation : Configuration_.RequiredAnnotations().Items())
  {
    switch (annotation)
    {
    case Configuration::Annotation::NumRvsdgNodes:
      AnnotateNumRvsdgNodes(rvsdg, annotationMap);
      break;
    default:
      JLM_UNREACHABLE("Unhandled RVSDG tree annotation.");
    }
  }

  return annotationMap;
}

void
RvsdgTreePrinter::AnnotateNumRvsdgNodes(
    const rvsdg::graph & rvsdg,
    util::AnnotationMap & annotationMap)
{
  static std::string_view label("NumRvsdgNodes");

  std::function<size_t(const rvsdg::region &)> annotateRegion = [&](const rvsdg::region & region)
  {
    for (auto & node : region.nodes)
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::structural_node *>(&node))
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

  annotateRegion(*rvsdg.root());
}

void
RvsdgTreePrinter::WriteTreeToFile(const RvsdgModule & rvsdgModule, const std::string & tree) const
{
  auto outputFile = CreateOutputFile(rvsdgModule);

  outputFile.open("w");
  fprintf(outputFile.fd(), "%s\n", tree.c_str());
  outputFile.close();
}

util::file
RvsdgTreePrinter::CreateOutputFile(const RvsdgModule & rvsdgModule) const
{
  auto fileName = util::strfmt(
      Configuration_.OutputDirectory().to_str(),
      "/",
      rvsdgModule.SourceFileName().base().c_str(),
      "-rvsdgTree-",
      GetOutputFileNameCounter(rvsdgModule));
  return util::filepath(fileName);
}

uint64_t
RvsdgTreePrinter::GetOutputFileNameCounter(const RvsdgModule & rvsdgModule)
{
  static std::unordered_map<std::string_view, uint64_t> RvsdgModuleCounterMap_;

  auto key = util::strfmt(&rvsdgModule, rvsdgModule.SourceFileName().to_str());
  return RvsdgModuleCounterMap_[key]++;
}

}
