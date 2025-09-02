/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <fstream>

namespace jlm::rvsdg
{

Transformation::~Transformation() noexcept = default;

class TransformationSequence::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::RvsdgOptimization, sourceFile)
  {}

  void
  StartMeasuring(const Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  EndMeasuring(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, nnodes(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

TransformationSequence::~TransformationSequence() noexcept = default;

void
TransformationSequence::Run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->StartMeasuring(rvsdgModule.Rvsdg());

  size_t numPasses = 0;
  if (DumpRvsdgDotGraphs_)
  {
    DumpDotGraphs(
        rvsdgModule,
        statisticsCollector.GetSettings().GetOutputDirectory(),
        "Pristine",
        numPasses);
    numPasses++;
  }

  for (const auto & transformation : Transformations_)
  {
    transformation->Run(rvsdgModule, statisticsCollector);
    if (DumpRvsdgDotGraphs_)
    {
      DumpDotGraphs(
          rvsdgModule,
          statisticsCollector.GetSettings().GetOutputDirectory(),
          "After" + std::string(transformation->GetName()),
          numPasses);
      numPasses++;
    }
  }

  statistics->EndMeasuring(rvsdgModule.Rvsdg());
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
TransformationSequence::DumpDotGraphs(
    RvsdgModule & rvsdgModule,
    const util::FilePath & outputDir,
    const std::string & passName,
    size_t numPass) const
{
  JLM_ASSERT(outputDir.Exists() && outputDir.IsDirectory());

  util::graph::Writer graphWriter;
  DotWriter_.WriteGraphs(graphWriter, rvsdgModule.Rvsdg().GetRootRegion(), true);

  std::stringstream filePath;
  filePath << outputDir.to_str() << "/" << std::setw(3) << std::setfill('0') << numPass << "-"
           << passName << ".dot";

  std::ofstream outputFile;
  outputFile.open(filePath.str().c_str());
  graphWriter.OutputAllGraphs(outputFile, util::graph::OutputFormat::Dot);
  outputFile.close();
}

}
