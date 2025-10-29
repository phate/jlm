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
  StartMeasuring() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  StartTransformationMeasuring(size_t passNumber, std::string_view passName, const Graph & graph)
  {
    const auto fullName = util::strfmt(std::setw(3), std::setfill('0'), passNumber, "-", passName);

    const auto totalNodes = nnodes(&graph.GetRootRegion());
    AddMeasurement(fullName + "-#RvsdgNodesBefore", totalNodes);

    JLM_ASSERT(currentTransformationTimer_ == nullptr);
    currentTransformationTimer_ = &AddTimer(fullName + "-Timer");
    currentTransformationTimer_->start();
  }

  void
  EndTransformationMeasuring()
  {
    JLM_ASSERT(currentTransformationTimer_ != nullptr);
    currentTransformationTimer_->stop();
    currentTransformationTimer_ = nullptr;
  }

  void
  EndMeasuring(const Graph & graph) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, nnodes(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

  static bool
  IsDemandedBy(const util::StatisticsCollector & collector)
  {
    return collector.IsDemanded(Id::RvsdgOptimization);
  }

private:
  util::Timer * currentTransformationTimer_ = nullptr;
};

TransformationSequence::~TransformationSequence() noexcept = default;

void
TransformationSequence::Run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  std::unique_ptr<Statistics> statistics;
  if (Statistics::IsDemandedBy(statisticsCollector))
    statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  if (statistics)
    statistics->StartMeasuring();

  size_t numPasses = 0;
  if (DumpRvsdgDotGraphs_)
  {
    DumpDotGraphs(
        rvsdgModule,
        statisticsCollector.GetSettings().GetOrCreateOutputDirectory(),
        "Pristine",
        numPasses);
    numPasses++;
  }

  for (const auto & transformation : Transformations_)
  {
    if (statistics)
      statistics->StartTransformationMeasuring(
          numPasses,
          transformation->GetName(),
          rvsdgModule.Rvsdg());

    transformation->Run(rvsdgModule, statisticsCollector);

    if (statistics)
      statistics->EndTransformationMeasuring();

    if (DumpRvsdgDotGraphs_)
    {
      DumpDotGraphs(
          rvsdgModule,
          statisticsCollector.GetSettings().GetOrCreateOutputDirectory(),
          "After" + std::string(transformation->GetName()),
          numPasses);
    }

    numPasses++;
  }

  if (statistics)
  {
    statistics->EndMeasuring(rvsdgModule.Rvsdg());
    statisticsCollector.CollectDemandedStatistics(std::move(statistics));
  }
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
  outputFile.open(filePath.str());
  graphWriter.outputAllGraphs(outputFile, util::graph::OutputFormat::Dot);
  outputFile.close();
}

}
