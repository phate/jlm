/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "RvsdgModule.hpp"
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/Transformation.hpp>

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

  for (const auto & optimization : Transformations_)
  {
    optimization->Run(rvsdgModule, statisticsCollector);
  }

  statistics->EndMeasuring(rvsdgModule.Rvsdg());
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}
