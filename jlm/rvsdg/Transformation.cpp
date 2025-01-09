/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{

Transformation::~Transformation() noexcept = default;

class TransformationSequence::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::RvsdgOptimization, sourceFile)
  {}

  void
  StartMeasuring(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  EndMeasuring(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

TransformationSequence::~OptimizationSequence() noexcept = default;

void
TransformationSequence::Run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
  statistics->StartMeasuring(rvsdgModule.Rvsdg());

  for (const auto & optimization : Optimizations_)
  {
    optimization->run(rvsdgModule, statisticsCollector);
  }

  statistics->EndMeasuring(rvsdgModule.Rvsdg());
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}