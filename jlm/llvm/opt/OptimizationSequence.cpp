/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/OptimizationSequence.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class OptimizationSequence::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

  explicit Statistics(util::filepath sourceFile)
      : util::Statistics(Statistics::Id::RvsdgOptimization),
        SourceFile_(std::move(sourceFile)),
        NumNodesBefore_(0),
        NumNodesAfter_(0)
  {}

  void
  StartMeasuring(const jlm::rvsdg::graph & graph) noexcept
  {
    NumNodesBefore_ = jlm::rvsdg::nnodes(graph.root());
    Timer_.start();
  }

  void
  EndMeasuring(const jlm::rvsdg::graph & graph) noexcept
  {
    Timer_.stop();
    NumNodesAfter_ = jlm::rvsdg::nnodes(graph.root());
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return util::strfmt(
        "RVSDGOPTIMIZATION ",
        SourceFile_.to_str(),
        " ",
        NumNodesBefore_,
        " ",
        NumNodesAfter_,
        " ",
        Timer_.ns());
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

private:
  util::timer Timer_;
  util::filepath SourceFile_;
  size_t NumNodesBefore_;
  size_t NumNodesAfter_;
};

OptimizationSequence::~OptimizationSequence() noexcept = default;

void
OptimizationSequence::run(
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
