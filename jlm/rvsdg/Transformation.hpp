/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRANSFORMATION_HPP
#define JLM_RVSDG_TRANSFORMATION_HPP

#include <jlm/util/Statistics.hpp>

namespace jlm::rvsdg
{

class RvsdgModule;

/**
 * \brief Represents an RVSDG transformation.
 */
class Transformation
{
public:
  virtual ~Transformation() noexcept;

  /**
   * \brief Perform transformation
   *
   * \note This method is expected to be called multiple times. An
   * implementation is required to reset the objects' internal state
   * to ensure correct behavior after every invocation.
   *
   * \param module RVSDG module the optimization is performed on.
   * \param statisticsCollector Statistics collector for collecting optimization statistics.
   */
  virtual void
  Run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector) = 0;

  /**
   * \brief Perform transformation
   *
   * \note This method is expected to be called multiple times. An
   * implementation is required to reset the objects' internal state
   * to ensure correct behavior after every invocation.
   *
   * @param module RVSDG module the optimization is performed on.
   */
  void
  Run(RvsdgModule & module)
  {
    util::StatisticsCollector statisticsCollector;
    Run(module, statisticsCollector);
  }
};

/**
 * Sequentially applies a list of optimizations to an Rvsdg.
 */
class TransformationSequence final : public Transformation
{
public:
  class Statistics;

  ~TransformationSequence() noexcept override;

  explicit TransformationSequence(std::vector<Transformation *> optimizations)
      : Optimizations_(std::move(optimizations))
  {}

  void
  Run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(
      RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      std::vector<Transformation *> optimizations)
  {
    TransformationSequence sequentialApplication(std::move(optimizations));
    sequentialApplication.Run(rvsdgModule, statisticsCollector);
  }

private:
  std::vector<Transformation *> Optimizations_;
};

}

#endif // JLM_RVSDG_TRANSFORMATION_HPP
