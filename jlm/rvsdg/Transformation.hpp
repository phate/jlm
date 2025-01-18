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
   * \brief Perform RVSDG transformation
   *
   * \note This method is expected to be called multiple times. An
   * implementation is required to reset the objects' internal state
   * to ensure correct behavior after every invocation.
   *
   * \param module RVSDG module the transformation is performed on.
   * \param statisticsCollector Statistics collector for collecting transformation statistics.
   */
  virtual void
  Run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector) = 0;

  /**
   * \brief Perform RVSDG transformation
   *
   * \note This method is expected to be called multiple times. An
   * implementation is required to reset the objects' internal state
   * to ensure correct behavior after every invocation.
   *
   * @param module RVSDG module the transformation is performed on.
   */
  void
  Run(RvsdgModule & module)
  {
    util::StatisticsCollector statisticsCollector;
    Run(module, statisticsCollector);
  }
};

/**
 * Sequentially applies a list of RVSDG transformations.
 */
class TransformationSequence final : public Transformation
{
  class Statistics;

public:

  ~TransformationSequence() noexcept override;

  explicit TransformationSequence(std::vector<Transformation *> transformations)
      : Transformations_(std::move(transformations))
  {}

  /**
   * \brief Perform RVSDG transformations
   *
   * @param rvsdgModule RVSDG module the transformation is performed on.
   * @param statisticsCollector Statistics collector for collecting transformation statistics.
   */
  void
  Run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  /**
   * \brief Creates a transformation sequence and invokes its Run() method.
   *
   * @param rvsdgModule RVSDG module the transformation is performed on.
   * @param statisticsCollector Statistics collector for collecting transformation statistics.
   * @param transformations The transformations that are sequentially applied to \p rvsdgModule.
   */
  static void
  CreateAndRun(
      RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      std::vector<Transformation *> transformations)
  {
    TransformationSequence sequentialApplication(std::move(transformations));
    sequentialApplication.Run(rvsdgModule, statisticsCollector);
  }

private:
  std::vector<Transformation *> Transformations_;
};

}

#endif // JLM_RVSDG_TRANSFORMATION_HPP
