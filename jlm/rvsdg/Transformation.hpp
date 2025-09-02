/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRANSFORMATION_HPP
#define JLM_RVSDG_TRANSFORMATION_HPP

#include <jlm/rvsdg/DotWriter.hpp>
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

  explicit Transformation(std::string_view Name)
      : Name_(Name)
  {}

  [[nodiscard]] const std::string_view &
  GetName() const noexcept
  {
    return Name_;
  }

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

private:
  std::string_view Name_;
};

/**
 * Sequentially applies a list of RVSDG transformations.
 */
class TransformationSequence final : public Transformation
{
  class Statistics;

public:
  ~TransformationSequence() noexcept override;

  explicit TransformationSequence(
      std::vector<Transformation *> transformations,
      DotWriter & dotWriter,
      const bool dumpRvsdgDotGraphs)
      : Transformation("TransformationSequence"),
        DotWriter_(dotWriter),
        DumpRvsdgDotGraphs_(dumpRvsdgDotGraphs),
        Transformations_(std::move(transformations))
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
   * @param dotWriter The DOT writer for dumping the RVSDG graphs.
   * @param dumpRvsdgDotGraphs Determines whether to dump the RVSDG graphs.
   */
  static void
  CreateAndRun(
      RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      std::vector<Transformation *> transformations,
      DotWriter & dotWriter,
      const bool dumpRvsdgDotGraphs)
  {
    TransformationSequence sequentialApplication(
        std::move(transformations),
        dotWriter,
        dumpRvsdgDotGraphs);
    sequentialApplication.Run(rvsdgModule, statisticsCollector);
  }

private:
  void
  DumpDotGraphs(
      RvsdgModule & rvsdgModule,
      const util::FilePath & filePath,
      const std::string & passName,
      size_t numPass) const;

  DotWriter & DotWriter_;
  bool DumpRvsdgDotGraphs_;
  std::vector<Transformation *> Transformations_;
};

}

#endif // JLM_RVSDG_TRANSFORMATION_HPP
