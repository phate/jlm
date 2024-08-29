/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP
#define JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/util/AnnotationMap.hpp>
#include <jlm/util/file.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::rvsdg
{
class graph;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

class RvsdgModule;

/** \brief RVSDG tree printer debug pass
 *
 * Prints an RVSDG tree to a file.
 */
class RvsdgTreePrinter final : public optimization
{
  class Statistics;

public:
  /**
   * Configuration for the \ref RvsdgTreePrinter.
   */
  class Configuration final
  {
  public:
    enum class Annotation
    {
      /**
       * Must always be the first enum value. Used for iteration.
       */
      FirstEnumValue,

      /**
       * Annotate regions and structural nodes with the number of RVSDG nodes.
       */
      NumRvsdgNodes,

      /**
       * Must always be the last enum value. Used for iteration.
       */
      LastEnumValue
    };

    Configuration(
        const util::filepath & outputDirectory,
        util::HashSet<Annotation> requiredAnnotations)
        : OutputDirectory_(std::move(outputDirectory)),
          RequiredAnnotations_(std::move(requiredAnnotations))
    {
      JLM_ASSERT(outputDirectory.IsDirectory());
      JLM_ASSERT(outputDirectory.Exists());
    }

    /**
     * The output directory for the RVSDG tree files.
     */
    [[nodiscard]] const util::filepath &
    OutputDirectory() const noexcept
    {
      return OutputDirectory_;
    }

    /**
     * The required annotations for the RVSDG tree.
     */
    [[nodiscard]] const util::HashSet<Annotation> &
    RequiredAnnotations() const noexcept
    {
      return RequiredAnnotations_;
    }

  private:
    util::filepath OutputDirectory_;
    util::HashSet<Annotation> RequiredAnnotations_ = {};
  };

  ~RvsdgTreePrinter() noexcept override;

  explicit RvsdgTreePrinter(Configuration configuration)
      : Configuration_(std::move(configuration))
  {}

  RvsdgTreePrinter(const RvsdgTreePrinter &) = delete;

  RvsdgTreePrinter(RvsdgTreePrinter &&) = delete;

  RvsdgTreePrinter &
  operator=(const RvsdgTreePrinter &) = delete;

  RvsdgTreePrinter &
  operator=(RvsdgTreePrinter &&) = delete;

  void
  run(RvsdgModule & rvsdgModule, jlm::util::StatisticsCollector & statisticsCollector) override;

  void
  run(RvsdgModule & rvsdgModule);

private:
  [[nodiscard]] util::AnnotationMap
  ComputeAnnotationMap(const rvsdg::graph & rvsdg) const;

  static void
  AnnotateNumRvsdgNodes(const rvsdg::graph & rvsdg, util::AnnotationMap & annotationMap);

  void
  WriteTreeToFile(const RvsdgModule & rvsdgModule, const std::string & tree) const;

  [[nodiscard]] util::file
  CreateOutputFile(const RvsdgModule & rvsdgModule) const;

  static uint64_t
  GetOutputFileNameCounter(const RvsdgModule & rvsdgModule);

  Configuration Configuration_;
};

}

#endif
