/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP
#define JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP

#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/AnnotationMap.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::rvsdg
{
class Graph;
class input;
class output;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

/** \brief RVSDG tree printer debug pass
 *
 * Prints an RVSDG tree to a file.
 */
class RvsdgTreePrinter final : public rvsdg::Transformation
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
       * Annotate region and structural nodes with the number of inputs/outputs of type
       * MemoryStateType.
       */
      NumMemoryStateInputsOutputs,

      /**
       * Must always be the last enum value. Used for iteration.
       */
      LastEnumValue
    };

    Configuration(util::HashSet<Annotation> requiredAnnotations)
        : RequiredAnnotations_(std::move(requiredAnnotations))
    {}

    /**
     * The required annotations for the RVSDG tree.
     */
    [[nodiscard]] const util::HashSet<Annotation> &
    RequiredAnnotations() const noexcept
    {
      return RequiredAnnotations_;
    }

  private:
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
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  /**
   * Computes a map with annotations based on the required \ref jlm::util::Annotation%s in the \ref
   * Configuration for the individual regions and structural nodes of the region tree.
   *
   * @param rvsdg The RVSDG for which to compute the annotations.
   * @return An instance of \ref AnnotationMap.
   */
  [[nodiscard]] util::AnnotationMap
  ComputeAnnotationMap(const rvsdg::Graph & rvsdg) const;

  /**
   * Adds an annotation to \p annotationMap that indicates the number of RVSDG nodes for regions
   * and structural nodes.
   *
   * @param rvsdg The RVSDG for which to compute the annotation.
   * @param annotationMap The annotation map in which the annotation is inserted.
   *
   * @see NumRvsdgNodes
   */
  static void
  AnnotateNumRvsdgNodes(const rvsdg::Graph & rvsdg, util::AnnotationMap & annotationMap);

  /**
   * Adds an annotation to \p annotationMap that indicates the number of inputs/outputs of type
   * MemoryStateType.
   *
   * @param rvsdg The RVSDG for which to compute the annotation.
   * @param annotationMap The annotation map in which the annotation is inserted.
   *
   * @see NumMemoryStateInputsOutputs
   */
  static void
  AnnotateNumMemoryStateInputsOutputs(
      const rvsdg::Graph & rvsdg,
      util::AnnotationMap & annotationMap);

  [[nodiscard]] static bool
  IsMemoryStateInput(const rvsdg::input * input) noexcept;

  [[nodiscard]] static bool
  IsMemoryStateOutput(const rvsdg::output * output) noexcept;

  Configuration Configuration_;
};

}

#endif
