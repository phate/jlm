/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_DUMPTRANSFORMATION_HPP
#define JLM_HLS_OPT_DUMPTRANSFORMATION_HPP

#include <jlm/hls/HlsDotWriter.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <string>
#include <vector>

namespace jlm::hls
{

/** \brief RVSDG dump transformation that writes the graph in various formats after a pass.
 *
 * This is a reusable transformation that can be interleaved in a transformation sequence
 * alongside regular optimization passes. After each Run() call, it captures the current state
 * of the RVSDG module and writes one or more files to the statistics collector's output
 * directory. The filename uses a sequential number prefix (e.g. "003-") combined with a label
 * set via setLabel() so that output files read like
 * "003-dead-node-elimination.dot".
 *
 * Construct with Disabled{} to create a zero-cost no-op that can stand in for an enabled dump
 * transformation — Run() returns immediately. Use this when the same sequence must compile with
 * or without debugging dumps, toggled by a constructor argument at setup time.
 */
class DumpTransformation final : public rvsdg::Transformation
{
public:
  /** Tag type for creating a disabled (no-op) instance. */
  struct DisabledTag
  {
  };

  static constexpr DisabledTag
      Disabled{}; /**< Convenience constant, e.g. DumpTransformation{Disabled} */

  /** Supported output formats. Each corresponds to an existing rendering backend. */
  enum class OutputFormat
  {
    Dot,           /**< GraphViz DOT (uses LlvmDotWriter + util::graph::Writer) */
    HlsDot,        /**< GraphViz DOT with HLS type annotations (HlsDotWriter) */
    StructuralDot, /**< RVSDG structural node ports (ToDot from view.hpp) */
    Json,          /**< Structured JSON per-region graph (LlvmDotWriter + json) */
    Ascii,         /**< Human-readable indented text (rvsdg::view) */
    Tree,          /**< Annotated tree view (rvsdg::Region::ToTree with annotations) */
    JsonTree       /**< Hierarchical JSON tree (rvsdg::Region::toJson) */
  };

  /** Lightweight configuration passed to the constructor. */
  struct Config
  {
    bool recursive = true;             /**< Include subregions (for Dot/Json only). */
    std::vector<OutputFormat> formats; /**< Formats to produce in a single Run(). */
  };

  ~DumpTransformation() noexcept override;

  /** Construct an enabled dump transformation with the given configuration. */
  explicit DumpTransformation(Config config);

  /** Construct a disabled (no-op) dump transformation. */
  explicit DumpTransformation(DisabledTag);

  DumpTransformation(const DumpTransformation &) = delete;
  DumpTransformation &
  operator=(const DumpTransformation &) = delete;

  /**
   * Set the label used in output filenames.
   *
   * The filename produced by Run() is: "{pass_number}-{label}.{ext}"
   * e.g. "003-dead-node-elimination.dot"
   *
   * Call this before each transformation so that the label reflects
   * the preceding transformation's name (i.e., what you're dumping after).
   */
  void
  setLabel(std::string label) noexcept
  {
    label_ = std::move(label);
  }

  /** Set the sequence number used in output filenames. */
  void
  setPassNumber(size_t num) noexcept
  {
    passNumber_ = num;
  }

  /** Add an output format to the list of formats produced by each Run() call. */
  void
  addFormat(OutputFormat format) noexcept
  {
    Config_.formats.push_back(format);
  }

  /**
   * Dump the RVSDG module in all configured formats.
   * Files are placed in the output directory of statisticsCollector,
   * with names like "[module]-[unique]-005-{label}.{ext}".
   */
  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  /** Output Dot, HlsDot, or Json via writer + util::graph::Writer. */
  void
  outputAsGraph(const rvsdg::Region & region, OutputFormat format, std::ofstream & out);

  /** Output HLS-structural DOT with default colors (ToDot from view.hpp). */
  static void
  outputAsStructuralDot(const rvsdg::Region & region, std::ofstream & out);

  /** Output human-readable ASCII tree via rvsdg::view(). */
  static void
  outputAsAscii(const rvsdg::Region & region, std::ofstream & out);

  /** Output annotated tree view (all annotation types are always applied). */
  static void
  outputAsTree(const rvsdg::Graph & graph, const rvsdg::Region & region, std::ofstream & out);

  /** Output hierarchical JSON tree. */
  static void
  outputAsJsonTree(const rvsdg::Region & region, std::ofstream & out);

  Config Config_;
  bool enabled_ = true;
  std::string label_ = "dump"; /**< Default label — set via setLabel() before Run(). */
  size_t passNumber_ = 0;
};

} // namespace jlm::hls

#endif
