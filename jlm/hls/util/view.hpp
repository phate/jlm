/*
 * Copyright 2024 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_UTIL_VIEW_HPP
#define JLM_BACKEND_HLS_UTIL_VIEW_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/Statistics.hpp>
#include <string>

namespace jlm::hls
{

enum ViewColors
{
  NONE,
  BLACK,
  RED
};

std::string
ViewcolorToString(const ViewColors & color);

std::string
RegionToDot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::Output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::Input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::Output *, ViewColors> & tailLabel);

std::string
ToDot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::Output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::Input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::Output *, ViewColors> & tailLabel);

void
ViewDot(rvsdg::Region * region, FILE * out);

void
ViewDot(
    rvsdg::Region * region,
    FILE * out,
    std::unordered_map<rvsdg::Output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::Input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::Output *, ViewColors> & tailLabel);

void
DumpDot(
    llvm::RvsdgModule & rvsdgModule,
    const std::string & fileName,
    std::unordered_map<rvsdg::Output *, ViewColors> outputColor,
    std::unordered_map<rvsdg::Input *, ViewColors> inputColor,
    std::unordered_map<rvsdg::Output *, ViewColors> tailLabel);
void

DumpDot(llvm::RvsdgModule & rvsdgModule, const std::string & fileName);

void
DumpDot(
    rvsdg::Region * region,
    const std::string & fileName,
    std::unordered_map<rvsdg::Output *, ViewColors> outputColor,
    std::unordered_map<rvsdg::Input *, ViewColors> inputColor,
    std::unordered_map<rvsdg::Output *, ViewColors> tailLabel);
void
DumpDot(rvsdg::Region * region, const std::string & fileName);

void
DotToSvg(const std::string & fileName);

/**
 * This transformation does nothing except dumping the RVSDG module to a dot file,
 * using the hls dot output.
 */
class DumpDotTransformation final : public rvsdg::Transformation
{
public:
  ~DumpDotTransformation() noexcept override;

  DumpDotTransformation();

  DumpDotTransformation(const DumpDotTransformation &) = delete;

  DumpDotTransformation &
  operator=(const DumpDotTransformation &) = delete;

  /**
   * Dumps the given \p rvsdgModule to an GraphViz dot file.
   * The file is placed in the output folder of the \p statisticsCollector.
   * @param rvsdgModule the module to dump
   * @param statisticsCollector the statistics collector whose output folder is used
   */
  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    DumpDotTransformation dumpDot;
    dumpDot.Run(rvsdgModule, statisticsCollector);
  }
};

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_UTIL_VIEW_HPP
