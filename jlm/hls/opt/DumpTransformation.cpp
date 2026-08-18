/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/opt/DumpTransformation.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::hls
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string_view
format_to_ext(DumpTransformation::OutputFormat format)
{
  using OF = DumpTransformation::OutputFormat;
  switch (format)
  {
  case OF::Dot:
    return "llvm.dot";
  case OF::HlsDot:
    return "hls.dot";
  case OF::StructuralDot:
    return "structural.dot";
  case OF::Json:
    return "json";
  case OF::Ascii:
    return "txt";
  case OF::Tree:
    return "tree.txt";
  case OF::JsonTree:
    return "tree.json";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DumpTransformation::~DumpTransformation() noexcept = default;

DumpTransformation::DumpTransformation(Config config)
    : Transformation("DumpTransformation"),
      Config_(std::move(config))
{}

DumpTransformation::DumpTransformation(DisabledTag)
    : Transformation("DumpTransformation"),
      enabled_(false)
{}

// ---------------------------------------------------------------------------
// Public Run
// ---------------------------------------------------------------------------

void
DumpTransformation::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  // Disabled variant returns immediately — zero overhead.
  if (!enabled_)
    return;

  const auto * graph = &rvsdgModule.Rvsdg();
  const auto & rootRegion = graph->GetRootRegion();

  // Build the shared prefix once: "{ModuleName}-{UniqueString}"
  std::string prefix;
  if (!statisticsCollector.GetSettings().GetModuleName().empty())
    prefix += statisticsCollector.GetSettings().GetModuleName() + "-";
  if (!statisticsCollector.GetSettings().GetUniqueString().empty())
    prefix += statisticsCollector.GetSettings().GetUniqueString() + "-";

  // Generate the base name so all formats share one unique string,
  // e.g. "jhls-bicg-AaG8EE-003-loop-unswitching"
  std::ostringstream base;
  base << prefix << std::setw(3) << std::setfill('0') << passNumber_++ << "-" << label_;

  for (auto format : Config_.formats)
  {
    auto fileName = base.str() + "." + std::string(format_to_ext(format));

    // Write directly to avoid createOutputFile generating a new unique string.
    // We still use createOutputFile once to validate the output directory exists,
    // then write files manually since all extensions are distinct now.
    std::ofstream fs(fileName);
    if (!fs)
      throw util::Error("DumpTransformation: failed to open file " + fileName);

    switch (format)
    {
    case OutputFormat::Dot:
    case OutputFormat::HlsDot:
    case OutputFormat::Json:
      outputAsGraph(rootRegion, format, fs);
      break;
    case OutputFormat::StructuralDot:
      outputAsStructuralDot(
          rootRegion,
          Config_.outputColor,
          Config_.inputColor,
          Config_.tailLabel,
          fs);
      break;
    case OutputFormat::Ascii:
      outputAsAscii(rootRegion, fs);
      break;
    case OutputFormat::Tree:
      outputAsTree(*graph, rootRegion, fs);
      break;
    case OutputFormat::JsonTree:
      outputAsJsonTree(rootRegion, fs);
      break;
    default:
      throw util::Error("DumpTransformation: unhandled output format");
    }
  }
}

// ---------------------------------------------------------------------------
// Graph-based formats (Dot / HlsDot / Json)
// ---------------------------------------------------------------------------

void
DumpTransformation::outputAsGraph(
    const rvsdg::Region & region,
    OutputFormat format,
    std::ofstream & out)
{
  util::graph::Writer writer;

  // Use HlsDotWriter for all DOT/JSON output — it extends LlvmDotWriter
  // with support for HLS types (TriggerType, BundleType). For plain Dot/HlsDot
  // the output is identical to what LlvmDotWriter would produce alone.
  HlsDotWriter hlsWriter;
  hlsWriter.WriteGraphs(writer, region, Config_.recursive);

  auto outFmt = (format == OutputFormat::Json) ? util::graph::OutputFormat::Json
                                               : util::graph::OutputFormat::Dot;

  writer.outputAllGraphs(out, outFmt);
}

// ---------------------------------------------------------------------------
// StructuralDot — HLS-structural rendering via ToDot() from view.hpp
// ---------------------------------------------------------------------------

void
DumpTransformation::outputAsStructuralDot(
    const rvsdg::Region & region,
    const std::unordered_map<rvsdg::Output *, ViewColors> & outputColor,
    const std::unordered_map<rvsdg::Input *, ViewColors> & inputColor,
    const std::unordered_map<rvsdg::Output *, ViewColors> & tailLabel,
    std::ofstream & out)
{
  auto dot = ToDot(
      const_cast<rvsdg::Region *>(&region),
      const_cast<std::unordered_map<rvsdg::Output *, ViewColors> &>(outputColor),
      const_cast<std::unordered_map<rvsdg::Input *, ViewColors> &>(inputColor),
      const_cast<std::unordered_map<rvsdg::Output *, ViewColors> &>(tailLabel));

  out << dot;
}

// ---------------------------------------------------------------------------
// ASCII format — reuses hls::view(region) to produce a human-readable indented text string.
// ---------------------------------------------------------------------------

void
DumpTransformation::outputAsAscii(const rvsdg::Region & region, std::ofstream & out)
{
  auto ascii = view(&region);
  out << ascii;
}

// ---------------------------------------------------------------------------
// Tree format — reuses Region::ToTree(region, annotationMap)
// ---------------------------------------------------------------------------

void
DumpTransformation::outputAsTree(
    const rvsdg::Graph & graph,
    const rvsdg::Region & region,
    std::ofstream & out)
{
  auto tree = llvm::RvsdgTreePrinter::RenderAnnotatedTree(graph, region);

  out << tree;
}

// ---------------------------------------------------------------------------
// JSON Tree format — reuses Region::toJson(region)
// ---------------------------------------------------------------------------

void
DumpTransformation::outputAsJsonTree(const rvsdg::Region & region, std::ofstream & out)
{
  auto json = rvsdg::Region::toJson(region);

  out << json;
}

} // namespace jlm::hls
