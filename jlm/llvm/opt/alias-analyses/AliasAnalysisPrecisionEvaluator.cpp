/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysisPrecisionEvaluator.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/GraphWriter.hpp>

#include <fstream>
#include <map>

namespace jlm::llvm::aa
{

class AliasAnalysisPrecisionEvaluator::PrecisionStatistics final : public util::Statistics
{
  // The output from calling ToString on the AliasAnalysis
  static constexpr auto PairwiseAliasAnalysisType_ = "PairwiseAliasAnalysisType";

  static constexpr auto LoadsConsideredClobbers_ = "LoadsConsideredClobbers";
  static constexpr auto DeduplicatingPointers_ = "DeduplicatingPointers";
  // The path of the file where per-function statistics are written, if enabled
  static constexpr auto PerFunctionOutputFile_ = "PerFunctionOutputFile";
  // The path of the file where the aliasing graph is written, if enabled
  static constexpr auto AliasingGraphOutputFile_ = "AliasingGraphOutputFile";

  // The total number of clobbering operations considered by the precision evaluator.
  // If deduplication is enabled, only unique clobbers are counted.
  static constexpr auto ModuleNumClobbers_ = "ModuleNumClobbers";
  // The rate of each response type, for the average clobbering operations. Should sum up to 1
  static constexpr auto ClobberAverageNoAlias = "ClobberAverageNoAlias";
  static constexpr auto ClobberAverageMayAlias = "ClobberAverageMayAlias";
  static constexpr auto ClobberAverageMustAlias = "ClobberAverageMustAlias";
  // The total number of alias query responses given, of each kind
  static constexpr auto NumTotalNoAlias_ = "#TotalNoAlias";
  static constexpr auto NumTotalMayAlias_ = "#TotalMayAlias";
  static constexpr auto NumTotalMustAlias_ = "#TotalMustAlias";

  static constexpr auto PrecisionEvaluationTimer_ = "PrecisionEvaluationTimer";

public:
  ~PrecisionStatistics() override = default;

  explicit PrecisionStatistics(const util::FilePath & sourceFile)
      : Statistics(Id::AliasAnalysisPrecisionEvaluation, sourceFile)
  {}

  void
  StartEvaluatingPrecision(
      const AliasAnalysisPrecisionEvaluator & evaluator,
      const AliasAnalysis & aliasAnalysis)
  {
    AddTimer(PrecisionEvaluationTimer_).start();
    AddMeasurement(PairwiseAliasAnalysisType_, aliasAnalysis.ToString());
    AddMeasurement(
        LoadsConsideredClobbers_,
        evaluator.AreLoadsConsideredClobbers() ? "true" : "false");
    AddMeasurement(DeduplicatingPointers_, evaluator.IsDeduplicatingPointers() ? "true" : "false");
  }

  void
  StopEvaluatingPrecision()
  {
    GetTimer(PrecisionEvaluationTimer_).stop();
  }

  void
  AddPerFunctionOutputFile(const util::FilePath & outputFile)
  {
    AddMeasurement(PerFunctionOutputFile_, outputFile.to_str());
  }

  void
  AddAliasingGraphOutputFile(const util::FilePath & outputFile)
  {
    AddMeasurement(AliasingGraphOutputFile_, outputFile.to_str());
  }

  void
  AddPrecisionSummaryStatistics(const AggregatedClobberInfos & clobberInfos)
  {
    AddMeasurement(ModuleNumClobbers_, clobberInfos.NumClobberOperations);
    AddMeasurement(ClobberAverageNoAlias, clobberInfos.ClobberAverageNoAlias);
    AddMeasurement(ClobberAverageMayAlias, clobberInfos.ClobberAverageMayAlias);
    AddMeasurement(ClobberAverageMustAlias, clobberInfos.ClobberAverageMustAlias);
    AddMeasurement(NumTotalNoAlias_, clobberInfos.TotalNoAlias);
    AddMeasurement(NumTotalMayAlias_, clobberInfos.TotalMayAlias);
    AddMeasurement(NumTotalMustAlias_, clobberInfos.TotalMustAlias);
  }

  static std::unique_ptr<PrecisionStatistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<PrecisionStatistics>(sourceFile);
  }
};

AliasAnalysisPrecisionEvaluator::AliasAnalysisPrecisionEvaluator() = default;

AliasAnalysisPrecisionEvaluator::~AliasAnalysisPrecisionEvaluator() noexcept = default;

void
AliasAnalysisPrecisionEvaluator::EvaluateAliasAnalysisClient(
    const rvsdg::RvsdgModule & rvsdgModule,
    AliasAnalysis & aliasAnalysis,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = PrecisionStatistics::Create(rvsdgModule.SourceFilePath().value());

  // If precision evaluation statistics are not demanded, skip doing it
  if (!statisticsCollector.IsDemanded(*statistics))
    return;

  Context_ = Context{};

  // If creating an aliasing graph is enabled, initialize an empty graph
  util::graph::Writer gw;
  if (IsAliasingGraphEnabled())
  {
    Context_.AliasingGraph_ = &gw.CreateGraph();

    // Also emit the RVSDG to the graph writer
    LlvmDotWriter writer;
    writer.WriteGraphs(gw, rvsdgModule.Rvsdg().GetRootRegion(), true);
  }

  // Do the pairwise alias queries
  statistics->StartEvaluatingPrecision(*this, aliasAnalysis);
  EvaluateAllFunctions(rvsdgModule.Rvsdg().GetRootRegion(), aliasAnalysis);
  statistics->StopEvaluatingPrecision();

  // If an aliasing graph was constructed during the evaluation, print it out now
  if (IsAliasingGraphEnabled())
  {
    auto out = statisticsCollector.createOutputFile("AliasingGraph.dot", true);
    std::ofstream fd(out.path().to_str());
    gw.outputAllGraphs(fd, util::graph::OutputFormat::Dot);
    statistics->AddAliasingGraphOutputFile(out.path());
  }

  // If precision metrics are demanded per function, create the output file
  std::optional<util::FilePath> perFunctionOutputFile;
  if (IsPerFunctionOutputEnabled())
  {
    perFunctionOutputFile =
        statisticsCollector.createOutputFile("AAPrecisionEvaluation.log", true).path();
    statistics->AddPerFunctionOutputFile(*perFunctionOutputFile);
  }

  // Calculate total and average precision statistics
  CalculateResults(perFunctionOutputFile, *statistics);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
AliasAnalysisPrecisionEvaluator::EvaluateAllFunctions(
    const rvsdg::Region & region,
    AliasAnalysis & aliasAnalysis)
{
  for (auto & node : region.Nodes())
  {
    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(&node))
    {
      EvaluateFunction(*lambda, aliasAnalysis);
    }
    else if (auto phi = dynamic_cast<const rvsdg::PhiNode *>(&node))
    {
      EvaluateAllFunctions(*phi->subregion(), aliasAnalysis);
    }
  }
}

void
AliasAnalysisPrecisionEvaluator::EvaluateFunction(
    const rvsdg::LambdaNode & function,
    AliasAnalysis & aliasAnalysis)
{
  // Starting a new function, reset previously collected pointer operations
  Context_.PointerOperations.clear();

  // Collect all pointer uses and clobbers from this function
  CollectPointersFromRegion(*function.subregion());

  // Pointers are normalized, to prevent pointers that trivially originate
  // from the same output to be regarded as different.
  NormalizePointerValues();

  // If IsDeduplicatingPointers() is true, duplicate pointers are discarded
  // Even if deduplicating is disabled, duplicates are still grouped, but only for performance.
  AggregateDuplicates();

  // Create a PrecisionInfo instance for this function
  auto & precisionEvaluation = Context_.PerFunctionPrecision[&function];

  // Go over all pointer usages, find the ratio of clobbering points that may alias with it
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto [p1, s1, p1IsClobber, p1Multiplier] = Context_.PointerOperations[i];

    precisionEvaluation.NumOperations += p1Multiplier;

    if (!p1IsClobber)
      continue;

    precisionEvaluation.NumClobberOperations += p1Multiplier;

    PrecisionInfo::ClobberInfo clobberInfo;
    clobberInfo.Multiplier = p1Multiplier;

    // If p1 represents more than one concrete clobber operation,
    // add the result of querying against all the other clobber operations
    if (p1Multiplier > 1)
      clobberInfo.NumMustAlias += p1Multiplier - 1;

    for (size_t j = 0; j < Context_.PointerOperations.size(); j++)
    {
      if (i == j)
        continue;

      auto [p2, s2, p2IsClobber, p2Multiplier] = Context_.PointerOperations[j];

      auto response = aliasAnalysis.Query(*p1, s1, *p2, s2);

      // Queries should always be symmetric, so double check that in debug builds
      // Note: LLVM's own alias analyses are not always symmetric, but ours should be
      JLM_ASSERT(response == aliasAnalysis.Query(*p2, s2, *p1, s1));

      // Add edge to aliasing graph if dumping a graph of alias analysis response edges is requested
      if (IsAliasingGraphEnabled())
      {
        AddToAliasingGraph(*p1, s1, *p2, s2, response);
      }

      if (response == AliasAnalysis::NoAlias)
        clobberInfo.NumNoAlias += p2Multiplier;
      else if (response == AliasAnalysis::MayAlias)
        clobberInfo.NumMayAlias += p2Multiplier;
      else if (response == AliasAnalysis::MustAlias)
        clobberInfo.NumMustAlias += p2Multiplier;
      else
        JLM_UNREACHABLE("Unknown AliasAnalysis query response");
    }

    precisionEvaluation.ClobberOperations.push_back(clobberInfo);
  }
}

void
AliasAnalysisPrecisionEvaluator::CollectPointersFromRegion(const rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      CollectPointersFromSimpleNode(*simpleNode);
    }
    else if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      CollectPointersFromStructuralNode(*structuralNode);
    }
    else
    {
      JLM_UNREACHABLE("Unknown node type");
    }
  }
}

void
AliasAnalysisPrecisionEvaluator::CollectPointersFromSimpleNode(const rvsdg::SimpleNode & node)
{
  if (const auto load = dynamic_cast<const LoadOperation *>(&node.GetOperation()))
  {
    // At the time of writing, these are the only loads we know about
    JLM_ASSERT(is<LoadNonVolatileOperation>(*load) || is<LoadVolatileOperation>(*load));

    const auto size = GetTypeSize(*load->GetLoadedType());
    CollectPointer(LoadOperation::AddressInput(node).origin(), size, AreLoadsConsideredClobbers());
  }
  else if (auto store = dynamic_cast<const StoreOperation *>(&node.GetOperation()))
  {
    // At the time of writing, these are the only stores we know about
    JLM_ASSERT(is<StoreNonVolatileOperation>(*store) || is<StoreVolatileOperation>(*store));

    const auto size = GetTypeSize(store->GetStoredType());
    CollectPointer(StoreOperation::AddressInput(node).origin(), size, true);
  }
}

void
AliasAnalysisPrecisionEvaluator::CollectPointersFromStructuralNode(
    const rvsdg::StructuralNode & node)
{
  for (auto & subregion : node.Subregions())
  {
    CollectPointersFromRegion(subregion);
  }
}

void
AliasAnalysisPrecisionEvaluator::CollectPointer(
    const rvsdg::Output * value,
    size_t size,
    bool isClobber)
{
  JLM_ASSERT(IsPointerCompatible(*value));
  Context_.PointerOperations.push_back({ value, size, isClobber, 1 });
}

void
AliasAnalysisPrecisionEvaluator::NormalizePointerValues()
{
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto & pointer = std::get<0>(Context_.PointerOperations[i]);
    pointer = &llvm::traceOutput(*pointer);
  }
}

void
AliasAnalysisPrecisionEvaluator::AggregateDuplicates()
{
  // Add up the multipliers of all (pointer, size, isClobber) values
  std::map<std::tuple<const rvsdg::Output *, size_t, bool>, size_t> pointerOpMultipliers;

  for (const auto & [pointer, size, isClobber, multiplier] : Context_.PointerOperations)
  {
    pointerOpMultipliers[{ pointer, size, isClobber }] += multiplier;
  }

  Context_.PointerOperations.clear();
  for (const auto & [pointerOp, multiplier] : pointerOpMultipliers)
  {
    const auto [pointer, size, isClobber] = pointerOp;

    if (IsDeduplicatingPointers())
    {
      // If Deduplication is being done, set all multipliers to 1
      // Also, if the operation exists both as a clobber and not a clobber, only add the former
      if (!isClobber && pointerOpMultipliers.count({ pointer, size, true }) > 0)
        continue;
      Context_.PointerOperations.push_back({ pointer, size, isClobber, 1 });
    }
    else
    {
      // When deduplication is not enabled, use the sum of multipliers as the final multiplier
      Context_.PointerOperations.push_back({ pointer, size, isClobber, multiplier });
    }
  }
}

void
AliasAnalysisPrecisionEvaluator::AddToAliasingGraph(
    const rvsdg::Output & p1,
    size_t s1,
    const rvsdg::Output & p2,
    size_t s2,
    AliasAnalysis::AliasQueryResponse response)
{
  // Create a node associated with the given output
  // but also attach it to the GraphElement that already represents it
  const auto GetOrCreateAliasGraphNode = [&](const rvsdg::Output & p) -> util::graph::Node &
  {
    const auto element = Context_.AliasingGraph_->GetElementFromProgramObject(p);
    const auto node = dynamic_cast<util::graph::Node *>(element);
    if (node)
      return *node;

    auto & newNode = Context_.AliasingGraph_->CreateNode();
    auto existingElement = Context_.AliasingGraph_->GetWriter().GetElementFromProgramObject(p);
    newNode.SetAttributeGraphElement("output", *existingElement);
    newNode.SetProgramObject(p);
    return newNode;
  };

  auto & p1Node = GetOrCreateAliasGraphNode(p1);
  auto & p2Node = GetOrCreateAliasGraphNode(p2);

  // Only create edges for MayAlias and MustAlias
  std::optional<std::string> edgeColor;
  if (response == AliasAnalysis::MayAlias)
    edgeColor = util::graph::Colors::Purple;
  else if (response == AliasAnalysis::MustAlias)
    edgeColor = util::graph::Colors::Orange;

  if (edgeColor)
  {
    auto & edge = Context_.AliasingGraph_->CreateEdge(p1Node, p2Node, false);
    edge.SetAttribute("s1", util::strfmt(s1));
    edge.SetAttribute("s2", util::strfmt(s2));
    edge.SetAttribute("color", *edgeColor);
  }
}

AliasAnalysisPrecisionEvaluator::AggregatedClobberInfos
AliasAnalysisPrecisionEvaluator::AggregateClobberInfos(
    const std::vector<PrecisionInfo::ClobberInfo> & clobberInfos)
{
  AggregatedClobberInfos result;

  for (auto & clobber : clobberInfos)
  {
    size_t total = clobber.NumNoAlias + clobber.NumMayAlias + clobber.NumMustAlias;

    // Avoid division by 0 by skipping clobbers that are alone in their function
    if (total == 0)
      continue;

    result.NumClobberOperations += clobber.Multiplier;

    result.ClobberAverageNoAlias +=
        static_cast<double>(clobber.NumNoAlias) / total * clobber.Multiplier;
    result.ClobberAverageMayAlias +=
        static_cast<double>(clobber.NumMayAlias) / total * clobber.Multiplier;
    result.ClobberAverageMustAlias +=
        static_cast<double>(clobber.NumMustAlias) / total * clobber.Multiplier;
    result.TotalNoAlias += clobber.NumNoAlias * clobber.Multiplier;
    result.TotalMayAlias += clobber.NumMayAlias * clobber.Multiplier;
    result.TotalMustAlias += clobber.NumMustAlias * clobber.Multiplier;
  }

  if (result.NumClobberOperations > 0)
  {
    // Perform the final division to get the average across all clobbers
    result.ClobberAverageNoAlias /= result.NumClobberOperations;
    result.ClobberAverageMayAlias /= result.NumClobberOperations;
    ;
    result.ClobberAverageMustAlias /= result.NumClobberOperations;
  }

  return result;
}

void
AliasAnalysisPrecisionEvaluator::PrintAggregatedClobberInfos(
    const AggregatedClobberInfos & clobberInfos,
    std::ostream & out)
{
  if (clobberInfos.NumClobberOperations == 0)
  {
    out << "No clobber operations" << std::endl;
    return;
  }

  out << "Number of clobbering operations: " << clobberInfos.NumClobberOperations << std::endl;
  out << "The average clobber has: " << (clobberInfos.ClobberAverageNoAlias * 100)
      << " % NoAlias \t";
  out << (clobberInfos.ClobberAverageMayAlias * 100) << " % MayAlias \t";
  out << (clobberInfos.ClobberAverageMustAlias * 100) << " % MustAlias" << std::endl;
  out << "Total responses: " << clobberInfos.TotalNoAlias << " NoAlias \t";
  out << clobberInfos.TotalMayAlias << " MayAlias \t";
  out << clobberInfos.TotalMustAlias << " MustAlias" << std::endl;
  out << std::endl;
}

void
AliasAnalysisPrecisionEvaluator::CalculateResults(
    std::optional<util::FilePath> perFunctionOutputFile,
    PrecisionStatistics & statistics) const
{
  // Adds up the total set of clobber operations in the entire module
  std::vector<PrecisionInfo::ClobberInfo> allClobberInfos;

  // Write precision info about each function to an output file
  std::ofstream out;
  if (perFunctionOutputFile)
    out.open(perFunctionOutputFile->to_str());

  for (auto [function, precision] : Context_.PerFunctionPrecision)
  {
    // Calculate and print information about the clobbers in this function
    if (perFunctionOutputFile)
    {
      const auto aggregated = AggregateClobberInfos(precision.ClobberOperations);
      out << function->GetOperation().debug_string() << " [";
      out << precision.NumOperations << " pointer operations]:" << std::endl;
      PrintAggregatedClobberInfos(aggregated, out);
    }

    // Add the clobber operations to the list of all clobbers
    for (auto & clobberInfo : precision.ClobberOperations)
      allClobberInfos.push_back(clobberInfo);
  }

  const auto aggregated = AggregateClobberInfos(allClobberInfos);

  if (perFunctionOutputFile)
  {
    out << "Module total:" << std::endl;
    PrintAggregatedClobberInfos(aggregated, out);
    out.close();
  }

  statistics.AddPrecisionSummaryStatistics(aggregated);
}

}
