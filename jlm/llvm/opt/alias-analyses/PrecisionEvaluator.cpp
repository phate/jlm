/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/dot/DotWriter.hpp>
#include <jlm/llvm/opt/alias-analyses/PrecisionEvaluator.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/GraphWriter.hpp>

#include <fstream>
#include <map>

namespace jlm::llvm::aa
{

/**
 * Controls if duplicated pointers should be removed before making alias queries.
 * If true, behavior is closer to LLVM's alias evaluation,
 * but arguably says less about the actual memory operations in the function.
 */
static constexpr bool RemoveDuplicatePointers = false;

static constexpr bool OutputAliasingGraph = false;

static constexpr auto DefaultMode = PrecisionEvaluator::Mode::ClobberingStores;

std::string_view
PrecisionEvaluationModeToString(PrecisionEvaluator::Mode mode)
{
  switch (mode)
  {
  case PrecisionEvaluator::Mode::ClobberingStores:
    return "ClobberingStores";
  case PrecisionEvaluator::Mode::AllLoadStorePairs:
    return "AllLoadStorePairs";
  default:
    JLM_UNREACHABLE("Unknown precision evaluation mode");
  }
}

class PrecisionEvaluator::PrecisionStatistics final : public util::Statistics
{
  // This statistic places additional information in a separate file. This is the path of the file.
  static constexpr auto PrecisionEvaluationMode_ = "PrecisionEvaluationMode";
  static constexpr auto PairwiseAliasAnalysisType_ = "PairwiseAliasAnalysisType";
  static constexpr auto IsRemovingDuplicatePointers_ = "IsRemovingDuplicatePointers";
  static constexpr auto PrecisionDumpFile_ = "DumpFile";
  static constexpr auto ModuleNumClobbers_ = "ModuleNumClobbers";
  static constexpr auto ClobberAverageNoAlias = "ClobberAverageNoAlias";
  static constexpr auto ClobberAverageMayAlias = "ClobberAverageMayAlias";
  static constexpr auto ClobberAverageMustAlias = "ClobberAverageMustAlias";
  static constexpr auto NumTotalNoAlias_ = "#TotalNoAlias";
  static constexpr auto NumTotalMayAlias_ = "#TotalMayAlias";
  static constexpr auto NumTotalMustAlias_ = "#TotalMustAlias";

  static constexpr auto PrecisionEvaluationTimer_ = "PrecisionEvaluationTimer";

public:
  ~PrecisionStatistics() override = default;

  explicit PrecisionStatistics(const util::filepath & sourceFile)
      : Statistics(Id::AliasAnalysisPrecisionEvaluation, sourceFile)
  {}

  void
  StartEvaluatingPrecision(PrecisionEvaluator::Mode mode, AliasAnalysis & aliasAnalysis)
  {
    AddTimer(PrecisionEvaluationTimer_).start();
    AddMeasurement(PrecisionEvaluationMode_, std::string(PrecisionEvaluationModeToString(mode)));
    AddMeasurement(PairwiseAliasAnalysisType_, aliasAnalysis.ToString());
    AddMeasurement(IsRemovingDuplicatePointers_, RemoveDuplicatePointers ? "true" : "false");
  }

  void
  StopEvaluatingPrecision()
  {
    GetTimer(PrecisionEvaluationTimer_).stop();
  }

  void
  AddPrecisionSummaryStatistics(
      const util::filepath & outputFile,
      uint64_t moduleNumClobbers,
      double clobberAverageNoAlias,
      double clobberAverageMayAlias,
      double clobberAverageMustAlias,
      uint64_t totalNoAlias,
      uint64_t totalMayAlias,
      uint64_t totalMustAlias)
  {
    AddMeasurement(PrecisionDumpFile_, outputFile.to_str());
    AddMeasurement(ModuleNumClobbers_, moduleNumClobbers);
    AddMeasurement(ClobberAverageNoAlias, clobberAverageNoAlias);
    AddMeasurement(ClobberAverageMayAlias, clobberAverageMayAlias);
    AddMeasurement(ClobberAverageMustAlias, clobberAverageMustAlias);
    AddMeasurement(NumTotalNoAlias_, totalNoAlias);
    AddMeasurement(NumTotalMayAlias_, totalMayAlias);
    AddMeasurement(NumTotalMustAlias_, totalMustAlias);
  }

  static std::unique_ptr<PrecisionStatistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<PrecisionStatistics>(sourceFile);
  }
};

PrecisionEvaluator::PrecisionEvaluator()
    : Mode_(DefaultMode)
{}

void
PrecisionEvaluator::EvaluateAliasAnalysisClient(
    const rvsdg::RvsdgModule & rvsdgModule,
    AliasAnalysis & aliasAnalysis,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = PrecisionStatistics::Create(rvsdgModule.SourceFilePath().value());

  // If a precision evaluation is not demanded, skip doing it
  if (!statisticsCollector.IsDemanded(*statistics))
    return;

  Context_ = Context{};

  util::GraphWriter gw;
  if (OutputAliasingGraph)
  {
    AliasingGraph_ = &gw.CreateGraph();
    dot::WriteGraphs(gw, rvsdgModule.Rvsdg().GetRootRegion(), true);
  }

  statistics->StartEvaluatingPrecision(Mode_, aliasAnalysis);

  EvaluateAllFunctions(rvsdgModule.Rvsdg().GetRootRegion(), aliasAnalysis);

  statistics->StopEvaluatingPrecision();

  // Calculate average precision in functions and for the whole module, and print to output
  const auto outputFile = statisticsCollector.CreateOutputFile("AAPrecisionEvaluation.log", true);
  CalculateAverageMayAliasRate(outputFile, *statistics);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  if (OutputAliasingGraph)
  {
    auto out = statisticsCollector.CreateOutputFile("AAGraph.dot", true);
    std::ofstream fd(out.path().to_str());
    gw.OutputAllGraphs(fd, util::GraphOutputFormat::Dot);
  }
}

void
PrecisionEvaluator::EvaluateAllFunctions(
    const rvsdg::Region & region,
    AliasAnalysis & aliasAnalysis)
{
  for (auto & node : region.Nodes())
  {
    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(&node))
    {
      EvaluateFunction(*lambda, aliasAnalysis);
    }
    else if (auto structural = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structural->nsubregions(); n++)
      {
        EvaluateAllFunctions(*structural->subregion(n), aliasAnalysis);
      }
    }
  }
}

void
PrecisionEvaluator::EvaluateFunction(
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

  if (RemoveDuplicatePointers)
    RemoveDuplicates();

  // Create a PrecisionInfo instance for this function
  auto & precisionEvaluation = Context_.PerFunctionPrecision[&function];

  // Go over all pointer usages, find the ratio of clobbering points that may alias with it
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto [llvmInst1, p1, s1, p1IsUse, p1IsClobber] = Context_.PointerOperations[i];

    precisionEvaluation.NumOperations++;
    precisionEvaluation.NumUseOperations += p1IsUse;

    if (!p1IsClobber)
      continue;

    PrecisionInfo::ClobberInfo clobberInfo;

    for (size_t j = 0; j < Context_.PointerOperations.size(); j++)
    {
      if (i == j)
        continue;

      auto [llvmInst2, p2, s2, p2IsUse, p2IsClobber] = Context_.PointerOperations[j];
      if (!p2IsUse)
        continue;

      auto response = aliasAnalysis.Query(llvmInst1, *p1, s1, llvmInst2, *p2, s2);

      // Queries should always be symmetric, so double check that in debug builds
      // TODO: Disabled when using LLVM, as they are not always symmetric (see bugs/no-must-alias)
      // JLM_ASSERT(response == aliasAnalysis.Query(llvmInst2, *p2, s2, llvmInst1, *p1, s1));

      // Add edge to aliasing graph if requested
      if (OutputAliasingGraph && p1 < p2)
      {
        // Create a node associated with the given output
        // but also attach it to the GraphElement that already represents it
        const auto GetAliasGraphNode = [&](const rvsdg::output & p) -> util::Node &
        {
          const auto element = AliasingGraph_->GetElementFromProgramObject(p);
          const auto node = dynamic_cast<util::Node *>(element);
          if (node)
            return *node;

          auto & newNode = AliasingGraph_->CreateNode();
          auto existingElement = AliasingGraph_->GetGraphWriter().GetElementFromProgramObject(p);
          newNode.SetAttributeGraphElement("output", *existingElement);
          newNode.SetProgramObject(p);
          return newNode;
        };

        auto & p1Node = GetAliasGraphNode(*p1);
        auto & p2Node = GetAliasGraphNode(*p2);

        std::optional<std::string> edgeColor;
        if (response == AliasAnalysis::MayAlias)
          edgeColor = util::Colors::Purple;
        else if (response == AliasAnalysis::MustAlias)
          edgeColor = util::Colors::Orange;

        if (edgeColor)
        {
          auto & edge = AliasingGraph_->CreateEdge(p1Node, p2Node, false);
          edge.SetAttribute("color", *edgeColor);
        }
      }

      if (response == AliasAnalysis::NoAlias)
        clobberInfo.NumNoAlias++;
      else if (response == AliasAnalysis::MayAlias)
        clobberInfo.NumMayAlias++;
      else if (response == AliasAnalysis::MustAlias)
        clobberInfo.NumMustAlias++;
      else
        JLM_UNREACHABLE("Unknown AliasAnalysis query response");
    }

    precisionEvaluation.ClobberOperations.push_back(clobberInfo);
  }
}

void
PrecisionEvaluator::CollectPointersFromRegion(const rvsdg::Region & region)
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
PrecisionEvaluator::CollectPointersFromSimpleNode(const rvsdg::SimpleNode & node)
{
  bool loadsClobber;
  if (Mode_ == Mode::AllLoadStorePairs)
    loadsClobber = true;
  else if (Mode_ == Mode::ClobberingStores)
    loadsClobber = false;
  else
    JLM_UNREACHABLE("Unknown mode");

  // In this mode, only (volatile) load and store operations count as uses and clobbers
  if (const auto load = dynamic_cast<const LoadOperation *>(&node.GetOperation()))
  {
    const auto size = GetLlvmTypeSize(*load->GetLoadedType());
    CollectPointer(
        load->GetLlvmLoad(),
        LoadOperation::AddressInput(node).origin(),
        size,
        true,
        loadsClobber);
  }
  else if (auto store = dynamic_cast<const StoreOperation *>(&node.GetOperation()))
  {
    const auto size = GetLlvmTypeSize(store->GetStoredType());
    CollectPointer(
        store->GetLlvmStore(),
        StoreOperation::AddressInput(node).origin(),
        size,
        true,
        true);
  }
}

void
PrecisionEvaluator::CollectPointersFromStructuralNode(const rvsdg::StructuralNode & node)
{
  for (size_t n = 0; n < node.nsubregions(); n++)
  {
    CollectPointersFromRegion(*node.subregion(n));
  }
}

void
PrecisionEvaluator::CollectPointer(
    ::llvm::Instruction * llvmInst,
    const rvsdg::output * value,
    size_t size,
    bool isUse,
    bool isClobber)
{
  JLM_ASSERT(IsPointerCompatible(*value));
  Context_.PointerOperations.push_back({ llvmInst, value, size, isUse, isClobber });
}

void
PrecisionEvaluator::NormalizePointerValues()
{
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto & pointer = std::get<1>(Context_.PointerOperations[i]);
    pointer = &NormalizePointerValue(*pointer);
  }
}

void
PrecisionEvaluator::RemoveDuplicates()
{
  // For each occurrence of a (pointer, size) pair, perform logical or to find the final isUse and
  // isClobber values
  std::map<std::pair<const rvsdg::output *, size_t>, std::tuple<::llvm::Instruction *, bool, bool>>
      uniquePointerOps;

  for (const auto & [llvmInst, pointer, size, isUse, isClobber] : Context_.PointerOperations)
  {
    auto & op = uniquePointerOps[{ pointer, size }];
    const auto [_, wasUse, wasClobber] = op;
    op = { llvmInst, isUse | wasUse, isClobber | wasClobber };
  }

  Context_.PointerOperations.clear();
  for (const auto & [pointerSize, op] : uniquePointerOps)
  {
    const auto [pointer, size] = pointerSize;
    const auto [llvmInst, isUse, isClobber] = op;
    Context_.PointerOperations.push_back({ llvmInst, pointer, size, isUse, isClobber });
  }
}

void
PrecisionEvaluator::AggregateClobberInfos(
    std::vector<PrecisionInfo::ClobberInfo> & clobberInfos,
    double & clobberAverageNoAlias,
    double & clobberAverageMayAlias,
    double & clobberAverageMustAlias,
    uint64_t & totalNoAlias,
    uint64_t & totalMayAlias,
    uint64_t & totalMustAlias)
{
  clobberAverageNoAlias = 0.0;
  clobberAverageMayAlias = 0.0;
  clobberAverageMustAlias = 0.0;
  totalNoAlias = 0;
  totalMayAlias = 0;
  totalMustAlias = 0;

  for (auto & clobber : clobberInfos)
  {
    size_t total = clobber.NumNoAlias + clobber.NumMayAlias + clobber.NumMustAlias;
    if (total == 0)
      continue;

    clobberAverageNoAlias += static_cast<double>(clobber.NumNoAlias) / total;
    clobberAverageMayAlias += static_cast<double>(clobber.NumMayAlias) / total;
    clobberAverageMustAlias += static_cast<double>(clobber.NumMustAlias) / total;
    totalNoAlias += clobber.NumNoAlias;
    totalMayAlias += clobber.NumMayAlias;
    totalMustAlias += clobber.NumMustAlias;
  }

  clobberAverageNoAlias /= clobberInfos.size();
  clobberAverageMayAlias /= clobberInfos.size();
  clobberAverageMustAlias /= clobberInfos.size();
}

void
PrecisionEvaluator::CalculateAverageMayAliasRate(
    const util::file & outputFile,
    PrecisionStatistics & statistics) const
{
  std::vector<PrecisionInfo::ClobberInfo> allClobberInfo;

  double clobberAverageNoAlias, clobberAverageMayAlias, clobberAverageMustAlias;
  uint64_t totalNoAlias, totalMayAlias, totalMustAlias;

  // Write precision info about each function to an output file
  std::ofstream out(outputFile.path().to_str());
  for (auto [function, precision] : Context_.PerFunctionPrecision)
  {
    for (auto & clobberInfo : precision.ClobberOperations)
      allClobberInfo.push_back(clobberInfo);

    AggregateClobberInfos(
        precision.ClobberOperations,
        clobberAverageNoAlias,
        clobberAverageMayAlias,
        clobberAverageMustAlias,
        totalNoAlias,
        totalMayAlias,
        totalMustAlias);

    out << function->GetOperation().debug_string() << " [";
    out << precision.NumOperations << " pointer operations: ";
    out << precision.NumUseOperations << " use operations, ";
    out << precision.ClobberOperations.size() << " clobbering operations]:" << std::endl;
    out << "The average clobber has: " << (clobberAverageNoAlias * 100) << " % NoAlias \t";
    out << (clobberAverageMayAlias * 100) << " % MayAlias \t";
    out << (clobberAverageMustAlias * 100) << " % MustAlias" << std::endl;
    out << "Total responses: " << totalNoAlias << " NoAlias \t";
    out << totalMayAlias << " MayAlias \t";
    out << totalMustAlias << " MustAlias" << std::endl;
    out << std::endl;
  }

  AggregateClobberInfos(
      allClobberInfo,
      clobberAverageNoAlias,
      clobberAverageMayAlias,
      clobberAverageMustAlias,
      totalNoAlias,
      totalMayAlias,
      totalMustAlias);

  out << "Module total:" << std::endl;
  out << "The average clobber has: " << (clobberAverageNoAlias * 100) << " % NoAlias \t";
  out << (clobberAverageMayAlias * 100) << " % MayAlias \t";
  out << (clobberAverageMustAlias * 100) << " % MustAlias" << std::endl;
  out << "Total responses: " << totalNoAlias << " NoAlias \t";
  out << totalMayAlias << " MayAlias \t";
  out << totalMustAlias << " MustAlias" << std::endl;
  out << std::endl;

  out.close();

  statistics.AddPrecisionSummaryStatistics(
      outputFile.path(),
      allClobberInfo.size(),
      clobberAverageNoAlias,
      clobberAverageMayAlias,
      clobberAverageMustAlias,
      totalNoAlias,
      totalMayAlias,
      totalMustAlias);
}

}
