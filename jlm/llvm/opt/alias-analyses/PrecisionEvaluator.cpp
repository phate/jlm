/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "MemoryStateEncoder.hpp"
#include <jlm/llvm/opt/alias-analyses/PrecisionEvaluator.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>

#include <fstream>
#include <map>

namespace jlm::llvm::aa
{

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
  static constexpr auto PrecisionDumpFile_ = "DumpFile";
  static constexpr auto ModuleNumUseOperations_ = "ModuleNumUseOperations";
  static constexpr auto ModuleAverageMayAliasRate_ = "ModuleAverageMayAliasRate";
  static constexpr auto NumNoAlias_ = "#NoAlias";
  static constexpr auto NumMayAlias_ = "#MayAlias";
  static constexpr auto NumMustAlias_ = "#MustAlias";

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
  }

  void
  StopEvaluatingPrecision()
  {
    GetTimer(PrecisionEvaluationTimer_).stop();
  }

  void
  AddPrecisionSummaryStatistics(
      const util::filepath & outputFile,
      uint64_t moduleNumUseOperations,
      double moduleAverageMayAliasRate,
      uint64_t numNoAlias,
      uint64_t numMayAlias,
      uint64_t numMustAlias)
  {
    AddMeasurement(PrecisionDumpFile_, outputFile.to_str());
    AddMeasurement(ModuleNumUseOperations_, moduleNumUseOperations);
    AddMeasurement(ModuleAverageMayAliasRate_, moduleAverageMayAliasRate);
    AddMeasurement(NumNoAlias_, numNoAlias);
    AddMeasurement(NumMayAlias_, numMayAlias);
    AddMeasurement(NumMustAlias_, numMustAlias);
  }

  static std::unique_ptr<PrecisionStatistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<PrecisionStatistics>(sourceFile);
  }
};

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

  statistics->StartEvaluatingPrecision(Mode_, aliasAnalysis);

  EvaluateAllFunctions(rvsdgModule.Rvsdg().GetRootRegion(), aliasAnalysis);

  statistics->StopEvaluatingPrecision();

  // Calculate average precision in functions and for the whole module, and print to output
  const auto outputFile = statisticsCollector.CreateOutputFile("AAPrecisionEvaluation.log", true);
  CalculateAverageMayAliasRate(outputFile, *statistics);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
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

  // In order to get results more comparable with LLVM, duplicates are removed.
  // First, pointers are normalized, to prevent pointers that trivially originate
  // from the same output to be regarded as different.
  NormalizePointerValues();
  RemoveDuplicates();

  // Create a PrecisionInfo instance for this function
  auto & precisionEvaluation = Context_.PerFunctionPrecision[&function];

  // Go over all pointer usages, find the ratio of clobbering points that may alias with it
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto [p1, s1, p1IsUse, p1IsClobber] = Context_.PointerOperations[i];

    precisionEvaluation.NumOperations++;
    precisionEvaluation.NumClobberingOperations += p1IsClobber;

    if (!p1IsUse)
      continue;

    PrecisionInfo::UseInfo useInfo;

    for (size_t j = 0; j < Context_.PointerOperations.size(); j++)
    {
      if (i == j)
        continue;

      auto [p2, s2, p2IsUse, p2IsClobber] = Context_.PointerOperations[j];
      if (!p2IsClobber)
        continue;

      auto response = aliasAnalysis.Query(*p1, s1, *p2, s2);
      if (response == AliasAnalysis::NoAlias)
        useInfo.NumNoAlias++;
      else if (response == AliasAnalysis::MayAlias)
        useInfo.NumMayAlias++;
      else if (response == AliasAnalysis::MustAlias)
        useInfo.NumMustAlias++;
      else
        JLM_UNREACHABLE("Unknown AliasAnalysis query response");
    }

    precisionEvaluation.UseOperations.push_back(useInfo);
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
    CollectPointer(LoadOperation::AddressInput(node).origin(), size, true, loadsClobber);
  }
  else if (auto store = dynamic_cast<const StoreOperation *>(&node.GetOperation()))
  {
    const auto size = GetLlvmTypeSize(store->GetStoredType());
    CollectPointer(StoreOperation::AddressInput(node).origin(), size, true, true);
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
    const rvsdg::output * value,
    size_t size,
    bool isUse,
    bool isClobber)
{
  JLM_ASSERT(IsPointerCompatible(*value));
  Context_.PointerOperations.push_back({ value, size, isUse, isClobber });
}

void
PrecisionEvaluator::NormalizePointerValues()
{
  for (size_t i = 0; i < Context_.PointerOperations.size(); i++)
  {
    auto & pointer = std::get<0>(Context_.PointerOperations[i]);
    pointer = &NormalizePointerValue(*pointer);
  }
}

void
PrecisionEvaluator::RemoveDuplicates()
{
  // For each occurrence of a (pointer, size) pair, perform logical or to find the final isUse and
  // isClobber values
  std::map<std::pair<const rvsdg::output *, size_t>, std::pair<bool, bool>> uniquePointerOps;

  for (const auto & [pointer, size, isUse, isClobber] : Context_.PointerOperations)
  {
    auto & op = uniquePointerOps[{ pointer, size }];
    op.first |= isUse;
    op.second |= isClobber;
  }

  Context_.PointerOperations.clear();
  for (const auto & [pointerSize, isUseClobber] : uniquePointerOps)
  {
    Context_.PointerOperations.push_back(
        { pointerSize.first, pointerSize.second, isUseClobber.first, isUseClobber.second });
  }
}

void
PrecisionEvaluator::CalculateAverageMayAliasRate(
    const util::file & outputFile,
    PrecisionStatistics & statistics) const
{
  // Average may alias ratio among uses in the whole module
  size_t moduleNumUsages = 0;
  double moduleUseMayAliasRatioSum = 0.0;

  // As an alternative measurement, add up all alias query responses
  size_t moduleTotalNoAlias = 0;
  size_t moduleTotalMayAlias = 0;
  size_t moduleTotalMustAlias = 0;

  // Write precision info about each function to an output file
  std::ofstream out(outputFile.path().to_str());
  for (auto [function, precision] : Context_.PerFunctionPrecision)
  {
    // Average may alias ratio among uses in the function
    double functionUseMayAliasRatioSum = 0.0;
    // Only usages with at least one clobber are counted, to avoid division by 0
    size_t functionNumUsages = 0;

    size_t functionTotalNoAlias = 0;
    size_t functionTotalMayAlias = 0;
    size_t functionTotalMustAlias = 0;

    for (const auto & use : precision.UseOperations)
    {
      functionTotalNoAlias += use.NumNoAlias;
      functionTotalMayAlias += use.NumMayAlias;
      functionTotalMustAlias += use.NumMustAlias;

      const auto totalQueries = use.NumNoAlias + use.NumMayAlias + use.NumMustAlias;
      if (totalQueries == 0)
        continue;

      functionUseMayAliasRatioSum += static_cast<double>(use.NumMayAlias) / totalQueries;
      functionNumUsages++;
    }
    // Calculate the average may use ratio for uses in the function
    const auto functionAverageMayAliasRatio = functionUseMayAliasRatioSum / functionNumUsages;

    // Add function's contributions to module totals
    moduleNumUsages += functionNumUsages;
    moduleUseMayAliasRatioSum += functionUseMayAliasRatioSum;
    moduleTotalNoAlias += functionTotalNoAlias;
    moduleTotalMayAlias += functionTotalMayAlias;
    moduleTotalMustAlias += functionTotalMustAlias;

    out << function->GetOperation().debug_string() << " [";
    out << precision.NumOperations << " pointer operations: ";
    out << functionNumUsages << " use operations, ";
    out << precision.NumClobberingOperations << " clobbering operations]:" << std::endl;
    out << "Average use MayAlias rate: " << functionAverageMayAliasRatio << std::endl;
    out << "In total: " << functionTotalNoAlias << " NoAlias \t\t";
    out << functionTotalMayAlias << " MayAlias \t\t";
    out << functionTotalMustAlias << " MustAlias" << std::endl;
    out << std::endl;
  }

  // Calculate the module-wide average precision for each pointer use
  const auto moduleAverageMayAliasRatio = moduleUseMayAliasRatioSum / moduleNumUsages;
  out << std::endl; // Empty line before the final result
  out << "Module total: " << moduleNumUsages << " pointer usages. ";
  out << "Average use MayAlias rate: " << moduleAverageMayAliasRatio << std::endl;
  out << "In total: " << moduleTotalNoAlias << " NoAlias \t\t";
  out << moduleTotalMayAlias << " MayAlias \t\t";
  out << moduleTotalMustAlias << " MustAlias" << std::endl;
  out.close();

  statistics.AddPrecisionSummaryStatistics(
      outputFile.path(),
      moduleNumUsages,
      moduleAverageMayAliasRatio,
      moduleTotalNoAlias,
      moduleTotalMayAlias,
      moduleTotalMustAlias);
}

}
