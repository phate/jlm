/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "MemoryStateEncoder.hpp"
#include <jlm/llvm/opt/alias-analyses/PrecisionEvaluator.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>

#include <fstream>

namespace jlm::llvm::aa
{
PairwiseAliasAnalysis::PairwiseAliasAnalysis() = default;

PairwiseAliasAnalysis::~PairwiseAliasAnalysis() = default;

bool
PairwiseAliasAnalysis::MayAlias(const rvsdg::output & p1, const rvsdg::output & p2)
{
  if (!MayAliasImpl(p1, p2))
    return false;

  // This analysis was unable to determine that p1 and p2 do not alias, do we have a backup?
  if (Backup_)
    return Backup_->MayAlias(p1, p2);

  return true;
}

std::string
PairwiseAliasAnalysis::ToString() const
{
  std::string result = ToStringImpl();
  if (Backup_)
    result = util::strfmt(result, "(", Backup_->ToString(), ")");
  return result;
}

PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
{}

PointsToGraphAliasAnalysis::~PointsToGraphAliasAnalysis() = default;

bool
PointsToGraphAliasAnalysis::MayAliasImpl(const rvsdg::output & p1, const rvsdg::output & p2)
{
  // Assume that all pointers actually exist in the PointsToGraph
  auto & p1RegisterNode = PointsToGraph_.GetRegisterNode(p1);
  auto & p2RegisterNode = PointsToGraph_.GetRegisterNode(p2);

  // If the registers are represented by the same node, they may alias
  if (&p1RegisterNode == &p2RegisterNode)
    return true;

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const auto & externalNode = PointsToGraph_.GetExternalMemoryNode();
  if (p1RegisterNode.HasTarget(externalNode) && p2RegisterNode.HasTarget(externalNode))
    return true;

  // Check if p1 and p2 share any target memory nodes
  for (auto & target : p1RegisterNode.Targets())
  {
    if (p2RegisterNode.HasTarget(target))
      return true;
  }

  return false;
}

std::string
PointsToGraphAliasAnalysis::ToStringImpl() const
{
  return "PointsToGraphAliasAnalysis";
}

std::string_view
PrecisionEvaluationModeToString(PrecisionEvaluationMode mode)
{
  switch (mode)
  {
  case PrecisionEvaluationMode::ClobberingStores:
    return "ClobberingStores";
  case PrecisionEvaluationMode::AllPointerPairs:
    return "AllPointerPairs";
  default:
    JLM_UNREACHABLE("Unknown precision evaluation mode");
  }
}

class PrecisionEvaluator::PrecisionStatistics final : public util::Statistics
{
  // This statistic places additional information in a separate file. This is the path of the file.
  static constexpr auto PrecisionEvaluationMode_ = "PrecisionEvaluationMode";
  static constexpr auto PairwiseAliasAnalysisType_ = "PairwiseAliasAnalysisType";
  static constexpr auto NumMayAliasQueries_ = "#MayAliasQueries";
  static constexpr auto PrecisionDumpFile_ = "DumpFile";
  static constexpr auto ModuleNumPointerUsages_ = "ModuleNumPointerUsages";
  static constexpr auto ModuleAverageMayAliasRate_ = "ModuleAverageMayAliasRate";

  static constexpr auto PrecisionEvaluationTimer_ = "PrecisionEvaluationTimer";

public:
  ~PrecisionStatistics() override = default;

  explicit PrecisionStatistics(const util::filepath & sourceFile)
      : Statistics(Id::AliasAnalysisPrecisionEvaluation, sourceFile)
  {}

  void
  StartEvaluatingPrecision(PrecisionEvaluationMode mode, PairwiseAliasAnalysis & aliasAnalysis)
  {
    AddTimer(PrecisionEvaluationTimer_).start();
    AddMeasurement(PrecisionEvaluationMode_, std::string(PrecisionEvaluationModeToString(mode)));
    AddMeasurement(PairwiseAliasAnalysisType_, aliasAnalysis.ToString());
  }

  void
  StopEvaluatingPrecision(uint64_t numMayAliasQueries)
  {
    GetTimer(PrecisionEvaluationTimer_).stop();
    AddMeasurement(NumMayAliasQueries_, numMayAliasQueries);
  }

  void
  AddPrecisionSummaryStatistics(
      const util::filepath & outputFile,
      uint64_t moduleNumPointerUsages,
      double moduleAverageMayAliasRate)
  {
    AddMeasurement(PrecisionDumpFile_, outputFile.to_str());
    AddMeasurement(ModuleNumPointerUsages_, moduleNumPointerUsages);
    AddMeasurement(ModuleAverageMayAliasRate_, moduleAverageMayAliasRate);
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
    PairwiseAliasAnalysis & aliasAnalysis,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = PrecisionStatistics::Create(rvsdgModule.SourceFilePath().value());

  // If a precision evaluation is not demanded, skip doing it
  if (!statisticsCollector.IsDemanded(*statistics))
    return;

  Context_ = Context{};

  statistics->StartEvaluatingPrecision(Mode_, aliasAnalysis);

  EvaluateAllFunctions(rvsdgModule.Rvsdg().GetRootRegion(), aliasAnalysis);

  statistics->StopEvaluatingPrecision(Context_.NumMayAliasQueries);

  // Calculate average precision in functions and for the whole module, and print to output
  const auto outputFile = statisticsCollector.CreateOutputFile("AAPrecisionEvaluation.log", true);
  CalculateAverageMayAliasRate(outputFile, *statistics);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
PrecisionEvaluator::EvaluateAllFunctions(
    const rvsdg::Region & region,
    PairwiseAliasAnalysis & aliasAnalysis)
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
    PairwiseAliasAnalysis & aliasAnalysis)
{
  // Reset collected pointer uses and pointer clobbers
  Context_.PointerUses.clear();
  Context_.PointerClobbers.clear();

  // Collect all pointer uses and clobbers from this function
  CollectPointersFromFunctionArguments(function);
  CollectPointersFromRegion(*function.subregion());

  // Create a PrecisionInfo instance for this function
  auto & precisionEvaluation = Context_.PerFunctionPrecision[&function];
  precisionEvaluation.NumClobberingPointers = Context_.PointerClobbers.size();

  // Go over all pointer usages, find the ratio of clobbering points that may alias with it
  for (auto [usedPointer, useIsClobber] : Context_.PointerUses)
  {
    uint64_t mayAliasClobbers = 0;
    for (auto clobberedPointer : Context_.PointerClobbers)
    {
      Context_.NumMayAliasQueries++;
      mayAliasClobbers += aliasAnalysis.MayAlias(*usedPointer, *clobberedPointer);
    }
    precisionEvaluation.AddPointerUse(useIsClobber, mayAliasClobbers);
  }
}

void
PrecisionEvaluator::CollectPointersFromFunctionArguments(const rvsdg::LambdaNode & function)
{
  // In this mode, only loads and stores constitute uses and clobbers, so ignore function arguments
  if (Mode_ == PrecisionEvaluationMode::ClobberingStores)
    return;

  JLM_ASSERT(Mode_ == PrecisionEvaluationMode::AllPointerPairs);
  for (const auto arg : function.GetFunctionArguments())
  {
    if (IsPointerCompatible(arg))
      CollectPointer(arg, true, true);
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
  if (Mode_ == PrecisionEvaluationMode::ClobberingStores)
  {
    // In this mode, only (volatile) load and store operations count as uses and clobbers
    if (auto load = dynamic_cast<const LoadNode *>(&node))
    {
      CollectPointer(load->GetAddressInput().origin(), true, false);
    }
    else if (auto store = dynamic_cast<const StoreNode *>(&node))
    {
      CollectPointer(store->GetAddressInput().origin(), true, true);
    }
  }
  else if (Mode_ == PrecisionEvaluationMode::AllPointerPairs)
  {
    // In this mode, all pointer compatible outputs are regarded as both uses and clobbers
    for (size_t n = 0; n < node.noutputs(); n++)
    {
      if (const auto output = node.output(n); IsPointerCompatible(output))
        CollectPointer(output, true, true);
    }
  }
  else
  {
    JLM_UNREACHABLE("Unknown precision evaluation mode");
  }
}

void
PrecisionEvaluator::CollectPointersFromStructuralNode(const rvsdg::StructuralNode & node)
{
  for (size_t n = 0; n < node.nsubregions(); n++)
  {
    CollectPointersFromRegion(*node.subregion(n));
  }

  if (Mode_ == PrecisionEvaluationMode::AllPointerPairs)
  {
    // In this mode, pointer compatible outputs from structural nodes represent new pointers.
    // This is to mimic LLVM phi nodes more closely
    // If the output has no users, such as a theta loop variable only used in the loop, it is
    // skipped
    for (size_t n = 0; n < node.noutputs(); n++)
    {
      const auto output = node.output(n);
      if (IsPointerCompatible(output) && output->nusers() > 0)
        CollectPointer(output, true, true);
    }
  }
}

bool
PrecisionEvaluator::IsPointerCompatible(const rvsdg::output * value)
{
  // Omit including function types as pointers, as direct calls are trivial for the analysis.
  // This is also closer to LLVM, as they do not include direct function calls in alias pairs.
  const auto & type = value->type();
  return IsOrContains<PointerType>(type);
}

void
PrecisionEvaluator::CollectPointer(const rvsdg::output * value, bool isUse, bool isClobber)
{
  JLM_ASSERT(IsPointerCompatible(value));

  if (isUse)
    Context_.PointerUses.push_back({ value, isClobber });

  if (isClobber)
    Context_.PointerClobbers.push_back(value);
}

void
PrecisionEvaluator::CalculateAverageMayAliasRate(
    const util::file & outputFile,
    PrecisionStatistics & statistics) const
{
  // Create one average alias analysis precision for the whole module
  size_t moduleNumUsages = 0;
  double moduleUseMayAliasTotal = 0.0;

  // Write precision info about each function to an output file
  std::ofstream out(outputFile.path().to_str());
  for (auto [function, precision] : Context_.PerFunctionPrecision)
  {
    // Calculate the average alias analysis precision rate for this function
    auto functionNumUsages = precision.UsedPointerMayAlias.size();
    double functionUseMayAliasTotal = 0.0;
    for (double mayAlias : precision.UsedPointerMayAlias)
      functionUseMayAliasTotal += mayAlias;

    // Also include the pointer uses in the module average
    moduleNumUsages += functionNumUsages;
    moduleUseMayAliasTotal += functionUseMayAliasTotal;

    const auto functionAverageMayAlias = functionUseMayAliasTotal / functionNumUsages;

    out << function->GetOperation().debug_string() << " [";
    out << precision.UsedPointerMayAlias.size() << " used pointers, ";
    out << precision.NumClobberingPointers << " clobbering pointers]: ";
    out << functionAverageMayAlias;
    out << std::endl;
  }

  // Calculate the module-wide average precision for each pointer use
  const auto moduleAverageMayAlias = moduleUseMayAliasTotal / moduleNumUsages;
  out << std::endl; // Empty line before the final result
  out << "Total: " << moduleNumUsages << " used pointers. ";
  out << "Average use may alias clobber rate: " << moduleAverageMayAlias << std::endl;

  out.close();

  statistics.AddPrecisionSummaryStatistics(
      outputFile.path(),
      moduleNumUsages,
      moduleAverageMayAlias);
}

}
