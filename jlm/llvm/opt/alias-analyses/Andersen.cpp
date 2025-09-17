/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/**
 * If values are pointers, or contain pointers to things, its set of pointees must be tracked.
 * @param type the rvsdg type to be checked
 * @return true if pointees should be tracked for all values of the given type, otherwise false
 */
bool
IsOrContainsPointerType(const rvsdg::Type & type)
{
  return IsOrContains<PointerType>(type) || is<rvsdg::FunctionType>(type);
}

std::string
Andersen::Configuration::ToString() const
{
  std::ostringstream str;
  if (EnableOfflineVariableSubstitution_)
    str << "OVS_";
  if (EnableOfflineConstraintNormalization_)
    str << "NORM_";
  if (Solver_ == Solver::Naive)
  {
    str << "Solver=Naive_";
  }
  else if (Solver_ == Solver::Worklist)
  {
    str << "Solver=Worklist_";
    str << "Policy=";
    str << PointerObjectConstraintSet::WorklistSolverPolicyToString(WorklistSolverPolicy_);
    str << "_";

    if (EnableOnlineCycleDetection_)
      str << "OnlineCD_";
    if (EnableHybridCycleDetection_)
      str << "HybridCD_";
    if (EnableLazyCycleDetection_)
      str << "LazyCD_";
    if (EnableDifferencePropagation_)
      str << "DP_";
    if (EnablePreferImplicitPointees_)
      str << "PIP_";
  }
  else
  {
    JLM_UNREACHABLE("Unknown solver type");
  }

  auto result = str.str();
  result.erase(result.size() - 1, 1); // Remove trailing '_'
  return result;
}

std::vector<Andersen::Configuration>
Andersen::Configuration::GetAllConfigurations()
{
  std::vector<Configuration> configs;
  auto PickPreferImplicitPointees = [&](Configuration config)
  {
    config.EnablePreferImplicitPointees(false);
    configs.push_back(config);
    config.EnablePreferImplicitPointees(true);
    configs.push_back(config);
  };
  auto PickDifferencePropagation = [&](Configuration config)
  {
    config.EnableDifferencePropagation(false);
    PickPreferImplicitPointees(config);
    config.EnableDifferencePropagation(true);
    PickPreferImplicitPointees(config);
  };
  auto PickLazyCycleDetection = [&](Configuration config)
  {
    config.EnableLazyCycleDetection(false);
    PickDifferencePropagation(config);
    config.EnableLazyCycleDetection(true);
    PickDifferencePropagation(config);
  };
  auto PickHybridCycleDetection = [&](Configuration config)
  {
    config.EnableHybridCycleDetection(false);
    PickLazyCycleDetection(config);
    // Hybrid Cycle Detection can only be enabled when OVS is enabled
    if (config.IsOfflineVariableSubstitutionEnabled())
    {
      config.EnableHybridCycleDetection(true);
      PickLazyCycleDetection(config);
    }
  };
  auto PickOnlineCycleDetection = [&](Configuration config)
  {
    config.EnableOnlineCycleDetection(false);
    PickHybridCycleDetection(config);
    config.EnableOnlineCycleDetection(true);
    // OnlineCD can not be combined with HybridCD or LazyCD
    PickDifferencePropagation(config);
  };
  auto PickWorklistPolicy = [&](Configuration config)
  {
    using Policy = PointerObjectConstraintSet::WorklistSolverPolicy;
    config.SetWorklistSolverPolicy(Policy::LeastRecentlyFired);
    PickOnlineCycleDetection(config);
    config.SetWorklistSolverPolicy(Policy::TwoPhaseLeastRecentlyFired);
    PickOnlineCycleDetection(config);
    config.SetWorklistSolverPolicy(Policy::LastInFirstOut);
    PickOnlineCycleDetection(config);
    config.SetWorklistSolverPolicy(Policy::FirstInFirstOut);
    PickOnlineCycleDetection(config);
    config.SetWorklistSolverPolicy(Policy::TopologicalSort);
    PickDifferencePropagation(config); // With topo, skip all cycle detection
  };
  auto PickOfflineNormalization = [&](Configuration config)
  {
    config.EnableOfflineConstraintNormalization(false);
    configs.push_back(config);
    config.EnableOfflineConstraintNormalization(true);
    configs.push_back(config);
  };
  auto PickSolver = [&](Configuration config)
  {
    config.SetSolver(Solver::Worklist);
    PickWorklistPolicy(config);
    config.SetSolver(Solver::Naive);
    PickOfflineNormalization(config);
  };
  auto PickOfflineVariableSubstitution = [&](Configuration config)
  {
    config.EnableOfflineVariableSubstitution(false);
    PickSolver(config);
    config.EnableOfflineVariableSubstitution(true);
    PickSolver(config);
  };

  // Adds one configuration for all valid combinations of features
  PickOfflineVariableSubstitution(NaiveSolverConfiguration());

  return configs;
}

/**
 * Class collecting statistics from a pass of Andersen's alias analysis
 */
class Andersen::Statistics final : public util::Statistics
{
  static constexpr const char * NumPointerObjects_ = "#PointerObjects";
  static constexpr const char * NumMemoryPointerObjects_ = "#MemoryPointerObjects";
  static constexpr const char * NumMemoryPointerObjectsCanPoint_ = "#MemoryPointerObjectsCanPoint";
  static constexpr const char * NumRegisterPointerObjects_ = "#RegisterPointerObjects";
  // A PointerObject of Register kind can represent multiple outputs in RVSDG. Sum them up.
  static constexpr const char * NumRegistersMappedToPointerObject_ =
      "#RegistersMappedToPointerObject";
  static constexpr const char * NumAllocaPointerObjects = "#AllocaPointerObjects";
  static constexpr const char * NumMallocPointerObjects = "#MallocPointerObjects";
  static constexpr const char * NumGlobalPointerObjects = "#GlobalPointerObjects";
  static constexpr const char * NumFunctionPointerObjects = "#FunctionPointerObjects";
  static constexpr const char * NumImportPointerObjects = "#ImportPointerObjects";

  static constexpr const char * NumBaseConstraints_ = "#BaseConstraints";
  static constexpr const char * NumSupersetConstraints_ = "#SupersetConstraints";
  static constexpr const char * NumStoreConstraints_ = "#StoreConstraints";
  static constexpr const char * NumLoadConstraints_ = "#LoadConstraints";
  static constexpr const char * NumFunctionCallConstraints_ = "#FunctionCallConstraints";
  static constexpr const char * NumScalarFlagConstraints_ = "#ScalarFlagConstraints";
  static constexpr const char * NumOtherFlagConstraints_ = "#OtherFlagConstraints";

  static constexpr const char * Configuration_ = "Configuration";

  // ====== Offline technique statistics ======
  static constexpr const char * NumUnificationsOvs_ = "#Unifications(OVS)";
  static constexpr const char * NumConstraintsRemovedOfflineNorm_ =
      "#ConstraintsRemoved(OfflineNorm)";

  // ====== Solver statistics ======
  static constexpr const char * NumNaiveSolverIterations_ = "#NaiveSolverIterations";

  static constexpr const char * WorklistPolicy_ = "WorklistPolicy";
  static constexpr const char * NumWorklistSolverWorkItemsPopped_ =
      "#WorklistSolverWorkItemsPopped";
  static constexpr const char * NumWorklistSolverWorkItemsNewPointees_ =
      "#WorklistSolverWorkItemsNewPointees";
  static constexpr const char * NumTopologicalWorklistSweeps_ = "#TopologicalWorklistSweeps";

  // ====== Online technique statistics ======
  static constexpr const char * NumOnlineCyclesDetected_ = "#OnlineCyclesDetected";
  static constexpr const char * NumOnlineCycleUnifications_ = "#OnlineCycleUnifications";

  static constexpr const char * NumHybridCycleUnifications_ = "#HybridCycleUnifications";

  static constexpr const char * NumLazyCycleDetectionAttempts_ = "#LazyCycleDetectionAttempts";
  static constexpr const char * NumLazyCyclesDetected_ = "#LazyCyclesDetected";
  static constexpr const char * NumLazyCycleUnifications_ = "#LazyCycleUnifications";

  static constexpr const char * NumPIPExplicitPointeesRemoved_ = "#PIPExplicitPointeesRemoved";

  // ====== During solving points-to set statistics ======
  // How many times a pointee has been attempted inserted into an explicit points-to set.
  // If a set with 10 elements is unioned into another set, that counts as 10 insertion attempts.
  static constexpr const char * NumSetInsertionAttempts_ = "#PointsToSetInsertionAttempts";
  // How many explicit pointees have been removed from points-to sets during solving.
  // Removal can only happen due to unification, or explicitly when using PIP
  static constexpr const char * NumExplicitPointeesRemoved_ = "#ExplicitPointeesRemoved";

  // ====== After solving statistics ======
  // How many disjoint sets of PointerObjects exist
  static constexpr const char * NumUnificationRoots_ = "#UnificationRoots";
  // How many memory objects where CanPoint() == true have escaped
  static constexpr const char * NumCanPointsEscaped_ = "#CanPointsEscaped";
  // How many memory objects where CanPoint() == false have escaped
  static constexpr const char * NumCantPointsEscaped_ = "#CantPointsEscaped";

  // The number of explicit pointees, counting only unification roots
  static constexpr const char * NumExplicitPointees_ = "#ExplicitPointees";
  // Only unification roots may have explicit pointees, but all PointerObjects in the unification
  // marked CanPoint effectively have those explicit pointees. Add up the number of such relations.
  static constexpr const char * NumExplicitPointsToRelations_ = "#ExplicitPointsToRelations";

  // The number of PointsToExternal flags, counting only unification roots
  static constexpr const char * NumPointsToExternalFlags_ = "#PointsToExternalFlags";
  // Among all PointerObjects marked CanPoint, how many are in a unification pointing to external
  static constexpr const char * NumPointsToExternalRelations_ = "#PointsToExternalRelations";

  // Among all PointerObjects marked CanPoint and NOT flagged as pointing to external,
  // add up how many pointer-pointee relations they have.
  static constexpr const char * NumExplicitPointsToRelationsAmongPrecise_ =
      "#ExplicitPointsToRelationsAmongPrecise";

  // The number of PointeesEscaping flags, counting only unification roots
  static constexpr const char * NumPointeesEscapingFlags_ = "#PointeesEscapingFlags";
  // Among all PointerObjects marked CanPoint, how many are in a unification where pointees escape.
  static constexpr const char * NumPointeesEscapingRelations_ = "#PointeesEscapingRelations";

  // The total number of pointer-pointee relations, counting both explicit and implicit.
  // In the case of doubled up pointees, the same pointer-pointee relation is not counted twice.
  static constexpr const char * NumPointsToRelations_ = "#PointsToRelations";

  // The number of doubled up pointees, only counting unification roots
  static constexpr const char * NumDoubledUpPointees_ = "#DoubledUpPointees";
  // The number of doubled up pointees, counting all PointerObjects marked CanPoint()
  static constexpr const char * NumDoubledUpPointsToRelations_ = "#DoubledUpPointsToRelations";

  // Number of unifications where no members have the CanPoint flag
  static constexpr const char * NumCantPointUnifications_ = "#CantPointUnifications";
  // In unifications where no member CanPoint, add up their explicit pointees
  static constexpr const char * NumCantPointExplicitPointees_ = "#CantPointExplicitPointees";

  static constexpr const char * AnalysisTimer_ = "AnalysisTimer";
  static constexpr const char * SetAndConstraintBuildingTimer_ = "SetAndConstraintBuildingTimer";
  static constexpr const char * OfflineVariableSubstitutionTimer_ = "OVSTimer";
  static constexpr const char * OfflineConstraintNormalizationTimer_ = "OfflineNormTimer";
  static constexpr const char * ConstraintSolvingNaiveTimer_ = "ConstraintSolvingNaiveTimer";
  static constexpr const char * ConstraintSolvingWorklistTimer_ = "ConstraintSolvingWorklistTimer";
  static constexpr const char * PointsToGraphConstructionTimer_ = "PointsToGraphConstructionTimer";
  static constexpr const char * PointsToGraphConstructionExternalToEscapedTimer_ =
      "PointsToGraphConstructionExternalToEscapedTimer";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::AndersenAnalysis, sourceFile)
  {}

  void
  StartAndersenStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(AnalysisTimer_).start();
  }

  void
  StartSetAndConstraintBuildingStatistics() noexcept
  {
    AddTimer(SetAndConstraintBuildingTimer_).start();
  }

  void
  StopSetAndConstraintBuildingStatistics(
      const PointerObjectSet & set,
      const PointerObjectConstraintSet & constraints) noexcept
  {
    GetTimer(SetAndConstraintBuildingTimer_).stop();

    // Measure the number of pointer objects of different kinds
    AddMeasurement(NumPointerObjects_, set.NumPointerObjects());
    AddMeasurement(NumMemoryPointerObjects_, set.NumMemoryPointerObjects());
    AddMeasurement(NumMemoryPointerObjectsCanPoint_, set.NumMemoryPointerObjectsCanPoint());
    AddMeasurement(NumRegisterPointerObjects_, set.NumRegisterPointerObjects());
    AddMeasurement(NumRegistersMappedToPointerObject_, set.GetRegisterMap().size());

    AddMeasurement(
        NumAllocaPointerObjects,
        set.NumPointerObjectsOfKind(PointerObjectKind::AllocaMemoryObject));
    AddMeasurement(
        NumMallocPointerObjects,
        set.NumPointerObjectsOfKind(PointerObjectKind::MallocMemoryObject));
    AddMeasurement(
        NumGlobalPointerObjects,
        set.NumPointerObjectsOfKind(PointerObjectKind::GlobalMemoryObject));
    AddMeasurement(
        NumFunctionPointerObjects,
        set.NumPointerObjectsOfKind(PointerObjectKind::FunctionMemoryObject));
    AddMeasurement(
        NumImportPointerObjects,
        set.NumPointerObjectsOfKind(PointerObjectKind::ImportMemoryObject));

    // Count the number of constraints of different kinds
    size_t numSupersetConstraints = 0;
    size_t numStoreConstraints = 0;
    size_t numLoadConstraints = 0;
    size_t numFunctionCallConstraints = 0;
    for (const auto & constraint : constraints.GetConstraints())
    {
      numSupersetConstraints += std::holds_alternative<SupersetConstraint>(constraint);
      numStoreConstraints += std::holds_alternative<StoreConstraint>(constraint);
      numLoadConstraints += std::holds_alternative<LoadConstraint>(constraint);
      numFunctionCallConstraints += std::holds_alternative<FunctionCallConstraint>(constraint);
    }
    AddMeasurement(NumBaseConstraints_, constraints.NumBaseConstraints());
    AddMeasurement(NumSupersetConstraints_, numSupersetConstraints);
    AddMeasurement(NumStoreConstraints_, numStoreConstraints);
    AddMeasurement(NumLoadConstraints_, numLoadConstraints);
    AddMeasurement(NumFunctionCallConstraints_, numFunctionCallConstraints);
    const auto [scalarFlags, otherFlags] = constraints.NumFlagConstraints();
    AddMeasurement(NumScalarFlagConstraints_, scalarFlags);
    AddMeasurement(NumOtherFlagConstraints_, otherFlags);
  }

  void
  StartOfflineVariableSubstitution() noexcept
  {
    AddTimer(OfflineVariableSubstitutionTimer_).start();
  }

  void
  StopOfflineVariableSubstitution(size_t numUnifications) noexcept
  {
    GetTimer(OfflineVariableSubstitutionTimer_).stop();
    AddMeasurement(NumUnificationsOvs_, numUnifications);
  }

  void
  StartOfflineConstraintNormalization() noexcept
  {
    AddTimer(OfflineConstraintNormalizationTimer_).start();
  }

  void
  StopOfflineConstraintNormalization(size_t numConstraintsRemoved) noexcept
  {
    GetTimer(OfflineConstraintNormalizationTimer_).stop();
    AddMeasurement(NumConstraintsRemovedOfflineNorm_, numConstraintsRemoved);
  }

  void
  StartConstraintSolvingNaiveStatistics() noexcept
  {
    AddTimer(ConstraintSolvingNaiveTimer_).start();
  }

  void
  StopConstraintSolvingNaiveStatistics(size_t numIterations) noexcept
  {
    GetTimer(ConstraintSolvingNaiveTimer_).stop();
    AddMeasurement(NumNaiveSolverIterations_, numIterations);
  }

  void
  StartConstraintSolvingWorklistStatistics() noexcept
  {
    AddTimer(ConstraintSolvingWorklistTimer_).start();
  }

  void
  StopConstraintSolvingWorklistStatistics(
      PointerObjectConstraintSet::WorklistStatistics & statistics) noexcept
  {
    GetTimer(ConstraintSolvingWorklistTimer_).stop();

    // What worklist policy was used
    AddMeasurement(
        WorklistPolicy_,
        PointerObjectConstraintSet::WorklistSolverPolicyToString(statistics.Policy));

    // How many work items were popped from the worklist in total
    AddMeasurement(NumWorklistSolverWorkItemsPopped_, statistics.NumWorkItemsPopped);
    AddMeasurement(NumWorklistSolverWorkItemsNewPointees_, statistics.NumWorkItemNewPointees);

    if (statistics.NumTopologicalWorklistSweeps)
      AddMeasurement(NumTopologicalWorklistSweeps_, *statistics.NumTopologicalWorklistSweeps);

    if (statistics.NumOnlineCyclesDetected)
      AddMeasurement(NumOnlineCyclesDetected_, *statistics.NumOnlineCyclesDetected);

    if (statistics.NumOnlineCycleUnifications)
      AddMeasurement(NumOnlineCycleUnifications_, *statistics.NumOnlineCycleUnifications);

    if (statistics.NumHybridCycleUnifications)
      AddMeasurement(NumHybridCycleUnifications_, *statistics.NumHybridCycleUnifications);

    if (statistics.NumLazyCyclesDetectionAttempts)
      AddMeasurement(NumLazyCycleDetectionAttempts_, *statistics.NumLazyCyclesDetectionAttempts);

    if (statistics.NumLazyCyclesDetected)
      AddMeasurement(NumLazyCyclesDetected_, *statistics.NumLazyCyclesDetected);

    if (statistics.NumLazyCycleUnifications)
      AddMeasurement(NumLazyCycleUnifications_, *statistics.NumLazyCycleUnifications);

    if (statistics.NumPipExplicitPointeesRemoved)
      AddMeasurement(NumPIPExplicitPointeesRemoved_, *statistics.NumPipExplicitPointeesRemoved);
  }

  void
  AddStatisticFromConfiguration(const Configuration & config)
  {
    AddMeasurement(Configuration_, config.ToString());
  }

  void
  AddStatisticsFromSolution(const PointerObjectSet & set)
  {
    AddMeasurement(NumSetInsertionAttempts_, set.GetNumSetInsertionAttempts());
    AddMeasurement(NumExplicitPointeesRemoved_, set.GetNumExplicitPointeesRemoved());

    size_t numUnificationRoots = 0;

    size_t numCanPointEscaped = 0;
    size_t numCantPointEscaped = 0;

    size_t numExplicitPointees = 0;
    size_t numExplicitPointsToRelations = 0;
    size_t numExplicitPointeeRelationsAmongPrecise = 0;

    size_t numPointsToExternalFlags = 0;
    size_t numPointsToExternalRelations = 0;
    size_t numPointeesEscapingFlags = 0;
    size_t numPointeesEscapingRelations = 0;

    size_t numDoubledUpPointees = 0;
    size_t numDoubledUpPointsToRelations = 0;

    std::vector<bool> unificationHasCanPoint(set.NumPointerObjects(), false);

    for (PointerObjectIndex i = 0; i < set.NumPointerObjects(); i++)
    {
      if (set.HasEscaped(i))
      {
        if (set.CanPoint(i))
          numCanPointEscaped++;
        else
          numCantPointEscaped++;
      }

      const auto & pointees = set.GetPointsToSet(i);

      if (set.CanPoint(i))
      {
        numExplicitPointsToRelations += pointees.Size();
        numPointeesEscapingRelations += set.HasPointeesEscaping(i);

        if (set.IsPointingToExternal(i))
        {
          numPointsToExternalRelations++;
          for (auto pointee : pointees.Items())
          {
            if (set.HasEscaped(pointee))
              numDoubledUpPointsToRelations++;
          }
        }
        else
        {
          // When comparing precision, the number of explicit pointees is more interesting among
          // pointers that do not also point to external.
          numExplicitPointeeRelationsAmongPrecise += pointees.Size();
        }

        // This unification has at least one CanPoint member
        unificationHasCanPoint[set.GetUnificationRoot(i)] = true;
      }

      // The rest of this loop is only concerned with unification roots, as they are the only
      // PointerObjects that actually have explicit pointees or flags
      if (!set.IsUnificationRoot(i))
        continue;

      numUnificationRoots++;
      if (set.IsPointingToExternal(i))
        numPointsToExternalFlags++;
      if (set.HasPointeesEscaping(i))
        numPointeesEscapingFlags++;

      numExplicitPointees += pointees.Size();

      // If the PointsToExternal flag is set, any explicit pointee that has escaped is doubled up
      if (set.IsPointingToExternal(i))
        for (auto pointee : pointees.Items())
          if (set.HasEscaped(pointee))
            numDoubledUpPointees++;
    }

    // Now find unifications where no member is marked CanPoint, as any explicit pointee is a waste
    size_t numCantPointUnifications = 0;
    size_t numCantPointExplicitPointees = 0;
    for (PointerObjectIndex i = 0; i < set.NumPointerObjects(); i++)
    {
      if (!set.IsUnificationRoot(i))
        continue;
      if (unificationHasCanPoint[i])
        continue;
      numCantPointUnifications++;
      numCantPointExplicitPointees += set.GetPointsToSet(i).Size();
    }

    AddMeasurement(NumUnificationRoots_, numUnificationRoots);
    AddMeasurement(NumCanPointsEscaped_, numCanPointEscaped);
    AddMeasurement(NumCantPointsEscaped_, numCantPointEscaped);

    AddMeasurement(NumExplicitPointees_, numExplicitPointees);
    AddMeasurement(NumExplicitPointsToRelations_, numExplicitPointsToRelations);
    AddMeasurement(
        NumExplicitPointsToRelationsAmongPrecise_,
        numExplicitPointeeRelationsAmongPrecise);

    AddMeasurement(NumPointsToExternalFlags_, numPointsToExternalFlags);
    AddMeasurement(NumPointsToExternalRelations_, numPointsToExternalRelations);
    AddMeasurement(NumPointeesEscapingFlags_, numPointeesEscapingFlags);
    AddMeasurement(NumPointeesEscapingRelations_, numPointeesEscapingRelations);

    // Calculate the total number of pointer-pointee relations by adding up all explicit and
    // implicit relations, and removing the doubled up relations.
    size_t numPointsToRelations =
        numExplicitPointsToRelations - numDoubledUpPointsToRelations
        + numPointsToExternalRelations * (numCanPointEscaped + numCantPointEscaped);

    AddMeasurement(NumPointsToRelations_, numPointsToRelations);

    AddMeasurement(NumDoubledUpPointees_, numDoubledUpPointees);
    AddMeasurement(NumDoubledUpPointsToRelations_, numDoubledUpPointsToRelations);

    AddMeasurement(NumCantPointUnifications_, numCantPointUnifications);
    AddMeasurement(NumCantPointExplicitPointees_, numCantPointExplicitPointees);
  }

  void
  StartPointsToGraphConstructionStatistics()
  {
    AddTimer(PointsToGraphConstructionTimer_).start();
  }

  void
  StartExternalToAllEscapedStatistics()
  {
    AddTimer(PointsToGraphConstructionExternalToEscapedTimer_).start();
  }

  void
  StopExternalToAllEscapedStatistics()
  {
    GetTimer(PointsToGraphConstructionExternalToEscapedTimer_).stop();
  }

  void
  StopPointsToGraphConstructionStatistics(const PointsToGraph & pointsToGraph)
  {
    GetTimer(PointsToGraphConstructionTimer_).stop();
    AddMeasurement(Label::NumPointsToGraphNodes, pointsToGraph.NumNodes());
    AddMeasurement(Label::NumPointsToGraphAllocaNodes, pointsToGraph.NumAllocaNodes());
    AddMeasurement(Label::NumPointsToGraphDeltaNodes, pointsToGraph.NumDeltaNodes());
    AddMeasurement(Label::NumPointsToGraphImportNodes, pointsToGraph.NumImportNodes());
    AddMeasurement(Label::NumPointsToGraphLambdaNodes, pointsToGraph.NumLambdaNodes());
    AddMeasurement(Label::NumPointsToGraphMallocNodes, pointsToGraph.NumMallocNodes());
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.NumMemoryNodes());
    AddMeasurement(Label::NumPointsToGraphRegisterNodes, pointsToGraph.NumRegisterNodes());
    AddMeasurement(
        Label::NumPointsToGraphEscapedNodes,
        pointsToGraph.GetEscapedMemoryNodes().Size());
    // The number of nodes pointing to external (and all nodes marked as escaped)
    AddMeasurement(
        Label::NumPointsToGraphExternalMemorySources,
        pointsToGraph.GetExternalMemoryNode().NumSources());
    auto [numEdges, numPointsToRelations] = pointsToGraph.NumEdges();
    AddMeasurement(Label::NumPointsToGraphEdges, numEdges);
    AddMeasurement(Label::NumPointsToGraphPointsToRelations, numPointsToRelations);
  }

  void
  StopAndersenStatistics() noexcept
  {
    GetTimer(AnalysisTimer_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

Andersen::Andersen() = default;

Andersen::~Andersen() noexcept = default;

void
Andersen::AnalyzeSimpleNode(const rvsdg::SimpleNode & node)
{
  rvsdg::MatchTypeWithDefault(
      node.GetOperation(),
      [&](const AllocaOperation &)
      {
        AnalyzeAlloca(node);
      },
      [&](const MallocOperation &)
      {
        AnalyzeMalloc(node);
      },
      [&](const LoadOperation &)
      {
        AnalyzeLoad(node);
      },
      [&](const StoreOperation &)
      {
        AnalyzeStore(node);
      },
      [&](const CallOperation &)
      {
        AnalyzeCall(node);
      },
      [&](const GetElementPtrOperation &)
      {
        AnalyzeGep(node);
      },
      [&](const BitCastOperation &)
      {
        AnalyzeBitcast(node);
      },
      [&](const IntegerToPointerOperation &)
      {
        AnalyzeBits2ptr(node);
      },
      [&](const PtrToIntOperation &)
      {
        AnalyzePtrToInt(node);
      },
      [&](const ConstantPointerNullOperation &)
      {
        AnalyzeConstantPointerNull(node);
      },
      [&](const UndefValueOperation &)
      {
        AnalyzeUndef(node);
      },
      [&](const MemCpyOperation &)
      {
        AnalyzeMemcpy(node);
      },
      [&](const ConstantArrayOperation &)
      {
        AnalyzeConstantArray(node);
      },
      [&](const ConstantStruct &)
      {
        AnalyzeConstantStruct(node);
      },
      [&](const ConstantAggregateZeroOperation &)
      {
        AnalyzeConstantAggregateZero(node);
      },
      [&](const ExtractValueOperation &)
      {
        AnalyzeExtractValue(node);
      },
      [&](const VariadicArgumentListOperation &)
      {
        AnalyzeValist(node);
      },
      [&](const PointerToFunctionOperation &)
      {
        AnalyzePointerToFunction(node);
      },
      [&](const FunctionToPointerOperation &)
      {
        AnalyzeFunctionToPointer(node);
      },
      [&](const IOBarrierOperation &)
      {
        AnalyzeIOBarrier(node);
      },
      [&](const FreeOperation &)
      {
        // Takes pointers as input, but does not affect any points-to sets
      },
      [&](const PtrCmpOperation &)
      {
        // Takes pointers as input, but does not affect any points-to sets
      },
      [&]()
      {
        // This node operation is unknown, make sure it doesn't consume any pointers
        for (size_t n = 0; n < node.ninputs(); n++)
          JLM_ASSERT(!IsOrContainsPointerType(*node.input(n)->Type()));
      });
}

void
Andersen::AnalyzeAlloca(const rvsdg::SimpleNode & node)
{
  const auto allocaOp = util::AssertedCast<const AllocaOperation>(&node.GetOperation());

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  const bool canPoint = IsOrContainsPointerType(*allocaOp->ValueType());
  const auto allocaPO = Set_->CreateAllocaMemoryObject(node, canPoint);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, allocaPO);
}

void
Andersen::AnalyzeMalloc(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<MallocOperation>(node.GetOperation()));

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  // We do not know what types will be stored in the malloc, so let it track pointers
  const auto mallocPO = Set_->CreateMallocMemoryObject(node, true);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, mallocPO);
}

void
Andersen::AnalyzeLoad(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<LoadOperation>(node.GetOperation()));

  const auto & addressRegister = *LoadOperation::AddressInput(node).origin();
  const auto & outputRegister = LoadOperation::LoadedValueOutput(node);

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);

  if (IsOrContainsPointerType(*outputRegister.Type()))
  {
    const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
    Constraints_->AddConstraint(LoadConstraint(outputRegisterPO, addressRegisterPO));
  }
  else
  {
    Set_->MarkAsLoadingAsScalar(addressRegisterPO);
  }
}

void
Andersen::AnalyzeStore(const rvsdg::SimpleNode & node)
{
  const auto & addressRegister = *StoreOperation::AddressInput(node).origin();
  const auto & valueRegister = *StoreOperation::StoredValueInput(node).origin();

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);

  // If the written value is not a pointer, be conservative and mark the address
  if (IsOrContainsPointerType(*valueRegister.Type()))
  {
    const auto valueRegisterPO = Set_->GetRegisterPointerObject(valueRegister);
    Constraints_->AddConstraint(StoreConstraint(addressRegisterPO, valueRegisterPO));
  }
  else
  {
    Set_->MarkAsStoringAsScalar(addressRegisterPO);
  }
}

void
Andersen::AnalyzeCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(callNode.GetOperation()));

  // The address being called by the call node
  const auto & callTarget = *CallOperation::GetFunctionInput(callNode).origin();
  const auto callTargetPO = Set_->GetRegisterPointerObject(callTarget);

  // Create PointerObjects for all output values of pointer type
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    const auto & outputRegister = *callNode.output(n);
    if (IsOrContainsPointerType(*outputRegister.Type()))
      (void)Set_->CreateRegisterPointerObject(outputRegister);
  }

  // We make no attempt at detecting what type of call this is here.
  // The logic handling external and indirect calls is done by the FunctionCallConstraint.
  // Passing points-to-sets from call-site to function bodies is done fully by this constraint.
  Constraints_->AddConstraint(FunctionCallConstraint(callTargetPO, callNode));
}

void
Andersen::AnalyzeGep(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(node.GetOperation()));

  // The analysis is field insensitive, so ignoring the offset and mapping the output
  // to the same PointerObject as the input is sufficient.
  const auto & baseRegister = *node.input(0)->origin();
  JLM_ASSERT(is<PointerType>(baseRegister.Type()));

  const auto baseRegisterPO = Set_->GetRegisterPointerObject(baseRegister);
  const auto & outputRegister = *node.output(0);
  Set_->MapRegisterToExistingPointerObject(outputRegister, baseRegisterPO);
}

void
Andersen::AnalyzeBitcast(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<BitCastOperation>(node.GetOperation()));

  const auto & inputRegister = *node.input(0)->origin();
  const auto & outputRegister = *node.output(0);

  JLM_ASSERT(!IsAggregateType(*inputRegister.Type()) && !IsAggregateType(*outputRegister.Type()));
  if (!IsOrContainsPointerType(*inputRegister.Type()))
    return;

  // If the input is a pointer type, the output must also be a pointer type
  JLM_ASSERT(IsOrContainsPointerType(*outputRegister.Type()));

  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  Set_->MapRegisterToExistingPointerObject(outputRegister, inputRegisterPO);
}

void
Andersen::AnalyzeBits2ptr(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<IntegerToPointerOperation>(node.GetOperation()));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.Type()));

  // This operation synthesizes a pointer from bytes.
  // Since no points-to information is tracked through integers, the resulting pointer must
  // be assumed to possibly point to any external or escaped memory object.
  const auto outputPO = Set_->CreateRegisterPointerObject(output);
  Constraints_->AddPointsToExternalConstraint(outputPO);
}

void
Andersen::AnalyzePtrToInt(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<PtrToIntOperation>(node.GetOperation()));
  const auto & inputRegister = *node.input(0)->origin();
  JLM_ASSERT(is<PointerType>(inputRegister.Type()));

  // This operation converts a pointer to bytes, exposing it as an integer, which we can't track.
  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
}

void
Andersen::AnalyzeConstantPointerNull(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(node.GetOperation()));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.Type()));

  // ConstantPointerNull cannot point to any memory location. We therefore only insert a register
  // node for it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeUndef(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<UndefValueOperation>(node.GetOperation()));
  const auto & output = *node.output(0);

  if (!IsOrContainsPointerType(*output.Type()))
    return;

  // UndefValue cannot point to any memory location. We therefore only insert a register node for
  // it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeMemcpy(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<MemCpyOperation>(node.GetOperation()));

  auto & dstAddressRegister = *node.input(0)->origin();
  auto & srcAddressRegister = *node.input(1)->origin();
  JLM_ASSERT(is<PointerType>(dstAddressRegister.Type()));
  JLM_ASSERT(is<PointerType>(srcAddressRegister.Type()));

  const auto dstAddressRegisterPO = Set_->GetRegisterPointerObject(dstAddressRegister);
  const auto srcAddressRegisterPO = Set_->GetRegisterPointerObject(srcAddressRegister);

  // Create an intermediate PointerObject representing the moved values
  const auto dummyPO = Set_->CreateDummyRegisterPointerObject();

  // Add a "load" constraint from the source into the dummy register
  Constraints_->AddConstraint(LoadConstraint(dummyPO, srcAddressRegisterPO));
  // Add a "store" constraint from the dummy register into the destination
  Constraints_->AddConstraint(StoreConstraint(dstAddressRegisterPO, dummyPO));
}

void
Andersen::AnalyzeConstantArray(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantArrayOperation>(node.GetOperation()));

  if (!IsOrContainsPointerType(*node.output(0)->Type()))
    return;

  // Make the resulting array point to everything its members are pointing to
  auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    JLM_ASSERT(IsOrContainsPointerType(*inputRegister.Type()));

    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, inputRegisterPO));
  }
}

void
Andersen::AnalyzeConstantStruct(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantStruct>(node.GetOperation()));

  if (!IsOrContainsPointerType(*node.output(0)->Type()))
    return;

  // Make the resulting struct point to everything its members are pointing to
  auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    if (!IsOrContainsPointerType(*inputRegister.Type()))
      continue;

    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, inputRegisterPO));
  }
}

void
Andersen::AnalyzeConstantAggregateZero(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantAggregateZeroOperation>(node.GetOperation()));
  auto & output = *node.output(0);

  if (!IsOrContainsPointerType(*output.Type()))
    return;

  // ConstantAggregateZero cannot point to any memory location.
  // We therefore only insert a register node for it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeExtractValue(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ExtractValueOperation>(node.GetOperation()));

  const auto & result = *node.output(0);
  if (!IsOrContainsPointerType(*result.Type()))
    return;

  const auto & inputRegister = *node.input(0)->origin();
  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  // The resulting element can point to anything the aggregate type points to
  Set_->MapRegisterToExistingPointerObject(result, inputRegisterPO);
}

void
Andersen::AnalyzeValist(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<VariadicArgumentListOperation>(node.GetOperation()));

  // Members of the valist are extracted using the va_arg macro, which loads from the va_list struct
  // on the stack. This struct will be marked as escaped from the call to va_start, and thus point
  // to external. All we need to do is mark all pointees of pointer varargs as escaping. When the
  // pointers are re-created inside the function, they will be marked as pointing to external.

  for (size_t i = 0; i < node.ninputs(); i++)
  {
    if (!IsOrContainsPointerType(*node.input(i)->Type()))
      continue;

    const auto & inputRegister = *node.input(i)->origin();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
  }
}

void
Andersen::AnalyzePointerToFunction(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<PointerToFunctionOperation>(node.GetOperation()));

  // For pointer analysis purposes, function objects and pointers
  // to functions are treated as being the same.
  const auto & baseRegister = *node.input(0)->origin();
  JLM_ASSERT(is<PointerType>(baseRegister.Type()));

  const auto baseRegisterPO = Set_->GetRegisterPointerObject(baseRegister);
  const auto & outputRegister = *node.output(0);
  Set_->MapRegisterToExistingPointerObject(outputRegister, baseRegisterPO);
}

void
Andersen::AnalyzeFunctionToPointer(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<FunctionToPointerOperation>(node.GetOperation()));

  // For pointer analysis purposes, function objects and pointers
  // to functions are treated as being the same.
  const auto & baseRegister = *node.input(0)->origin();
  JLM_ASSERT(is<rvsdg::FunctionType>(baseRegister.Type()));

  const auto baseRegisterPO = Set_->GetRegisterPointerObject(baseRegister);
  const auto & outputRegister = *node.output(0);
  Set_->MapRegisterToExistingPointerObject(outputRegister, baseRegisterPO);
}

void
Andersen::AnalyzeIOBarrier(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<IOBarrierOperation>(node.GetOperation()));

  const auto operation = util::AssertedCast<const IOBarrierOperation>(&node.GetOperation());
  if (!IsOrContainsPointerType(*operation->Type()))
    return;

  const auto & inputRegister = *node.input(0)->origin();
  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  const auto & outputRegister = *node.output(0);
  Set_->MapRegisterToExistingPointerObject(outputRegister, inputRegisterPO);
}

void
Andersen::AnalyzeStructuralNode(const rvsdg::StructuralNode & node)
{
  MatchTypeOrFail(
      node,
      [this](const rvsdg::LambdaNode & lambdaNode)
      {
        AnalyzeLambda(lambdaNode);
      },
      [this](const rvsdg::DeltaNode & deltaNode)
      {
        AnalyzeDelta(deltaNode);
      },
      [this](const rvsdg::PhiNode & phiNode)
      {
        AnalyzePhi(phiNode);
      },
      [this](const rvsdg::GammaNode & gammaNode)
      {
        AnalyzeGamma(gammaNode);
      },
      [this](const rvsdg::ThetaNode & thetaNode)
      {
        AnalyzeTheta(thetaNode);
      });
}

void
Andersen::AnalyzeLambda(const rvsdg::LambdaNode & lambda)
{
  // Handle context variables
  for (const auto & cv : lambda.GetContextVars())
  {
    if (!IsOrContainsPointerType(*cv.input->Type()))
      continue;

    auto & inputRegister = *cv.input->origin();
    auto & argumentRegister = *cv.inner;
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each argument of pointing type in the function
  for (auto argument : lambda.GetFunctionArguments())
  {
    if (IsOrContainsPointerType(*argument->Type()))
      (void)Set_->CreateRegisterPointerObject(*argument);
  }

  AnalyzeRegion(*lambda.subregion());

  // Create a lambda PointerObject for the lambda itself
  const auto lambdaPO = Set_->CreateFunctionMemoryObject(lambda);

  // Make the lambda node's output point to the lambda PointerObject
  const auto & lambdaOutput = *lambda.output();
  const auto lambdaOutputPO = Set_->CreateRegisterPointerObject(lambdaOutput);
  Constraints_->AddPointerPointeeConstraint(lambdaOutputPO, lambdaPO);
}

void
Andersen::AnalyzeDelta(const rvsdg::DeltaNode & delta)
{
  // Handle context variables
  for (auto & cv : delta.GetContextVars())
  {
    if (!IsOrContainsPointerType(*cv.input->Type()))
      continue;

    auto & inputRegister = *cv.input->origin();
    auto & argumentRegister = *cv.inner;
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  AnalyzeRegion(*delta.subregion());

  // Get the result register from the subregion
  auto & resultRegister = *delta.result().origin();

  // If the type of the delta can point, the analysis should track its set of possible pointees
  bool canPoint = IsOrContainsPointerType(*delta.Type());

  // Create a global memory object representing the global variable
  const auto globalPO = Set_->CreateGlobalMemoryObject(delta, canPoint);

  // If the initializer subregion result is a pointer, make the global point to what it points to
  if (canPoint)
  {
    const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
    Constraints_->AddConstraint(SupersetConstraint(globalPO, resultRegisterPO));
  }

  // Finally create a Register PointerObject for the delta's output, pointing to the memory object
  auto & outputRegister = delta.output();
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, globalPO);
}

void
Andersen::AnalyzePhi(const rvsdg::PhiNode & phi)
{
  // Handle context variables
  for (auto var : phi.GetContextVars())
  {
    if (!IsOrContainsPointerType(*var.inner->Type()))
      continue;

    auto & inputRegister = *var.input->origin();
    auto & argumentRegister = *var.inner;
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each fixpoint variable argument
  for (auto var : phi.GetFixVars())
  {
    if (!IsOrContainsPointerType(*var.output->Type()))
      continue;

    auto & argumentRegister = *var.recref;
    (void)Set_->CreateRegisterPointerObject(argumentRegister);
  }

  AnalyzeRegion(*phi.subregion());

  // Handle recursive definition results
  for (auto var : phi.GetFixVars())
  {
    if (!IsOrContainsPointerType(*var.output->Type()))
      continue;

    // Make the recursion variable argument point to what the result register points to
    auto & argumentRegister = *var.recref;
    auto & resultRegister = *var.result->origin();
    const auto argumentRegisterPO = Set_->GetRegisterPointerObject(argumentRegister);
    const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
    Constraints_->AddConstraint(SupersetConstraint(argumentRegisterPO, resultRegisterPO));

    // Map the output register to the recursion result's pointer object
    auto & outputRegister = *var.output;
    Set_->MapRegisterToExistingPointerObject(outputRegister, resultRegisterPO);
  }
}

void
Andersen::AnalyzeGamma(const rvsdg::GammaNode & gamma)
{
  // Handle input variables
  for (const auto & ev : gamma.GetEntryVars())
  {
    if (!IsOrContainsPointerType(*ev.input->Type()))
      continue;

    auto & inputRegister = *ev.input->origin();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);

    for (auto & argument : ev.branchArgument)
      Set_->MapRegisterToExistingPointerObject(*argument, inputRegisterPO);
  }

  // Handle subregions
  for (size_t n = 0; n < gamma.nsubregions(); n++)
    AnalyzeRegion(*gamma.subregion(n));

  // Handle exit variables
  for (const auto & ex : gamma.GetExitVars())
  {
    if (!IsOrContainsPointerType(*ex.output->Type()))
      continue;

    auto & outputRegister = ex.output;
    const auto outputRegisterPO = Set_->CreateRegisterPointerObject(*outputRegister);

    for (auto result : ex.branchResult)
    {
      const auto resultRegisterPO = Set_->GetRegisterPointerObject(*result->origin());
      Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, resultRegisterPO));
    }
  }
}

void
Andersen::AnalyzeTheta(const rvsdg::ThetaNode & theta)
{
  // Create a PointerObject for each argument in the inner region
  // And make it point to a superset of the corresponding input register
  for (const auto & loopVar : theta.GetLoopVars())
  {
    if (!IsOrContainsPointerType(*loopVar.input->Type()))
      continue;

    auto & inputReg = *loopVar.input->origin();
    auto & innerArgumentReg = *loopVar.pre;
    const auto inputRegPO = Set_->GetRegisterPointerObject(inputReg);
    const auto innerArgumentRegPO = Set_->CreateRegisterPointerObject(innerArgumentReg);

    // The inner argument can point to anything the input did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, inputRegPO));
  }

  AnalyzeRegion(*theta.subregion());

  // Iterate over loop variables again, making the inner arguments point to a superset
  // of what the corresponding result registers point to
  for (const auto & loopVar : theta.GetLoopVars())
  {
    if (!IsOrContainsPointerType(*loopVar.input->Type()))
      continue;

    auto & innerArgumentReg = *loopVar.pre;
    auto & innerResultReg = *loopVar.post->origin();
    auto & outputReg = *loopVar.output;

    const auto innerArgumentRegPO = Set_->GetRegisterPointerObject(innerArgumentReg);
    const auto innerResultRegPO = Set_->GetRegisterPointerObject(innerResultReg);

    // The inner argument can point to anything the result of last iteration did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, innerResultRegPO));

    // Due to theta nodes running at least once, the output always comes from the inner results
    Set_->MapRegisterToExistingPointerObject(outputReg, innerResultRegPO);
  }
}

void
Andersen::AnalyzeRegion(rvsdg::Region & region)
{
  // Check that all region arguments of pointing types have PointerObjects
  for (size_t i = 0; i < region.narguments(); i++)
  {
    if (IsOrContainsPointerType(*region.argument(i)->Type()))
      JLM_ASSERT(Set_->GetRegisterMap().count(region.argument(i)));
  }

  // The use of the top-down traverser is vital, as it ensures all input origins
  // of pointer type are mapped to PointerObjects by the time a node is processed.
  rvsdg::TopDownTraverser traverser(&region);

  // While visiting the node we have the responsibility of creating
  // PointerObjects for any of the node's outputs of pointer type
  for (const auto node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(node))
      AnalyzeSimpleNode(*simpleNode);
    else if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(node))
      AnalyzeStructuralNode(*structuralNode);
    else
      JLM_UNREACHABLE("Unknown node type");

    // Check that all outputs with pointing types have PointerObjects created
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      if (IsOrContainsPointerType(*node->output(i)->Type()))
        JLM_ASSERT(Set_->GetRegisterMap().count(node->output(i)));
    }
  }
}

void
Andersen::AnalyzeRvsdg(const rvsdg::Graph & graph)
{
  auto & rootRegion = graph.GetRootRegion();

  // Iterate over all arguments to the root region - symbols imported from other modules
  // These symbols can either be global variables or functions
  for (size_t n = 0; n < rootRegion.narguments(); n++)
  {
    auto & argument = *util::AssertedCast<GraphImport>(rootRegion.argument(n));

    // Only care about imported pointer values
    if (!IsOrContainsPointerType(*argument.Type()))
      continue;

    // Create a memory PointerObject representing the target of the external symbol
    // We can assume that two external symbols don't alias, clang does.
    // Imported memory objects are always marked as CanPoint() == false, due to the fact that
    // the analysis can't ever hope to track points-to sets of external memory with any precision.
    const auto importObjectPO = Set_->CreateImportMemoryObject(argument);

    // Create a register PointerObject representing the address value itself
    const auto importRegisterPO = Set_->CreateRegisterPointerObject(argument);
    Constraints_->AddPointerPointeeConstraint(importRegisterPO, importObjectPO);
  }

  AnalyzeRegion(rootRegion);

  // Mark all results escaping the root module as escaped
  for (size_t n = 0; n < rootRegion.nresults(); n++)
  {
    auto & escapedRegister = *rootRegion.result(n)->origin();
    if (!IsOrContainsPointerType(*escapedRegister.Type()))
      continue;

    const auto escapedRegisterPO = Set_->GetRegisterPointerObject(escapedRegister);
    Constraints_->AddRegisterContentEscapedConstraint(escapedRegisterPO);
  }
}

void
Andersen::SetConfiguration(Configuration config)
{
  Config_ = std::move(config);
}

const Andersen::Configuration &
Andersen::GetConfiguration() const
{
  return Config_;
}

void
Andersen::AnalyzeModule(const rvsdg::RvsdgModule & module, Statistics & statistics)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  statistics.StartSetAndConstraintBuildingStatistics();
  AnalyzeRvsdg(module.Rvsdg());
  statistics.StopSetAndConstraintBuildingStatistics(*Set_, *Constraints_);
}

void
Andersen::SolveConstraints(
    PointerObjectConstraintSet & constraints,
    const Configuration & config,
    Statistics & statistics)
{
  statistics.AddStatisticFromConfiguration(config);

  if (config.IsOfflineVariableSubstitutionEnabled())
  {
    statistics.StartOfflineVariableSubstitution();
    // If the solver uses hybrid cycle detection, tell OVS to store info about ref node cycles
    bool hasHCD = config.IsHybridCycleDetectionEnabled();
    auto numUnifications = constraints.PerformOfflineVariableSubstitution(hasHCD);
    statistics.StopOfflineVariableSubstitution(numUnifications);
  }

  if (config.IsOfflineConstraintNormalizationEnabled())
  {
    statistics.StartOfflineConstraintNormalization();
    auto numConstraintsRemoved = constraints.NormalizeConstraints();
    statistics.StopOfflineConstraintNormalization(numConstraintsRemoved);
  }

  if (config.GetSolver() == Configuration::Solver::Naive)
  {
    statistics.StartConstraintSolvingNaiveStatistics();
    size_t numIterations = constraints.SolveNaively();
    statistics.StopConstraintSolvingNaiveStatistics(numIterations);
  }
  else if (config.GetSolver() == Configuration::Solver::Worklist)
  {
    statistics.StartConstraintSolvingWorklistStatistics();
    auto worklistStatistics = constraints.SolveUsingWorklist(
        config.GetWorklistSoliverPolicy(),
        config.IsOnlineCycleDetectionEnabled(),
        config.IsHybridCycleDetectionEnabled(),
        config.IsLazyCycleDetectionEnabled(),
        config.IsDifferencePropagationEnabled(),
        config.IsPreferImplicitPointeesEnabled());
    statistics.StopConstraintSolvingWorklistStatistics(worklistStatistics);
  }
  else
    JLM_UNREACHABLE("Unknown solver");
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(
    const rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(module.SourceFilePath().value());
  statistics->StartAndersenStatistics(module.Rvsdg());

  // Check environment variables for debugging flags
  size_t testAllConfigsIterations = 0;
  if (auto testAllConfigsString = std::getenv(ENV_TEST_ALL_CONFIGS))
    testAllConfigsIterations = std::stoi(testAllConfigsString);
  std::optional<size_t> useExactConfig;
  if (auto useExactConfigString = std::getenv(ENV_USE_EXACT_CONFIG))
    useExactConfig = std::stoi(useExactConfigString);
  const bool doubleCheck = std::getenv(ENV_DOUBLE_CHECK);

  const bool dumpGraphs = std::getenv(ENV_DUMP_SUBSET_GRAPH);
  util::graph::Writer writer;

  AnalyzeModule(module, *statistics);

  // If solving multiple times, make a copy of the original constraint set
  std::pair<std::unique_ptr<PointerObjectSet>, std::unique_ptr<PointerObjectConstraintSet>> copy;
  if (testAllConfigsIterations || doubleCheck)
    copy = Constraints_->Clone();

  // Draw subset graph both before and after solving
  if (dumpGraphs)
    Constraints_->DrawSubsetGraph(writer);

  auto config = Config_;
  if (useExactConfig.has_value())
  {
    auto allConfigs = Configuration::GetAllConfigurations();
    config = allConfigs.at(*useExactConfig);
  }

  SolveConstraints(*Constraints_, config, *statistics);
  statistics->AddStatisticsFromSolution(*Set_);

  if (dumpGraphs)
  {
    auto & graph = Constraints_->DrawSubsetGraph(writer);
    graph.AppendToLabel("After Solving with " + config.ToString());
    writer.OutputAllGraphs(std::cout, util::graph::OutputFormat::Dot);
  }

  auto result = ConstructPointsToGraphFromPointerObjectSet(*Set_, *statistics);

  statistics->StopAndersenStatistics();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Solve again if double-checking against naive is enabled
  if (testAllConfigsIterations || doubleCheck)
  {
    if (doubleCheck)
      std::cerr << "Double checking Andersen analysis using naive solving" << std::endl;

    // If double-checking, only use the naive configuration. Otherwise, try all configurations
    std::vector<Configuration> configs;
    if (testAllConfigsIterations)
      configs = Configuration::GetAllConfigurations();
    else
      configs.push_back(Configuration::NaiveSolverConfiguration());

    // If testing all configurations, do it as many times as requested.
    // Otherwise, do it at least once
    const auto iterations = std::max<size_t>(testAllConfigsIterations, 1);

    for (size_t i = 0; i < iterations; i++)
    {
      for (const auto & config : configs)
      {
        // Create a clone of the unsolved pointer object set and constraint set
        auto workingCopy = copy.second->Clone();
        // These statistics will only contain solving data
        auto solvingStats = Statistics::Create(module.SourceFilePath().value());
        SolveConstraints(*workingCopy.second, config, *solvingStats);
        solvingStats->AddStatisticsFromSolution(*workingCopy.first);
        statisticsCollector.CollectDemandedStatistics(std::move(solvingStats));

        // Only double check on the first iteration
        if (doubleCheck && i == 0)
        {
          if (workingCopy.first->HasIdenticalSolAs(*Set_))
            continue;
          std::cerr << "Solving with original config: " << Config_.ToString()
                    << " did not produce the same solution as the config " << config.ToString()
                    << std::endl;
          JLM_UNREACHABLE("Andersen solver double checking uncovered differences!");
        }
      }
    }
  }

  // Cleanup
  Constraints_.reset();
  Set_.reset();
  return result;
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule & module)
{
  util::StatisticsCollector statisticsCollector;
  return Analyze(module, statisticsCollector);
}

std::unique_ptr<PointsToGraph>
Andersen::ConstructPointsToGraphFromPointerObjectSet(
    const PointerObjectSet & set,
    Statistics & statistics)
{
  statistics.StartPointsToGraphConstructionStatistics();

  auto pointsToGraph = PointsToGraph::Create();

  // memory nodes are the nodes that can be pointed to in the points-to graph.
  // This vector has the same indexing as the nodes themselves, register nodes become nullptr.
  std::vector<PointsToGraph::MemoryNode *> memoryNodes(set.NumPointerObjects());

  // Nodes that should point to external in the final graph.
  // They also get explicit edges connecting them to all escaped memory nodes.
  std::vector<PointsToGraph::Node *> pointsToExternal;

  // A list of all memory nodes that have been marked as escaped
  std::vector<PointsToGraph::MemoryNode *> escapedMemoryNodes;

  // First all memory nodes are created
  for (auto [allocaNode, pointerObjectIndex] : set.GetAllocaMap())
  {
    auto & node = PointsToGraph::AllocaNode::Create(*pointsToGraph, *allocaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [mallocNode, pointerObjectIndex] : set.GetMallocMap())
  {
    auto & node = PointsToGraph::MallocNode::Create(*pointsToGraph, *mallocNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [deltaNode, pointerObjectIndex] : set.GetGlobalMap())
  {
    auto & node = PointsToGraph::DeltaNode::Create(*pointsToGraph, *deltaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [lambdaNode, pointerObjectIndex] : set.GetFunctionMap())
  {
    auto & node = PointsToGraph::LambdaNode::Create(*pointsToGraph, *lambdaNode);
    memoryNodes[pointerObjectIndex] = &node;
  }
  for (auto [argument, pointerObjectIndex] : set.GetImportMap())
  {
    auto & node = PointsToGraph::ImportNode::Create(*pointsToGraph, *argument);
    memoryNodes[pointerObjectIndex] = &node;
  }

  // Helper function for attaching PointsToGraph nodes to their pointees, based on the
  // PointerObject's points-to set.
  auto applyPointsToSet = [&](PointsToGraph::Node & node, PointerObjectIndex index)
  {
    // Add all PointsToGraph nodes who should point to external to the list
    if (set.IsPointingToExternal(index))
      pointsToExternal.push_back(&node);

    for (const auto targetIdx : set.GetPointsToSet(index).Items())
    {
      // Only PointerObjects corresponding to memory nodes can be members of points-to sets
      JLM_ASSERT(memoryNodes[targetIdx]);
      node.AddEdge(*memoryNodes[targetIdx]);
    }
  };

  // First group RVSDG registers by the PointerObject they are mapped to.
  // If the PointerObject is part of a unification, all Register PointerObjects in the unification
  // share points-to set, so they can all become one RegisterNode in the PointsToGraph.
  std::unordered_map<PointerObjectIndex, util::HashSet<const rvsdg::Output *>> outputsInRegister;
  for (auto [outputNode, registerIdx] : set.GetRegisterMap())
  {
    auto root = set.GetUnificationRoot(registerIdx);
    outputsInRegister[root].Insert(outputNode);
  }

  // Create PointsToGraph::RegisterNodes for each PointerObject of register kind, and add edges
  for (auto & [registerIdx, outputNodes] : outputsInRegister)
  {
    auto & node = PointsToGraph::RegisterNode::Create(*pointsToGraph, std::move(outputNodes));
    applyPointsToSet(node, registerIdx);
  }

  // Now add all edges from memory node to memory node.
  // Also checks and informs the PointsToGraph which memory nodes are marked as escaping the module
  for (PointerObjectIndex idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (memoryNodes[idx] == nullptr)
      continue; // Skip all nodes that are not MemoryNodes

    // Add outgoing edges to nodes representing pointer values
    if (set.CanPoint(idx))
      applyPointsToSet(*memoryNodes[idx], idx);

    if (set.HasEscaped(idx))
    {
      memoryNodes[idx]->MarkAsModuleEscaping();
      escapedMemoryNodes.push_back(memoryNodes[idx]);
    }
  }

  // Finally make all nodes marked as pointing to external, point to all escaped memory nodes
  statistics.StartExternalToAllEscapedStatistics();
  for (const auto source : pointsToExternal)
  {
    for (const auto target : escapedMemoryNodes)
    {
      source->AddEdge(*target);
    }
    // Add an edge to the special PointsToGraph node called "external" as well
    source->AddEdge(pointsToGraph->GetExternalMemoryNode());
  }
  statistics.StopExternalToAllEscapedStatistics();

  // We do not use the unknown node, and do not give the external node any targets
  JLM_ASSERT(pointsToGraph->GetExternalMemoryNode().NumTargets() == 0);
  JLM_ASSERT(pointsToGraph->GetUnknownMemoryNode().NumSources() == 0);
  JLM_ASSERT(pointsToGraph->GetUnknownMemoryNode().NumTargets() == 0);

  statistics.StopPointsToGraphConstructionStatistics(*pointsToGraph);
  return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Andersen::ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set)
{
  // Create a throwaway instance of statistics
  Statistics statistics(util::FilePath(""));
  return ConstructPointsToGraphFromPointerObjectSet(set, statistics);
}

}
