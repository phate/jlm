/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
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
IsOrContainsPointerType(const rvsdg::type & type)
{
  return IsOrContains<PointerType>(type);
}

Andersen::Configuration
Andersen::Configuration::DefaultConfiguration()
{
  Configuration config;

  const auto configString = std::getenv(ENV_CONFIG_OVERRIDE);
  if (configString == nullptr)
    return config;

  std::istringstream configStream(configString);
  std::string option;
  while (true)
  {
    configStream >> option;
    if (configStream.fail())
      break;

    using Policy = PointerObjectConstraintSet::WorklistSolverPolicy;

    if (option == CONFIG_OVS_ON)
      config.EnableOfflineVariableSubstitution(true);
    else if (option == CONFIG_OVS_OFF)
      config.EnableOfflineVariableSubstitution(false);

    else if (option == CONFIG_NORMALIZE_ON)
      config.EnableOfflineConstraintNormalization(true);
    else if (option == CONFIG_NORMALIZE_OFF)
      config.EnableOfflineConstraintNormalization(false);

    else if (option == CONFIG_SOLVER_NAIVE)
      config.SetSolver(Solver::Naive);
    else if (option == CONFIG_SOLVER_WL)
      config.SetSolver(Solver::Worklist);

    else if (option == CONFIG_WL_POLICY_LRF)
      config.SetWorklistSolverPolicy(Policy::LeastRecentlyFired);
    else if (option == CONFIG_WL_POLICY_TWO_PHASE_LRF)
      config.SetWorklistSolverPolicy(Policy::TwoPhaseLeastRecentlyFired);
    else if (option == CONFIG_WL_POLICY_FIFO)
      config.SetWorklistSolverPolicy(Policy::FirstInFirstOut);
    else if (option == CONFIG_WL_POLICY_LIFO)
      config.SetWorklistSolverPolicy(Policy::LastInFirstOut);

    else if (option == CONFIG_ONLINE_CYCLE_DETECTION_ON)
      config.EnableOnlineCycleDetection(true);
    else if (option == CONFIG_ONLINE_CYCLE_DETECTION_OFF)
      config.EnableOnlineCycleDetection(false);

    else if (option == CONFIG_LAZY_CYCLE_DETECTION_ON)
      config.EnableLazyCycleDetection(true);
    else if (option == CONFIG_LAZY_CYCLE_DETECTION_OFF)
      config.EnableLazyCycleDetection(false);

    else if (option == CONFIG_DIFFERENCE_PROPAGATION_ON)
      config.EnableDifferencePropagation(true);
    else if (option == CONFIG_DIFFERENCE_PROPAGATION_OFF)
      config.EnableDifferencePropagation(false);

    else if (option == CONFIG_PREFER_IMPLICIT_PROPAGATION_ON)
      config.EnablePreferImplicitPropagation(true);
    else if (option == CONFIG_PREFER_IMPLICIT_PROPAGATION_OFF)
      config.EnablePreferImplicitPropagation(false);

    else
    {
      std::cerr << "Unknown config option string: '" << option << "'" << std::endl;
      JLM_UNREACHABLE("Andersen default config override string broken");
    }
  }

  return config;
}

/**
 * Class collecting statistics from a pass of Andersen's alias analysis
 */
class Andersen::Statistics final : public util::Statistics
{
  static constexpr const char * NumPointerObjects_ = "#PointerObjects";
  static constexpr const char * NumPointerObjectsWithImplicitPointees_ =
      "#PointerObjectsWithImplicitPointees";
  static constexpr const char * NumRegisterPointerObjects_ = "#RegisterPointerObjects";
  static constexpr const char * NumRegistersMappedToPointerObject_ =
      "#RegistersMappedToPointerObject";

  static constexpr const char * NumSupersetConstraints_ = "#SupersetConstraints";
  static constexpr const char * NumStoreConstraints_ = "#StoreConstraints";
  static constexpr const char * NumLoadConstraints_ = "#LoadConstraints";
  static constexpr const char * NumFunctionCallConstraints_ = "#FunctionCallConstraints";
  // Includes base constraints and flags
  static constexpr const char * NumConstraintsTotal_ = "#ConstraintsTotal";

  // Offline technique statistics
  static constexpr const char * NumUnificationsOvs_ = "#Unifications(OVS)";
  static constexpr const char * NumConstraintsRemovedOfflineNorm_ =
      "#ConstraintsRemoved(OfflineNorm)";

  // Solver statistics
  static constexpr const char * NumNaiveSolverIterations_ = "#NaiveSolverIterations";

  static constexpr const char * WorklistPolicy_ = "WorklistPolicy";
  static constexpr const char * NumWorklistSolverWorkItemsPopped_ =
      "#WorklistSolverWorkItemsPopped";

  // Online technique statistics
  static constexpr const char * NumOnlineCyclesDetected_ = "#OnlineCyclesDetected";
  static constexpr const char * NumOnlineCycleUnifications_ = "#OnlineCycleUnifications";

  static constexpr const char * NumLazyCycleDetectionAttempts_ = "#LazyCycleDetectionAttempts";
  static constexpr const char * NumLazyCyclesDetected_ = "#LazyCyclesDetected";
  static constexpr const char * NumLazyCycleUnifications_ = "#LazyCycleUnifications";

  // After solving statistics
  static constexpr const char * NumEscapedMemoryObjects_ = "#EscapedMemoryObjects";
  static constexpr const char * NumUnificationRoots_ = "#UnificationRoots";
  // These next measurements only count flags and pointees of unification roots
  static constexpr const char * NumPointsToExternalFlags_ = "#PointsToExternalFlags";
  static constexpr const char * NumPointeesEscapingFlags_ = "#PointeesEscapingFlags";
  static constexpr const char * NumExplicitPointees_ = "#ExplicitPointees";
  // If a pointee is both implicit (through PointsToExternal flag) and explicit
  static constexpr const char * NumDoubledUpPointees_ = "#DoubledUpPointees";

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

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::AndersenAnalysis, sourceFile)
  {}

  void
  StartAndersenStatistics(const rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(graph.root()));
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

    AddMeasurement(NumPointerObjects_, set.NumPointerObjects());
    AddMeasurement(
        NumPointerObjectsWithImplicitPointees_,
        set.NumPointerObjectsWithImplicitPointees());
    AddMeasurement(
        NumRegisterPointerObjects_,
        set.NumPointerObjectsOfKind(PointerObjectKind::Register));
    AddMeasurement(NumRegistersMappedToPointerObject_, set.GetRegisterMap().size());

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
    AddMeasurement(NumSupersetConstraints_, numSupersetConstraints);
    AddMeasurement(NumStoreConstraints_, numStoreConstraints);
    AddMeasurement(NumLoadConstraints_, numLoadConstraints);
    AddMeasurement(NumFunctionCallConstraints_, numFunctionCallConstraints);
    AddMeasurement(NumConstraintsTotal_, constraints.GetTotalConstraintCount());
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

    if (statistics.NumOnlineCyclesDetected)
      AddMeasurement(NumOnlineCyclesDetected_, *statistics.NumOnlineCyclesDetected);

    if (statistics.NumOnlineCycleUnifications)
      AddMeasurement(NumOnlineCycleUnifications_, *statistics.NumOnlineCycleUnifications);

    if (statistics.NumLazyCyclesDetectionAttempts)
      AddMeasurement(NumLazyCycleDetectionAttempts_, *statistics.NumLazyCyclesDetectionAttempts);

    if (statistics.NumLazyCyclesDetected)
      AddMeasurement(NumLazyCyclesDetected_, *statistics.NumLazyCyclesDetected);

    if (statistics.NumLazyCycleUnifications)
      AddMeasurement(NumLazyCycleUnifications_, *statistics.NumLazyCycleUnifications);
  }

  void
  AddStatisticsFromSolution(const PointerObjectSet & set)
  {
    size_t numEscapedMemoryObjects = 0;
    size_t numUnificationRoots = 0;
    size_t numPointsToExternalFlags = 0;
    size_t numPointeesEscapingFlags = 0;
    size_t numExplicitPointees = 0;
    size_t numDoubleUpPointees = 0;

    for (PointerObjectIndex i = 0; i < set.NumPointerObjects(); i++)
    {
      if (set.HasEscaped(i))
        numEscapedMemoryObjects++;

      if (!set.IsUnificationRoot(i))
        continue;

      numUnificationRoots++;
      if (set.IsPointingToExternal(i))
        numPointsToExternalFlags++;
      if (set.HasPointeesEscaping(i))
        numPointeesEscapingFlags++;

      const auto & pointees = set.GetPointsToSet(i);
      numExplicitPointees += pointees.Size();

      // If the PointsToExternal flag is set, any explicit pointee that has escaped is doubled up
      if (set.IsPointingToExternal(i))
        for (auto pointee : pointees.Items())
          if (set.HasEscaped(pointee))
            numDoubleUpPointees++;
    }
    AddMeasurement(NumEscapedMemoryObjects_, numEscapedMemoryObjects);
    AddMeasurement(NumUnificationRoots_, numUnificationRoots);
    AddMeasurement(NumPointsToExternalFlags_, numPointsToExternalFlags);
    AddMeasurement(NumPointeesEscapingFlags_, numPointeesEscapingFlags);
    AddMeasurement(NumExplicitPointees_, numExplicitPointees);
    AddMeasurement(NumDoubledUpPointees_, numDoubleUpPointees);
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
  }

  void
  StopAndersenStatistics() noexcept
  {
    GetTimer(AnalysisTimer_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

void
Andersen::AnalyzeSimpleNode(const rvsdg::simple_node & node)
{
  const auto & op = node.operation();

  if (is<alloca_op>(op))
    AnalyzeAlloca(node);
  else if (is<malloc_op>(op))
    AnalyzeMalloc(node);
  else if (const auto loadNode = dynamic_cast<const LoadNode *>(&node))
    AnalyzeLoad(*loadNode);
  else if (const auto storeNode = dynamic_cast<const StoreNode *>(&node))
    AnalyzeStore(*storeNode);
  else if (const auto callNode = dynamic_cast<const CallNode *>(&node))
    AnalyzeCall(*callNode);
  else if (is<GetElementPtrOperation>(op))
    AnalyzeGep(node);
  else if (is<bitcast_op>(op))
    AnalyzeBitcast(node);
  else if (is<bits2ptr_op>(op))
    AnalyzeBits2ptr(node);
  else if (is<ptr2bits_op>(op))
    AnalyzePtr2bits(node);
  else if (is<ConstantPointerNullOperation>(op))
    AnalyzeConstantPointerNull(node);
  else if (is<UndefValueOperation>(op))
    AnalyzeUndef(node);
  else if (is<MemCpyOperation>(op))
    AnalyzeMemcpy(node);
  else if (is<ConstantArray>(op))
    AnalyzeConstantArray(node);
  else if (is<ConstantStruct>(op))
    AnalyzeConstantStruct(node);
  else if (is<ConstantAggregateZero>(op))
    AnalyzeConstantAggregateZero(node);
  else if (is<ExtractValue>(op))
    AnalyzeExtractValue(node);
  else if (is<valist_op>(op))
    AnalyzeValist(node);
  else if (is<FreeOperation>(op) || is<ptrcmp_op>(op))
  {
    // These operations take pointers as input, but do not affect any points-to sets
  }
  else
  {
    // This node operation is unknown, make sure it doesn't consume any pointers
    for (size_t i = 0; i < node.ninputs(); i++)
    {
      JLM_ASSERT(!IsOrContainsPointerType(node.input(i)->type()));
    }
  }
}

void
Andersen::AnalyzeAlloca(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  const auto allocaPO = Set_->CreateAllocaMemoryObject(node);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, allocaPO);
}

void
Andersen::AnalyzeMalloc(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  const auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  const auto mallocPO = Set_->CreateMallocMemoryObject(node);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, mallocPO);
}

void
Andersen::AnalyzeLoad(const LoadNode & loadNode)
{
  const auto & addressRegister = *loadNode.GetAddressInput().origin();
  const auto & outputRegister = loadNode.GetLoadedValueOutput();

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);

  if (!IsOrContainsPointerType(outputRegister.type()))
  {
    // TODO: When reading address as an integer, some of address' target might still pointers,
    // which should now be considered as having escaped
    return;
  }

  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  Constraints_->AddConstraint(LoadConstraint(outputRegisterPO, addressRegisterPO));
}

void
Andersen::AnalyzeStore(const StoreNode & storeNode)
{
  const auto & addressRegister = *storeNode.GetAddressInput().origin();
  const auto & valueRegister = *storeNode.GetStoredValueInput().origin();

  // If the written value is not a pointer, be conservative and mark the address
  if (!IsOrContainsPointerType(valueRegister.type()))
  {
    // TODO: We are writing an integer to *address,
    // which really should mark all of address' targets as pointing to external
    // in case they are ever read as pointers.
    return;
  }

  const auto addressRegisterPO = Set_->GetRegisterPointerObject(addressRegister);
  const auto valueRegisterPO = Set_->GetRegisterPointerObject(valueRegister);
  Constraints_->AddConstraint(StoreConstraint(addressRegisterPO, valueRegisterPO));
}

void
Andersen::AnalyzeCall(const CallNode & callNode)
{
  // The address being called by the call node
  const auto & callTarget = *callNode.GetFunctionInput()->origin();
  const auto callTargetPO = Set_->GetRegisterPointerObject(callTarget);

  // Create PointerObjects for all output values of pointer type
  for (size_t n = 0; n < callNode.NumResults(); n++)
  {
    const auto & outputRegister = *callNode.Result(n);
    if (IsOrContainsPointerType(outputRegister.type()))
      (void)Set_->CreateRegisterPointerObject(outputRegister);
  }

  // We make no attempt at detecting what type of call this is here.
  // The logic handling external and indirect calls is done by the FunctionCallConstraint.
  // Passing points-to-sets from call-site to function bodies is done fully by this constraint.
  Constraints_->AddConstraint(FunctionCallConstraint(callTargetPO, callNode));
}

void
Andersen::AnalyzeGep(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(&node));

  // The analysis is field insensitive, so ignoring the offset and mapping the output
  // to the same PointerObject as the input is sufficient.
  const auto & baseRegister = *node.input(0)->origin();
  JLM_ASSERT(is<PointerType>(baseRegister.type()));

  const auto baseRegisterPO = Set_->GetRegisterPointerObject(baseRegister);
  const auto & outputRegister = *node.output(0);
  Set_->MapRegisterToExistingPointerObject(outputRegister, baseRegisterPO);
}

void
Andersen::AnalyzeBitcast(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bitcast_op>(&node));

  const auto & inputRegister = *node.input(0)->origin();
  const auto & outputRegister = *node.output(0);

  JLM_ASSERT(!IsAggregateType(inputRegister.type()) && !IsAggregateType(outputRegister.type()));
  if (!IsOrContainsPointerType(inputRegister.type()))
    return;

  // If the input is a pointer type, the output must also be a pointer type
  JLM_ASSERT(IsOrContainsPointerType(outputRegister.type()));

  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  Set_->MapRegisterToExistingPointerObject(outputRegister, inputRegisterPO);
}

void
Andersen::AnalyzeBits2ptr(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bits2ptr_op>(&node));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.type()));

  // This operation synthesizes a pointer from bytes.
  // Since no points-to information is tracked through integers, the resulting pointer must
  // be assumed to possibly point to any external or escaped memory object.
  const auto outputPO = Set_->CreateRegisterPointerObject(output);
  Constraints_->AddPointsToExternalConstraint(outputPO);
}

void
Andersen::AnalyzePtr2bits(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ptr2bits_op>(&node));
  const auto & inputRegister = *node.input(0)->origin();
  JLM_ASSERT(is<PointerType>(inputRegister.type()));

  // This operation converts a pointer to bytes, exposing it as an integer, which we can't track.
  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
}

void
Andersen::AnalyzeConstantPointerNull(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));
  const auto & output = *node.output(0);
  JLM_ASSERT(is<PointerType>(output.type()));

  // ConstantPointerNull cannot point to any memory location. We therefore only insert a register
  // node for it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeUndef(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  const auto & output = *node.output(0);

  if (!IsOrContainsPointerType(output.type()))
    return;

  // UndefValue cannot point to any memory location. We therefore only insert a register node for
  // it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeMemcpy(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<MemCpyOperation>(&node));

  auto & dstAddressRegister = *node.input(0)->origin();
  auto & srcAddressRegister = *node.input(1)->origin();
  JLM_ASSERT(is<PointerType>(dstAddressRegister.type()));
  JLM_ASSERT(is<PointerType>(srcAddressRegister.type()));

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
Andersen::AnalyzeConstantArray(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantArray>(&node));

  if (!IsOrContainsPointerType(node.output(0)->type()))
    return;

  // Make the resulting array point to everything its members are pointing to
  auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    JLM_ASSERT(IsOrContainsPointerType(inputRegister.type()));

    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, inputRegisterPO));
  }
}

void
Andersen::AnalyzeConstantStruct(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  if (!IsOrContainsPointerType(node.output(0)->type()))
    return;

  // Make the resulting struct point to everything its members are pointing to
  auto & outputRegister = *node.output(0);
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    const auto & inputRegister = *node.input(n)->origin();
    if (!IsOrContainsPointerType(inputRegister.type()))
      continue;

    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, inputRegisterPO));
  }
}

void
Andersen::AnalyzeConstantAggregateZero(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantAggregateZero>(&node));
  auto & output = *node.output(0);

  if (!IsOrContainsPointerType(output.type()))
    return;

  // ConstantAggregateZero cannot point to any memory location.
  // We therefore only insert a register node for it, but let this node not point to anything.
  (void)Set_->CreateRegisterPointerObject(output);
}

void
Andersen::AnalyzeExtractValue(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ExtractValue>(&node));

  const auto & result = *node.output(0);
  if (!IsOrContainsPointerType(result.type()))
    return;

  const auto & inputRegister = *node.input(0)->origin();
  const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
  // The resulting element can point to anything the aggregate type points to
  Set_->MapRegisterToExistingPointerObject(result, inputRegisterPO);
}

void
Andersen::AnalyzeValist(const rvsdg::simple_node & node)
{
  JLM_ASSERT(is<valist_op>(&node));

  // Members of the valist are extracted using the va_arg macro, which loads from the va_list struct
  // on the stack. This struct will be marked as escaped from the call to va_start, and thus point
  // to external. All we need to do is mark all pointees of pointer varargs as escaping. When the
  // pointers are re-created inside the function, they will be marked as pointing to external.

  for (size_t i = 0; i < node.ninputs(); i++)
  {
    if (!IsOrContainsPointerType(node.input(i)->type()))
      continue;

    const auto & inputRegister = *node.input(i)->origin();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Constraints_->AddRegisterContentEscapedConstraint(inputRegisterPO);
  }
}

void
Andersen::AnalyzeStructuralNode(const rvsdg::structural_node & node)
{
  if (const auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
    AnalyzeLambda(*lambdaNode);
  else if (const auto deltaNode = dynamic_cast<const delta::node *>(&node))
    AnalyzeDelta(*deltaNode);
  else if (const auto phiNode = dynamic_cast<const phi::node *>(&node))
    AnalyzePhi(*phiNode);
  else if (const auto gammaNode = dynamic_cast<const rvsdg::gamma_node *>(&node))
    AnalyzeGamma(*gammaNode);
  else if (const auto thetaNode = dynamic_cast<const rvsdg::theta_node *>(&node))
    AnalyzeTheta(*thetaNode);
  else
    JLM_UNREACHABLE("Unknown structural node operation");
}

void
Andersen::AnalyzeLambda(const lambda::node & lambda)
{
  // Handle context variables
  for (auto & cv : lambda.ctxvars())
  {
    if (!IsOrContainsPointerType(cv.type()))
      continue;

    auto & inputRegister = *cv.origin();
    auto & argumentRegister = *cv.argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each argument of pointing type in the function
  for (auto & argument : lambda.fctarguments())
  {
    if (IsOrContainsPointerType(argument.type()))
      (void)Set_->CreateRegisterPointerObject(argument);
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
Andersen::AnalyzeDelta(const delta::node & delta)
{
  // Handle context variables
  for (auto & cv : delta.ctxvars())
  {
    if (!IsOrContainsPointerType(cv.type()))
      continue;

    auto & inputRegister = *cv.origin();
    auto & argumentRegister = *cv.argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  AnalyzeRegion(*delta.subregion());

  // Get the result register from the subregion
  auto & resultRegister = *delta.result()->origin();

  // Create a global memory object representing the global variable
  const auto globalPO = Set_->CreateGlobalMemoryObject(delta);

  // If the subregion result is a pointer, make the global point to the same variables
  if (IsOrContainsPointerType(resultRegister.type()))
  {
    const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
    Constraints_->AddConstraint(SupersetConstraint(globalPO, resultRegisterPO));
  }

  // Finally create a Register PointerObject for the delta's output, pointing to the memory object
  auto & outputRegister = *delta.output();
  const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);
  Constraints_->AddPointerPointeeConstraint(outputRegisterPO, globalPO);
}

void
Andersen::AnalyzePhi(const phi::node & phi)
{
  // Handle context variables
  for (auto cv = phi.begin_cv(); cv != phi.end_cv(); ++cv)
  {
    if (!IsOrContainsPointerType(cv->type()))
      continue;

    auto & inputRegister = *cv->origin();
    auto & argumentRegister = *cv->argument();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);
    Set_->MapRegisterToExistingPointerObject(argumentRegister, inputRegisterPO);
  }

  // Create Register PointerObjects for each recursion variable argument
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); ++rv)
  {
    if (!IsOrContainsPointerType(rv->type()))
      continue;

    auto & argumentRegister = *rv->argument();
    (void)Set_->CreateRegisterPointerObject(argumentRegister);
  }

  AnalyzeRegion(*phi.subregion());

  // Handle recursion variable results
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); ++rv)
  {
    if (!IsOrContainsPointerType(rv->type()))
      continue;

    // Make the recursion variable argument point to what the result register points to
    auto & argumentRegister = *rv->argument();
    auto & resultRegister = *rv->result()->origin();
    const auto argumentRegisterPO = Set_->GetRegisterPointerObject(argumentRegister);
    const auto resultRegisterPO = Set_->GetRegisterPointerObject(resultRegister);
    Constraints_->AddConstraint(SupersetConstraint(argumentRegisterPO, resultRegisterPO));

    // Map the output register to the recursion result's pointer object
    auto & outputRegister = *rv;
    Set_->MapRegisterToExistingPointerObject(outputRegister, resultRegisterPO);
  }
}

void
Andersen::AnalyzeGamma(const rvsdg::gamma_node & gamma)
{
  // Handle input variables
  for (auto ev = gamma.begin_entryvar(); ev != gamma.end_entryvar(); ++ev)
  {
    if (!IsOrContainsPointerType(ev->type()))
      continue;

    auto & inputRegister = *ev->origin();
    const auto inputRegisterPO = Set_->GetRegisterPointerObject(inputRegister);

    for (auto & argument : *ev)
      Set_->MapRegisterToExistingPointerObject(argument, inputRegisterPO);
  }

  // Handle subregions
  for (size_t n = 0; n < gamma.nsubregions(); n++)
    AnalyzeRegion(*gamma.subregion(n));

  // Handle exit variables
  for (auto ex = gamma.begin_exitvar(); ex != gamma.end_exitvar(); ++ex)
  {
    if (!IsOrContainsPointerType(ex->type()))
      continue;

    auto & outputRegister = *ex.output();
    const auto outputRegisterPO = Set_->CreateRegisterPointerObject(outputRegister);

    for (auto & result : *ex)
    {
      const auto resultRegisterPO = Set_->GetRegisterPointerObject(*result.origin());
      Constraints_->AddConstraint(SupersetConstraint(outputRegisterPO, resultRegisterPO));
    }
  }
}

void
Andersen::AnalyzeTheta(const rvsdg::theta_node & theta)
{
  // Create a PointerObject for each argument in the inner region
  // And make it point to a superset of the corresponding input register
  for (const auto thetaOutput : theta)
  {
    if (!IsOrContainsPointerType(thetaOutput->type()))
      continue;

    auto & inputReg = *thetaOutput->input()->origin();
    auto & innerArgumentReg = *thetaOutput->argument();
    const auto inputRegPO = Set_->GetRegisterPointerObject(inputReg);
    const auto innerArgumentRegPO = Set_->CreateRegisterPointerObject(innerArgumentReg);

    // The inner argument can point to anything the input did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, inputRegPO));
  }

  AnalyzeRegion(*theta.subregion());

  // Iterate over loop variables again, making the inner arguments point to a superset
  // of what the corresponding result registers point to
  for (const auto thetaOutput : theta)
  {
    if (!IsOrContainsPointerType(thetaOutput->type()))
      continue;

    auto & innerArgumentReg = *thetaOutput->argument();
    auto & innerResultReg = *thetaOutput->result()->origin();
    auto & outputReg = *thetaOutput;

    const auto innerArgumentRegPO = Set_->GetRegisterPointerObject(innerArgumentReg);
    const auto innerResultRegPO = Set_->GetRegisterPointerObject(innerResultReg);

    // The inner argument can point to anything the result of last iteration did
    Constraints_->AddConstraint(SupersetConstraint(innerArgumentRegPO, innerResultRegPO));

    // Due to theta nodes running at least once, the output always comes from the inner results
    Set_->MapRegisterToExistingPointerObject(outputReg, innerResultRegPO);
  }
}

void
Andersen::AnalyzeRegion(rvsdg::region & region)
{
  // Check that all region arguments of pointing types have PointerObjects
  for (size_t i = 0; i < region.narguments(); i++)
  {
    if (IsOrContainsPointerType(region.argument(i)->type()))
      JLM_ASSERT(Set_->GetRegisterMap().count(region.argument(i)));
  }

  // The use of the top-down traverser is vital, as it ensures all input origins
  // of pointer type are mapped to PointerObjects by the time a node is processed.
  rvsdg::topdown_traverser traverser(&region);

  // While visiting the node we have the responsibility of creating
  // PointerObjects for any of the node's outputs of pointer type
  for (const auto node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::simple_node *>(node))
      AnalyzeSimpleNode(*simpleNode);
    else if (auto structuralNode = dynamic_cast<const rvsdg::structural_node *>(node))
      AnalyzeStructuralNode(*structuralNode);
    else
      JLM_UNREACHABLE("Unknown node type");

    // Check that all outputs with pointing types have PointerObjects created
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      if (IsOrContainsPointerType(node->output(i)->type()))
        JLM_ASSERT(Set_->GetRegisterMap().count(node->output(i)));
    }
  }
}

void
Andersen::AnalyzeRvsdg(const rvsdg::graph & graph)
{
  auto & rootRegion = *graph.root();

  // Iterate over all arguments to the root region - symbols imported from other modules
  // These symbols can either be global variables or functions
  for (size_t n = 0; n < rootRegion.narguments(); n++)
  {
    auto & argument = *rootRegion.argument(n);

    // Only care about imported pointer values
    if (!IsOrContainsPointerType(argument.type()))
      continue;

    // TODO: Mark the created ImportMemoryObject based on it being a function or a variable
    // Functions and non-pointer typed globals can not point to other MemoryObjects,
    // so letting them be ShouldTrackPointees() == false aids analysis.

    // Create a memory PointerObject representing the target of the external symbol
    // We can assume that two external symbols don't alias, clang does.
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
    if (!IsOrContainsPointerType(escapedRegister.type()))
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
Andersen::AnalyzeModule(const RvsdgModule & module, Statistics & statistics)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  statistics.StartSetAndConstraintBuildingStatistics();
  AnalyzeRvsdg(module.Rvsdg());
  statistics.StopSetAndConstraintBuildingStatistics(*Set_, *Constraints_);
}

void
Andersen::SolveConstraints(const Configuration & config, Statistics & statistics)
{
  if (config.IsOfflineVariableSubstitutionEnabled())
  {
    statistics.StartOfflineVariableSubstitution();
    auto numUnifications = Constraints_->PerformOfflineVariableSubstitution();
    statistics.StopOfflineVariableSubstitution(numUnifications);
  }

  if (config.IsOfflineConstraintNormalizationEnabled())
  {
    statistics.StartOfflineConstraintNormalization();
    auto numConstraintsRemoved = Constraints_->NormalizeConstraints();
    statistics.StopOfflineConstraintNormalization(numConstraintsRemoved);
  }

  if (config.GetSolver() == Configuration::Solver::Naive)
  {
    statistics.StartConstraintSolvingNaiveStatistics();
    size_t numIterations = Constraints_->SolveNaively();
    statistics.StopConstraintSolvingNaiveStatistics(numIterations);
  }
  else if (config.GetSolver() == Configuration::Solver::Worklist)
  {
    statistics.StartConstraintSolvingWorklistStatistics();
    auto worklistStatistics = Constraints_->SolveUsingWorklist(
        config.GetWorklistSoliverPolicy(),
        config.IsOnlineCycleDetectionEnabled(),
        config.IsLazyCycleDetectionEnabled(),
        config.IsDifferencePropagationEnabled(),
        config.IsPreferImplicitPropagationEnabled());
    statistics.StopConstraintSolvingWorklistStatistics(worklistStatistics);
  }
  else
    JLM_UNREACHABLE("Unknown solver");

  statistics.AddStatisticsFromSolution(*Set_);
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(module.SourceFileName());
  statistics->StartAndersenStatistics(module.Rvsdg());

  // Check environment variables for debugging flags
  const bool checkAgainstNaive =
      std::getenv(ENV_COMPARE_SOLVE_NAIVE) && Config_ != Configuration::NaiveSolverConfiguration();
  const bool dumpGraphs = std::getenv(ENV_DUMP_SUBSET_GRAPH);
  util::GraphWriter writer;

  AnalyzeModule(module, *statistics);

  // If double-checking against the naive solver, make a copy of the original constraint set
  std::pair<std::unique_ptr<PointerObjectSet>, std::unique_ptr<PointerObjectConstraintSet>> copy;
  if (checkAgainstNaive)
    copy = Constraints_->Clone();

  // Draw subset graph both before and after solving
  if (dumpGraphs)
    Constraints_->DrawSubsetGraph(writer);

  SolveConstraints(Config_, *statistics);

  if (dumpGraphs)
  {
    auto & graph = Constraints_->DrawSubsetGraph(writer);
    graph.AppendToLabel("After Solving");

    writer.OutputAllGraphs(std::cout, util::GraphOutputFormat::Dot);
  }

  auto result = ConstructPointsToGraphFromPointerObjectSet(*Set_, *statistics);

  statistics->StopAndersenStatistics();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Solve again if double-checking against naive is enabled
  if (checkAgainstNaive)
  {
    std::cerr << "Double checking Andersen analysis using naive solving" << std::endl;
    // Restore the problem to before solving started
    Set_ = std::move(copy.first);
    Constraints_ = std::move(copy.second);

    // Create a separate Statistics instance for naive statistics
    auto naiveStatistics = Statistics::Create(module.SourceFileName());
    SolveConstraints(Configuration::NaiveSolverConfiguration(), *naiveStatistics);

    auto naiveResult = ConstructPointsToGraphFromPointerObjectSet(*Set_, *naiveStatistics);

    statisticsCollector.CollectDemandedStatistics(std::move(naiveStatistics));

    // Check if the PointsToGraphs are identical
    bool error = false;
    if (!naiveResult->IsSupergraphOf(*result))
    {
      std::cerr << "The naive PointsToGraph is NOT a supergraph of the PointsToGraph" << std::endl;
      error = true;
    }
    if (!result->IsSupergraphOf(*naiveResult))
    {
      std::cerr << "The PointsToGraph is NOT a supergraph of the naive PointsToGraph" << std::endl;
      error = true;
    }
    if (error)
    {
      std::cout << PointsToGraph::ToDot(*result) << std::endl;
      std::cout << PointsToGraph::ToDot(*naiveResult) << std::endl;
      JLM_UNREACHABLE("PointsToGraph double checking uncovered differences!");
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
    // PointerObjects marked as not tracking pointees should not point to anything
    if (!set.ShouldTrackPointees(index))
      return;

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
  std::unordered_map<PointerObjectIndex, util::HashSet<const rvsdg::output *>> outputsInRegister;
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

  statistics.StopPointsToGraphConstructionStatistics(*pointsToGraph);
  return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Andersen::ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set)
{
  // Create a throwaway instance of statistics
  Statistics statistics(util::filepath(""));
  return ConstructPointsToGraphFromPointerObjectSet(set, statistics);
}

}
