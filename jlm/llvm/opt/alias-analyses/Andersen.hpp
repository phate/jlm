/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToAnalysis.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm::aa
{

/**
 * class implementing Andersen's set constraint based pointer analysis, based on the Ph.D. thesis
 * Lars Ole Andersen - Program Analysis and Specialization for the C Programming Language
 * The analysis is inter-procedural, field-insensitive, context-insensitive,
 * flow-insensitive, and uses a static heap model.
 */
class Andersen final : public PointsToAnalysis
{
  class Statistics;

public:
  /**
   * Environment variable that when set, triggers analyzing the program with every single
   * valid combination of Configuration flags.
   * Must be set to a number, that determines how many times each config is used.
   */
  static inline const char * const ENV_TEST_ALL_CONFIGS = "JLM_ANDERSEN_TEST_ALL_CONFIGS";

  /**
   * Alternative to testing all configs, this environment variable specifies exactly which config to
   * use. It must be an index into the Configuration::GetAllConfigurations() vector.
   * If ENV_TEST_ALL_CONFIGS is also set, it specifies how many times the config is tested.
   */
  static inline const char * const ENV_USE_EXACT_CONFIG = "JLM_ANDERSEN_USE_EXACT_CONFIG";

  /**
   * Environment variable that will trigger double-checking of the analysis.
   * If ENV_TEST_ALL_CONFIGS is set, the output is double-checked against them all.
   * Otherwise, the output is double-checked only against the default naive solver.
   */
  static inline const char * const ENV_DOUBLE_CHECK = "JLM_ANDERSEN_DOUBLE_CHECK";

  /**
   * Environment variable that will trigger dumping the subset graph before and after solving.
   */
  static inline const char * const ENV_DUMP_SUBSET_GRAPH = "JLM_ANDERSEN_DUMP_SUBSET_GRAPH";

  /**
   * class for configuring the Andersen pass, such as what solver to use.
   */
  class Configuration
  {
  private:
    Configuration() = default;

  public:
    enum class Solver
    {
      Naive,
      Worklist
    };

    /**
     * Sets which solver algorithm to use.
     * Not all solvers are compatible with all online techniques.
     */
    void
    SetSolver(Solver solver) noexcept
    {
      Solver_ = solver;
    }

    [[nodiscard]] Solver
    GetSolver() const noexcept
    {
      return Solver_;
    }

    /**
     * Sets which policy to be used by the worklist.
     * Only applies to the worklist solver.
     */
    void
    SetWorklistSolverPolicy(PointerObjectConstraintSet::WorklistSolverPolicy policy) noexcept
    {
      WorklistSolverPolicy_ = policy;
    }

    [[nodiscard]] PointerObjectConstraintSet::WorklistSolverPolicy
    GetWorklistSoliverPolicy() const noexcept
    {
      return WorklistSolverPolicy_;
    }

    /**
     * Enables or disables the use of offline variable substitution to pre-process
     * the constraint set before applying the solving algorithm.
     * The substitution only performs constraint variable unification,
     * which may create opportunities for constraint normalization.
     */
    void
    EnableOfflineVariableSubstitution(bool enable) noexcept
    {
      EnableOfflineVariableSubstitution_ = enable;
    }

    [[nodiscard]] bool
    IsOfflineVariableSubstitutionEnabled() const noexcept
    {
      return EnableOfflineVariableSubstitution_;
    }

    /**
     * Enables or disables offline constraint normalization.
     * If enabled, it is the last step in offline processing.
     * @see PointerObjectConstraintSet::NormalizeConstraints()
     */
    void
    EnableOfflineConstraintNormalization(bool enable) noexcept
    {
      EnableOfflineConstraintNormalization_ = enable;
    }

    [[nodiscard]] bool
    IsOfflineConstraintNormalizationEnabled() const noexcept
    {
      return EnableOfflineConstraintNormalization_;
    }

    /**
     * Enables or disables online cycle detection in the Worklist solver, as described by
     *   Pearce, 2003: "Online cycle detection and difference propagation for pointer analysis"
     * It detects all cycles, so it can not be combined with other cycle detection techniques.
     */
    void
    EnableOnlineCycleDetection(bool enable) noexcept
    {
      EnableOnlineCycleDetection_ = enable;
    }

    [[nodiscard]] bool
    IsOnlineCycleDetectionEnabled() const noexcept
    {
      return EnableOnlineCycleDetection_;
    }

    /**
     * Enables or disables hybrid cycle detection in the Worklist solver, as described by
     *   Hardekopf and Lin, 2007: "The Ant & the Grasshopper"
     * It detects some cycles, so it can not be combined with techniques that find all cycles.
     */
    void
    EnableHybridCycleDetection(bool enable) noexcept
    {
      EnableHybridCycleDetection_ = enable;
    }

    [[nodiscard]] bool
    IsHybridCycleDetectionEnabled() const noexcept
    {
      return EnableHybridCycleDetection_;
    }

    /**
     * Enables or disables lazy cycle detection in the Worklist solver, as described by
     *   Hardekopf and Lin, 2007: "The Ant & the Grasshopper"
     * It detects some cycles, so it can not be combined with techniques that find all cycles.
     */
    void
    EnableLazyCycleDetection(bool enable) noexcept
    {
      EnableLazyCycleDetection_ = enable;
    }

    [[nodiscard]] bool
    IsLazyCycleDetectionEnabled() const noexcept
    {
      return EnableLazyCycleDetection_;
    }

    /**
     * Enables or disables difference propagation in the Worklist solver, as described by
     *   Pearce, 2003: "Online cycle detection and difference propagation for pointer analysis"
     * Only used by the worklist solver.
     */
    void
    EnableDifferencePropagation(bool enable) noexcept
    {
      EnableDifferencePropagation_ = enable;
    }

    [[nodiscard]] bool
    IsDifferencePropagationEnabled() const noexcept
    {
      return EnableDifferencePropagation_;
    }

    /**
     * Enables or disables preferring implicit pointees in the Worklist solver
     */
    void
    EnablePreferImplicitPointees(bool enable) noexcept
    {
      EnablePreferImplicitPointees_ = enable;
    }

    [[nodiscard]] bool
    IsPreferImplicitPointeesEnabled() const noexcept
    {
      return EnablePreferImplicitPointees_;
    }

    [[nodiscard]] std::string
    ToString() const;

    /**
     * @return the default configuration
     */
    static Configuration
    DefaultConfiguration()
    {
      Configuration config;
      config.EnableOfflineVariableSubstitution(true);
      // Constraints are normalized inside the Worklist's representation either way
      config.EnableOfflineConstraintNormalization(false);
      config.SetSolver(Solver::Worklist);
      config.SetWorklistSolverPolicy(
          PointerObjectConstraintSet::WorklistSolverPolicy::LeastRecentlyFired);
      config.EnableOnlineCycleDetection(false);
      config.EnableHybridCycleDetection(true);
      config.EnableLazyCycleDetection(true);
      config.EnableDifferencePropagation(true);
      config.EnablePreferImplicitPointees(true);
      return config;
    }

    /**
     * Creates a solver configuration using the naive solver,
     * with all offline and online speedup techniques disabled.
     * @return the solver configuration
     */
    [[nodiscard]] static Configuration
    NaiveSolverConfiguration() noexcept
    {
      Configuration config;
      config.EnableOfflineVariableSubstitution(false);
      config.EnableOfflineConstraintNormalization(false);
      config.SetSolver(Solver::Naive);
      return config;
    }

    /**
     * @return a list containing all possible Configurations,
     * avoiding useless combinations of techniques.
     */
    [[nodiscard]] static std::vector<Configuration>
    GetAllConfigurations();

  private:
    // All techniques are turned off by default
    bool EnableOfflineVariableSubstitution_ = false;
    bool EnableOfflineConstraintNormalization_ = false;
    Solver Solver_ = Solver::Naive;
    PointerObjectConstraintSet::WorklistSolverPolicy WorklistSolverPolicy_ =
        PointerObjectConstraintSet::WorklistSolverPolicy::LeastRecentlyFired;
    bool EnableOnlineCycleDetection_ = false;
    bool EnableHybridCycleDetection_ = false;
    bool EnableLazyCycleDetection_ = false;
    bool EnableDifferencePropagation_ = false;
    bool EnablePreferImplicitPointees_ = false;
  };

  ~Andersen() noexcept override = default;

  Andersen() = default;

  Andersen(const Andersen &) = delete;

  Andersen(Andersen &&) = delete;

  Andersen &
  operator=(const Andersen &) = delete;

  Andersen &
  operator=(Andersen &&) = delete;

  /**
   * Specify the PassConfiguration the Andersen pass should use when analyzing
   * @param config
   */
  void
  SetConfiguration(Configuration config);

  /**
   * @return the PassConfiguration used by the Andersen pass when analyzing
   */
  [[nodiscard]] const Configuration &
  GetConfiguration() const;

  /**
   * Performs Andersen's alias analysis on the rvsdg \p module,
   * producing a PointsToGraph describing what memory objects exists,
   * and which values in the rvsdg program may point to them.
   * @param module the module to analyze
   * @param statisticsCollector the collector that will receive pass statistics
   * @return A PointsToGraph for the module
   * @see SetConfiguration to configure settings for the analysis
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
      override;

  /**
   * @brief Shorthand for Analyze, ignoring collecting any statistics.
   * @see Analyze
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & module);

  /**
   * Converts a PointerObjectSet into PointsToGraph nodes,
   * and points-to-graph set memberships into edges.
   *
   * In the PointerObjectSet, the PointsToExternal flag encodes pointing to an address available
   * outside the module. This may however be the address of a memory object within the module, that
   * has escaped. In the final PointsToGraph, any node marked as pointing to external, will get an
   * edge to the special "external" node, as well as to every memory object node marked as escaped.
   *
   * @param set the PointerObjectSet to convert
   * @param statistics the statistics instance used to collect statistics about the process
   * @return the newly created PointsToGraph
   */
  [[nodiscard]] static std::unique_ptr<PointsToGraph>
  ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set, Statistics & statistics);

  [[nodiscard]] static std::unique_ptr<PointsToGraph>
  ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set);

private:
  void
  AnalyzeRegion(rvsdg::Region & region);

  void
  AnalyzeSimpleNode(const rvsdg::SimpleNode & node);

  void
  AnalyzeAlloca(const rvsdg::SimpleNode & node);

  void
  AnalyzeMalloc(const rvsdg::SimpleNode & node);

  void
  AnalyzeLoad(const rvsdg::SimpleNode & node);

  void
  AnalyzeStore(const rvsdg::SimpleNode & node);

  void
  AnalyzeCall(const rvsdg::SimpleNode & callNode);

  void
  AnalyzeGep(const rvsdg::SimpleNode & node);

  void
  AnalyzeBitcast(const rvsdg::SimpleNode & node);

  void
  AnalyzeBits2ptr(const rvsdg::SimpleNode & node);

  void
  AnalyzePtrToInt(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantPointerNull(const rvsdg::SimpleNode & node);

  void
  AnalyzeUndef(const rvsdg::SimpleNode & node);

  void
  AnalyzeMemcpy(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantArray(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantStruct(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantAggregateZero(const rvsdg::SimpleNode & node);

  void
  AnalyzeExtractValue(const rvsdg::SimpleNode & node);

  void
  AnalyzeValist(const rvsdg::SimpleNode & node);

  void
  AnalyzePointerToFunction(const rvsdg::SimpleNode & node);

  void
  AnalyzeFunctionToPointer(const rvsdg::SimpleNode & node);

  void
  AnalyzeIOBarrier(const rvsdg::SimpleNode & node);

  void
  AnalyzeStructuralNode(const rvsdg::StructuralNode & node);

  void
  AnalyzeLambda(const rvsdg::LambdaNode & node);

  void
  AnalyzeDelta(const DeltaNode & node);

  void
  AnalyzePhi(const rvsdg::PhiNode & node);

  void
  AnalyzeGamma(const rvsdg::GammaNode & node);

  void
  AnalyzeTheta(const rvsdg::ThetaNode & node);

  void
  AnalyzeRvsdg(const rvsdg::Graph & graph);

  /**
   * Traverses the given module, and initializes the members Set_ and Constraints_ with
   * PointerObjects and constraints corresponding to the module.
   * @param module the module to analyze
   * @param statistics the Statistics instance used to track info about the analysis
   */
  void
  AnalyzeModule(const rvsdg::RvsdgModule & module, Statistics & statistics);

  /**
   * Solves the constraint problem using the techniques and solver specified in the given config.
   * @param constraints the instance of PointerObjectConstraintSet being operated on
   * @param config settings for the solving
   * @param statistics the Statistics instance used to track info about the analysis
   */
  static void
  SolveConstraints(
      PointerObjectConstraintSet & constraints,
      const Configuration & config,
      Statistics & statistics);

  Configuration Config_ = Configuration::DefaultConfiguration();

  std::unique_ptr<PointerObjectSet> Set_;
  std::unique_ptr<PointerObjectConstraintSet> Constraints_;
};

}

#endif
