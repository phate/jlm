/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

namespace jlm::llvm::aa
{

/**
 * class implementing Andersen's set constraint based pointer analysis, based on the Ph.D. thesis
 * Lars Ole Andersen - Program Analysis and Specialization for the C Programming Language
 * The analysis is inter-procedural, field-insensitive, context-insensitive,
 * flow-insensitive, and uses a static heap model.
 */
class Andersen final : public AliasAnalysis
{
  class Statistics;

public:
  /**
   * Environment variable that will trigger double checking of the analysis,
   * by running analysis again with the naive solver and no extra processing.
   * Any differences in the produced PointsToGraph result in an error.
   */
  static inline const char * const ENV_COMPARE_SOLVE_NAIVE = "JLM_ANDERSEN_COMPARE_SOLVE_NAIVE";

  /**
   * Environment variable that will trigger dumping the subset graph before and after solving.
   */
  static inline const char * const ENV_DUMP_SUBSET_GRAPH = "JLM_ANDERSEN_DUMP_SUBSET_GRAPH";

  /**
   * Environment variable for overriding the default configuration.
   * The variable should something look like
   * "+OVS +Normalize -OnlineCD Solver=Worklist WLPolicy=LRF"
   */
  static inline const char * const ENV_CONFIG_OVERRIDE = "JLM_ANDERSEN_CONFIG_OVERRIDE";
  static inline const char * const CONFIG_OVS_ON = "+OVS";
  static inline const char * const CONFIG_OVS_OFF = "-OVS";
  static inline const char * const CONFIG_NORMALIZE_ON = "+Normalize";
  static inline const char * const CONFIG_NORMALIZE_OFF = "-Normalize";
  static inline const char * const CONFIG_SOLVER_WL = "Solver=Worklist";
  static inline const char * const CONFIG_SOLVER_NAIVE = "Solver=Naive";
  static inline const char * const CONFIG_WL_POLICY_LRF = "WLPolicy=LRF";
  static inline const char * const CONFIG_WL_POLICY_TWO_PHASE_LRF = "WLPolicy=2LRF";
  static inline const char * const CONFIG_WL_POLICY_FIFO = "WLPolicy=FIFO";
  static inline const char * const CONFIG_WL_POLICY_LIFO = "WLPolicy=LIFO";

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

    [[nodiscard]] bool
    operator==(const Configuration & other) const
    {
      return EnableOfflineVariableSubstitution_ == other.EnableOfflineVariableSubstitution_
          && EnableOfflineConstraintNormalization_ == other.EnableOfflineConstraintNormalization_
          && Solver_ == other.Solver_ && WorklistSolverPolicy_ == other.WorklistSolverPolicy_;
    }

    [[nodiscard]] bool
    operator!=(const Configuration & other) const
    {
      return !operator==(other);
    }

    /**
     * Sets which solver algorithm to use.
     * Not all solvers are compatible with all online techniques.
     */
    void
    SetSolver(Solver solver)
    {
      Solver_ = solver;
    }

    [[nodiscard]] Solver
    GetSolver() const noexcept
    {
      return Solver_;
    }

    void
    SetWorklistSolverPolicy(PointerObjectConstraintSet::WorklistSolverPolicy policy)
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
    EnableOfflineVariableSubstitution(bool enable)
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
    EnableOfflineConstraintNormalization(bool enable)
    {
      EnableOfflineConstraintNormalization_ = enable;
    }

    [[nodiscard]] bool
    IsOfflineConstraintNormalizationEnabled() const noexcept
    {
      return EnableOfflineConstraintNormalization_;
    }

    /**
     * Creates the default Andersen constraint set solver configuration
     * @return the solver configuration
     */
    [[nodiscard]] static Configuration
    DefaultConfiguration();

    /**
     * Creates a solver configuration using the naive solver,
     * with all offline and online speedup techniques disabled.
     * @return the solver configuration
     */
    [[nodiscard]] static Configuration
    NaiveSolverConfiguration()
    {
      auto config = Configuration();
      config.EnableOfflineVariableSubstitution(false);
      config.EnableOfflineConstraintNormalization(false);
      config.SetSolver(Solver::Naive);
      return config;
    }

  private:
    bool EnableOfflineVariableSubstitution_ = true;
    bool EnableOfflineConstraintNormalization_ = true;
    Solver Solver_ = Solver::Worklist;
    PointerObjectConstraintSet::WorklistSolverPolicy WorklistSolverPolicy_ =
        PointerObjectConstraintSet::WorklistSolverPolicy::LeastRecentlyFired;
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
  Analyze(const RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

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
  AnalyzeRegion(rvsdg::region & region);

  void
  AnalyzeSimpleNode(const rvsdg::simple_node & node);

  void
  AnalyzeAlloca(const rvsdg::simple_node & node);

  void
  AnalyzeMalloc(const rvsdg::simple_node & node);

  void
  AnalyzeLoad(const LoadNode & loadNode);

  void
  AnalyzeStore(const StoreNode & storeNode);

  void
  AnalyzeCall(const CallNode & callNode);

  void
  AnalyzeGep(const rvsdg::simple_node & node);

  void
  AnalyzeBitcast(const rvsdg::simple_node & node);

  void
  AnalyzeBits2ptr(const rvsdg::simple_node & node);

  void
  AnalyzePtr2bits(const rvsdg::simple_node & node);

  void
  AnalyzeConstantPointerNull(const rvsdg::simple_node & node);

  void
  AnalyzeUndef(const rvsdg::simple_node & node);

  void
  AnalyzeMemcpy(const rvsdg::simple_node & node);

  void
  AnalyzeConstantArray(const rvsdg::simple_node & node);

  void
  AnalyzeConstantStruct(const rvsdg::simple_node & node);

  void
  AnalyzeConstantAggregateZero(const rvsdg::simple_node & node);

  void
  AnalyzeExtractValue(const rvsdg::simple_node & node);

  void
  AnalyzeValist(const rvsdg::simple_node & node);

  void
  AnalyzeStructuralNode(const rvsdg::structural_node & node);

  void
  AnalyzeLambda(const lambda::node & node);

  void
  AnalyzeDelta(const delta::node & node);

  void
  AnalyzePhi(const phi::node & node);

  void
  AnalyzeGamma(const rvsdg::gamma_node & node);

  void
  AnalyzeTheta(const rvsdg::theta_node & node);

  void
  AnalyzeRvsdg(const rvsdg::graph & graph);

  /**
   * Traverses the given module, and initializes the members Set_ and Constraints_ with
   * PointerObjects and constraints corresponding to the module.
   * @param module the module to analyze
   * @param statistics the Statistics instance used to track info about the analysis
   */
  void
  AnalyzeModule(const RvsdgModule & module, Statistics & statistics);

  /**
   * Works with the members Set_ and Constraints_, and solves the constraint problem
   * using the techniques and solver specified in the given configuration
   * @param config settings for the solving
   * @param statistics the Statistics instance used to track info about the analysis
   */
  void
  SolveConstraints(const Configuration & config, Statistics & statistics);

  Configuration Config_ = Configuration::DefaultConfiguration();

  std::unique_ptr<PointerObjectSet> Set_;
  std::unique_ptr<PointerObjectConstraintSet> Constraints_;
};

} // namespace

#endif
