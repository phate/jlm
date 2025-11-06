/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PREDICATECORRELATION_HPP
#define JLM_LLVM_OPT_PREDICATECORRELATION_HPP

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace jlm
{

namespace rvsdg
{
class Region;
class ThetaNode;
}

namespace util
{
class StatisticsCollector;
}
}

namespace jlm::llvm
{

/**
 * The different types of predicate correlations that are supported.
 */
enum class CorrelationType
{
  /**
   * The predicate correlates with control constants from subregions of a gamma node.
   */
  ControlConstantCorrelation,

  /**
   * The predicate correlates with a MatchNode that gets its input from constants originating from
   * subregions of a gamma node.
   */
  MatchConstantCorrelation,

  /**
   * The predicate correlates with a MatchNode that also serves as predicate producer for a gamma
   * node.
   */
  MatchCorrelation,
};

/**
 * Represents the predicate correlation between a theta node and a gamma node that resides in the
 * theta node's subregion.
 */
class ThetaGammaPredicateCorrelation final
{
public:
  using ControlConstantCorrelationData = std::vector<uint64_t>;

  struct MatchConstantCorrelationData
  {
    rvsdg::Node * matchNode{};
    std::vector<uint64_t> alternatives{};
  };

  struct MatchCorrelationData
  {
    rvsdg::Node * matchNode{};
  };

  using CorrelationData = std::
      variant<ControlConstantCorrelationData, MatchConstantCorrelationData, MatchCorrelationData>;

private:
  ThetaGammaPredicateCorrelation(
      const CorrelationType type,
      rvsdg::ThetaNode & thetaNode,
      rvsdg::GammaNode & gammaNode,
      CorrelationData correlationData)
      : type_(type),
        thetaNode_(thetaNode),
        gammaNode_(gammaNode),
        data_(std::move(correlationData))
  {}

public:
  [[nodiscard]] CorrelationType
  type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] rvsdg::ThetaNode &
  thetaNode() const noexcept
  {
    return thetaNode_;
  }

  [[nodiscard]] rvsdg::GammaNode &
  gammaNode() const noexcept
  {
    return gammaNode_;
  }

  [[nodiscard]] const CorrelationData &
  data() const noexcept
  {
    return data_;
  }

  static std::unique_ptr<ThetaGammaPredicateCorrelation>
  CreateControlConstantCorrelation(
      rvsdg::ThetaNode & thetaNode,
      rvsdg::GammaNode & gammaNode,
      ControlConstantCorrelationData data)
  {
    return std::unique_ptr<ThetaGammaPredicateCorrelation>(new ThetaGammaPredicateCorrelation(
        CorrelationType::ControlConstantCorrelation,
        thetaNode,
        gammaNode,
        std::move(data)));
  }

  static std::unique_ptr<ThetaGammaPredicateCorrelation>
  CreateMatchConstantCorrelation(
      rvsdg::ThetaNode & thetaNode,
      rvsdg::GammaNode & gammaNode,
      MatchConstantCorrelationData data)
  {
    return std::unique_ptr<ThetaGammaPredicateCorrelation>(new ThetaGammaPredicateCorrelation(
        CorrelationType::MatchConstantCorrelation,
        thetaNode,
        gammaNode,
        std::move(data)));
  }

  static std::unique_ptr<ThetaGammaPredicateCorrelation>
  CreateMatchCorrelation(
      rvsdg::ThetaNode & thetaNode,
      rvsdg::GammaNode & gammaNode,
      MatchCorrelationData data)
  {
    return std::unique_ptr<ThetaGammaPredicateCorrelation>(new ThetaGammaPredicateCorrelation(
        CorrelationType::MatchCorrelation,
        thetaNode,
        gammaNode,
        std::move(data)));
  }

private:
  CorrelationType type_;
  rvsdg::ThetaNode & thetaNode_;
  rvsdg::GammaNode & gammaNode_;
  CorrelationData data_{};
};

/**
 * Computes a theta-gamma predicate correlation for \p thetaNode if there is any.
 *
 * @param thetaNode The theta node for which to compute the predicate correlation.
 * @return A theta-gamma predicate correlation if any, otherwise std::nullopt.
 */
std::optional<std::unique_ptr<ThetaGammaPredicateCorrelation>>
computeThetaGammaPredicateCorrelation(rvsdg::ThetaNode & thetaNode);

/**
 * Predicate Correlation correlates the predicates between theta and gamma nodes, and
 * redirects their respective predicates to match operation nodes.
 *
 * ### Theta-Gamma Predicate Correlation
 * If a theta node's predicate originates from a gamma node with two control flow constants, then
 * the theta node's predicate is redirected to the gamma node's predicate match node.
 */
class PredicateCorrelation final : public rvsdg::Transformation
{
public:
  ~PredicateCorrelation() noexcept override;

  PredicateCorrelation()
      : Transformation("PredicateCorrelation")
  {}

  PredicateCorrelation(const PredicateCorrelation &) = delete;

  PredicateCorrelation &
  operator=(const PredicateCorrelation &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  correlatePredicatesInRegion(rvsdg::Region & region);

  /**
   * Performs theta-gamma predicate correlation
   *
   * @param thetaNode The theta node for which to correlate.
   */
  static void
  correlatePredicatesInTheta(rvsdg::ThetaNode & thetaNode);

  static void
  handleControlConstantCorrelation(const ThetaGammaPredicateCorrelation & correlation);

  static void
  handleMatchConstantCorrelation(const ThetaGammaPredicateCorrelation & correlation);
};

}

#endif
