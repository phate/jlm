/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REGIONPREDICATETRACE_HPP
#define JLM_RVSDG_REGIONPREDICATETRACE_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>

#include <unordered_map>

namespace jlm::rvsdg
{

/**
 * \brief Value range for a predicate.
 *
 * Describes which values a predicate can take.
 */
class PredicateValueRange
{
public:
  /**
   * \brief Constructs empty value range (unsatisfiable predicate range).
   */
  static inline PredicateValueRange
  CreateEmpty(const ControlType & type)
  {
    return PredicateValueRange(type.nalternatives(), false);
  }

  /**
   * \brief Constructs full value range (every value possible).
   */
  static inline PredicateValueRange
  CreateUnknown(const ControlType & type)
  {
    return PredicateValueRange(type.nalternatives(), true);
  }

  /**
   * \brief Definite value range (exactly one value possible).
   */
  static inline PredicateValueRange
  CreateSingleValue(const ControlValueRepresentation & value)
  {
    auto pred = PredicateValueRange(value.nalternatives(), false);
    pred.values_[value.alternative()] = true;
    return pred;
  }

  /**
   * \brief Takes union of two value ranges.
   */
  void
  UpdateUnion(const PredicateValueRange & other)
  {
    auto size = std::min(values_.size(), other.values_.size());
    for (std::size_t n = 0; n < size; ++n)
    {
      values_[n] = values_[n] || other.values_[n];
    }
  }

  /**
   * \brief Checks whether value range allows a specific value.
   */
  bool
  AllowsValue(std::size_t alternative) const noexcept
  {
    return values_.size() > alternative && values_[alternative];
  }

private:
  inline PredicateValueRange(std::size_t nalternatives, bool init_value)
      : values_(nalternatives, init_value)
  {}

  std::vector<bool> values_;
};

/**
 * \brief Describes which predicates need to be satisfied.
 *
 * For a given region, this describes which predicates determine
 * entry into that region and the necessary values. For example,
 * a region may be nested into three gamma nodes, and this structure
 * describes the values that each of the three gamma control predicates
 * need to take in order to reach this region.
 */
using PredicateSatRequired = std::vector<std::pair<Input *, std::size_t>>;

/**
 * \brief Traces region reachability by predicate assertions
 *
 * Traces predicate def/use patterns in the graph and determines
 * assignments of values to predicates in different regions,
 * as well as the predicate assignments necessary to reach
 * a given region.
 */
class RegionPredicateTrace
{
public:
  ~RegionPredicateTrace();

  RegionPredicateTrace();

  /**
   * \brief Computes value range for a predicate when exiting a region
   *
   * \param region
   *   Region to check at exit
   *
   * \param predUse
   *   Use site of the predicate of interest
   *
   * \returns
   *   The value range that can be reached coming out of this
   *   region, and that might be effective at the use site
   *
   * Traces the definitions of the predicate used at \p preUse
   * back to its definition sites. Computes the possible values
   * that the predicate can obtain from its definition sites
   * at the end of the region queried.
   *
   * For example, if a constant value is assigned to a predicate
   * within this region, then this will determine its value range.
   * If OTOH there are multiple alternative paths within or
   * leading into this region with multiple possible predicate
   * assignments, then this will report the union of all possible
   * values.
   */
  PredicateValueRange
  GetRegionPredicateAssignConstraints(Region & region, Input & predUse);

  /**
   * \brief Computes required predicate assignments for region
   *
   * \param region
   *   The region that we want to check for reachability
   *
   * \returns
   *   Necessary predicate / value pairs that need to be
   *   satisfied in order to reach this region.
   *
   * Computes which predicate needs to be assigned which value
   * in order to reach a specific region. E.g. for a region
   * nested inside three gamma nodes, this gives the required
   * assignments to the three control predicates in question
   * that are needed in order to reach the inner region.
   */
  PredicateSatRequired
  GetRegionSatRequired(Region & region);

  /**
   * \brief Checks for dynamic reachability between two regions
   *
   * \param originRegion
   *   The "upper" region, from which we want to check whether
   *   another regino can be reached.
   *
   * \param targetRegion
   *   The "lower" region which we want to check whether it can
   *   be reached.
   *
   * \returns
   *   True iff \p targetRegion is dynamically reachable assuming
   *   that \p originRegion has been reached before.
   *
   * Computes predicates value ranges that are necessarily
   * assigned assuming that \p originRegion has been entered,
   * and checks whether these predicates allow \p targetRegion
   * to be entered.
   *
   * This allows to dynamically discriminate whether a value generated
   * in \p originRegion can affect a use site in \p targetRegion:
   * - if this returns true, then a value generated in \p originRegion
   *   _may_ dynamically be forwarded and used in \p targetRegion
   * - if this returns false, then any value generated in \p originRegion
   *   cannot be the value ultimately used in \p targetRegion -- the
   *   effective value at the use site within \p targetRegion must
   *   originate from somewhere else (effectively a gamma branch
   *   that is parallel to \p originRegion).
   */
  bool
  CheckPredicatesSatisfiable(Region & originRegion, Region & targetRegion);

private:
  // For a given input, its RegionPredRange provides a map from regions to the
  // set of values than can end up being routed to the input, from results of the region.
  using RegionPredRange = std::unordered_map<Region *, PredicateValueRange>;
  class Observer;

  void
  Clear();

  void
  ObserveRegion(Region & region);

  /**
   * Traces from the given \p input to find the regions that may provide its value.
   * @param regionPredRange the resulting map of possible values provided in each region.
   * @param input the input being traced from
   * @param visitedInputs set of seen inputs, to avoid re-visiting
   * @param type the type of the input
   */
  const PredicateValueRange &
  ComputeAndRecord(
      RegionPredRange & regionPredRange,
      Input & input,
      std::unordered_map<Input *, PredicateValueRange> & visitedInputs,
      const ControlType & type);

  /**
   * Helper function for the above \ref ComputeAndRecord
   */
  PredicateValueRange
  Compute(
      RegionPredRange & regionPredRange,
      Input & input,
      std::unordered_map<Input *, PredicateValueRange> & visitedInputs,
      const ControlType & type);

  // For a given input, gives the regions where we know which the set of values
  // the region may provide to the input
  std::unordered_map<Input *, RegionPredRange> predAssignment_;

  // For a given target region, what predicates must be satisfied to reach it
  std::unordered_map<Region *, PredicateSatRequired> predSat_;

  // Observers registered on region to inform the tracer when caches must be invalidated
  std::unordered_map<Region *, std::unique_ptr<Observer>> observers_;
};

class AlternativeRegionPredicateTracer final
{
  public:
    AlternativeRegionPredicateTracer();

    /**
     * Determines if it is possible for control flow to go from the given
     * \p originRegion to the given \p targetRegion.
     * The method assuemes that control flow does not follow any back-edges
     * in any theta node that contains the \p originRegion.
     * No assumptions are made about theta nodes surrounding the \p targetRegion,
     * or control flow entering theta nodes between the two regions.
     */
    [[nodiscard]] bool
    canRegionReachRegion(Region & originRegion, Region & targetRegion);

  private:

    /**
     * Updates the internal target region and clears caches if it is new.
     */
    void
    setTargetRegion(Region & targetRegion);

    /**
     * Adds the fact that a given output needs to have a specific value in order to
     * reach the target region without following back-edges around its origin region.
     * If the value is not possible to achieve in a region, that region is marked
     * as an impossible origin region.
     * @param output an output of type ControlType
     * @param value the value the output must have.
     * @return false if the output can not satisfy the requirement
     */
    bool
    setRequiredPredicateValue(rvsdg::Output & output, size_t value);

    /**
     * Traces the target region one step further up the region tree,
     * and traces region predicate requirements when leaving gamma nodes.
     */
    void
    visitNextTargetRegionAncestor();

    // The target region of the last query.
    // All caches are cleared when the target region changes
    Region * targetRegion_ = nullptr;

    util::HashSet<Region*> targetRegionAncestors_;
    Region * topSeenTargetAncestor_ = nullptr;

    // Outputs that have been processed by \ref setRequiredPredicateValue.
    // Maps to the ability of the output to provide the required value.
    std::unordered_map<Output*, bool> processedOutputs_;

    // Regions that are known to never be origin regions
    util::HashSet<Region*> impossibleOriginRegions_;
};

}

#endif // JLM_RVSDG_REGIONTRACE_HPP
