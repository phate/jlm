/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REGIONPREDICATETRACE_HPP
#define JLM_RVSDG_REGIONPREDICATETRACE_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>

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
  using RegionPredRange = std::unordered_map<Region *, PredicateValueRange>;
  class Observer;

  void
  Clear();

  void
  ObserveRegion(Region & region);

  PredicateValueRange
  ComputeAndRecord(RegionPredRange & regionPredRange, Input & input, const ControlType & type);

  PredicateValueRange
  Compute(RegionPredRange & regionPredRange, Input & input, const ControlType & type);

  std::unordered_map<Input *, RegionPredRange> predAssignment_;
  std::unordered_map<Region *, PredicateSatRequired> predSat_;
  std::unordered_map<Region *, std::unique_ptr<Observer>> observers_;
};

}

#endif // JLM_RVSDG_REGIONTRACE_HPP
