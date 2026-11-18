/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REGIONPREDICATETRACE_HPP
#define JLM_RVSDG_REGIONPREDICATETRACE_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/HashSet.hpp>

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
 * \brief Provides guarantees about unreachability of regions from other regions.
 *
 * If any control type edges are modified, or if any regions are deleted,
 * during the lifetime of the tracer, the tracer's caches must be cleared manually.
 * @see clearCaches()
 */
class RegionPredicateTracer final
{
public:
  RegionPredicateTracer();

  /**
   * Determines if it is possible for control flow to go from the given
   * \p originRegion to the given \p targetRegion,
   * without following any back-edges in any theta node that contains the \p originRegion.
   * No assumptions are made about theta nodes surrounding the \p targetRegion,
   * or control flow entering theta nodes between the two regions.
   */
  [[nodiscard]] bool
  isReachableFromRegion(Region & targetRegion, Region & originRegion);

  /**
   * Removes all cached information from the tracer.
   * This must be done if edges or constants of ControlType have been modified in
   * a way that changes the static reachability guarantees between regions.
   * Also, if regions get removed from the graph, the caches must be cleared
   * to prevent new regions re-useing the address from causing cache collisions.
   */
  void
  clearCaches();

private:
  /**
   * Determines the set of possible values for the given \p output.
   * @param output the output in question, must be of type ControlType.
   * @return the possible values of the control type output.
   */
  [[nodiscard]] PredicateValueRange &
  getPossibleValues(Output & output);

  [[nodiscard]] PredicateValueRange
  getPossibleValuesInternal(Output & output);

  /**
   * Processes the fact that a given output needs to have a specific value,
   * and adds all origin regions that are unable to satisfy the requirement to a list.
   * Assumes no back-edges are taken around the considered origin regions.
   *
   * @param output an output of type ControlType
   * @param value the value the output must have in order to reach/leave the target region.
   * @param impossibleOrigins a list to which all impossible origin regions are added.
   * @return false if the output can never satisfy the requirement
   */
  bool
  markRequiredPredicateValue(
      Output & output,
      size_t value,
      std::vector<Region *> & impossibleOrigins);

  bool
  markRequiredPredicateValueInternal(
      Output & output,
      size_t value,
      std::vector<Region *> & impossibleOrigins);

  /**
   * Checks if the current origin region or any of its ancestors are unable to
   * satisfy the requirement that the given output gets the given value.
   * If the (output, value) pair has not yet been processed,
   * it is processed first via \ref markRequiredPredicateValue.
   *
   * @param output an output of type ControlType
   * @param value the value the output must have in order to reach/leave the target region.
   * @return false if the origin region or one of its ancestors are unable to meet the requirement
   */
  [[nodiscard]] bool
  canCurrentOriginSatisfyRequirement(Output & output, size_t value);

  // The possible values an output of control type may have.
  std::unordered_map<Output *, PredicateValueRange> predicateValueRanges_;

  // Map from (output, required value) to a list of origin regions that cannot provide the value,
  // without control flow taking any back-edges around the origin region.
  std::unordered_map<
      std::pair<Output *, size_t>,
      std::vector<Region *>,
      util::Hash<std::pair<Output *, size_t>>>
      impossibleOriginRegions_;

  // The ancestors of the current origin region.
  // Not a cache, updates for every call to \ref isReachableFromRegion.
  util::HashSet<Region *> currentOriginRegionAncestors_;
};

}

#endif // JLM_RVSDG_REGIONTRACE_HPP
