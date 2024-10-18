/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_DIFFERENCEPROPAGATION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_DIFFERENCEPROPAGATION_HPP

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <vector>

namespace jlm::llvm::aa
{

class DifferencePropagation
{

public:
  explicit DifferencePropagation(PointerObjectSet & set)
      : Set_(set)
  {}

  /**
   * Must be called before using any other methods.
   */
  void
  Initialize()
  {
    NewPointees_.resize(Set_.NumPointerObjects());
    NewPointeesTracked_.resize(Set_.NumPointerObjects(), false);
#ifndef ANDERSEN_NO_FLAGS
    PointsToExternalFlagSeen_.resize(Set_.NumPointerObjects(), false);
    PointeesEscapeFlagSeen_.resize(Set_.NumPointerObjects(), false);
#endif
  }

  [[nodiscard]] bool
  IsInitialized() const noexcept
  {
    return NewPointees_.size() == Set_.NumPointerObjects();
  }

  /**
   * Starts tracking any pointees added to \p index from this point onwards.
   * @param index the index of the PointerObject, must be a unification root.
   */
  void
  ClearNewPointees(PointerObjectIndex index)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(index));

    NewPointees_[index].Clear();
    NewPointeesTracked_[index] = true;
  };

  /**
   * Makes P(pointer) contain pointee
   * @param pointer the PointerObject which should contain pointee in its points-to set.
   *   Must be a unification root.
   * @param pointee the PointerObject which should be pointed to by pointer
   * @return true if this operation added a new pointee to P(pointer)
   */
  bool
  AddToPointsToSet(PointerObjectIndex pointer, PointerObjectIndex pointee)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(pointer));

    // If pointees added to the superset are being tracked, use the tracking version
    bool newPointee = Set_.AddToPointsToSet(pointer, pointee);
    if (newPointee && NewPointeesTracked_[pointer])
      return NewPointees_[pointer].Insert(pointee);

    return newPointee;
  }

  /**
   * Makes P(superset) a superset of P(subset), and propagates the PointsToExternal flag if set.
   * @param superset the PointerObject which should point to everything the subset points to
   *   Must be a unification root.
   * @param subset a PointerObject which should have all its pointees also be pointees of superset.
   * @return true if this operation adds any pointees or flags to superset.
   */
  bool
  MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(superset));

    // If pointees added to the superset are being tracked, use the tracking version
    if (NewPointeesTracked_[superset])
      return Set_.MakePointsToSetSuperset(superset, subset, NewPointees_[superset]);
    else
      return Set_.MakePointsToSetSuperset(superset, subset);
  }

  /**
   * Gets the pointees of a PointerObject that have been added since the last time
   * ClearNewPointees(index) was called.
   * If new pointees of index are not being tracked, all pointees are returned.
   * @param index the index of the PointerObject, must be a unification root.
   * @return a reference to either all new pointees, or all pointees of index.
   */
  [[nodiscard]] const util::HashSet<PointerObjectIndex> &
  GetNewPointees(PointerObjectIndex index) const
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    if (NewPointeesTracked_[index])
      return NewPointees_[index];
    return Set_.GetPointsToSet(index);
  }

  /**
   * Clears the tracked set of new pointees of \p index, and stops tracking it.
   * @param index the index of the PointerObject, must be a unification root.
   */
  void
  OnRemoveAllPointees(PointerObjectIndex index)
  {
    NewPointeesTracked_[index] = false;
    NewPointees_[index].Clear();
  }

#ifndef ANDERSEN_NO_FLAGS
  /**
   * If the given PointerObject has the PointsToExternal flag now,
   * and MarkPointsToExternalAsHandled(index) has not been called, return true. Otherwise false.
   * An exception is if the PointerObject has been unified with other PointerObjects,
   * and some of the other PointerObjects did not have that flag handled already.
   * @param index the index of the PointerObject being queried. Must be a unification root
   * @return true if the PointerObject is newly flagged as PointsToExternal.
   */
  [[nodiscard]] bool
  PointsToExternalIsNew(PointerObjectIndex index)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    if (!Set_.IsPointingToExternal(index))
      return false;
    return !PointsToExternalFlagSeen_[index];
  }

  /**
   * Call once the addition of the new flag PointsToExternal has been handled.
   * After this call, PointsToExternalIsNew(index) will return false (unless new unifications).
   * @param index the index of the PointerObject whose PointsToExternal flag has been handled.
   * Must be a unification root.
   */
  void
  MarkPointsToExternalAsHandled(PointerObjectIndex index)
  {
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    JLM_ASSERT(Set_.IsPointingToExternal(index));
    PointsToExternalFlagSeen_[index] = true;
  }

  /**
   * If the given PointerObject has the PointeesEscape flag now,
   * but didn't have it the last time ClearDifferenceTracking(index) was called, return true.
   * An exception is if the PointerObject has been unified with other PointerObjects,
   * and some of the other PointerObjects had not have that flag seen.
   * @param index the index of the PointerObject being queried. Must be a unification root
   * @return true if the PointerObject is newly flagged as AllPointeesEscape.
   */
  [[nodiscard]] bool
  PointeesEscapeIsNew(PointerObjectIndex index)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    if (!Set_.HasPointeesEscaping(index))
      return false;
    return !PointeesEscapeFlagSeen_[index];
  }

  /**
   * Call once the addition of the new flag PointeesEscape has been handled.
   * After this call, PointeesEscapeIsNew(index) will return false (unless new unifications).
   * @param index the index of the PointerObject whose PointeesEscape flag has been handled.
   * Must be a unification root.
   */
  void
  MarkPointeesEscapeAsHandled(PointerObjectIndex index)
  {
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    JLM_ASSERT(Set_.HasPointeesEscaping(index));
    PointeesEscapeFlagSeen_[index] = true;
  }
#endif

  /**
   * Performs conservative clearing of tracked differences, after unification.
   * The set of tracked pointees is fully cleared, since all pointees might be new to
   * constraints previously owned by the opposite PointerObject.
   *
   * If the worklist has seen the PointsToExternal and PointeesEscape flags on both operands,
   * the flags will also be considered already seen for the resulting unification root.
   *
   * @param root the operand of the unification that ended up at the new root.
   * @param nonRoot the root of the other unification, that is now no longer a root.
   */
  void
  OnPointerObjectsUnified(PointerObjectIndex root, PointerObjectIndex nonRoot)
  {
    JLM_ASSERT(IsInitialized());

    // After unification, forget everything about tracked differences in points-to sets
    NewPointees_[nonRoot].Clear();
    NewPointees_[root].Clear();
    NewPointeesTracked_[root] = false;

#ifndef ANDERSEN_NO_FLAGS
    PointsToExternalFlagSeen_[root] =
        PointsToExternalFlagSeen_[root] && PointsToExternalFlagSeen_[nonRoot];
    PointeesEscapeFlagSeen_[root] =
        PointeesEscapeFlagSeen_[root] && PointeesEscapeFlagSeen_[nonRoot];
#endif
  }

private:
  PointerObjectSet & Set_;

  // Only unification roots matter in these vectors

  // Tracks all new pointees added to a unification root i,
  // since ClearNewPointees(i) was last called.
  std::vector<util::HashSet<PointerObjectIndex>> NewPointees_;
  // Becomes true for a unification root i when CleanNewPointees(i) is called for the first time.
  // Becomes false again when unification fully resets difference propagation
  std::vector<bool> NewPointeesTracked_;

#ifndef ANDERSEN_NO_FLAGS
  // These are set to true after the _IsNew have returned true
  // When two PointerObjects a and b are unified, the flag only remains "seen",
  // if the flag has already been "seen" on both a and b.
  std::vector<bool> PointsToExternalFlagSeen_;
  std::vector<bool> PointeesEscapeFlagSeen_;
#endif
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_DIFFERENCEPROPAGATION_HPP
