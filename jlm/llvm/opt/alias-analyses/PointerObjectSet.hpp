/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/util/BijectiveMap.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/Math.hpp>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jlm::llvm::aa
{

enum class PointerObjectKind : uint8_t
{
  // Registers can not be pointed to, only point.
  Register = 0,
  // All other pointer objects represent storage instances in memory
  AllocaMemoryObject,
  MallocMemoryObject,
  GlobalMemoryObject,
  // Represents functions, they can not point to any memory objects.
  FunctionMemoryObject,
  // Represents functions and global variables imported from other modules.
  ImportMemoryObject,

  COUNT
};

using PointerObjectIndex = uint32_t;

/**
 * A class containing a set of PointerObjects, and their points-to-sets,
 * as well as mappings from RVSDG nodes/outputs to the PointerObjects.
 * For brevity, P(x) denotes the points-to-set of a PointerObject x.
 */
class PointerObjectSet final
{
  /**
   * Struct used internally to store information about each PointerObject.
   * When PointerObjects are unified, some flags are shared.
   * This is handled by the accessor methods defined on PointerObjectSet
   */
  struct PointerObject final
  {
    // The kind of pointer object
    PointerObjectKind Kind : util::BitWidthOfEnum(PointerObjectKind::COUNT);

    // This memory object's address is known outside the module.
    // If CanBePointee() is false, this object has no address, but its value has escaped.
    uint8_t HasEscaped : 1;

    // If this PointerObject is the parent of its own unification,
    // and this flag is set, this pointer object is pointing to external.
    uint8_t PointsToExternal : 1;

    explicit PointerObject(PointerObjectKind kind)
        : Kind(kind),
          HasEscaped(0),
          PointsToExternal(0)
    {
      JLM_ASSERT(kind != PointerObjectKind::COUNT);
    }

    /**
     * Some PointerObjects are not capable of pointing to anything else.
     * Their points-to-set will always be empty, and constraints that attempt
     * to add pointees should be no-ops that are silently ignored.
     * The same applies to attempts at setting the PointsToExternal-flag.
     * @return true if this PointerObject can point to other PointerObjects
     */
    [[nodiscard]] bool
    CanPoint() const noexcept
    {
      return Kind != PointerObjectKind::FunctionMemoryObject;
    }

    /**
     * Some PointerObjects don't have addresses, and can as such not be pointed to.
     * Any attempt at adding them to a points-to-set is a fatal error.
     * @return true if this PointerObject can be pointed to by another PointerObject
     */
    [[nodiscard]] bool
    CanBePointee() const noexcept
    {
      return Kind != PointerObjectKind::Register;
    }
  };

  // All PointerObjects in the set
  std::vector<PointerObject> PointerObjects_;

  // The parent of each PointerObject in the disjoint-set forest. Roots are their own parent.
  // Marked as mutable to allow path compression in const qualified methods.
  mutable std::vector<PointerObjectIndex> PointerObjectParents_;

  // Metadata enabling union by rank, where rank is an upper bound for tree height
  // Size of a disjoint set is at least 2^rank, making a uint8_t plenty big enough.
  std::vector<uint8_t> PointerObjectRank_;

  // For each PointerObject, a set of the other PointerObjects it points to
  // Only unification roots may have a non-empty set,
  // other PointerObjects refer to their root's set.
  std::vector<util::HashSet<PointerObjectIndex>> PointsToSets_;

  // Mapping from register to PointerObject
  // Unlike the other maps, several rvsdg::output* can share register PointerObject
  std::unordered_map<const rvsdg::output *, PointerObjectIndex> RegisterMap_;

  std::unordered_map<const rvsdg::node *, PointerObjectIndex> AllocaMap_;

  std::unordered_map<const rvsdg::node *, PointerObjectIndex> MallocMap_;

  std::unordered_map<const delta::node *, PointerObjectIndex> GlobalMap_;

  util::BijectiveMap<const lambda::node *, PointerObjectIndex> FunctionMap_;

  std::unordered_map<const rvsdg::argument *, PointerObjectIndex> ImportMap_;

  /**
   * Internal helper function for adding PointerObjects, use the Create* methods instead
   */
  [[nodiscard]] PointerObjectIndex
  AddPointerObject(PointerObjectKind kind);

public:
  [[nodiscard]] size_t
  NumPointerObjects() const noexcept;

  /**
   * @return the number of PointerObjects in the set matching the specified \p kind.
   */
  [[nodiscard]] size_t
  NumPointerObjectsOfKind(PointerObjectKind kind) const noexcept;

  /**
   * Creates a PointerObject of Register kind and maps the rvsdg output to the new PointerObject.
   * The rvsdg output can not already be associated with a PointerObject.
   * @param rvsdgOutput the rvsdg output associated with the register PointerObject
   * @return the index of the new PointerObject in the PointerObjectSet
   */
  [[nodiscard]] PointerObjectIndex
  CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput);

  /**
   * Retrieves a previously created PointerObject of Register kind.
   * @param rvsdgOutput an rvsdg::output that already corresponds to a PointerObject in the set
   * @return the index of the PointerObject associated with the rvsdg::output
   */
  [[nodiscard]] PointerObjectIndex
  GetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const;

  /**
   * Retrieves a previously created PointerObject of Register kind, associated with an rvsdg output.
   * If no PointerObject is associated with the given output, nullopt is returned.
   * @param rvsdgOutput the rvsdg::output that might correspond to a PointerObject in the set
   * @return the index of the PointerObject associated with rvsdgOutput, if it exists
   */
  [[nodiscard]] std::optional<PointerObjectIndex>
  TryGetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const;

  /**
   * Reuses an existing PointerObject of register type for an additional rvsdg output.
   * This is useful when values are passed from outer regions to inner regions.
   * @param rvsdgOutput the new rvsdg output providing the register value
   * @param pointerObject the index of the existing PointerObject, must be of register kind
   */
  void
  MapRegisterToExistingPointerObject(
      const rvsdg::output & rvsdgOutput,
      PointerObjectIndex pointerObject);

  /**
   * Creates a PointerObject of Register kind, without any association to any node in the program.
   * It can not be pointed to, and will not be included in the final PointsToGraph,
   * but it can be used as a temporary object to string together constraints.
   * @see Andersen::AnalyzeMemcpy.
   * @return the index of the new PointerObject
   */
  [[nodiscard]] PointerObjectIndex
  CreateDummyRegisterPointerObject();

  [[nodiscard]] PointerObjectIndex
  CreateAllocaMemoryObject(const rvsdg::node & allocaNode);

  [[nodiscard]] PointerObjectIndex
  CreateMallocMemoryObject(const rvsdg::node & mallocNode);

  [[nodiscard]] PointerObjectIndex
  CreateGlobalMemoryObject(const delta::node & deltaNode);

  /**
   * Creates a PointerObject of Function kind associated with the given \p lambdaNode.
   * The lambda node can not be associated with a PointerObject already.
   * @param lambdaNode the RVSDG node defining the function,
   * @return the index of the new PointerObject in the PointerObjectSet
   */
  [[nodiscard]] PointerObjectIndex
  CreateFunctionMemoryObject(const lambda::node & lambdaNode);

  /**
   * Retrieves the PointerObject of Function kind associated with the given lambda node
   * @param lambdaNode the lambda node associated with the existing PointerObject
   * @return the index of the associated PointerObject
   */
  [[nodiscard]] PointerObjectIndex
  GetFunctionMemoryObject(const lambda::node & lambdaNode) const;

  /**
   * Gets the lambda node associated with a given PointerObject.
   * @param index the index of the PointerObject
   * @return the lambda node associated with the PointerObject
   */
  [[nodiscard]] const lambda::node &
  GetLambdaNodeFromFunctionMemoryObject(PointerObjectIndex index) const;

  [[nodiscard]] PointerObjectIndex
  CreateImportMemoryObject(const rvsdg::argument & importNode);

  const std::unordered_map<const rvsdg::output *, PointerObjectIndex> &
  GetRegisterMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObjectIndex> &
  GetAllocaMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObjectIndex> &
  GetMallocMap() const noexcept;

  const std::unordered_map<const delta::node *, PointerObjectIndex> &
  GetGlobalMap() const noexcept;

  const util::BijectiveMap<const lambda::node *, PointerObjectIndex> &
  GetFunctionMap() const noexcept;

  const std::unordered_map<const rvsdg::argument *, PointerObjectIndex> &
  GetImportMap() const noexcept;

  /**
   * @return the kind of the PointerObject with the given \p index
   */
  [[nodiscard]] PointerObjectKind
  GetPointerObjectKind(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index can point, otherwise false
   */
  [[nodiscard]] bool
  CanPointerObjectPoint(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index can be a pointee, otherwise false
   */
  [[nodiscard]] bool
  CanPointerObjectBePointee(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index has escaped, otherwise false
   */
  [[nodiscard]] bool
  HasEscaped(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as having escaped the module.
   * @return true if the flag was changed by this operation
   */
  bool
  MarkAsEscaped(PointerObjectIndex index);

  /**
   * @return true if the PointerObject with the given \p index points to external, otherwise false
   */
  [[nodiscard]] bool
  IsPointingToExternal(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as pointing to external.
   * If the PointerObject has CanPoint() = false, this is a no-op.
   * @return true if the flag was changed by this operation
   */
  bool
  MarkAsPointingToExternal(PointerObjectIndex index);

  /**
   * @return the root in the unification the PointerObject with the given \p index belongs to.
   * PointerObjects that have not been unified will always be their own root.
   */
  [[nodiscard]] PointerObjectIndex
  GetUnificationRoot(PointerObjectIndex index) const noexcept;

  /**
   * Unifies two PointerObjects, such that they will forever share their set of pointees.
   * If any object in the unification points to external, they will all point to external.
   * The HasEscaped flags are not shared, and can still be set individually.
   * Only PointerObjects where CanPoint() is true may be unified.
   * If the objects already belong to the same disjoint set, this is a no-op.
   * @param object1 the index of the first PointerObject to unify
   * @param object2 the index of the second PointerObject to unify
   * @return the index of the root PointerObject in the unification
   */
  PointerObjectIndex
  UnifyPointerObjects(PointerObjectIndex object1, PointerObjectIndex object2);

  /**
   * @return the PointsToSet of the PointerObject with the given \p index.
   */
  [[nodiscard]] const util::HashSet<PointerObjectIndex> &
  GetPointsToSet(PointerObjectIndex index) const;

  /**
   * Adds \p pointee to P(\p pointer)
   * @param pointer the index of the PointerObject that shall point to \p pointee
   * @param pointee the index of the PointerObject that is pointed at, can not be a register.
   *
   * If the pointer is of a PointerObjectKind that can't point, this is a no-op.
   *
   * @return true if P(\p pointer) was changed by this operation
   */
  bool
  AddToPointsToSet(PointerObjectIndex pointer, PointerObjectIndex pointee);

  /**
   * Makes P(\p superset) a superset of P(\p subset), by adding any elements in the set difference
   * @param superset the index of the PointerObject that shall point to everything subset points to
   * @param subset the index of the PointerObject whose pointees shall all be pointed to by superset
   * as well
   *
   * If the superset is of a PointerObjectKind that can't point, this is a no-op.
   *
   * @return true if P(\p superset) was modified by this operation
   */
  bool
  MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset);

  /**
   * Adds the Escaped flag to all PointerObjects in the P(\p pointer) set
   * @param pointer the pointer whose pointees should be marked as escaped
   * @return true if any PointerObjects had their flag modified by this operation
   */
  bool
  MarkAllPointeesAsEscaped(PointerObjectIndex pointer);
};

/**
 * A constraint of the form:
 * P(superset) supseteq P(subset)
 * Example of application is when a register has multiple source values
 */
class SupersetConstraint final
{
  PointerObjectIndex Superset_;
  PointerObjectIndex Subset_;

public:
  SupersetConstraint(PointerObjectIndex superset, PointerObjectIndex subset)
      : Superset_(superset),
        Subset_(subset)
  {}

  /**
   * @return the PointerObject that should point to everything the subset points to
   */
  [[nodiscard]] PointerObjectIndex
  GetSuperset() const noexcept
  {
    return Superset_;
  }

  /**
   * @param superset the new PointerObject that should point to everything the subset points to
   */
  void
  SetSuperset(PointerObjectIndex superset)
  {
    Superset_ = superset;
  }

  /**
   * @return the PointerObject whose points-to set should be contained within the superset
   */
  [[nodiscard]] PointerObjectIndex
  GetSubset() const noexcept
  {
    return Subset_;
  }

  /**
   * @param subset the new PointerObject whose points-to set should be contained within the superset
   */
  void
  SetSubset(PointerObjectIndex subset)
  {
    Subset_ = subset;
  }

  /**
   * Apply this constraint to \p set once.
   * @return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  ApplyDirectly(PointerObjectSet & set);
};

/**
 * The constraint:
 *   *P(pointer) supseteq P(value)
 * Written out as
 *   for all x in P(pointer), P(x) supseteq P(value)
 * Which corresponds to *pointer = value
 */
class StoreConstraint final
{
  PointerObjectIndex Pointer_;
  PointerObjectIndex Value_;

public:
  StoreConstraint(PointerObjectIndex pointer, PointerObjectIndex value)
      : Pointer_(pointer),
        Value_(value)
  {}

  /**
   * @return the PointerObject representing the value written by the store instruction
   */
  [[nodiscard]] PointerObjectIndex
  GetValue() const noexcept
  {
    return Value_;
  }

  /**
   * @param value the new PointerObject representing the value written by the store instruction
   */
  void
  SetValue(PointerObjectIndex value)
  {
    Value_ = value;
  }

  /**
   * @return the PointerObject representing the pointer written to by the store instruction
   */
  [[nodiscard]] PointerObjectIndex
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the pointer written to by the store.
   */
  void
  SetPointer(PointerObjectIndex pointer)
  {
    Pointer_ = pointer;
  }

  /**
   * Apply this constraint to \p set once.
   * @return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  ApplyDirectly(PointerObjectSet & set);
};

/**
 * A constraint of the form:
 *   P(value) supseteq *P(pointer)
 * Written out as
 *   for all x in P(pointer), P(value) supseteq P(x)
 * Which corresponds to value = *pointer
 */
class LoadConstraint final
{
  PointerObjectIndex Value_;
  PointerObjectIndex Pointer_;

public:
  LoadConstraint(PointerObjectIndex value, PointerObjectIndex pointer)
      : Value_(value),
        Pointer_(pointer)
  {}

  /**
   * @return the PointerObject representing the value returned by the load instruction
   */
  [[nodiscard]] PointerObjectIndex
  GetValue() const noexcept
  {
    return Value_;
  }

  /**
   * @param value the new PointerObject representing the value returned by the load instruction
   */
  void
  SetValue(PointerObjectIndex value)
  {
    Value_ = value;
  }

  /**
   * @return the PointerObject representing the pointer loaded by the load instruction
   */
  [[nodiscard]] PointerObjectIndex
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the pointer loaded by the load instruction.
   */
  void
  SetPointer(PointerObjectIndex pointer)
  {
    Pointer_ = pointer;
  }

  /**
   * Apply this constraint to \p set once.
   * @return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  ApplyDirectly(PointerObjectSet & set);
};

/**
 * A constraint making the given call site communicate with the functions it may call.
 *
 * It follows the given pseudocode:
 * for f in P(CallTarget):
 *   if f is not a lambda, or the signature doesn't match:
 *     continue
 *   for each function argument/input pair (a, i):
 *     make P(a) a superset of P(i)
 *   for each result/output pair (r, o):
 *     make P(o) a superset of P(r)
 *
 * if CallTarget is flagged as PointsToExternal:
 *   mark all of CallNode's parameters as HasEscaped
 *   mark the CallNode's return values as PointsToExternal
 */
class FunctionCallConstraint final
{
  /**
   * A PointerObject of Register kind, representing the function pointer being called
   */
  PointerObjectIndex Pointer_;

  /**
   * The RVSDG node representing the function call
   */
  const jlm::llvm::CallNode & CallNode_;

public:
  FunctionCallConstraint(PointerObjectIndex pointer, const jlm::llvm::CallNode & callNode)
      : Pointer_(pointer),
        CallNode_(callNode)
  {}

  /**
   * @return the PointerObject representing the function pointer being called
   */
  [[nodiscard]] PointerObjectIndex
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the function pointer being called
   */
  void
  SetPointer(PointerObjectIndex pointer)
  {
    Pointer_ = pointer;
  }

  /**
   * @return the RVSDG call node for the function call
   */
  [[nodiscard]] const jlm::llvm::CallNode &
  GetCallNode() const noexcept
  {
    return CallNode_;
  }

  /**
   * Apply this constraint to \p set once.
   * @return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  ApplyDirectly(PointerObjectSet & set);
};

/**
 * Helper class representing the global constraint:
 *   For all PointerObjects x marked as escaping, all pointees in P(x) are escaping
 */
class EscapeFlagConstraint final
{
public:
  EscapeFlagConstraint() = delete;

  /**
   * Performs the minimum set of changes required to satisfy the constraint.
   * @param set the PointerObjectSet representing this module
   * @return true if the function modified any flags.
   */
  static bool
  PropagateEscapedFlagsDirectly(PointerObjectSet & set);
};

/**
 * Helper class representing the global constraint:
 *   For each function x marked as escaped, all parameters point to external, and all results escape
 * Parameters and results that do not have PointerObjects corresponding to them are ignored.
 */
class EscapedFunctionConstraint final
{
public:
  EscapedFunctionConstraint() = delete;

  /**
   * Performs the minimum set of changes required to satisfy the constraint.
   * @param set the PointerObjectSet representing this module
   * @return true if the function modified any flags
   */
  static bool
  PropagateEscapedFunctionsDirectly(PointerObjectSet & set);
};

/**
 * A class for adding and applying constraints to the points-to-sets of the PointerObjectSet.
 * Unlike the set modification methods on PointerObjectSet, constraints can be added in any order,
 * with the same result. Use SolveNaively() to calculate the final points-to-sets.
 *
 * Some additional constraints on the PointerObject flags are built in.
 */
class PointerObjectConstraintSet final
{
public:
  using ConstraintVariant =
      std::variant<SupersetConstraint, StoreConstraint, LoadConstraint, FunctionCallConstraint>;

  explicit PointerObjectConstraintSet(PointerObjectSet & set)
      : Set_(set)
  {}

  PointerObjectConstraintSet(const PointerObjectConstraintSet & other) = delete;

  PointerObjectConstraintSet(PointerObjectConstraintSet && other) = delete;

  PointerObjectConstraintSet &
  operator=(const PointerObjectConstraintSet & other) = delete;

  PointerObjectConstraintSet &
  operator=(PointerObjectConstraintSet && other) = delete;

  /**
   * The simplest constraint, on the form: pointee in P(pointer)
   * @param pointer the PointerObject that should have the pointee in its points-to set
   * @param pointee the PointerObject that should be in the points-to-set
   */
  void
  AddPointerPointeeConstraint(PointerObjectIndex pointer, PointerObjectIndex pointee);

  /**
   * Adds a constraint making \p pointer flagged as pointing to external
   * @param pointer the PointerObject that should be marked as pointing to external
   */
  void
  AddPointsToExternalConstraint(PointerObjectIndex pointer);

  /**
   * Ensures that any PointerObject in P(registerIndex) will be marked as escaped.
   * @param registerIndex the register whose content leaves the module, thus exposing any memory it
   * may point to
   */
  void
  AddRegisterContentEscapedConstraint(PointerObjectIndex registerIndex);

  /**
   * Generic add function for all struct based constraints
   * @param c an instance of a constraint struct, passed as a ConstraintVariant
   */
  void
  AddConstraint(ConstraintVariant c);

  /**
   * Retrieves all added constraints that were not simple one-off flag changes
   */
  [[nodiscard]] const std::vector<ConstraintVariant> &
  GetConstraints() const noexcept;

  /**
   * Finds a least solution satisfying all constraints, using the Worklist algorithm.
   * Descriptions of the algorithm can be found in
   *  - Pearce et al. 2003: "Online cycle detection and difference propagation for pointer analysis"
   *  - Hardekopf et al. 2007, "The Ant and the Grasshopper".
   * @return the total number of work items handled by the WorkList algorithm
   */
  size_t
  SolveUsingWorklist();

  /**
   * Iterates over and applies constraints until all points-to-sets satisfy them.
   * Also applies inference rules on the escaped and pointing to external flags.
   * This operation potentially has a long runtime, with an upper bound of O(n^5).
   * @return the number of iterations until a fixed solution was established. At least 1.
   */
  size_t
  SolveNaively();

private:
  // The PointerObjectSet being built upon
  PointerObjectSet & Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;
};

} // namespace jlm::llvm::aa

#endif
