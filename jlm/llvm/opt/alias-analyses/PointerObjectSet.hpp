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

/**
 * Class representing a single entry in the PointerObjectSet.
 */
class PointerObject final
{
  PointerObjectKind Kind_ : util::BitWidthOfEnum(PointerObjectKind::COUNT);

  // May point to a memory object defined outside the module, or any escaped memory object
  uint8_t PointsToExternal_ : 1;

  // This memory object's address is known outside the module.
  // If set and Kind_ is Register, it only means the pointees of the Reigister have escaped.
  uint8_t HasEscaped_ : 1;

public:
  using Index = std::uint32_t;

  explicit PointerObject(PointerObjectKind kind)
      : Kind_(kind),
        PointsToExternal_(0),
        HasEscaped_(0)
  {
    JLM_ASSERT(kind != PointerObjectKind::COUNT);

    // Memory objects from other modules are definitely not private to this module
    if (kind == PointerObjectKind::ImportMemoryObject)
      MarkAsEscaped();
  }

  [[nodiscard]] PointerObjectKind
  GetKind() const noexcept
  {
    return Kind_;
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
    return Kind_ != PointerObjectKind::FunctionMemoryObject;
  }

  /**
   * Some PointerObjects don't have addresses, and can as such not be pointed to.
   * Any attempt at adding them to a points-to-set is a fatal error.
   * @return true if this PointerObject can be pointed to by another PointerObject
   */
  [[nodiscard]] bool
  CanBePointee() const noexcept
  {
    return Kind_ != PointerObjectKind::Register;
  }

  /**
   * When the PointsToExternal-flag is set, the PointerObject possibly points to a storage
   * instance declared outside to module, or to a memory object from this same module, that has
   * escaped.
   * @return true if the PointsToExternal flag is set.
   */
  [[nodiscard]] bool
  PointsToExternal() const noexcept
  {
    return PointsToExternal_;
  }

  /**
   * Sets the PointsToExternal-flag, if possible.
   * If CanPoint() is false, this is a no-op.
   * @return true if the PointsToExternal flag was modified, otherwise false
   */
  bool
  MarkAsPointsToExternal() noexcept
  {
    if (!CanPoint())
      return false;

    bool modified = !PointsToExternal_;
    PointsToExternal_ = 1;
    return modified;
  }

  /**
   * When set, the PointerObject's value is accessible from outside the module.
   * Anything it points to can also be accessed outside the module, and should also be marked as
   * escaped.
   *
   * If CanBePointee() is true, this PointerObjects's address is available outside the module,
   * and can potentially be written to. Therefore, HasEscaped implies the PointsToEscaped flag,
   * if it is possible to set it.
   *
   * @return true if the PointerObject is marked as having escaped
   */
  [[nodiscard]] bool
  HasEscaped() const noexcept
  {
    return HasEscaped_;
  }

  /**
   * Sets the HasEscaped-flag, indicating that this PointerObject's value is available outside
   * the module. If CanBePointee() is true, its address is also escaped, and PointsToExternal
   * will be set as well, if possible.
   *
   * @see HasEscaped()
   * @return true if the HasEscaped or PointsToExternal flags were modified, otherwise false.
   */
  bool
  MarkAsEscaped() noexcept
  {
    bool modified = !HasEscaped_;
    HasEscaped_ = 1;
    if (CanBePointee())
      modified |= MarkAsPointsToExternal();
    return modified;
  }
};

/**
 * A class containing a set of PointerObjects, and their points-to-sets,
 * as well as mappings from RVSDG nodes/outputs to the PointerObjects.
 * For brevity, P(x) denotes the points-to-set of a PointerObject x.
 */
class PointerObjectSet final
{
  // All PointerObjects in the set
  std::vector<PointerObject> PointerObjects_;

  // For each PointerObject, a set of the other PointerObjects it points to
  std::vector<util::HashSet<PointerObject::Index>> PointsToSets_;

  // Mapping from register to PointerObject
  // Unlike the other maps, several rvsdg::output* can share register PointerObject
  std::unordered_map<const rvsdg::output *, PointerObject::Index> RegisterMap_;

  std::unordered_map<const rvsdg::node *, PointerObject::Index> AllocaMap_;

  std::unordered_map<const rvsdg::node *, PointerObject::Index> MallocMap_;

  std::unordered_map<const delta::node *, PointerObject::Index> GlobalMap_;

  util::BijectiveMap<const lambda::node *, PointerObject::Index> FunctionMap_;

  std::unordered_map<const rvsdg::argument *, PointerObject::Index> ImportMap_;

  /**
   * Internal helper function for adding PointerObjects, use the Create* methods instead
   */
  [[nodiscard]] PointerObject::Index
  AddPointerObject(PointerObjectKind kind);

public:
  [[nodiscard]] size_t
  NumPointerObjects() const noexcept;

  /**
   * Returns the number of PointerObjects in the set matching the specified \p kind.
   * @return the number of matches
   */
  [[nodiscard]] size_t
  NumPointerObjectsOfKind(PointerObjectKind kind) const noexcept;

  /**
   * Gets the PointerObject associated with the given \p index
   * @return a reference to the PointerObject with the given index
   */
  [[nodiscard]] PointerObject &
  GetPointerObject(PointerObject::Index index);

  [[nodiscard]] const PointerObject &
  GetPointerObject(PointerObject::Index index) const;

  /**
   * Creates a PointerObject of Register kind and maps the rvsdg output to the new PointerObject.
   * The rvsdg output can not already be associated with a PointerObject.
   * @param rvsdgOutput the rvsdg output associated with the register PointerObject
   * @return the index of the new PointerObject in the PointerObjectSet
   */
  [[nodiscard]] PointerObject::Index
  CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput);

  /**
   * Retrieves a previously created PointerObject of Register kind.
   * @param rvsdgOutput an rvsdg::output that already corresponds to a PointerObject in the set
   * @return the index of the PointerObject associated with the rvsdg::output
   */
  [[nodiscard]] PointerObject::Index
  GetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const;

  /**
   * Retrieves a previously created PointerObject of Register kind, associated with an rvsdg output.
   * If no PointerObject is associated with the given output, nullopt is returned.
   * @param rvsdgOutput the rvsdg::output that might correspond to a PointerObject in the set
   * @return the index of the PointerObject associated with rvsdgOutput, if it exists
   */
  [[nodiscard]] std::optional<PointerObject::Index>
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
      PointerObject::Index pointerObject);

  /**
   * Creates a PointerObject of Register kind, without any association to any node in the program.
   * It can not be pointed to, and will not be included in the final PointsToGraph,
   * but it can be used as a temporary object to string together constraints.
   * @see Andersen::AnalyzeMemcpy.
   * @return the index of the new PointerObject
   */
  [[nodiscard]] PointerObject::Index
  CreateDummyRegisterPointerObject();

  [[nodiscard]] PointerObject::Index
  CreateAllocaMemoryObject(const rvsdg::node & allocaNode);

  [[nodiscard]] PointerObject::Index
  CreateMallocMemoryObject(const rvsdg::node & mallocNode);

  [[nodiscard]] PointerObject::Index
  CreateGlobalMemoryObject(const delta::node & deltaNode);

  /**
   * Creates a PointerObject of Function kind associated with the given \p lambdaNode.
   * The lambda node can not be associated with a PointerObject already.
   * @param lambdaNode the RVSDG node defining the function,
   * @return the index of the new PointerObject in the PointerObjectSet
   */
  [[nodiscard]] PointerObject::Index
  CreateFunctionMemoryObject(const lambda::node & lambdaNode);

  /**
   * Retrieves the PointerObject of Function kind associated with the given lambda node
   * @param lambdaNode the lambda node associated with the existing PointerObject
   * @return the index of the associated PointerObject
   */
  [[nodiscard]] PointerObject::Index
  GetFunctionMemoryObject(const lambda::node & lambdaNode) const;

  /**
   * Gets the lambda node associated with a given PointerObject.
   * @param index the index of the PointerObject
   * @return the lambda node associated with the PointerObject
   */
  [[nodiscard]] const lambda::node &
  GetLambdaNodeFromFunctionMemoryObject(PointerObject::Index index) const;

  [[nodiscard]] PointerObject::Index
  CreateImportMemoryObject(const rvsdg::argument & importNode);

  const std::unordered_map<const rvsdg::output *, PointerObject::Index> &
  GetRegisterMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
  GetAllocaMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
  GetMallocMap() const noexcept;

  const std::unordered_map<const delta::node *, PointerObject::Index> &
  GetGlobalMap() const noexcept;

  const util::BijectiveMap<const lambda::node *, PointerObject::Index> &
  GetFunctionMap() const noexcept;

  const std::unordered_map<const rvsdg::argument *, PointerObject::Index> &
  GetImportMap() const noexcept;

  [[nodiscard]] const util::HashSet<PointerObject::Index> &
  GetPointsToSet(PointerObject::Index idx) const;

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
  AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee);

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
  MakePointsToSetSuperset(PointerObject::Index superset, PointerObject::Index subset);

  /**
   * Adds the Escaped flag to all PointerObjects in the P(\p pointer) set
   * @param pointer the pointer whose pointees should be marked as escaped
   * @return true if any PointerObjects had their flag modified by this operation
   */
  bool
  MarkAllPointeesAsEscaped(PointerObject::Index pointer);
};

/**
 * A constraint of the form:
 * P(superset) supseteq P(subset)
 * Example of application is when a register has multiple source values
 */
class SupersetConstraint final
{
  PointerObject::Index Superset_;
  PointerObject::Index Subset_;

public:
  SupersetConstraint(PointerObject::Index superset, PointerObject::Index subset)
      : Superset_(superset),
        Subset_(subset)
  {}

  /**
   * @return the PointerObject that should point to everything the subset points to
   */
  [[nodiscard]] PointerObject::Index
  GetSuperset() const noexcept
  {
    return Superset_;
  }

  /**
   * @param superset the new PointerObject that should point to everything the subset points to
   */
  void
  SetSuperset(PointerObject::Index superset)
  {
    Superset_ = superset;
  }

  /**
   * @return the PointerObject whose points-to set should be contained within the superset
   */
  [[nodiscard]] PointerObject::Index
  GetSubset() const noexcept
  {
    return Subset_;
  }

  /**
   * @param subset the new PointerObject whose points-to set should be contained within the superset
   */
  void
  SetSubset(PointerObject::Index subset)
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
  PointerObject::Index Pointer_;
  PointerObject::Index Value_;

public:
  StoreConstraint(PointerObject::Index pointer, PointerObject::Index value)
      : Pointer_(pointer),
        Value_(value)
  {}

  /**
   * @return the PointerObject representing the value written by the store instruction
   */
  [[nodiscard]] PointerObject::Index
  GetValue() const noexcept
  {
    return Value_;
  }

  /**
   * @param value the new PointerObject representing the value written by the store instruction
   */
  void
  SetValue(PointerObject::Index value)
  {
    Value_ = value;
  }

  /**
   * @return the PointerObject representing the pointer written to by the store instruction
   */
  [[nodiscard]] PointerObject::Index
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the pointer written to by the store.
   */
  void
  SetPointer(PointerObject::Index pointer)
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
  PointerObject::Index Value_;
  PointerObject::Index Pointer_;

public:
  LoadConstraint(PointerObject::Index value, PointerObject::Index pointer)
      : Value_(value),
        Pointer_(pointer)
  {}

  /**
   * @return the PointerObject representing the value returned by the load instruction
   */
  [[nodiscard]] PointerObject::Index
  GetValue() const noexcept
  {
    return Value_;
  }

  /**
   * @param value the new PointerObject representing the value returned by the load instruction
   */
  void
  SetValue(PointerObject::Index value)
  {
    Value_ = value;
  }

  /**
   * @return the PointerObject representing the pointer loaded by the load instruction
   */
  [[nodiscard]] PointerObject::Index
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the pointer loaded by the load instruction.
   */
  void
  SetPointer(PointerObject::Index pointer)
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
  PointerObject::Index Pointer_;

  /**
   * The RVSDG node representing the function call
   */
  const jlm::llvm::CallNode & CallNode_;

public:
  FunctionCallConstraint(PointerObject::Index pointer, const jlm::llvm::CallNode & callNode)
      : Pointer_(pointer),
        CallNode_(callNode)
  {}

  /**
   * @return the PointerObject representing the function pointer being called
   */
  [[nodiscard]] PointerObject::Index
  GetPointer() const noexcept
  {
    return Pointer_;
  }

  /**
   * @param pointer the new PointerObject representing the function pointer being called
   */
  void
  SetPointer(PointerObject::Index pointer)
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
  AddPointerPointeeConstraint(PointerObject::Index pointer, PointerObject::Index pointee);

  /**
   * Adds a constraint making \p pointer flagged as pointing to external
   * @param pointer the PointerObject that should be marked as pointing to external
   */
  void
  AddPointsToExternalConstraint(PointerObject::Index pointer);

  /**
   * Ensures that any PointerObject in P(registerIndex) will be marked as escaped.
   * @param registerIndex the register whose content leaves the module, thus exposing any memory it
   * may point to
   */
  void
  AddRegisterContentEscapedConstraint(PointerObject::Index registerIndex);

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
