/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/BijectiveMap.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/GraphWriter.hpp>
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

#ifdef ANDERSEN_NO_FLAGS
  // A special object representing all external memory, of which only one exists
  ExternalObject,
#endif

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

    // If set, this pointer object may point to other pointer objects.
    // If unset, the analysis should make no attempt at tracking what this PointerObject may target.
    // The final PointsToGraph will not have any outgoing edges for this object.
    const uint8_t CanPointFlag : 1;

#ifndef ANDERSEN_NO_FLAGS
    // This memory object's address is known outside the module.
    // Can only be true on memory objects.
    uint8_t HasEscaped : 1;

    // Any pointee of this PointerObject has its address escaped.
    // The unification root is the source of truth for this flag!
    // This flag is implied by HasEscaped
    uint8_t PointeesEscaping : 1;

    // If set, this pointer object is pointing to external.
    // The unification root is the source of truth for this flag!
    // This flag is implied by HasEscaped
    uint8_t PointsToExternal : 1;

    // If set, any pointee of this object should point to external.
    // The unification root is the source of truth for this flag!
    uint8_t StoredAsScalar : 1;

    // If set, any pointee of this object should mark its pointees as escaping.
    // The unification root is the source of truth for this flag!
    uint8_t LoadedAsScalar : 1;
#endif

    explicit PointerObject(PointerObjectKind kind, bool canPoint)
        : Kind(kind),
          CanPointFlag(canPoint)
#ifndef ANDERSEN_NO_FLAGS
          ,
          HasEscaped(0),
          PointeesEscaping(0),
          PointsToExternal(0),
          StoredAsScalar(0),
          LoadedAsScalar(0)
#endif
    {
      JLM_ASSERT(kind != PointerObjectKind::COUNT);

      // Ensure that certain kinds of PointerObject always CanPoint or never CanPoint
      if (kind == PointerObjectKind::FunctionMemoryObject
          || kind == PointerObjectKind::ImportMemoryObject)
        JLM_ASSERT(!CanPoint());
      else if (kind == PointerObjectKind::Register)
        JLM_ASSERT(CanPoint());

#ifndef ANDERSEN_NO_FLAGS
      if (!CanPoint())
      {
        // No attempt is made at tracking pointees, so use these flags to inform others
        PointeesEscaping = 1;
        PointsToExternal = 1;
      }
#endif
    }

    /**
     * If a PointerObject is marked with both PointsToExternal and PointeesEscaping,
     * its final points-to set will be identical to the set of escaped memory objects.
     * There is no need to track its pointees, since all pointees can be marked as HasEscaped,
     * and instead be an implicit pointee.
     * @return true if this PointerObject's points-to set can be tracked fully implicitly.
     */
    [[nodiscard]] bool
    CanTrackPointeesImplicitly() const noexcept
    {
#ifdef ANDERSEN_NO_FLAGS
      JLM_UNREACHABLE("ANDERSEN_NO_FLAGS");
#else
      return PointsToExternal && PointeesEscaping;
#endif
    }

    /**
     * Some memory objects can only be pointed to, but never themselves contain pointers.
     * When converting the analysis result to a PointsToGraph, these PointerObjects get no pointees.
     * @return true if the analysis tracks the points-to set of this PointerObject.
     */
    [[nodiscard]] bool
    CanPoint() const noexcept
    {
      return CanPointFlag;
    }

    /**
     * Registers are the only PointerObjects that may not be pointed to, only point.
     * @return true if the kind of this PointerObject is Register, false otherwise
     */
    [[nodiscard]] bool
    IsRegister() const noexcept
    {
      return Kind == PointerObjectKind::Register;
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

  std::unordered_map<const rvsdg::Node *, PointerObjectIndex> AllocaMap_;

  std::unordered_map<const rvsdg::Node *, PointerObjectIndex> MallocMap_;

  std::unordered_map<const delta::node *, PointerObjectIndex> GlobalMap_;

  util::BijectiveMap<const rvsdg::LambdaNode *, PointerObjectIndex> FunctionMap_;

  std::unordered_map<const GraphImport *, PointerObjectIndex> ImportMap_;

#ifdef ANDERSEN_NO_FLAGS
  // The first pointer object index is reserved for the external object
  static constexpr PointerObjectIndex ExternalPointerObject_ = 0;
#endif

  // How many items have been attempted added to explicit points-to sets
  size_t NumSetInsertionAttempts_ = 0;

  // How many pointees have been removed from points-to sets.
  // Explicit pointees can only be removed through unification, and the remove method
  size_t NumExplicitPointeesRemoved_ = 0;

  /**
   * Internal helper function for adding PointerObjects, use the Create* methods instead
   */
  [[nodiscard]] PointerObjectIndex
  AddPointerObject(PointerObjectKind kind, bool canPoint);

  /**
   * Internal helper function for making P(superset) a superset of P(subset), with a callback.
   * @see MakePointsToSetSuperset
   */
  template<typename NewPointeeFunctor>
  bool
  PropagateNewPointees(
      PointerObjectIndex superset,
      PointerObjectIndex subset,
      NewPointeeFunctor & onNewPointee);

public:
  PointerObjectSet();

  [[nodiscard]] size_t
  NumPointerObjects() const noexcept;

  /**
   * @return the number of PointerObjects in the set matching the specified \p kind.
   */
  [[nodiscard]] size_t
  NumPointerObjectsOfKind(PointerObjectKind kind) const noexcept;

  /**
   * @return the number of PointerObjects in the set representing virtual registers
   */
  [[nodiscard]] size_t
  NumRegisterPointerObjects() const noexcept;

  [[nodiscard]] size_t
  NumMemoryPointerObjects() const noexcept;

  [[nodiscard]] size_t
  NumMemoryPointerObjectsCanPoint() const noexcept;

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
  CreateAllocaMemoryObject(const rvsdg::Node & allocaNode, bool canPoint);

  [[nodiscard]] PointerObjectIndex
  CreateMallocMemoryObject(const rvsdg::Node & mallocNode, bool canPoint);

  [[nodiscard]] PointerObjectIndex
  CreateGlobalMemoryObject(const delta::node & deltaNode, bool canPoint);

  /**
   * Creates a PointerObject of Function kind associated with the given \p lambdaNode.
   * The lambda node can not be associated with a PointerObject already.
   * @param lambdaNode the RVSDG node defining the function,
   * @return the index of the new PointerObject in the PointerObjectSet
   */
  [[nodiscard]] PointerObjectIndex
  CreateFunctionMemoryObject(const rvsdg::LambdaNode & lambdaNode);

  /**
   * Retrieves the PointerObject of Function kind associated with the given lambda node
   * @param lambdaNode the lambda node associated with the existing PointerObject
   * @return the index of the associated PointerObject
   */
  [[nodiscard]] PointerObjectIndex
  GetFunctionMemoryObject(const rvsdg::LambdaNode & lambdaNode) const;

  /**
   * Gets the lambda node associated with a given PointerObject.
   * @param index the index of the PointerObject
   * @return the lambda node associated with the PointerObject
   */
  [[nodiscard]] const rvsdg::LambdaNode &
  GetLambdaNodeFromFunctionMemoryObject(PointerObjectIndex index) const;

  [[nodiscard]] PointerObjectIndex
  CreateImportMemoryObject(const GraphImport & importNode);

  const std::unordered_map<const rvsdg::output *, PointerObjectIndex> &
  GetRegisterMap() const noexcept;

  const std::unordered_map<const rvsdg::Node *, PointerObjectIndex> &
  GetAllocaMap() const noexcept;

  const std::unordered_map<const rvsdg::Node *, PointerObjectIndex> &
  GetMallocMap() const noexcept;

  const std::unordered_map<const delta::node *, PointerObjectIndex> &
  GetGlobalMap() const noexcept;

  const util::BijectiveMap<const rvsdg::LambdaNode *, PointerObjectIndex> &
  GetFunctionMap() const noexcept;

  const std::unordered_map<const GraphImport *, PointerObjectIndex> &
  GetImportMap() const noexcept;

#ifdef ANDERSEN_NO_FLAGS
  PointerObjectIndex
  GetExternalObject() const noexcept
  {
    return ExternalPointerObject_;
  }
#endif

  /**
   * @return the kind of the PointerObject with the given \p index
   */
  [[nodiscard]] PointerObjectKind
  GetPointerObjectKind(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index can point, otherwise false
   */
  [[nodiscard]] bool
  CanPoint(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index is a Register
   */
  [[nodiscard]] bool
  IsPointerObjectRegister(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index has its address escaped
   */
  [[nodiscard]] bool
  HasEscaped(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as having escaped the module.
   * Can only be called on non-register PointerObjects.
   * Implies both the PointeesEscaping flag, and the PointsToExternal flag.
   * @return true if the flag was changed by this operation, false otherwise
   */
  bool
  MarkAsEscaped(PointerObjectIndex index);

#ifndef ANDERSEN_NO_FLAGS
  /**
   * @return true if the PointerObject with the given \p index makes all its pointees escape
   */
  [[nodiscard]] bool
  HasPointeesEscaping(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as having all pointees escaping.
   * The flag is applied to the unification root.
   * @return true if the flag was changed by this operation, false otherwise
   */
  bool
  MarkAsPointeesEscaping(PointerObjectIndex index);

  /**
   * @return true if the PointerObject with the given \p index points to external, otherwise false
   */
  [[nodiscard]] bool
  IsPointingToExternal(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as pointing to external.
   * The flag is applied to the unification root.
   * @return true if the flag was changed by this operation, false otherwise
   */
  bool
  MarkAsPointingToExternal(PointerObjectIndex index);

  /**
   * @return true if the PointerObject with the given \p index is flagged as both
   * PointsToExternal and PointeesEscaping.
   * In that case, any explicit pointee will also be implicit, so it is better to avoid explicit.
   */
  [[nodiscard]] bool
  CanTrackPointeesImplicitly(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as holding the target of a scalar store.
   * @return true if the flags was changed by this operation, false otherwise
   */
  bool
  MarkAsStoringAsScalar(PointerObjectIndex index);

  /**
   * @return true if the PointerObject with the given \p index is the target of a scalar store,
   * false otherwise. If it is, any pointee of \p index will be marked as pointing to external.
   */
  [[nodiscard]] bool
  IsStoredAsScalar(PointerObjectIndex index) const noexcept;

  /**
   * Marks the PointerObject with the given \p index as holding the target of a scalar load.
   * @return true if the flags was changed by this operation, false otherwise
   */
  bool
  MarkAsLoadingAsScalar(PointerObjectIndex index);

  /**
   * @return true if the PointerObject with the given \p index is the target of a scalar load, false
   * otherwise. If it is, any pointee of \p index will be marked as making its pointees escape.
   */
  [[nodiscard]] bool
  IsLoadedAsScalar(PointerObjectIndex index) const noexcept;

#endif

  /**
   * @return the root in the unification the PointerObject with the given \p index belongs to.
   * PointerObjects that have not been unified will always be their own root.
   */
  [[nodiscard]] PointerObjectIndex
  GetUnificationRoot(PointerObjectIndex index) const noexcept;

  /**
   * @return true if the PointerObject with the given \p index is its own unification root
   */
  [[nodiscard]] bool
  IsUnificationRoot(PointerObjectIndex index) const noexcept;

  /**
   * Unifies two PointerObjects, such that they will forever share their set of pointees.
   * If any object in the unification points to external, they will all point to external.
   * The HasEscaped flags are not shared, and can still be set individually.
   * If the objects already belong to the same disjoint set, this is a no-op.
   * @param object1 the index of the first PointerObject to unify
   * @param object2 the index of the second PointerObject to unify
   * @return the index of the root PointerObject in the unification
   */
  PointerObjectIndex
  UnifyPointerObjects(PointerObjectIndex object1, PointerObjectIndex object2);

  /**
   * Looks up the pointees of the given PointerObject \p index.
   * If index is part of a unification, the unification root's points-to set is returned.
   * @return the PointsToSet of the PointerObject.
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
   * Makes P(\p superset) a superset of P(\p subset), by adding any elements in the set difference.
   * Also propagates the PointsToExternal flag.
   * @param superset the index of the PointerObject that shall point to everything subset points to
   * @param subset the index of the PointerObject whose pointees shall all be pointed to by superset
   * as well
   *
   * @return true if P(\p superset) or any flags were modified by this operation
   */
  bool
  MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset);

  /**
   * A version of MakePointsToSetSuperset that adds any new pointees of \p superset,
   * to the set \p newPointees.
   */
  bool
  MakePointsToSetSuperset(
      PointerObjectIndex superset,
      PointerObjectIndex subset,
      util::HashSet<PointerObjectIndex> & newPointees);

  /**
   * Removes all pointees from the PointerObject with the given \p index.
   * Can be used, e.g., when the PointerObject already points to all its pointees implicitly.
   */
  void
  RemoveAllPointees(PointerObjectIndex index);

  /**
   * @param pointer the PointerObject possibly pointing to \p pointee
   * @param pointee the PointerObject possibly being pointed at
   * @return true if \p pointer points to \p pointee, either explicitly, implicitly, or both.
   */
  bool
  IsPointingTo(PointerObjectIndex pointer, PointerObjectIndex pointee) const;

  /**
   * Creates a clone of this PointerObjectSet, with all the same PointerObjects,
   * flags, unifications and points-to sets.
   * @return an owned clone of this
   */
  [[nodiscard]] std::unique_ptr<PointerObjectSet>
  Clone() const;

  /**
   * Compares the Sol sets of all PointerObjects between two PointerObjectSets.
   * Assumes that this and \p other represent the same set of PointerObjects, and in the same order.
   * Only the final Sol set of each PointerObject matters, so unifications do not need to match.
   * The set of escaped PointerObjects must match.
   * @param other the set being compared to
   * @return true if this and \p other are identical, false otherwise
   */
  [[nodiscard]] bool
  HasIdenticalSolAs(const PointerObjectSet & other) const;

  /**
   * @return the number of pointees that have been inserted, or were attempted inserted
   * but already existed, among all points-to sets in this PointerObjectSet.
   * Unioning a set x into another makes |x| insertion attempts.
   */
  [[nodiscard]] size_t
  GetNumSetInsertionAttempts() const noexcept;

  /**
   * @return the number of pointees that have been removed from points-to sets,
   * due to either unification, or the RemoveAllPointees() method.
   */
  [[nodiscard]] size_t
  GetNumExplicitPointeesRemoved() const noexcept;
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
 *   if f is not a lambda:
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

#ifndef ANDERSEN_NO_FLAGS
/**
 * Helper class representing the global constraint:
 *   For all PointerObjects x marked as PointeesEscaping, all pointees in P(x) are escaping
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
#endif

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
 * with the same result. Multiple solvers can be used to solve for the final points-to sets.
 *
 * Some additional constraints on the PointerObject flags are built in.
 */
class PointerObjectConstraintSet final
{
public:
  using ConstraintVariant =
      std::variant<SupersetConstraint, StoreConstraint, LoadConstraint, FunctionCallConstraint>;

  enum class WorklistSolverPolicy
  {
    /**
     * A worklist policy based on selecting the work item that was least recently selected. From:
     *   A. Kanamori and D. Weise "Worklist management strategies for Dataflow Analysis" (1994)
     * @see jlm::util::LrfWorklist
     */
    LeastRecentlyFired,

    /**
     * A worklist policy like LeastRecentlyFired, but using two lists instead of a priority queue.
     * Described by:
     *   B. Hardekopf and C. Lin "The And and the Grasshopper: Fast and Accurate Pointer Analysis
     *   for Millions of Lines of Code" (2007)
     * @see jlm::util::TwoPhaseLrfWorklist
     */
    TwoPhaseLeastRecentlyFired,

    /**
     * Not a real worklist policy.
     * For each "sweep", all nodes are visited in topological order.
     * Any cycles found during topological sorting are eliminated.
     * This continues until a full sweep has been done with no attempts at pushing to the worklist.
     * Described by:
     *   Pearce 2007: "Efficient field-sensitive pointer analysis of C"
     */
    TopologicalSort,

    /**
     * A worklist policy based on a queue.
     * @see jlm::util::FifoWorklist
     */
    FirstInFirstOut,

    /**
     * A worklist policy based on a stack.
     * @see jlm::util::LifoWorklist
     */
    LastInFirstOut
  };

  [[nodiscard]] static const char *
  WorklistSolverPolicyToString(WorklistSolverPolicy policy);

  /**
   * Struct holding statistics from solving the constraint set using the worklist solver.
   */
  struct WorklistStatistics
  {
    explicit WorklistStatistics(WorklistSolverPolicy policy)
        : Policy(policy)
    {}

    /**
     * The policy used for the worklist.
     */
    WorklistSolverPolicy Policy;

    /**
     * The number of items that were popped from the worklist before the solution converged.
     */
    size_t NumWorkItemsPopped{};

    /**
     * The sum of the number of new pointees, for each visited work item.
     * If Difference Propagation is not enabled, all pointees are always regarded as new.
     */
    size_t NumWorkItemNewPointees{};

    /**
     * The number of times the topological worklist orders the whole set of work items
     * and visits them all in topological order.
     */
    std::optional<size_t> NumTopologicalWorklistSweeps;

    /**
     * The number of cycles detected by online cycle detection,
     * and number of unifications made to eliminate the cycles,
     * if Online Cycle Detection is enabled.
     */
    std::optional<size_t> NumOnlineCyclesDetected;

    /**
     * The number of unifications made by online cycle detection, if enabled.
     */
    std::optional<size_t> NumOnlineCycleUnifications;

    /**
     * The number of unifications performed due to hybrid cycle detection.
     */
    std::optional<size_t> NumHybridCycleUnifications;

    /**
     * The number of DFSs started in attempts at detecting cycles,
     * the number of cycles detected by lazy cycle detection,
     * and number of unifications made to eliminate the cycles,
     * if Lazy Cycle Detection is enabled.
     */
    std::optional<size_t> NumLazyCyclesDetectionAttempts;
    std::optional<size_t> NumLazyCyclesDetected;
    std::optional<size_t> NumLazyCycleUnifications;

    /**
     * When Prefer Implicit Pointees is enabled, and a node's pointees can be tracked fully
     * implicitly, its set of explicit pointees is cleared.
     */
    std::optional<size_t> NumPipExplicitPointeesRemoved;
  };

  /**
   * Struct holding statistics from solving using Wave Propagation
   */
  struct WavePropagationStatistics
  {
    size_t NumIterations{};

    size_t NumUnifications{};
  };

  explicit PointerObjectConstraintSet(PointerObjectSet & set)
      : Set_(set),
        Constraints_(),
        ConstraintSetFrozen_(false)
  {
#ifdef ANDERSEN_NO_FLAGS
    AddConstraint(StoreConstraint(set.GetExternalObject(), set.GetExternalObject()));
    AddConstraint(LoadConstraint(set.GetExternalObject(), set.GetExternalObject()));
#endif
  }

  PointerObjectConstraintSet(const PointerObjectConstraintSet & other) = delete;

  PointerObjectConstraintSet(PointerObjectConstraintSet && other) = delete;

  PointerObjectConstraintSet &
  operator=(const PointerObjectConstraintSet & other) = delete;

  PointerObjectConstraintSet &
  operator=(PointerObjectConstraintSet && other) = delete;

  /**
   * Some offline processing relies on knowing about all constraints that will ever be added.
   * After doing such processing, the constraint set is frozen, which prevents any new constraints
   * from being added. Offline processing and solving can still be performed while frozen.
   * @return true if the constraint set has been frozen, false otherwise
   */
  [[nodiscard]] bool
  IsFrozen() const noexcept;

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
   * @return all added constraints that were not simple one-off pointee inclusions or flag changes
   */
  [[nodiscard]] const std::vector<ConstraintVariant> &
  GetConstraints() const noexcept;

  /**
   * @return the number of base constraints
   */
  [[nodiscard]] size_t
  NumBaseConstraints() const noexcept;

  /**
   * Gets the number of flag constraints, among all PointerObjetcs.
   * Flags that are unified are only counted once (on the unification root).
   * The count is divided into two: flags for loads/stores of scalars, and the other flags
   * @return a pair (num flags on scalar operations, num other flags)
   */
  [[nodiscard]] std::pair<size_t, size_t>
  NumFlagConstraints() const noexcept;

  /**
   * Creates a subset graph containing all PointerObjects, their current points-to sets,
   * and edges representing the current set of constraints.
   */
  util::Graph &
  DrawSubsetGraph(util::GraphWriter & writer) const;

  /**
   * Performs off-line detection of PointerObjects that can be shown to always contain
   * the same pointees, and unifies them.
   * It is a version of Rountev and Chandra, 2000: "Off-line variable substitution",
   * modified to support SSA based constraint graphs.
   *
   * The algorithm uses an offline constraint graph, consisting of two nodes per PointerObject v:
   *  n(v) represents the points-to set of v
   *  n(*v) represents the union of points-to sets of all pointees of v
   *  Edges in the graph represent points-to set inclusion.
   *
   * In this graph, strongly connected components (SCCs) are collapsed into single equivalence sets.
   *
   * If an SCC consists of only "direct" nodes, and all predecessors share equivalence
   * set label, the SCC gets the same label.
   * See PointerObjectConstraintSet::CreateOvsSubsetGraph() for a description of direct nodes.
   *
   * All PointerObjects v1, ... vN where n(v1), ... n(vN) share equivalence set label, get unified.
   * The run time is linear in the amount of PointerObjects and constraints.
   *
   * @param storeRefCycleUnificationRoot if true, ref nodes in cycles with regular nodes are stored,
   *   to be used by hybrid cycle detection during solving.
   * @return the number PointerObject unifications made
   * @see NormalizeConstraints() call it afterwards to remove constraints made unnecessary.
   */
  size_t
  PerformOfflineVariableSubstitution(bool storeRefCycleUnificationRoot);

  /**
   * Traverses the list of constraints, and does the following:
   *  - Redirects constraints to reference the root of unifications
   *  - Removes no-op constraints (e.g. X is a superset of X)
   *  - Removes duplicate constraints
   *
   * @return the number of constraints that were removed
   */
  size_t
  NormalizeConstraints();

  /**
   * Finds a least solution satisfying all constraints, using the Worklist algorithm.
   * Descriptions of the algorithm can be found in
   *  - Pearce et al. 2003: "Online cycle detection and difference propagation for pointer analysis"
   *  - Hardekopf and Lin, 2007: "The Ant and the Grasshopper".
   * These papers also describe a set of techniques that potentially improve solving performance:
   *  - Online Cycle Detection (Pearce, 2003)
   *  - Hybrid Cycle Detection (Hardekopf 2007)
   *  - Lazy Cycle Detection (Hardekopf 2007)
   *  - Difference Propagation (Pearce, 2003)
   * @param policy the worklist iteration order policy to use
   * @param enableOnlineCycleDetection if true, online cycle detection will be performed.
   * @param enableHybridCycleDetection if true, hybrid cycle detection will be performed.
   * @param enableLazyCycleDetection if true, lazy cycle detection will be performed.
   * @param enableDifferencePropagation if true, difference propagation will be enabled.
   * @param enablePreferImplicitPropation if true, enables PIP, which is novel to this codebase
   * @return an instance of WorklistStatistics describing solver statistics
   */
  WorklistStatistics
  SolveUsingWorklist(
      WorklistSolverPolicy policy,
      bool enableOnlineCycleDetection,
      bool enableHybridCycleDetection,
      bool enableLazyCycleDetection,
      bool enableDifferencePropagation,
      bool enablePreferImplicitPropation);

  /**
   * Iterates over and applies constraints until all points-to-sets satisfy them.
   * Also applies inference rules on the escaped and pointing to external flags.
   * This operation potentially has a long runtime, with an upper bound of O(n^5).
   * @return the number of iterations until a fixed solution was established. At least 1.
   */
  size_t
  SolveNaively();

  /**
   * Solves the constraint set using the Wave propagation technique described in
   * Pereira and Berlin, 2009, "Wave Propagation and Deep Propagation for Pointer Analysis".
   * The algorithm is an evolution on Pearce's topological worklist policy + difference propagation.
   * It has three phases that loop until a fixed point is reached:
   *  - collapse cycles (by finding SCCs)
   *  - Propagate in topological order
   *  - Add new edges
   * @return statistics about the solving
   */
  WavePropagationStatistics
  SolveUsingWavePropagation();

  /**
   * Creates a clone of this constraint set, and the underlying PointerObjectSet.
   * The result is an identical copy, containing no references to the original.
   * @return the cloned PointerObjectSet and PointerObjectConstraintSet
   */
  std::pair<std::unique_ptr<PointerObjectSet>, std::unique_ptr<PointerObjectConstraintSet>>
  Clone() const;

private:
  /**
   * Creates a special subset graph containing both regular nodes n(v) and dereference nodes n(*v).
   * The graph is used by PointerObjectConstraintSet::PerformOfflineVariableSubstitution().
   *
   * Some nodes are marked as direct, when all subset predecessors are known offline.
   * They can:
   *  - only be n(v) nodes, where v is a register
   *  - not be the return value of a function call
   *  - not be an argument of a function body
   *
   *  @return a tuple containing:
   *   - the total number of nodes N
   *   - a vector of length N containing the successors of each node
   *   - a boolean vector of length N, containing true on direct nodes, and false otherwise
   */
  std::tuple<size_t, std::vector<util::HashSet<PointerObjectIndex>>, std::vector<bool>>
  CreateOvsSubsetGraph();

  /**
   * The worklist solver, with configuration passed at compile time as templates.
   * @param statistics the WorklistStatistics instance that will get information about this run.
   * @tparam Worklist a type supporting the worklist interface with PointerObjectIndex as work items
   * @tparam EnableOnlineCycleDetection if true, online cycle detection is enabled.
   * @tparam EnableHybridCycleDetection if true, hybrid cycle detection is enabled.
   * @tparam EnableLazyCycleDetection if true, lazy cycle detection is enabled.
   * @tparam EnableDifferencePropagation if true, difference propagation is enabled.
   * @tparam EnablePreferImplicitPointees if true, prefer implicit pointees is enabled
   * @see SolveUsingWorklist() for the public interface.
   */
  template<
      typename Worklist,
      bool EnableOnlineCycleDetection,
      bool EnableHybridCycleDetection,
      bool EnableLazyCycleDetection,
      bool EnableDifferencePropagation,
      bool EnablePreferImplicitPointees>
  void
  RunWorklistSolver(WorklistStatistics & statistics);

  // The PointerObjectSet being built upon
  PointerObjectSet & Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;

  // When true, no new constraints can be added.
  // Only offline processing is allowed to modify the constraint set.
  bool ConstraintSetFrozen_;

  // Offline Variable Substitution can determine that all pointees of a node p,
  // should be unified together, possibly with some other PointerObjects.
  // This happens when *p is in a cycle with regular nodes
  std::unordered_map<PointerObjectIndex, PointerObjectIndex> RefNodeUnificationRoot_;
};

} // namespace jlm::llvm::aa

#endif
