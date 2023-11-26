/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/util/BiMap.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/Math.hpp>

#include <cstdint>
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

  // When this flag is set, the PointerObject possibly points to a storage instance declared outside
  // to module. The flag also means that the PointerObject possibly points to any escaped storage
  // instance from this module.
  uint8_t PointsToExternal_ : 1;

  // When set, the PointerObject is known to be accessible from outside the module.
  // Anything it points to can also be accessed outside the module, and should also be marked as
  // escaped. Escaped memory object can be overridden outside the module, so HasEscaped implies
  // PointsToExternal.
  uint8_t HasEscaped_ : 1;

public:
  using Index = size_t;

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

  PointerObjectKind
  GetKind() const noexcept
  {
    return Kind_;
  }

  bool
  PointsToExternal() const noexcept
  {
    return PointsToExternal_;
  }

  /**
   * Sets the PointsToExternal-flag.
   * @return True, if the PointsToExternal flag was modified, otherwise false
   */
  bool
  MarkAsPointsToExternal() noexcept
  {
    bool modified = !PointsToExternal_;
    PointsToExternal_ = 1;
    return modified;
  }

  bool
  HasEscaped() const noexcept
  {
    return HasEscaped_;
  }

  /**
   * Sets the Escaped-flag.
   * Also sets the PointsToExternal flag, if unset
   * @return true if HasEscaped or PointsToExternal flag were modified, otherwise false
   */
  bool
  MarkAsEscaped() noexcept
  {
    bool modified = !HasEscaped_;
    HasEscaped_ = 1;
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

  // Mapping from alloca node to PointerObject
  std::unordered_map<const rvsdg::node *, PointerObject::Index> AllocaMap_;

  // Mapping from malloc call node to PointerObject
  std::unordered_map<const rvsdg::node *, PointerObject::Index> MallocMap_;

  // Mapping from global variables declared with delta nodes to PointerObject
  std::unordered_map<const delta::node *, PointerObject::Index> GlobalMap_;

  // Bidirectional mapping from functions declared with lambda nodes to PointerObject
  util::BiMap<const lambda::node *, PointerObject::Index> FunctionMap_;

  // Mapping from symbols imported into the module to PointerObject
  std::unordered_map<const rvsdg::argument *, PointerObject::Index> ImportMap_;

  /**
   * Internal helper function for adding PointerObjects, use the Create* methods instead
   */
  PointerObject::Index
  AddPointerObject(PointerObjectKind kind);

public:
  [[nodiscard]] size_t
  NumPointerObjects() const noexcept;

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
   * @throws jlm::util::error if no associated PointerObject exists
   */
  [[nodiscard]] PointerObject::Index
  GetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const;

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
   * The lambda node can not already
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
   * \brief
   * \param index
   * \return
   */
  [[nodiscard]] const lambda::node &
  GetLambdaNodeFromFunctionMemoryObject(PointerObject::Index index) const;

  PointerObject::Index
  CreateImportMemoryObject(const rvsdg::argument & importNode);

  const std::unordered_map<const rvsdg::output *, PointerObject::Index> &
  GetRegisterMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
  GetAllocaMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
  GetMallocMap() const noexcept;

  const std::unordered_map<const delta::node *, PointerObject::Index> &
  GetGlobalMap() const noexcept;

  const std::unordered_map<const lambda::node *, PointerObject::Index> &
  GetFunctionMap() const noexcept;

  const std::unordered_map<const rvsdg::argument *, PointerObject::Index> &
  GetImportMap() const noexcept;

  [[nodiscard]] const util::HashSet<PointerObject::Index> &
  GetPointsToSet(PointerObject::Index idx) const;

  /**
   * Adds \p pointee to P(\p pointer)
   * @param pointer the index of the PointerObject that shall point to \p pointee
   * @param pointee the index of the PointerObject that is pointed at, can not be a register.
   * @return true if P(\p pointer) was changed by this operation
   */
  bool
  AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee);

  /**
   * Makes P(\p superset) a superset of P(\p subset), by adding any elements in the set difference
   * @param superset the index of the PointerObject that shall point to everything subset points to
   * @param subset the index of the PointerObject whose pointees shall all be pointed to by superset
   * as well
   * @return true if P(\p superset) was modified by this operation
   */
  bool
  MakePointsToSetSuperset(PointerObject::Index superset, PointerObject::Index subset);

  /**
   * Adds the Escaped flag to all PointerObjects in the P(\ pointer) set
   * @param pointer the pointer whose pointees should be marked as escaped
   * @return true if any PointerObjects had their flag modified by this operation
   */
  bool
  MarkAllPointeesAsEscaped(PointerObject::Index pointer);
};

/**
 * A constraint of the form:
 * P(superset) is a superset of P(subset)
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
   * \brief Applies the constraint to the \p set
   * \return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  Apply(PointerObjectSet & set);
};

/**
 * A constraint of the form:
 * for all x in P(pointer1), make P(x) a superset of P(pointer2)
 * Example of application is a store, e.g. when *pointer1 = pointer2
 */
class AllPointeesPointToSupersetConstraint final
{
  PointerObject::Index Pointer1_;
  PointerObject::Index Pointer2_;

public:
  AllPointeesPointToSupersetConstraint(PointerObject::Index pointer1, PointerObject::Index pointer2)
      : Pointer1_(pointer1),
        Pointer2_(pointer2)
  {}

  /**
   * \brief Applies the constraint to the \p set
   * \return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  Apply(PointerObjectSet & set);
};

/**
 * A constraint of the form:
 * P(loaded) is a superset of P(x) for all x in P(pointer)
 * Example of application is a load, e.g. when loaded = *pointer
 */
class SupersetOfAllPointeesConstraint final
{
  PointerObject::Index Loaded_;
  PointerObject::Index Pointer_;

public:
  SupersetOfAllPointeesConstraint(PointerObject::Index loaded, PointerObject::Index pointer)
      : Loaded_(loaded),
        Pointer_(pointer)
  {}

  /**
   * \brief Applies the constraint to the \p set
   * \return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  Apply(PointerObjectSet & set);
};

/**
 * A constraint of the form:
 * If the function escapes the module, its return value should be marked as escaping the module,
 * and all arguments should be marked as possibly pointing to external
 */
class HandleEscapingFunctionConstraint final
{
  PointerObject::Index Lambda_;

  /**
   * Once the function has been determined to be escaping,
   * the flags only need to be applied to its arguments and results once.
   * Afterwards, this boolean will be true, preventing additional unnecessary work.
   */
  bool EscapeHandled_;

public:
  explicit HandleEscapingFunctionConstraint(PointerObject::Index lambda)
      : Lambda_(lambda),
        EscapeHandled_(false)
  {}

  /**
   * \brief Applies the constraint to the \p set
   * \return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  Apply(PointerObjectSet & set);
};

/**
 * A constraint making the given call site communicate with the functions it may call.
 *
 * It follows the given pseudocode:
 * for f in P(CallTarget):
 *   if f is not a lambda:
 *     continue
 *   for each function argument/input pair (a, i):
 *     make P(x) a superset of P(y)
 *   for each return value/output pair (r, o):
 *     make P(o) a superset of P(r)
 *
 * if CallTarget is flagged as PointsToExternal:
 *   mark all of CallNode's parameters as HasEscaped
 *   mark the CallNode's return values as PointsToExternal
 */
class FunctionCallConstraint final
{
  /**
   * A PointerObject of Register kind, representing the source of the function pointer
   */
  PointerObject::Index CallTarget_;

  /**
   *
   */
  const jlm::llvm::CallNode & CallNode_;

public:
  explicit FunctionCallConstraint(
      PointerObject::Index callTarget,
      const jlm::llvm::CallNode & callNode)
      : CallTarget_(callTarget),
        CallNode_(callNode)
  {}

  /**
   * \brief Applies the constraint to the \p set
   * \return true if this operation modified any PointerObjects or points-to-sets
   */
  bool
  Apply(PointerObjectSet & set);
};

/**
 * A class for adding and applying constraints to the points-to-sets of the PointerObjectSet.
 * Unlike the set modification methods on PointerObjectSet, constraints can be added in any order,
 * with the same result. Use Solve() to calculate the final points-to-sets.
 *
 * Some additional constraints on the PointerObject flags are built in.
 */
class PointerObjectConstraintSet final
{
public:
  using ConstraintVariant = std::variant<
      SupersetConstraint,
      AllPointeesPointToSupersetConstraint,
      SupersetOfAllPointeesConstraint,
      HandleEscapingFunctionConstraint>;

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
   * Iterates over and applies constraints until all points-to-sets satisfy them.
   * This operation potentially has a long runtime, with an upper bound of O(n^3).
   */
  void
  Solve();

private:
  /**
   * Ensures that the escaped flag is set for all pointees of any pointer object that is marked as
   * escaped.
   * @return true if the function modified any flags
   */
  bool
  PropagateEscapedFlag();

  // The PointerObjectSet being built upon
  PointerObjectSet & Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;
};

} // namespace jlm::llvm::aa

#endif
