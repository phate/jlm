/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTEROBJECTSET_HPP

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Math.hpp>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <variant>

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
  PointerObjectKind Kind_ : jlm::util::BitWidthOfEnum(PointerObjectKind::COUNT);

  // When this flag is set, the PointerObject possibly points to a storage instance declared outside to module.
  // The flag also means that the PointerObject possibly points to any escaped storage instance from this module.
  uint8_t PointsToExternal_ : 1;

  // When set, the PointerObject is known to be accessible from outside the module.
  // Anything it points to can also be accessed outside the module, and should also be marked as escaped.
  // Escaped memory object can be overridden outside the module, so HasEscaped implies PointsToExternal.
  uint8_t HasEscaped_ : 1;

public:
  using Index = size_t;

  explicit PointerObject(PointerObjectKind kind) : Kind_(kind),
                                                   PointsToExternal_(0), HasEscaped_(0)
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
   * @return true if the PointerObject used to not point to external, before this call
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
   * @return true if the PointerObject's flags were modified by this call
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
 * A class containing a set of PointerObjects, their points-to-sets,
 * as well as mappings from RVSDG nodes/outputs to the PointerObjects.
 */
class PointerObjectSet final {
  // All PointerObjects in the set
  std::vector<PointerObject> PointerObjects_;

  // For each PointerObject, a set of the other PointerObjects it points to
  std::vector<std::unordered_set<PointerObject::Index>> PointsToSets_;

  // Mapping from register to PointerObject
  // Unlike the other maps, several rvsdg::output* can share register PointerObject
  std::unordered_map<const rvsdg::output *, PointerObject::Index> RegisterMap_;
  // Mapping from alloca node to PointerObject
  std::unordered_map<const rvsdg::node *, PointerObject::Index> AllocaMap_;
  // Mapping from malloc call node to PointerObject
  std::unordered_map<const rvsdg::node *, PointerObject::Index> MallocMap_;
  // Mapping from global variables declared with delta nodes to PointerObject
  std::unordered_map<const delta::node *, PointerObject::Index> GlobalMap_;
  // Mapping from functions declared with lambda nodes to PointerObject
  std::unordered_map<const lambda::node *, PointerObject::Index> FunctionMap_;
  // Mapping from symbols imported into the module to PointerObject
  std::unordered_map<const rvsdg::argument *, PointerObject::Index> ImportMap_;

  /**
   * Internal helper function for adding PointerObjects, use the Create* methods instead
   */
  PointerObject::Index
  AddPointerObject(PointerObjectKind kind);

public:

  /**
   * Creates a PointerObject of register kind, and maps the rvsdg output to the new PointerObject.
   * @param rvsdgOutput the rvsdg output associated with the register PointerObject
   * @return the index of the new PointerObject, in the PointerObjectSet
   */
  PointerObject::Index
  CreateRegisterPointerObject(const rvsdg::output &rvsdgOutput);

  /**
   * Reuses an existing PointerObject of register type for an additional rvsdg output.
   * This is useful when two rvsdg outputs can be shown to always hold the exact same value.
   * @param rvsdgOutput the new rvsdg output providing the register value
   * @param pointerObject the index of the existing PointerObject, must be of register kind
   */
  void
  MapRegisterToExistingPointerObject(const rvsdg::output &rvsdgOutput, PointerObject::Index pointerObject);

  PointerObject::Index
  CreateAllocaMemoryObject(const rvsdg::node &allocaNode);

  PointerObject::Index
  CreateMallocMemoryObject(const rvsdg::node &mallocNode);

  PointerObject::Index
  CreateGlobalMemoryObject(const delta::node &deltaNode);

  PointerObject::Index
  CreateFunctionMemoryObject(const lambda::node &lambdaNode);

  PointerObject::Index
  CreateImportMemoryObject(const rvsdg::argument &importNode);

  const std::unordered_map<const rvsdg::output *, PointerObject::Index>&
  GetRegisterMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index>&
  GetAllocaMap() const noexcept;

  const std::unordered_map<const rvsdg::node *, PointerObject::Index>&
  GetMallocMap() const noexcept;

  const std::unordered_map<const delta::node *, PointerObject::Index>&
  GetGlobalMap() const noexcept;

  const std::unordered_map<const lambda::node *, PointerObject::Index>&
  GetFunctionMap() const noexcept;

  const std::unordered_map<const rvsdg::argument *, PointerObject::Index>&
  GetImportMap() const noexcept;

  PointerObject::Index
  NumPointerObjects() const noexcept;

  PointerObject &
  GetPointerObject(PointerObject::Index index);

  const PointerObject &
  GetPointerObject(PointerObject::Index index) const;

  const std::unordered_set<PointerObject::Index> &
  GetPointsToSet(PointerObject::Index idx) const;

  /**
   * Adds pointee to P(pointer)
   * @param pointer the index of the PointerObject that points
   * @param pointee the index of the PointerObject that is pointed at, can not be a register.
   * @return true if P(pointer) was changed by this operation
   */
  bool
  AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee);

  /**
   * Makes P(superset) a superset of P(subset) by adding any elements in the set difference
   * @param superset the index of the PointerObject that shall point to everything subset points to
   * @param subset the index of the PointerObject whose pointees shall all be pointed to by superset as well
   * @return true if P(superset) was modified by this operation
   */
  bool
  MakePointsToSetSuperset(PointerObject::Index superset, PointerObject::Index subset);

  /**
   * Adds the Escaped flag to all PointerObjects in the P(pointer) set
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
          : Superset_(superset), Subset_(subset) {}

  bool
  Apply(PointerObjectSet& set);
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
                                      : Pointer1_(pointer1), Pointer2_(pointer2) {}

  bool
  Apply(PointerObjectSet& set);
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
                                 : Loaded_(loaded), Pointer_(pointer) {}

  bool
  Apply(PointerObjectSet& set);
};

/**
 * A class for adding and applying constraints to the points-to-sets of the PointerObjectSet.
 * Unlike the set modification methods on PointerObjectSet, constraints can be added in any order, with the same result.
 * Use Solve() to calculate the final points-to-sets.
 *
 * Constraints on the special nodes, external and escaped, are built in.
 */
class PointerObjectConstraintSet final
{
public:
  using ConstraintVariant = std::variant<
          SupersetConstraint,
          AllPointeesPointToSupersetConstraint,
          SupersetOfAllPointeesConstraint>;

  explicit PointerObjectConstraintSet(PointerObjectSet& set) : Set_(set) {}

  PointerObjectConstraintSet(const PointerObjectConstraintSet& other) = delete;

  PointerObjectConstraintSet(PointerObjectConstraintSet&& other) = delete;

  PointerObjectConstraintSet&
  operator =(const PointerObjectConstraintSet& other) = delete;

  PointerObjectConstraintSet&
  operator =(PointerObjectConstraintSet&& other) = delete;

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
   * @param registerIndex the register whose content leaves the module, thus exposing any memory it may point to
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
   * Iterates over and applies constraints until the all points-to-sets satisfy them
   */
  void
  Solve();

private:

  /**
   * Makes sure any PointerObject marked as escaped, also makes all its pointees get the HasEscaped flag set.
   * @return true if the function modified any flags
   */
  bool
  PropagateEscapedFlag();

  // The PointerObjectSet being built upon
  PointerObjectSet& Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;
};

} // namespace jlm::llvm::aa

#endif
