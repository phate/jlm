/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/math.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <variant>
#include <queue>

namespace jlm::llvm::aa
{

enum class PointerObjectKind : uint8_t
{
  // Registers can not be pointed to, only point.
  Register = 0,
  AllocaMemoryObject,
  MallocMemoryObject,
  // Represents global memory objects, such as global variables and thread local variables.
  GlobalMemoryObject,
  Function,
  // Represents functions and global variables imported from other modules.
  ImportMemoryObject,

  // Sentinel enum value
  COUNT
};

/**
 * Class representing a single entry in the PointerObjectSet.
 */
class PointerObject final
{
  PointerObjectKind Kind_ : jlm::util::BitWidthOfEnum(PointerObjectKind::COUNT);

  // Instead of adding the special nodes *external* and *escaped*, flags are used.
  // The constraint solver knows their meaning.
  uint8_t PointsToExternal_ : 1;
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
   * @return true if the PointerObject used to not be marked as escaped, before this call
   */
  bool
  MarkAsEscaped() noexcept
  {
    bool modified = !HasEscaped_;
    HasEscaped_ = 1;
    return modified;
  }
};

/**
 * A class containing all PointerObjects, their points-to-sets,
 * as well as mappings from RVSDG nodes/outputs to the PointerObjects.
 */
class PointerObjectSet final
{
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
  AddPointerObject(PointerObjectKind kind)
  {
    PointerObjects_.emplace_back(kind);
    PointsToSets_.emplace_back(); // Add empty points-to-set
    return PointerObjects_.size() - 1;
  }

public:
  PointerObject::Index
  CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput)
  {
    JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
    return RegisterMap_[&rvsdgOutput] = AddPointerObject(PointerObjectKind::Register);
  }

  void
  MapRegisterToExistingPointerObject(const rvsdg::output & rvsdgOutput, PointerObject::Index pointerObject)
  {
    JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
    JLM_ASSERT(pointerObject < NumPointerObjects());
    RegisterMap_[&rvsdgOutput] = pointerObject;
  }

  PointerObject::Index
  CreateAllocaMemoryObject(const rvsdg::node & allocaNode)
  {
    JLM_ASSERT(AllocaMap_.count(&allocaNode) == 0);
    return AllocaMap_[&allocaNode] = AddPointerObject(PointerObjectKind::AllocaMemoryObject);
  }

  PointerObject::Index
  CreateMallocMemoryObject(const rvsdg::node & mallocNode)
  {
    JLM_ASSERT(MallocMap_.count(&mallocNode) == 0);
    return MallocMap_[&mallocNode] = AddPointerObject(PointerObjectKind::MallocMemoryObject);
  }

  PointerObject::Index
  CreateGlobalMemoryObject(const delta::node & deltaNode)
  {
    JLM_ASSERT(GlobalMap_.count(&deltaNode) == 0);
    return GlobalMap_[&deltaNode] = AddPointerObject(PointerObjectKind::GlobalMemoryObject);
  }

  PointerObject::Index
  CreateFunction(const lambda::node & lambdaNode)
  {
    JLM_ASSERT(FunctionMap_.count(&lambdaNode) == 0);
    return FunctionMap_[&lambdaNode] = AddPointerObject(PointerObjectKind::Function);
  }

  PointerObject::Index
  CreateImportMemoryObject(const rvsdg::argument & importNode)
  {
    JLM_ASSERT(ImportMap_.count(&importNode) == 0);
    return ImportMap_[&importNode] = AddPointerObject(PointerObjectKind::ImportMemoryObject);
  }

  PointerObject &
  GetPointerObject(PointerObject::Index index)
  {
    JLM_ASSERT(index <= NumPointerObjects());
    return PointerObjects_[index];
  }

  PointerObject::Index
  NumPointerObjects() const noexcept
  {
    return PointerObjects_.size();
  }

  const std::unordered_set<PointerObject::Index> &
  GetPointsToSet(PointerObject::Index idx)
  {
    return PointsToSets_[idx];
  }

  /**
   * Adds pointee to P(pointer)
   * @param pointer the index of the PointerObject that points
   * @param pointee the index of the PointerObject that is pointed at
   * @return true if P(pointer) was changed by this operation
   */
  bool
  AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee)
  {
    JLM_ASSERT(pointer < NumPointerObjects());
    JLM_ASSERT(pointee < NumPointerObjects());
    JLM_ASSERT(GetPointerObject(pointee).GetKind() != PointerObjectKind::Register);

    auto sizeBefore = PointsToSets_[pointer].size();
    PointsToSets_[pointer].insert(pointee);

    return PointsToSets_[pointer].size() != sizeBefore;
  }

  /**
   * Makes P(superset) a superset of P(subset) by adding any elements in the set difference
   * @param superset the index of the PointerObject that shall point to everything subset points to
   * @param subset the index of the PointerObject that only points to
   * @return true if P(superset) was modified by this operation
   */
  bool
  MakePointsToSetSuperset(PointerObject::Index superset, PointerObject::Index subset)
  {
    JLM_ASSERT(superset <= NumPointerObjects());
    JLM_ASSERT(subset <= NumPointerObjects());

    auto& P_super = PointsToSets_[superset];
    auto& P_sub = PointsToSets_[subset];

    bool modified = false;

    auto sizeBefore = P_super.size();
    P_super.insert(P_sub.begin(), P_sub.end());
    modified |= P_super.size() != sizeBefore;

    // If the external node is in the subset, it must also be part of the superset
    if (GetPointerObject(subset).PointsToExternal())
      modified |= GetPointerObject(superset).MarkAsPointsToExternal();

    return modified;
  }

  /**
   * Adds the Escaped flag to all PointerObjects in the P(pointer) set
   * @param pointer the pointer whose pointees should be marked as escaped
   * @return true if any PointerObjects had their flag modified by this operation
   */
  bool
  MarkAllPointeesAsEscaped(PointerObject::Index pointer)
  {
    bool modified = false;
    for (PointerObject::Index pointee : PointsToSets_[pointer])
      modified |= GetPointerObject(pointee).MarkAsEscaped();
    return modified;
  }

  /**
   * Converts the PointerObjectSet nodes into PointsToGraph nodes,
   * and points-to-graph set memberships into edges.
   *
   * Note that registers sharing PointerObject, become separate PointsToGraph nodes.
   *
   * The *escaped* node is not included.
   * Instead implicit edges through escaped+external, are added as explicit edges.
   * @return
   */
  std::unique_ptr<PointsToGraph>
  ConstructPointsToGraph()
  {
    auto pointsToGraph = PointsToGraph::Create();

    // memory nodes are the nodes that can be pointed to in the points-to graph.
    // This vector has the same indexing as the nodes themselves, register nodes become nullptr.
    std::vector<PointsToGraph::MemoryNode*> memoryNodes(NumPointerObjects());

    // Nodes that should point to external in the final graph.
    // They also get explicit edges connecting them to all escaped memory nodes.
    std::vector<PointsToGraph::Node *> pointsToExternal;

    // A list of all memory nodes that have been marked as escaped
    std::vector<PointsToGraph::MemoryNode*> escapedMemoryNodes;

    // First all memory nodes are created
    for (auto [allocaNode, pointerObjectIndex] : AllocaMap_) {
      auto& node = PointsToGraph::AllocaNode::Create(*pointsToGraph, *allocaNode);
      memoryNodes[pointerObjectIndex] = &node;
    }
    for (auto [mallocNode, pointerObjectIndex] : MallocMap_) {
      auto& node = PointsToGraph::MallocNode::Create(*pointsToGraph, *mallocNode);
      memoryNodes[pointerObjectIndex] = &node;
    }
    for (auto [deltaNode, pointerObjectIndex] : GlobalMap_) {
      auto& node = PointsToGraph::DeltaNode::Create(*pointsToGraph, *deltaNode);
      memoryNodes[pointerObjectIndex] = &node;
    }
    for (auto [lambdaNode, pointerObjectIndex] : FunctionMap_) {
      auto& node = PointsToGraph::LambdaNode::Create(*pointsToGraph, *lambdaNode);
      memoryNodes[pointerObjectIndex] = &node;
    }
    for (auto [argument, pointerObjectIndex] : ImportMap_) {
      auto& node = PointsToGraph::ImportNode::Create(*pointsToGraph, *argument);
      memoryNodes[pointerObjectIndex] = &node;
    }

    auto applyPointsToSet = [&](PointsToGraph::Node & node, PointerObject::Index index)
    {
      // Add all PointsToGraph nodes who should point to external to the list
      if (GetPointerObject(index).PointsToExternal())
        pointsToExternal.push_back(&node);

      for (PointerObject::Index targetIdx : PointsToSets_[index]) {
        // Only add edges to memory nodes
        if (memoryNodes[targetIdx]) {
          node.AddEdge(*memoryNodes[targetIdx]);
        }
      }
    };

    // Now add register nodes last. While adding them, also add any edges from them to the previously created memoryNodes
    for (auto [outputNode, registerIdx] : RegisterMap_) {
      auto &registerNode = PointsToGraph::RegisterNode::Create(*pointsToGraph, *outputNode);
      applyPointsToSet(registerNode, registerIdx);
    }

    // Now add all edges from memory node to memory node. Also tracks which memory nodes are marked as escaped
    for (PointerObject::Index idx = 0; idx < NumPointerObjects(); idx++) {
      if (memoryNodes[idx] == nullptr)
        continue; // Skip all nodes that are not MemoryNodes

      applyPointsToSet(*memoryNodes[idx], idx);

      if (GetPointerObject(idx).HasEscaped())
        escapedMemoryNodes.push_back(memoryNodes[idx]);
    }

    // Finally make all nodes marked as pointing to external, point to all escaped memory nodes in the graph
    for (auto source : pointsToExternal) {
      for (auto target : escapedMemoryNodes) {
        source->AddEdge(*target);
      }
      // Add an edge to the special PtG node called "external" as well
      source->AddEdge(pointsToGraph->GetExternalMemoryNode());
    }

    return pointsToGraph;
  }
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
  AllPointeesPointToSupersetConstraint(
          PointerObject::Index pointer1, PointerObject::Index pointer2)
          : Pointer1_(pointer1), Pointer2_(pointer2) {}

  bool
  Apply(PointerObjectSet& set)
  {
    bool modified = false;

    for (PointerObject::Index x : set.GetPointsToSet(Pointer1_)) {
      modified |= set.MakePointsToSetSuperset(x, Pointer2_);
    }

    // If external in P(Pointer1_), P(external) should become a superset of P(Pointer2)
    // In practice, this means everything in P(Pointer2) escapes
    if (set.GetPointerObject(Pointer1_).PointsToExternal())
      modified |= set.MarkAllPointeesAsEscaped(Pointer2_);

    return modified;
  }
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
  SupersetOfAllPointeesConstraint(
          PointerObject::Index loaded, PointerObject::Index pointer)
          : Loaded_(loaded), Pointer_(pointer) {}

  bool
  Apply(PointerObjectSet& set)
  {
    bool modified = false;

    for (PointerObject::Index x : set.GetPointsToSet(Pointer_)) {
      modified |= set.MakePointsToSetSuperset(Loaded_, x);
    }

    // Handling pointing to external is done by MakePointsToSetSuperset,
    // Propagating escaped status is handled by different constraints

    return modified;
  }
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
  using ConstraintVariant = std::variant<AllPointeesPointToSupersetConstraint, SupersetOfAllPointeesConstraint>;

  explicit PointerObjectConstraintSet(PointerObjectSet& set) : Set_(set) {}

  PointerObjectConstraintSet(const PointerObjectConstraintSet& other) = delete;

  PointerObjectConstraintSet(PointerObjectConstraintSet&& other) = delete;

  PointerObjectConstraintSet&
  operator =(const PointerObjectConstraintSet& other) = delete;

  PointerObjectConstraintSet&
  operator =(PointerObjectConstraintSet&& other) = delete;

  /**
   * The simplest constraint, on the form: pointee in P(pointer)
   */
  void
  AddPointerPointeeConstraint(PointerObject::Index pointer, PointerObject::Index pointee)
  {
    // All set constraints are additive, so simple constraints like this can be directly applied and forgotten.
    Set_.AddToPointsToSet(pointer, pointee);
  }

  void
  AddRegisterContentEscapedConstraint(PointerObject::Index registerIndex)
  {
    // Registers themselves can't really escape, since they don't have an address
    // We can however mark it as escaped, and let escape flag propagation ensure everything it ever points to is marked.
    auto& registerPointerObject = Set_.GetPointerObject(registerIndex);
    JLM_ASSERT(registerPointerObject.GetKind() == PointerObjectKind::Register);
    registerPointerObject.MarkAsEscaped();
  }

  void
  AddConstraint(ConstraintVariant c)
  {
    Constraints_.push_back(c);
  }

  /**
   * Iterates over and applies constraints until the all points-to-sets satisfy them
   */
  void
  Solve() {
    // Keep applying constraints until no sets are modified
    bool modified = true;

    while (modified) {
      modified = false;

      for (auto& constraint : Constraints_)
        std::visit([&](auto constraint) {
          modified |= constraint.Apply(Set_);
        }, constraint);

      modified |= PropagateEscapedFlag();
    }
  }

private:

  bool
  PropagateEscapedFlag()
  {
    bool modified = false;

    std::queue<PointerObject::Index> escapers;

    // First add all already escaped PointerObjects to the queue
    for (PointerObject::Index idx = 0; idx < Set_.NumPointerObjects(); idx++) {
      if (Set_.GetPointerObject(idx).HasEscaped())
        escapers.push(idx);
    }

    // For all escapers, check if they point to any PointerObjects not marked as escaped
    while (!escapers.empty()) {
      PointerObject::Index escaper = escapers.front();
      escapers.pop();

      for (PointerObject::Index pointee : Set_.GetPointsToSet(escaper)) {
        if (Set_.GetPointerObject(pointee).MarkAsEscaped()) {
          // Add the newly marked PointerObject to the queue, in case the flag can be propagated further
          escapers.push(pointee);
          modified = true;
        }
      }
    }

    return modified;
  }

  // The PointerObjectSet being built upon
  PointerObjectSet& Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;
};

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule &module, jlm::util::StatisticsCollector &statisticsCollector)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  AnalyzeRvsdg(module.Rvsdg());

  auto result = Set_->ConstructPointsToGraph();

  Constraints_.reset();
  Set_.reset();

  return result;
}

}