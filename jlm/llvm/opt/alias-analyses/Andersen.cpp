/*
 * Copyright 2023
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/math.hpp>
#include <vector>
#include <set>
#include <unordered_map>
#include <variant>

namespace jlm::llvm::aa
{

enum class PointerObjectKind : uint8_t
{
  /*
   * Registers represent values that are calculated once per region execution.
   * They are the outputs of nodes, or the region's arguments.
   * Only registers of pointer type get a PointerObject created for them.
   * Registers can not be pointed to, only point.
   */
  Register = 0,
  AllocaMemoryObject,
  MallocMemoryObject,

  /*
   * Represents global memory objects, such as global variables and thread local variables.
   * Global variables often, but not always, escape the module through exported symbols.
   */
  GlobalMemoryObject,

  /*
   * Functions often, but not always, escape the module through exported symbols.
   */
  Function,

  /*
   * Represents functions and global variables imported from other modules.
   * Are always marked as escaping the current module.
   */
  ImportMemoryObject,

  // Sentinel enum value
  COUNT
};
// The number of bits required to hold the integer representation of any PointerObjectKind value.
static constexpr int BITS_NEEDED_FOR_KIND = jlm::util::BitWidthOfEnum(PointerObjectKind::COUNT);

class PointerObjectSet;
class PointerObjectConstraintSet;

/**
 * Class representing a single entry in the PointerObjectSet.
 */
class PointerObject final
{
  friend PointerObjectSet;
private:
  uint8_t Kind_ : BITS_NEEDED_FOR_KIND;

  // Instead of adding the special nodes *external* and *escaped*, flags are used.
  // The constraint solver knows their meaning.
  uint8_t PointsToExternal_ : 1;
  uint8_t HasEscaped_ : 1;

  explicit PointerObject(PointerObjectKind kind) : Kind_(static_cast<std::underlying_type_t<PointerObjectKind>>(kind)),
                                          PointsToExternal_(0), HasEscaped_(0) {
    JLM_ASSERT(kind != PointerObjectKind::COUNT);

    // Memory objects from other modules are definitely not private to this module
    if (kind == PointerObjectKind::ImportMemoryObject)
      MarkAsEscaped();
  }

public:
  PointerObjectKind
  GetKind()
  {
    return static_cast<PointerObjectKind>(Kind_);
  }

  bool
  PointsToExternal()
  {
    return PointsToExternal_;
  }

  void
  PointToExternal()
  {
    PointsToExternal_ = 1;
  }

  bool
  HasEscaped()
  {
    return HasEscaped_;
  }

  void
  MarkAsEscaped()
  {
    HasEscaped_ = 1;
  }
};

// We use indices, this allows up to ~4 billion pointer objects
using PointerObjectIndex = uint32_t;

/**
 * A class containing all PointerObjects, their points-to-sets,
 * as well as mappings from RVSDG nodes/outputs to the PointerObjects.
 */
class PointerObjectSet final
{
  // All nodes in the graph
  std::vector<PointerObject> Nodes_;

  // For each node, a set of the other nodes it points to
  std::vector<std::set<PointerObjectIndex>> PointsToSets_;

  // Mapping from register to PointerObject
  // Unlike the other maps, several rvsdg::output* can share PointerObject
  std::unordered_map<jlm::rvsdg::output *, PointerObjectIndex> RegisterMap_;
  // Mapping from alloca node to PointerObject
  std::unordered_map<jlm::rvsdg::node *, PointerObjectIndex> AllocaMap_;
  // Mapping from malloc call node to PointerObject
  std::unordered_map<jlm::rvsdg::node *, PointerObjectIndex> MallocMap_;
  // Mapping from global variables declared with delta nodes to PointerObject
  std::unordered_map<jlm::llvm::delta::node *, PointerObjectIndex> GlobalMap_;
  // Mapping from functions declared with lambda nodes to PointerObject
  std::unordered_map<jlm::llvm::lambda::node *, PointerObjectIndex> FunctionMap_;
  // Mapping from symbols imported into the module to PointerObject
  std::unordered_map<jlm::rvsdg::argument *, PointerObjectIndex> ImportMap_;

  /**
   * Internal helper function for adding nodes, use the Create* methods instead
   */
  PointerObjectIndex
  AddNode(PointerObjectKind kind)
  {
    Nodes_.push_back(PointerObject(kind));
    PointsToSets_.emplace_back(); // Add empty points-to-set
    return Nodes_.size() - 1;
  }

public:
  PointerObject &
  GetNode(PointerObjectIndex index)
  {
    JLM_ASSERT(index <= NodeCount());
    return Nodes_[index];
  }

  PointerObjectIndex
  NodeCount()
  {
    return Nodes_.size();
  }

  const std::set<PointerObjectIndex> &
  GetPointsToSet(PointerObjectIndex idx)
  {
    return PointsToSets_[idx];
  }

  /**
   * Adds pointee to P(pointer)
   * @return true if P(pointer) was changed by this operation
   */
  bool
  AddToPointsToSet(PointerObjectIndex pointer, PointerObjectIndex pointee)
  {
    JLM_ASSERT(pointer < NodeCount());
    JLM_ASSERT(pointee < NodeCount());
    JLM_ASSERT(GetNode(pointee).GetKind() != PointerObjectKind::Register);

    auto sizeBefore = PointsToSets_[pointer].size();
    PointsToSets_[pointer].insert(pointee);

    return PointsToSets_[pointer].size() != sizeBefore;
  }

  /**
   * Makes P(superset) a superset of P(subset) by adding any elements in the set difference
   * @return true if P(superset) was changed by this
   */
  bool
  MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset)
  {
    auto& P_super = PointsToSets_[superset];
    auto& P_sub = PointsToSets_[subset];

    auto sizeBefore = P_super.size();
    P_super.insert(P_sub.begin(), P_sub.end());

    return P_super.size() != sizeBefore;
  }

  PointerObjectIndex
  CreateRegisterNode(jlm::rvsdg::output *rvsdgOutput)
  {
    JLM_ASSERT(RegisterMap_.count(rvsdgOutput) == 0);
    return RegisterMap_[rvsdgOutput] = AddNode(PointerObjectKind::Register);
  }

  void
  MapRegisterToExistingNode(jlm::rvsdg::output *rvsdgOutput, PointerObjectIndex node)
  {
    JLM_ASSERT(RegisterMap_.count(rvsdgOutput) == 0);
    JLM_ASSERT(node < NodeCount());
    RegisterMap_[rvsdgOutput] = node;
  }

  PointerObjectIndex
  CreateAllocaMemoryObject(jlm::rvsdg::node *allocaNode)
  {
    JLM_ASSERT(AllocaMap_.count(allocaNode) == 0);
    return AllocaMap_[allocaNode] = AddNode(PointerObjectKind::AllocaMemoryObject);
  }

  PointerObjectIndex
  CreateMallocMemoryObject(jlm::rvsdg::node *mallocNode)
  {
    JLM_ASSERT(MallocMap_.count(mallocNode) == 0);
    return MallocMap_[mallocNode] = AddNode(PointerObjectKind::MallocMemoryObject);
  }

  PointerObjectIndex
  CreateGlobalMemoryObject(jlm::llvm::delta::node *deltaNode)
  {
    JLM_ASSERT(GlobalMap_.count(deltaNode) == 0);
    return GlobalMap_[deltaNode] = AddNode(PointerObjectKind::GlobalMemoryObject);
  }

  PointerObjectIndex
  CreateFunction(jlm::llvm::lambda::node *lambdaNode)
  {
    JLM_ASSERT(FunctionMap_.count(lambdaNode) == 0);
    return FunctionMap_[lambdaNode] = AddNode(PointerObjectKind::Function);
  }

  PointerObjectIndex
  CreateImportMemoryObject(jlm::rvsdg::argument *importNode)
  {
    JLM_ASSERT(ImportMap_.count(importNode) == 0);
    return ImportMap_[importNode] = AddNode(PointerObjectKind::ImportMemoryObject);
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

    // memory nodes are the nodes that can be pointed to in the PtG.
    // This vector has the same indexing as the nodes themselves, register nodes become nullptr.
    std::vector<PointsToGraph::MemoryNode*> memoryNodes(NodeCount());

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

    auto applyPointsToSet = [&](PointsToGraph::Node* node, PointerObjectIndex index) {
      // Add all PointsToGraph nodes who should point to external to the list
      if (GetNode(index).PointsToExternal())
        pointsToExternal.push_back(node);

      for (PointerObjectIndex targetIdx : PointsToSets_[index]) {
        // Only add edges to memory nodes
        if (memoryNodes[targetIdx]) {
          node->AddEdge(*memoryNodes[targetIdx]);
        }
      }
    };

    // Now add register nodes last. This is done due to the fact that several registers can share PointerObject.
    // While adding the register nodes, also add any edges from them to memoryNodes
    for (auto [outputNode, registerIdx] : RegisterMap_) {
      auto &registerNode = PointsToGraph::RegisterNode::Create(*pointsToGraph, *outputNode);
      applyPointsToSet(&registerNode, registerIdx);
    }

    // Now add all edges from memory node to memory node
    // Also add all escaped memory nodes to
    for (PointerObjectIndex idx = 0; idx < NodeCount(); idx++) {
      if (memoryNodes[idx] == nullptr)
        continue; // Skip all nodes that are not MemoryNodes

      applyPointsToSet(memoryNodes[idx], idx);

      if (GetNode(idx).HasEscaped())
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
class AllPointeesPointToSupersetConstraint final {
  PointerObjectIndex Pointer1_;
  PointerObjectIndex Pointer2_;

public:
  AllPointeesPointToSupersetConstraint(PointerObjectIndex pointer1, PointerObjectIndex pointer2)
          : Pointer1_(pointer1), Pointer2_(pointer2) {}

  bool
  Apply(PointerObjectSet& set)
  {
    bool modified = false;

    for (PointerObjectIndex x : set.GetPointsToSet(Pointer1_)) {
      modified |= set.MakePointsToSetSuperset(x, Pointer2_);
    }

    // TODO: external and escaped flag stuff

    return modified;
  }
};

/**
 * A constraint of the form:
 * P(loaded) = union of P(x) for all x in P(pointer)
 * Example of application is a load, e.g. when loaded = *pointer
 */
class SupersetOfAllPointeesConstraint final {
  PointerObjectIndex Loaded_;
  PointerObjectIndex Pointer_;

public:
  SupersetOfAllPointeesConstraint(PointerObjectIndex loaded, PointerObjectIndex pointer)
          : Loaded_(loaded), Pointer_(pointer) {}

  bool
  Apply(PointerObjectSet& set)
  {
    bool modified = false;

    for (PointerObjectIndex x : set.GetPointsToSet(Pointer_)) {
      modified |= set.MakePointsToSetSuperset(Loaded_, x);
    }

    // TODO: external and escaped flag stuff

    return modified;
  }
};

using ConstraintVariant = std::variant<AllPointeesPointToSupersetConstraint, SupersetOfAllPointeesConstraint>;

/**
 * A class for adding and applying constraints to the points-to-sets of the PointerObjectSet
 * Some constraints are applied immediately, while others are kept in lists for later constraint solving.
 * Use Solve() to calculate the final points-to-sets.
 *
 * Constraints on the special external and escaped nodes are built in.
 */
class PointerObjectConstraintSet final
{
public:
  explicit PointerObjectConstraintSet(PointerObjectSet& set) : Set_(set) {}

  PointerObjectConstraintSet(const PointerObjectConstraintSet& other) = delete;

  PointerObjectConstraintSet&
  operator =(const PointerObjectConstraintSet& other) = delete;

  /**
   * The simplest constraint, on the form: pointee in P(pointer)
   */
  void
  AddPointerPointeeConstraint(PointerObjectIndex pointer, PointerObjectIndex pointee)
  {
    JLM_ASSERT(pointer <= Set_.NodeCount());
    JLM_ASSERT(pointee <= Set_.NodeCount());

    // All set constraints are additive, so simple constraints like this can be directly applied and forgotten.
    Set_.AddToPointsToSet(pointer, pointee);
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
  Solve();

private:

  /**
   * Internal function used to implement the constraints on flags
   */
  bool
  ApplyEscapedFlagPropagation();

  // The PointerObjectSet being built upon
  PointerObjectSet& Set_;

  // Lists of all constraints, of all different types
  std::vector<ConstraintVariant> Constraints_;
};

void
PointerObjectConstraintSet::Solve()
{
  // Keep applying constraints until no sets are modified
  bool modified = true;
  while (modified) {
    modified = false;

    for (auto& constraint : Constraints_)
      std::visit([&](auto constraint) {
        modified |= constraint.Apply(Set_);
      }, constraint);

    modified |= ApplyEscapedFlagPropagation();
  }
}

bool
PointerObjectConstraintSet::ApplyEscapedFlagPropagation()
{
  // This function only applies one iteration of escape flag propagation, so it might need to be run again
  bool modified = false;

  for (PointerObjectIndex idx = 0; idx < Set_.NodeCount(); idx++)
  {
    if (!Set_.GetNode(idx).HasEscaped())
      continue;

    for (PointerObjectIndex pointee : Set_.GetPointsToSet(idx)) {
      auto& pointeeNode = Set_.GetNode(pointee);
      modified |= !pointeeNode.HasEscaped();
      pointeeNode.MarkAsEscaped();
    }
  }

  return modified;
}

std::unique_ptr<PointsToGraph>
Andersen::Analyze(const RvsdgModule &module, jlm::util::StatisticsCollector &statisticsCollector)
{
  Set_ = std::make_unique<PointerObjectSet>();
  Constraints_ = std::make_unique<PointerObjectConstraintSet>(*Set_);

  AnalyzeRvsdg(module.Rvsdg());

  return Set_->ConstructPointsToGraph();
}

}