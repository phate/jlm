/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>
#include <map>

#include <typeindex>

namespace jlm::llvm
{

class CommonNodeElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * DivertTimerLabel_ = "DivertTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::CommonNodeElimination, sourceFile)
  {}

  void
  startMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  endMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  startDivertStatistics() noexcept
  {
    AddTimer(DivertTimerLabel_).start();
  }

  void
  endDivertStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(DivertTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

/**
 * Class representing congruence sets over elements of the given type.
 * Different instances of \p T can be grouped together into congruence sets,
 * where all members of a set are considered identical.
 *
 * One element per set is the leader, and the rest are followers.
 * During analysis of loops, Ts can be speculatively placed in the same congruence set.
 * Later passes can then quickly verify if a T does not belong, by comparing against only
 * the congruence set leader.
 *
 * @tparam T the element type of the congruence sets
 */
template<typename T>
class CongruenceSets
{
public:
  /**
   * Represents a set of T where all Ts in the set are considered identical.
   * All the Ts must be in the same region.
   * One T is the leader, and will eventually become the single source of truth.
   */
  struct CongruenceSet
  {
    // The set leader. Never changes.
    // Once a T has become a leader, it will never be a follower.
    // A leader always comes earlier than its followers in the traverser.
    const T * leader;
    // The set of followers. Can both grow and shrink during the marking phase.
    // Does not include the leader.
    util::HashSet<const T *> followers;
  };

  /** \brief Type safe index type for referencing congruence sets
   */
  class Index
  {
    size_t value_;

    explicit constexpr Index(size_t value)
        : value_{ value }
    {}
    friend class CongruenceSets;

  public:
    [[nodiscard]] bool
    operator==(const Index & other) const noexcept
    {
      return value_ == other.value_;
    }

    [[nodiscard]] bool
    operator!=(const Index & other) const noexcept
    {
      return !(*this == other);
    }

    [[nodiscard]] bool
    operator<(const Index & other) const noexcept
    {
      return value_ < other.value_;
    }
  };

  static constexpr auto NoCongruenceSet = Index(std::numeric_limits<size_t>::max());

  Index
  tryGetSetFor(const T & element)
  {
    if (const auto it = congruenceSetMapping_.find(&element); it != congruenceSetMapping_.end())
    {
      return it->second;
    }
    return NoCongruenceSet;
  }

  bool
  hasSet(const T & element)
  {
    return congruenceSetMapping_.find(&element) != congruenceSetMapping_.end();
  }

  Index
  makeSetForLeader(const T & leader)
  {
    auto nextSet = Index(sets_.size());
    sets_.emplace_back(CongruenceSet{ &leader, {} });
    const auto [_, added] = congruenceSetMapping_.emplace(&leader, nextSet);
    JLM_ASSERT(added);
    return nextSet;
  }

  /**
   * Gets the leader of the congruence set with the given \p index.
   * @param index the index of the congruence set
   * @return the output that leads the set
   */
  const T &
  getLeader(Index index) const
  {
    JLM_ASSERT(index.value_ < sets_.size());
    return *sets_[index.value_].leader;
  }

  void
  addFollower(Index index, const T & follower)
  {
    JLM_ASSERT(index.value_ < sets_.size());
    sets_[index.value_].followers.insert(&follower);
    const auto [_, added] = congruenceSetMapping_.emplace(&follower, index);
    JLM_ASSERT(added);
  }

  /**
   * Removes the given \p follower from its congruence set.
   * The \p follower must not be a leader.
   * @param follower the element to remove
   */
  void
  removeFollower(const T & follower)
  {
    if (const auto it = congruenceSetMapping_.find(&follower); it != congruenceSetMapping_.end())
    {
      const auto index = it->second;
      const auto removed = sets_[index.value_].followers.Remove(&follower);
      JLM_ASSERT(removed);
      congruenceSetMapping_.erase(it);
    }
  }

  /**
   * Gets the set of followers in the congruence set with the given \p index.
   * @param index the index of the congruence set
   * @return the followers
   */
  const util::HashSet<const T *> &
  getFollowers(Index index) const
  {
    JLM_ASSERT(index.value_ < sets_.size());
    return sets_[index.value_].followers;
  }

private:
  // The list of congruence sets
  std::vector<CongruenceSet> sets_;
  // A mapping from T to the congruence set it belongs to, either as leader or follower
  std::unordered_map<const T *, Index> congruenceSetMapping_;
};

using ArgumentCongruenceSets = CongruenceSets<rvsdg::RegionArgument>;
using NodeCongruenceSets = CongruenceSets<rvsdg::Node>;

class CommonNodeElimination::Context final
{

public:
  [[nodiscard]] ArgumentCongruenceSets &
  getArgumentCongruenceSets()
  {
    return argumentCongruenceSets_;
  }

  [[nodiscard]] NodeCongruenceSets &
  getNodeCongruenceSets()
  {
    return nodeCongruenceSets_;
  }

private:
  ArgumentCongruenceSets argumentCongruenceSets_;
  NodeCongruenceSets nodeCongruenceSets_;
};

/**
 * Checks if the given outputs are congruent by using the existing context.
 * The outputs must belong to the same region.
 * Both outputs must already belong to either an ArgumentCongruenceSet or NodeCongruenceSet.
 * @param o1 the first output
 * @param o2 the second output
 * @param context the common node elimination context
 * @return true if the outputs are considered congruent
 */
static bool
areOutputsCongruent(
    const rvsdg::Output & o1,
    const rvsdg::Output & o2,
    CommonNodeElimination::Context & context)
{
  if (*o1.Type() != *o2.Type())
    return false;

  // Check if both o1 and o2 originate from nodes
  const auto o1Node = rvsdg::TryGetOwnerNode<rvsdg::Node>(o1);
  const auto o2Node = rvsdg::TryGetOwnerNode<rvsdg::Node>(o2);
  if (o1Node && o2Node)
  {
    // If o1 and o2 originate from nodes, they must have the same output index to be congruent
    if (o1.index() != o2.index())
      return false;

    auto o1Set = context.getNodeCongruenceSets().tryGetSetFor(*o1Node);
    auto o2Set = context.getNodeCongruenceSets().tryGetSetFor(*o2Node);
    JLM_ASSERT(o1Set != NodeCongruenceSets::NoCongruenceSet);
    JLM_ASSERT(o2Set != NodeCongruenceSets::NoCongruenceSet);
    return o1Set == o2Set;
  }

  // If only one of the outputs is a node output, there is no congruence
  if (o1Node || o2Node)
    return false;

  // If we get here, both o1 and o2 should be region arguments
  const auto o1Argument = dynamic_cast<const rvsdg::RegionArgument *>(&o1);
  const auto o2Argument = dynamic_cast<const rvsdg::RegionArgument *>(&o2);
  if (o1Argument && o2Argument)
  {
    auto o1Set = context.getArgumentCongruenceSets().tryGetSetFor(*o1Argument);
    auto o2Set = context.getArgumentCongruenceSets().tryGetSetFor(*o2Argument);
    JLM_ASSERT(o1Set != ArgumentCongruenceSets::NoCongruenceSet);
    JLM_ASSERT(o2Set != ArgumentCongruenceSets::NoCongruenceSet);
    return o1Set == o2Set;
  }

  throw std::logic_error("Unknown type of output");
}

/**
 * Checks if the given nodes appear congruent by comparing their operations
 * and using the context to compare the origins of their inputs.
 * All inputs of both nodes must have origins that already have congruence sets.
 *
 * If the nodes already belong to a congruence set, it is ignored, as it might be wrong.
 *
 * This function can only detect congruence between simple nodes.
 *
 * @param node1 the first node
 * @param node2 the second node
 * @param context the current context of the mark phase
 * @return true if the nodes appear congruent, otherwise false
 */
static bool
checkNodesCongruent(
    const rvsdg::Node & node1,
    const rvsdg::Node & node2,
    CommonNodeElimination::Context & context)
{
  const auto simpleNode1 = dynamic_cast<const rvsdg::SimpleNode *>(&node1);
  const auto simpleNode2 = dynamic_cast<const rvsdg::SimpleNode *>(&node2);
  if (!simpleNode1 || !simpleNode2)
    return false;

  if (simpleNode1->ninputs() != simpleNode2->ninputs())
    return false;

  if (simpleNode1->GetOperation() != simpleNode2->GetOperation())
    return false;

  // For each pair of corresponding inputs of simpleNode1 and simpleNode2,
  // they must have origins that are congruent
  for (auto & input : simpleNode1->Inputs())
  {
    const auto origin1 = input.origin();
    const auto origin2 = simpleNode2->input(input.index())->origin();
    if (!areOutputsCongruent(*origin1, *origin2, context))
      return false;
  }

  return true;
}

/**
 * Uses the provided list of partition indices to partition the arguments of the given region.
 * Each argument has a corresponding integer in the \p partitions list.
 * Only arguments with identical integers can stay in the same partition.
 * If a pair of arguments are already in different congruence sets, they will remain separate.
 *
 * @param region the region whose arguments should be partitioned
 * @param partitions integers used to partition arguments. Must have length equal to narguments
 * @param context the current context of the marking phase
 * @return true if any congruence sets were created from this operation
 */
static bool
partitionArguments(
    const rvsdg::Region & region,
    const std::vector<size_t> & partitions,
    CommonNodeElimination::Context & context)
{
  auto & argumentCongruenceSets = context.getArgumentCongruenceSets();

  JLM_ASSERT(region.narguments() == partitions.size());

  bool anyChanged = false;

  // Keys in the map are old argument congruence sets, and provided partition key
  // Values in the map are the new congruence sets
  std::map<std::pair<ArgumentCongruenceSets::Index, size_t>, ArgumentCongruenceSets::Index> newSets;
  for (auto argument : region.Arguments())
  {
    const auto currentPartition = argumentCongruenceSets.tryGetSetFor(*argument);
    const auto key = std::make_pair(currentPartition, partitions[argument->index()]);

    // If this argument is the first with the given key, it should be a leader
    // otherwise it should be a follower

    if (const auto it = newSets.find(key); it != newSets.end())
    {
      // This argument should be a follower of the given congruence set
      const auto toFollow = it->second;

      // If we are already a follower, we are done
      if (currentPartition == toFollow)
        continue;

      // If we follow someone else, remove that
      if (currentPartition != ArgumentCongruenceSets::NoCongruenceSet)
        argumentCongruenceSets.removeFollower(*argument);

      // Start following our leader
      argumentCongruenceSets.addFollower(toFollow, *argument);
      anyChanged = true;
    }
    else
    {
      // This argument should be the leader of its congruence set

      // Check if the argument already belongs to a congruence set
      if (currentPartition != ArgumentCongruenceSets::NoCongruenceSet)
      {
        // If the argument is already the leader of its set, continue
        if (&argumentCongruenceSets.getLeader(currentPartition) == argument)
        {
          newSets.emplace(key, currentPartition);
          continue;
        }

        // otherwise we must remove the argument from its current (wrong) congruence set
        argumentCongruenceSets.removeFollower(*argument);
      }

      const auto newSet = argumentCongruenceSets.makeSetForLeader(*argument);
      anyChanged = true;
      newSets.emplace(key, newSet);
    }
  }

  return anyChanged;
}

/**
 * Divides a set of inputs into partitions by assigning partition keys to each input.
 * Two inputs have identical partition keys iff their origins are congruent outputs.
 *
 * @tparam InputGetter a functor for getting inputs (size_t) -> const rvsdg::Input &
 * @param numInputs the number of inputs
 * @param inputGetter an instantiation of the input getter functor
 * @param context the current context of the marking phase
 * @return partition keys for each input
 */
template<typename InputGetter>
static std::vector<size_t>
createPartitionKeys(
    size_t numInputs,
    const InputGetter & inputGetter,
    CommonNodeElimination::Context & context)
{
  // For each partition of congruent inputs, the partition key is the index of the leftmost member
  std::vector<const rvsdg::Input *> partitionLeaders;
  std::vector<size_t> partitionKeys;

  // TODO: We could use unordered_maps here to avoid comparing against all partitionLeaders
  for (size_t i = 0; i < numInputs; i++)
  {
    const rvsdg::Input & input = inputGetter(i);

    bool foundMatch = false;
    for (size_t j = 0; j < partitionLeaders.size(); j++)
    {
      auto & leader = *partitionLeaders[j];
      if (areOutputsCongruent(*input.origin(), *leader.origin(), context))
      {
        partitionKeys.push_back(j);
        foundMatch = true;
        break;
      }
    }

    if (!foundMatch)
    {
      partitionLeaders.push_back(&input);
      partitionKeys.push_back(partitionLeaders.size() - 1);
    }
  }

  return partitionKeys;
}

/**
 * Marks all arguments in all regions of the given structural node.
 *
 * Arguments that correspond to structural inputs are made congruent
 * if the structural inputs have congruent origins.
 *
 * This function can be called several times on the same node,
 * and will progressively partition the argument congruence sets.
 * It will never combine arguments from distinct congruence sets back together.
 *
 * @param node the structural node
 * @param context the current context of the marking phase
 * @return a vector of the subregions whose argument congruence sets were modified
 */
static std::vector<const rvsdg::Region *>
markArgumentsFromInputs(
    const rvsdg::StructuralNode & node,
    CommonNodeElimination::Context & context)
{
  const auto inputPartitionKeys = createPartitionKeys(
      node.ninputs(),
      [&](size_t i) -> const rvsdg::Input &
      {
        return *node.input(i);
      },
      context);

  std::vector<const rvsdg::Region *> changedRegions;

  for (auto & subregion : node.Subregions())
  {
    // create a partitioning of the region arguments
    std::vector<size_t> partitions(subregion.narguments());
    // Arguments that do not belong to any input are given partition keys higher than any input
    size_t nextUniquePartitionKey = inputPartitionKeys.size();

    for (auto & argument : subregion.Arguments())
    {
      if (argument->input())
      {
        // If the argument corresponds to an input, use the partition key of the input
        partitions[argument->index()] = inputPartitionKeys[argument->input()->index()];
      }
      else
      {
        // Otherwise make sure the argument is not partitioned with any other argument
        partitions[argument->index()] = nextUniquePartitionKey++;
      }
    }

    if (partitionArguments(subregion, partitions, context))
      changedRegions.push_back(&subregion);
  }

  return changedRegions;
}

/**
 * Makes all arguments as the leader of their own congruence set.
 * @param region the region whose arguments are
 * @param context the current context of the marking phase
 */
static void
markGraphImports(const rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  auto & argumentCongruenceSets = context.getArgumentCongruenceSets();

  for (auto argument : region.Arguments())
  {
    argumentCongruenceSets.makeSetForLeader(*argument);
  }
}

static void
markRegion(const rvsdg::Region &, CommonNodeElimination::Context & context);

static void
markGamma(const rvsdg::GammaNode & gamma, CommonNodeElimination::Context & context)
{
  const auto changedRegions = markArgumentsFromInputs(gamma, context);
  // Only visit subregions whose argument congruence sets have changed
  for (auto changedRegion : changedRegions)
    markRegion(*changedRegion, context);

  // Gamma nodes are never considered common to other gamma nodes. Make sure it has a set it leads
  if (!context.getNodeCongruenceSets().hasSet(gamma))
    context.getNodeCongruenceSets().makeSetForLeader(gamma);
}

static void
markTheta(const rvsdg::ThetaNode & theta, CommonNodeElimination::Context & context)
{
  auto changedRegions = markArgumentsFromInputs(theta, context);
  bool changed = !changedRegions.empty();
  while (changed)
  {
    // Propagate changes in argument congruence sets through the theta region
    markRegion(*theta.subregion(), context);

    // Use the theta region results to further partition region argument congruence sets, if needed
    const auto loopVariables = theta.GetLoopVars();
    const auto partitionKeys = createPartitionKeys(
        loopVariables.size(),
        [&](size_t i) -> const rvsdg::Input &
        {
          return *loopVariables[i].post;
        },
        context);
    changed = partitionArguments(*theta.subregion(), partitionKeys, context);
  }

  // Theta nodes are never considered common to other theta nodes. Make sure it has a set it leads
  if (!context.getNodeCongruenceSets().hasSet(theta))
    context.getNodeCongruenceSets().makeSetForLeader(theta);
}

static void
markLambda(const rvsdg::LambdaNode & lambda, CommonNodeElimination::Context & context)
{
  const auto changedRegions = markArgumentsFromInputs(lambda, context);
  // Only visit subregions whose argument congruence sets have changed
  for (auto changedRegion : changedRegions)
    markRegion(*changedRegion, context);

  // Lambda nodes are never considered common to other nodes. Make sure it has a set it leads
  if (!context.getNodeCongruenceSets().hasSet(lambda))
    context.getNodeCongruenceSets().makeSetForLeader(lambda);
}

static void
markPhi(const rvsdg::PhiNode & phi, CommonNodeElimination::Context & context)
{
  const auto changedRegions = markArgumentsFromInputs(phi, context);
  // Only visit subregions whose argument congruence sets have changed
  for (auto changedRegion : changedRegions)
    markRegion(*changedRegion, context);

  // Phi nodes are never considered common to other nodes. Make sure it has a set it leads
  if (!context.getNodeCongruenceSets().hasSet(phi))
    context.getNodeCongruenceSets().makeSetForLeader(phi);
}

using TopNodeLeaderList = std::vector<std::pair<const rvsdg::Node *, NodeCongruenceSets::Index>>;

static void
markTopNode(
    const rvsdg::Node & node,
    TopNodeLeaderList & leaders,
    CommonNodeElimination::Context & context)
{
  // TODO: Use some sort of hashing to make this not O(n * m)
  // where n is the number of outputs and m is the number of congruence sets

  // Check if the node is congruent with any existing leader in the TopNodeLeaderList
  for (auto [leader, set] : leaders)
  {
    if (checkNodesCongruent(node, *leader, context))
    {
      context.getNodeCongruenceSets().addFollower(set, node);
      return;
    }
  }

  // No existing congruence set leader found, create a new set with node as the leader
  auto newSet = context.getNodeCongruenceSets().makeSetForLeader(node);
  leaders.emplace_back(&node, newSet);
}

/**
 * Places the given \p node in a congruence set, based on its operation and inputs.
 * If the node already has a congruence set, it gets double-checked against the leader.
 * The node must not be a top node. @see markTopNode.
 *
 * @param node the node to find a congruence set for.
 * @param context the current context of the marking phase
 */
static void
markSimpleNode(const rvsdg::SimpleNode & node, CommonNodeElimination::Context & context)
{
  auto & nodeCongruenceSets = context.getNodeCongruenceSets();

  // If node is already in a congruence set, check that it actually belongs there
  const auto setIndex = nodeCongruenceSets.tryGetSetFor(node);
  if (setIndex != NodeCongruenceSets::NoCongruenceSet)
  {
    // If node is its own leader, it definitely belongs, and we are done
    const auto & leader = nodeCongruenceSets.getLeader(setIndex);
    if (&node == &leader)
      return;

    // Double-check that we are congruent with our leader
    if (checkNodesCongruent(node, leader, context))
      return;

    // It turns out that node didn't belong to this congruence set after all,
    // remove it and try to find a new set
    nodeCongruenceSets.removeFollower(node);
  }

  // This function should never be called for TopNodes
  JLM_ASSERT(node.ninputs() > 0);

  // Use the origin of the first input to find other potential candidates
  const auto origin = node.input(0)->origin();

  // This function looks at all nodes who has the given output as the origin for its first input.
  // If the other node is its own leader, and the current node is congruent, it becomes a follower.
  const auto tryFindCongruentUserOf = [&](const rvsdg::Output & output) -> bool
  {
    // TODO: It would be possible to maintain a list of only users that are leader nodes,
    // to avoid needing to check every user
    for (auto & user : output.Users())
    {
      if (user.index() != 0)
        continue;

      const auto otherNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user);
      if (!otherNode)
        continue;

      if (otherNode == &node)
        continue;

      auto otherNodeSet = nodeCongruenceSets.tryGetSetFor(*otherNode);
      if (otherNodeSet == NodeCongruenceSets::NoCongruenceSet)
        continue;

      if (&nodeCongruenceSets.getLeader(otherNodeSet) != otherNode)
        continue;

      if (checkNodesCongruent(node, *otherNode, context))
      {
        // When nodes are congruent, they should always have the same amount of outputs
        JLM_ASSERT(node.noutputs() == otherNode->noutputs());

        nodeCongruenceSets.addFollower(otherNodeSet, node);
        return true;
      }
    }
    return false;
  };

  // Check the two cases: either the origin is a node output, or a region argument
  if (const auto originNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(*origin); originNode)
  {
    auto originNodeSet = nodeCongruenceSets.tryGetSetFor(*originNode);
    JLM_ASSERT(originNodeSet != NodeCongruenceSets::NoCongruenceSet);

    const auto originIndex = origin->index();

    // go through all nodes congruent with the originNode, and look at the output with originIndex.
    if (tryFindCongruentUserOf(*nodeCongruenceSets.getLeader(originNodeSet).output(originIndex)))
      return;
    for (auto follower : nodeCongruenceSets.getFollowers(originNodeSet).Items())
    {
      if (tryFindCongruentUserOf(*follower->output(originIndex)))
        return;
    }
  }
  else if (auto argument = dynamic_cast<const rvsdg::RegionArgument *>(origin))
  {
    auto & argumentCongruenceSets = context.getArgumentCongruenceSets();
    auto argumentSet = argumentCongruenceSets.tryGetSetFor(*argument);
    JLM_ASSERT(argumentSet != ArgumentCongruenceSets::NoCongruenceSet);

    // go through all users of congruent region arguments to find candiates
    if (tryFindCongruentUserOf(argumentCongruenceSets.getLeader(argumentSet)))
      return;
    for (auto follower : argumentCongruenceSets.getFollowers(argumentSet).Items())
    {
      if (tryFindCongruentUserOf(*follower))
        return;
    }
  }
  else
  {
    throw std::logic_error("Unknown output type");
  }

  // If we got here, no congruence set was found, so become our own leaders
  nodeCongruenceSets.makeSetForLeader(node);
}

static void
markStructuralNode(const rvsdg::StructuralNode & node, CommonNodeElimination::Context & context)
{
  rvsdg::MatchTypeOrFail(
      node,
      [&](const rvsdg::GammaNode & gamma)
      {
        markGamma(gamma, context);
      },
      [&](const rvsdg::ThetaNode & theta)
      {
        markTheta(theta, context);
      },
      [&](const rvsdg::LambdaNode & lambda)
      {
        markLambda(lambda, context);
      },
      [&](const rvsdg::PhiNode & phi)
      {
        markPhi(phi, context);
      },
      [&](const rvsdg::DeltaNode & delta)
      {
        // Nothing to do.
      });
}

/**
 * Marks every argument and node in the given region, and recursively marks subregions.
 * Expects region arguments to already belong to argument congruence sets.
 * @param region the region to perform marking in
 * @param context the current marking context
 */
static void
markRegion(const rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  TopNodeLeaderList leaders;

  for (const auto & node : rvsdg::TopDownConstTraverser(&region))
  {
    // Handle top nodes as a special case
    if (node->ninputs() == 0)
    {
      markTopNode(*node, leaders, context);
    }
    else if (auto simple = dynamic_cast<const rvsdg::SimpleNode *>(node))
    {
      markSimpleNode(*simple, context);
    }
    else if (auto structural = dynamic_cast<const rvsdg::StructuralNode *>(node))
    {
      markStructuralNode(*structural, context);
    }
    else
      throw std::logic_error("Unknown node type");
  }
}

/* divert phase */

static void
divertArguments(rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  auto & argumentCongruenceSets = context.getArgumentCongruenceSets();

  for (auto argument : region.Arguments())
  {
    const auto argumentSet = argumentCongruenceSets.tryGetSetFor(*argument);
    JLM_ASSERT(argumentSet != ArgumentCongruenceSets::NoCongruenceSet);

    auto & leader = argumentCongruenceSets.getLeader(argumentSet);
    if (&leader == argument)
      continue;

    argument->divert_users(const_cast<rvsdg::RegionArgument *>(&leader));
  }
}

static void
divertSimpleNode(rvsdg::Node & node, CommonNodeElimination::Context & context)
{
  auto & nodeCongruenceSets = context.getNodeCongruenceSets();

  const auto nodeSet = nodeCongruenceSets.tryGetSetFor(node);
  JLM_ASSERT(nodeSet != NodeCongruenceSets::NoCongruenceSet);

  // If this node is the leader of its set, do nothing
  auto & leader = nodeCongruenceSets.getLeader(nodeSet);
  if (&leader == &node)
    return;

  // Otherwise redirect all our outputs to our leader
  for (auto & output : node.Outputs())
  {
    output.divert_users(leader.output(output.index()));
  }
}

static void
divertInRegion(rvsdg::Region &, CommonNodeElimination::Context &);

static void
divertGamma(rvsdg::GammaNode & gamma, CommonNodeElimination::Context & context)
{
  for (auto & subregion : gamma.Subregions())
  {
    divertInRegion(subregion, context);
  }
}

static void
divertTheta(rvsdg::ThetaNode & theta, CommonNodeElimination::Context & ctx)
{
  divertInRegion(*theta.subregion(), ctx);
}

static void
divertLambda(rvsdg::LambdaNode & lambda, CommonNodeElimination::Context & ctx)
{
  divertInRegion(*lambda.subregion(), ctx);
}

static void
divertPhi(rvsdg::PhiNode & phi, CommonNodeElimination::Context & ctx)
{
  divertInRegion(*phi.subregion(), ctx);
}

static void
divertStructuralNode(rvsdg::StructuralNode & node, CommonNodeElimination::Context & ctx)
{
  rvsdg::MatchTypeOrFail(
      node,
      [&](rvsdg::GammaNode & gamma)
      {
        divertGamma(gamma, ctx);
      },
      [&](rvsdg::ThetaNode & theta)
      {
        divertTheta(theta, ctx);
      },
      [&](rvsdg::LambdaNode & lambda)
      {
        divertLambda(lambda, ctx);
      },
      [&](rvsdg::PhiNode & phi)
      {
        divertPhi(phi, ctx);
      },
      [&](rvsdg::DeltaNode & delta)
      {
        // Nothing to do.
      });
}

static void
divertInRegion(rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  divertArguments(region, context);

  for (const auto & node : rvsdg::TopDownTraverser(&region))
  {
    if (auto simple = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      divertSimpleNode(*simple, context);
    }
    else if (auto structural = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      divertStructuralNode(*structural, context);
    }
    else
    {
      throw std::logic_error("Unknown node type");
    }
  }
}

CommonNodeElimination::~CommonNodeElimination() noexcept = default;

void
CommonNodeElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  const auto & rvsdg = module.Rvsdg();
  auto & rootRegion = rvsdg.GetRootRegion();

  Context context;
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->startMarkStatistics(rvsdg);
  markGraphImports(rootRegion, context);
  markRegion(rootRegion, context);
  statistics->endMarkStatistics();

  statistics->startDivertStatistics();
  divertInRegion(rvsdg.GetRootRegion(), context);
  statistics->endDivertStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}
