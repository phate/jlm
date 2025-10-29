/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/CommonNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>
#include <map>

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
 * Class representing congruence sets over outputs.
 * Different outputs can be grouped together into congruence sets,
 * where all members of a set are considered identical.
 *
 * One output per set is the leader, and the rest are followers.
 * During analysis of loops, outputs can be speculatively placed in the same congruence set.
 * Later passes can then quickly verify if a outputs does not belong, by comparing against only
 * the congruence set leader.
 */
class CommonNodeElimination::Context
{
public:
  /**
   * Represents a set of outputs where all outputs in the set are considered identical.
   * One output is the leader, and will eventually become the single source of truth.
   *
   * When making congruence sets, the following invariants must be satisfied:
   *  - In a congruence set consisting of region arguments, all arguments belong to the same region,
   *    and the leader is the argument with the lowest index.
   *  - In a congruence set consisting of node outputs, all nodes belong to the same region.
   *    The leader must be the output of the node that comes earliest in the top-down traversal
   *    of this region.
   *
   * Maintaining these invariants ensures that any search for a suitable leader only needs
   * to consider arguments / nodes that have already been visited in the current traversal.
   */
  struct CongruenceSet
  {
    explicit CongruenceSet(const rvsdg::Output & leader)
        : leader(&leader)
    {}

    // The set leader. Never changes.
    // Once an output has become a leader, it will never be a follower.
    const rvsdg::Output * leader;
    // The set of followers. Can both grow and shrink during the marking phase.
    // Does not include the leader.
    util::HashSet<const rvsdg::Output *> followers;
  };

  using CongruenceSetIndex = size_t;
  static constexpr auto NoCongruenceSetIndex = std::numeric_limits<CongruenceSetIndex>::max();

  /**
   * @return the number of congruence sets in the data structure.
   * Any congruence set that already exists when calling this method
   * is guaranteed to have a CongruenceSetIndex strictly less than the returned value.
   */
  [[nodiscard]] CongruenceSetIndex
  numCongruenceSets() const
  {
    return sets_.size();
  }

  /**
   * Gets the index of the congruence set representing the given \p output, if it has one.
   * @param output the output
   * @return the index of the congruence set if the output belongs to one, otherwise \ref
   * NoCongruenceSet.
   */
  [[nodiscard]] CongruenceSetIndex
  tryGetSetFor(const rvsdg::Output & output) const
  {
    if (const auto it = congruenceSetMapping_.find(&output); it != congruenceSetMapping_.end())
    {
      return it->second;
    }
    return NoCongruenceSetIndex;
  }

  /**
   * Gets the index of the congruence set representing the given \p output.
   * @param output the output
   * @return the index of the congruence set the output belongs to
   * @throws std::logic_error if the output does not belong to a congruence set
   */
  [[nodiscard]] CongruenceSetIndex
  getSetFor(const rvsdg::Output & output) const
  {
    const auto index = tryGetSetFor(output);
    if (index == NoCongruenceSetIndex)
      throw std::logic_error("Output does not belong to a congruence set");
    return index;
  }

  /**
   * Checks if the given \p output is associated with a congruence set.
   * @param output the output
   * @return true if the output is associated with a congruence set, false otherwise
   */
  [[nodiscard]] bool
  hasSet(const rvsdg::Output & output) const
  {
    return tryGetSetFor(output) != NoCongruenceSetIndex;
  }

  /**
   * Creates a new congruence set, in which the given \p leader becomes the leader.
   * If \p leader already belongs to a congruence set as a follower, it is removed from that set.
   * If \p leader is already a leader of a set, its index is returned, and this is a no-op.
   * @param leader the output that should lead a congruence set
   * @return the index of the congruence set led by the given leader
   */
  CongruenceSetIndex
  getOrCreateSetForLeader(const rvsdg::Output & leader)
  {
    // The index of the new set, if this operation actually creates one
    auto nextSet = sets_.size();
    auto [it, added] = congruenceSetMapping_.try_emplace(&leader, nextSet);

    if (!added)
    {
      // If the leader already has its own set, we are done
      if (sets_[it->second].leader == &leader)
      {
        return it->second;
      }

      // Remove the output from the congruence set it is following
      sets_[it->second].followers.Remove(&leader);
      it->second = nextSet;
    }

    // Create the new set, lead by \p leader
    sets_.emplace_back(leader);
    return nextSet;
  }

  /**
   * Gets the leader of the congruence set with the given \p index.
   * @param index the index of the congruence set
   * @return the output that is the leader of the set
   */
  const rvsdg::Output &
  getLeader(CongruenceSetIndex index) const
  {
    JLM_ASSERT(index < sets_.size());
    return *sets_[index].leader;
  }

  /**
   * Makes the given \p output a follower of the congruence set with the given \p index.
   * If \p output already follows the congruence set, this is a no-op
   * If \p output already follows a different congruence set, it is moved.
   * \p output can not be the leader of a congruence set.
   * @param index the index of the congruence set to follow
   * @param follower the output that should become a follower
   */
  void
  addFollower(CongruenceSetIndex index, const rvsdg::Output & follower)
  {
    JLM_ASSERT(index < sets_.size());

    const bool newFollower = sets_[index].followers.insert(&follower);

    // If the follower is already following the correct set, do nothing
    if (!newFollower)
      return;

    const auto [it, added] = congruenceSetMapping_.try_emplace(&follower, index);

    // If the follower already belonged to a congruence set, remove it from the old set
    if (!added)
    {
      JLM_ASSERT(it->second != index);

      if (sets_[it->second].leader == &follower)
        throw std::logic_error("Cannot turn a leader into a follower");

      const bool removed = sets_[it->second].followers.Remove(&follower);
      JLM_ASSERT(removed);

      it->second = index;
    }
  }

  /**
   * Gets the set of followers in the congruence set with the given \p index.
   * @param index the index of the congruence set
   * @return the followers
   */
  const util::HashSet<const rvsdg::Output *> &
  getFollowers(CongruenceSetIndex index) const
  {
    JLM_ASSERT(index < sets_.size());
    return sets_[index].followers;
  }

private:
  // The list of congruence sets
  std::vector<CongruenceSet> sets_;
  // A mapping from T to the congruence set it belongs to, either as leader or follower
  std::unordered_map<const rvsdg::Output *, CongruenceSetIndex> congruenceSetMapping_;
};

/**
 * Checks if the given outputs are congruent by using the existing context.
 * The outputs must belong to the same region.
 * Both outputs must already belong to a congruence set.
 * @param o1 the first output
 * @param o2 the second output
 * @param context the common node elimination context
 * @return true if the outputs are considered congruent, false otherwise
 */
static bool
areOutputsCongruent(
    const rvsdg::Output & o1,
    const rvsdg::Output & o2,
    CommonNodeElimination::Context & context)
{
  if (*o1.Type() != *o2.Type())
    return false;

  const auto o1Set = context.getSetFor(o1);
  const auto o2Set = context.getSetFor(o2);

  return o1Set == o2Set;
}

/**
 * Checks if the given nodes appear congruent by comparing their operations
 * and using the context to compare the origins of their inputs.
 * All inputs of both nodes must have origins that already have congruence sets.
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
 * Makes each output of the given node be the leader of its own congruence set.
 * @param leader the node whose outputs should all be leaders.
 * @param context the current context of the mark phase.
 */
void
markNodeAsLeader(const rvsdg::Node & leader, CommonNodeElimination::Context & context)
{
  for (auto & output : leader.Outputs())
  {
    context.getOrCreateSetForLeader(output);
  }
}

/**
 * Marks every output of the \p follower node as a follower
 * of the \p leader node's respective output.
 * The two nodes must have the same number of outputs, and be from the same region.
 * The leader must come before the follower in the TopDown traverser.
 * The leader must already have congruence sets associated with its outputs.
 * @param leader the node whose outputs' congruence sets are used,
 * @param follower the node whose outputs will follow the other congruence sets,
 * @param context the current context of the mark phase.
 */
void
markNodesAsCongruent(
    const rvsdg::Node & leader,
    const rvsdg::Node & follower,
    CommonNodeElimination::Context & context)
{
  JLM_ASSERT(leader.noutputs() == follower.noutputs());
  JLM_ASSERT(leader.region() == follower.region());

  for (size_t i = 0; i < leader.noutputs(); i++)
  {
    const auto & leaderOutput = *leader.output(i);
    const auto & followerOutput = *follower.output(i);
    const auto leaderSet = context.getSetFor(leaderOutput);
    context.addFollower(leaderSet, followerOutput);
  }
}

/**
 * Checks if the given \p node has been visited already during the mark phase.
 * If so, it returns the leader it is congruent with.
 * All outputs of \p node are congruent with the respective output of the leader.
 * @param node the node in question.
 * @param context the current context of the mark phase.
 * @return the leader of this node's congruence sets, or nullptr if this node has not been marked.
 */
[[nodiscard]] const rvsdg::Node *
tryGetLeaderNode(const rvsdg::Node & node, CommonNodeElimination::Context & context)
{
  // Nodes with 0 outputs are never congruent with anything, so let the node be its own leader.
  if (node.noutputs() == 0)
    return &node;

  // Check the congruence set of the first output
  const auto output0Set = context.tryGetSetFor(*node.output(0));
  // If the output has not gotten a congruence set yet, the node has yet to be marked
  if (output0Set == CommonNodeElimination::Context::NoCongruenceSetIndex)
    return nullptr;

  // Structural nodes can have outputs that are congruent with completely different nodes
  // due to invariant values. Structural nodes are always considered their own leaders.
  if (dynamic_cast<const rvsdg::StructuralNode *>(&node))
    return &node;

  // A simple node can only be congruent with other simple nodes
  const auto & output0Leader = context.getLeader(output0Set);
  const auto & leaderNode = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(output0Leader);
  JLM_ASSERT(leaderNode.noutputs() == node.noutputs());
  return &leaderNode;
}

using TopNodeLeaderList = std::vector<const rvsdg::Node *>;

/**
 * Function for marking the given top node \p node, marking it congruent with a previously marked
 * top node if their operations are identical. Otherwise it becomes a leader.
 * Maintains a separate list of leader top nodes in the region.
 * @param node the top node to mark.
 * @param leaders the list of all leader top nodes in the region.
 * @param context the current context of the marking phase.
 */
static void
markSimpleTopNode(
    const rvsdg::SimpleNode & node,
    TopNodeLeaderList & leaders,
    CommonNodeElimination::Context & context)
{
  // If the node has already been marked, check if it is still congruent with its leader
  if (const auto existingLeaderNode = tryGetLeaderNode(node, context))
  {
    // We are our own leader, nothing to check
    if (existingLeaderNode == &node)
    {
      leaders.push_back(&node);
      return;
    }

    // Check if we are still congruent with the leader
    if (checkNodesCongruent(*existingLeaderNode, node, context))
    {
      return;
    }

    // This node is no longer congruent with its leader, continue looking for a new leader
  }

  // TODO: Use some sort of hashing to make this not O(n * m)
  // where n is the number of outputs and m is the number of congruence sets

  // Check if the node is congruent with any existing leader in the TopNodeLeaderList
  for (auto leader : leaders)
  {
    if (checkNodesCongruent(node, *leader, context))
    {
      markNodesAsCongruent(*leader, node, context);
      return;
    }
  }

  // No existing leader top node found, create new congruence sets for each of node's outputs
  markNodeAsLeader(node, context);
  leaders.push_back(&node);
}

/**
 * Places the given \p node in a congruence set, based on its operation and inputs.
 * If the node already has a congruence set, it gets double-checked against the leader.
 * The node must not be a top node. @see markSimpleTopNode.
 *
 * @param node the node to find a congruence set for.
 * @param context the current context of the marking phase
 */
static void
markSimpleNode(const rvsdg::SimpleNode & node, CommonNodeElimination::Context & context)
{
  // This function should never be called for TopNodes
  JLM_ASSERT(node.ninputs() > 0);

  // If node is already in a congruence set, check that it actually belongs there
  if (const auto leaderNode = tryGetLeaderNode(node, context))
  {
    // If node is its own leader, it definitely belongs, and we are done
    if (leaderNode == &node)
      return;

    // Double-check that we are congruent with our leader
    if (checkNodesCongruent(node, *leaderNode, context))
      return;

    // Otherwise we need to continue looking for a new leader
  }

  // This function looks at all nodes that take the given output as the origin for its first input.
  // If the other node is its own leader, and the current node is congruent with it,
  // the current node becomes a follower.
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

      // Do not compare against ourselves
      if (otherNode == &node)
        continue;

      // Only consider other nodes that are leaders
      auto otherNodeLeader = tryGetLeaderNode(*otherNode, context);
      if (otherNode != otherNodeLeader)
        continue;

      if (checkNodesCongruent(node, *otherNode, context))
      {
        // When nodes are congruent, they should always have the same amount of outputs
        JLM_ASSERT(node.noutputs() == otherNode->noutputs());

        markNodesAsCongruent(*otherNode, node, context);
        return true;
      }
    }
    return false;
  };

  // Use the origin of the first input to find other potential candidates
  const auto origin = node.input(0)->origin();
  const auto origin0Set = context.getSetFor(*origin);
  const auto & origin0Leader = context.getLeader(origin0Set);
  const auto & origin0Followers = context.getFollowers(origin0Set);
  if (tryFindCongruentUserOf(origin0Leader))
    return;
  for (auto follower : origin0Followers.Items())
  {
    if (tryFindCongruentUserOf(*follower))
      return;
  }

  // If we got here, no congruent node was found, so make the node its own leader
  markNodeAsLeader(node, context);
}

static void
markRegion(const rvsdg::Region &, CommonNodeElimination::Context & context);

/**
 * Makes all arguments as the leader of their own congruence set.
 * @param region the region whose arguments should be marked.
 * @param context the current context of the marking phase.
 */
static void
markGraphImports(const rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  // FIXME: Multiple imports with identical names could in theory be aliases.
  // For now this function ignores that and makes all imports distinct. This is sound
  for (auto argument : region.Arguments())
  {
    context.getOrCreateSetForLeader(*argument);
  }
}

/**
 * Uses the provided list of partition indices to partition the arguments of the given region.
 * Each argument has a corresponding value in the \p partitions list.
 * The exact partition value is not important, but only arguments with identical
 * partition value are allowed to remain in the same congruence set.
 * If a pair of arguments are already in different congruence sets, they will remain separate.
 *
 * @param region the region whose arguments should be partitioned.
 * @param partitions integers used to partition arguments.
 * Must have length equal to the number of arguments in \p region.
 * @param context the current context of the marking phase.
 * @return true if any congruence sets were created from this operation.
 */
static bool
partitionArguments(
    const rvsdg::Region & region,
    const std::vector<CommonNodeElimination::Context::CongruenceSetIndex> & partitions,
    CommonNodeElimination::Context & context)
{
  JLM_ASSERT(region.narguments() == partitions.size());

  const auto numCongruenceSets = context.numCongruenceSets();

  // Keys in the map are (old congruence set index, provided partition key)
  // Values in the map are the new congruence set indices
  std::map<
      std::pair<
          CommonNodeElimination::Context::CongruenceSetIndex,
          CommonNodeElimination::Context::CongruenceSetIndex>,
      CommonNodeElimination::Context::CongruenceSetIndex>
      newSets;
  for (auto argument : region.Arguments())
  {
    const auto currentPartition = context.tryGetSetFor(*argument);
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

      // Start following our leader
      context.addFollower(toFollow, *argument);
    }
    else
    {
      // This argument should be the leader of its congruence set
      newSets.emplace(key, context.getOrCreateSetForLeader(*argument));
    }
  }

  // Return true iff any new congruence sets were created
  return context.numCongruenceSets() != numCongruenceSets;
}

/**
 * Marks all arguments in all subregions of the given structural node.
 *
 * Arguments that correspond to structural inputs can only be congruent
 * if the structural inputs have congruent origins.
 *
 * This function can be called several times on the same node,
 * and will progressively partition the argument congruence sets.
 * It will never combine arguments from distinct congruence sets back together.
 *
 * Whenever this method causes changes in the congruence sets of a subregion's arguments,
 * it also marks the subregion.
 *
 * @param node the structural node
 * @param context the current context of the marking phase
 * @return true if this operation changed any congruence sets
 */
static bool
markSubregionsFromInputs(
    const rvsdg::StructuralNode & node,
    CommonNodeElimination::Context & context)
{
  bool anyChanges = false;

  for (auto & subregion : node.Subregions())
  {
    // create a partitioning of the region arguments
    std::vector<size_t> partitions(subregion.narguments());
    // Arguments that do not belong to any input are given partition keys higher than any real index
    size_t nextUniquePartitionKey = context.numCongruenceSets();

    for (const auto argument : subregion.Arguments())
    {
      if (const auto input = argument->input())
      {
        // If the argument corresponds to an input, use the partition key of the input
        partitions[argument->index()] = context.getSetFor(*input->origin());
      }
      else
      {
        // Otherwise make sure the argument is not partitioned with any other argument
        partitions[argument->index()] = nextUniquePartitionKey++;
      }
    }

    if (partitionArguments(subregion, partitions, context))
    {
      anyChanges = true;
      markRegion(subregion, context);
    }
  }

  return anyChanges;
}

/**
 * Checks if the given gamma \p exitVar is always a copy of some other origin in the region
 * the gamma is in.
 * This happens if each branch result of the exit var is a copy of an entry variable,
 * and all copied entry variables take their value from congruent origins outside the gamma.
 * If this is the case, the congruence set of that origin is returned.
 * @param exitVar the exit variable in question
 * @param context the current context of the marking phase.
 * @return the congruence set index of the shared origin, or nullopt if no shared origin exists.
 */
static std::optional<CommonNodeElimination::Context::CongruenceSetIndex>
tryGetGammaExitVarCongruenceSet(
    rvsdg::GammaNode::ExitVar & exitVar,
    CommonNodeElimination::Context & context)
{
  std::optional<CommonNodeElimination::Context::CongruenceSetIndex> sharedCongruenceSet;

  for (auto result : exitVar.branchResult)
  {
    if (const auto argument = dynamic_cast<rvsdg::RegionArgument *>(result->origin()))
    {
      const auto inputCongruenceSet = context.getSetFor(*argument->input()->origin());
      if (!sharedCongruenceSet.has_value())
      {
        sharedCongruenceSet = inputCongruenceSet;
      }
      else if (*sharedCongruenceSet != inputCongruenceSet)
      {
        // We have multiple different non-congruent origins
        return std::nullopt;
      }
    }
    else
    {
      // The branch result was not invariant
      return std::nullopt;
    }
  }

  return sharedCongruenceSet;
}

/**
 * Marks the arguments of the gamma subregions, and the nodes within the subregions.
 * Uses the origins of the branch results of exit variables to assign congruence sets to
 * each exit variable's output.
 * @param gamma the gamma node to mark
 * @param context the current context of the marking phase.
 */
static void
markGamma(const rvsdg::GammaNode & gamma, CommonNodeElimination::Context & context)
{
  markSubregionsFromInputs(gamma, context);

  if (tryGetLeaderNode(gamma, context))
    return;

  // Go through the outputs of the gamma node and create congruence sets for them.
  // By default, each exit variable output gets a distinct congruence set.
  // Only if an exit variable is a copy of an entry variable in each region,
  // and all the entry variables have congruent origins, does the gamma output become a follower.
  for (auto exitVar : gamma.GetExitVars())
  {
    if (const auto entryVarCongruenceSet = tryGetGammaExitVarCongruenceSet(exitVar, context);
        entryVarCongruenceSet.has_value())
    {
      context.addFollower(*entryVarCongruenceSet, *exitVar.output);
    }
    else
    {
      context.getOrCreateSetForLeader(*exitVar.output);
    }
  }
}

/**
 * Marks the loop variables and subregion of the given \p theta node.
 * The loop variables are initially partitioned based on the origins of their inputs.
 * After marking the subregion, the partitioning of the loop variable post results
 * are used to further partition loop variables. The subregion is re-marked until the
 * loop variable partitioning reaches a fixed point.
 * The outputs of the theta node are partitioned based on the loop variables,
 * except for loop invariants, which are made congruent with their origin.
 * @param theta the theta node to mark.
 * @param context the current context of the mark phase.
 */
static void
markTheta(const rvsdg::ThetaNode & theta, CommonNodeElimination::Context & context)
{
  bool anyChanges = markSubregionsFromInputs(theta, context);
  if (!anyChanges)
    return;

  const auto loopVars = theta.GetLoopVars();

  // Use the loop variable post results to refine partitioning of loop variable arguments
  while (anyChanges)
  {
    // Create partition keys for each loop variable
    std::vector<CommonNodeElimination::Context::CongruenceSetIndex> partitions;
    for (const auto & loopVar : loopVars)
    {
      partitions.push_back(context.getSetFor(*loopVar.post->origin()));
    }

    anyChanges = partitionArguments(*theta.subregion(), partitions, context);
    if (anyChanges)
    {
      // Propagate refinement of argument congruence sets into the region
      markRegion(*theta.subregion(), context);
    }
  }

  // Partition theta outputs
  std::unordered_map<
      CommonNodeElimination::Context::CongruenceSetIndex,
      CommonNodeElimination::Context::CongruenceSetIndex>
      resultToOutputSetMapping;
  for (auto & loopVar : loopVars)
  {
    // invariant loop variables become followers of the input origin
    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
    {
      const auto inputOriginSet = context.getSetFor(*loopVar.input->origin());
      context.addFollower(inputOriginSet, *loopVar.output);
      continue;
    }

    // Other loop variable outputs are partitioned based on the origins of the post results.
    const auto resultSet = context.getSetFor(*loopVar.post->origin());
    const auto it = resultToOutputSetMapping.find(resultSet);
    if (it != resultToOutputSetMapping.end())
    {
      context.addFollower(it->second, *loopVar.output);
    }
    else
    {
      const auto outputSet = context.getOrCreateSetForLeader(*loopVar.output);
      resultToOutputSetMapping.emplace(resultSet, outputSet);
    }
  }
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
        // Context variables are congruent if their origins are congruent.
        // All other arguments are given distinct congruence sets.
        markSubregionsFromInputs(lambda, context);

        // A lambda output is always unique
        markNodeAsLeader(lambda, context);
      },
      [&](const rvsdg::PhiNode & phi)
      {
        // Context variables are congruent if their origins are congruent.
        // Recursion variables are given distinct congruence sets.
        markSubregionsFromInputs(phi, context);

        // A phi node is always unique
        markNodeAsLeader(phi, context);
      },
      [&](const rvsdg::DeltaNode & delta)
      {
        // We skip doing CNE inside delta nodes

        // A delta node is always unique
        markNodeAsLeader(delta, context);
      });
}

/**
 * Traverses every node in the region and places their outputs in congruence sets.
 * Also recurses into the subregions of structural nodes.
 * Expects the arguments of \p region to already belong to congruence sets.
 * @param region the region to perform marking in.
 * @param context the current context of the marking phase.
 */
static void
markRegion(const rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  TopNodeLeaderList leaders;

  for (const auto node : rvsdg::TopDownConstTraverser(&region))
  {
    rvsdg::MatchTypeOrFail(
        *node,
        [&](const rvsdg::SimpleNode & simple)
        {
          // Handle top nodes as a special case
          if (node->ninputs() == 0)
          {
            markSimpleTopNode(simple, leaders, context);
          }
          else
          {
            markSimpleNode(simple, context);
          }
        },
        [&](const rvsdg::StructuralNode & structural)
        {
          markStructuralNode(structural, context);
        });
  }
}

/* divert phase */

static void
divertOutput(rvsdg::Output & output, CommonNodeElimination::Context & context)
{
  const auto outputSet = context.getSetFor(output);

  auto & leader = context.getLeader(outputSet);
  if (&leader == &output)
    return;

  output.divert_users(const_cast<rvsdg::Output *>(&leader));
}

static void
divertInRegion(rvsdg::Region &, CommonNodeElimination::Context &);

static void
divertInStructuralNode(rvsdg::StructuralNode & node, CommonNodeElimination::Context & context)
{
  bool divertInSubregions = false;
  rvsdg::MatchTypeOrFail(
      node,
      [&]([[maybe_unused]] rvsdg::GammaNode & gamma)
      {
        divertInSubregions = true;
      },
      [&]([[maybe_unused]] rvsdg::ThetaNode & theta)
      {
        divertInSubregions = true;
      },
      [&]([[maybe_unused]] rvsdg::LambdaNode & lambda)
      {
        divertInSubregions = true;
      },
      [&]([[maybe_unused]] rvsdg::PhiNode & phi)
      {
        divertInSubregions = true;
      },
      [&]([[maybe_unused]] rvsdg::DeltaNode & delta)
      {
        // Inside a delta node we can not perform diverting,
        // since we never marked the outputs there
      });

  if (divertInSubregions)
  {
    for (auto & subregion : node.Subregions())
    {
      divertInRegion(subregion, context);
    }
  }
}

static void
divertInRegion(rvsdg::Region & region, CommonNodeElimination::Context & context)
{
  // First divert all region arguments
  for (auto argument : region.Arguments())
  {
    divertOutput(*argument, context);
  }

  // Divert all nodes
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    // When diverting structural nodes, also recurse into their subregions
    rvsdg::MatchType(
        *node,
        [&](rvsdg::StructuralNode & structural)
        {
          divertInStructuralNode(structural, context);
        });

    // Divert all outputs of the node
    for (auto & output : node->Outputs())
    {
      divertOutput(output, context);
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
  divertInRegion(rootRegion, context);
  statistics->endDivertStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}
}
