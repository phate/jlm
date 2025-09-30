/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REGION_HPP
#define JLM_RVSDG_REGION_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>

namespace jlm::util
{
class Annotation;
class AnnotationMap;
}

namespace jlm::rvsdg
{

class Node;
class SimpleNode;
class SimpleOperation;
class StructuralInput;
class StructuralNode;
class StructuralOutput;
class SubstitutionMap;
class RegionObserver;

/**
 * \brief Represents the argument of a region.
 *
 * Region arguments represent the initial values of the region's acyclic graph. These values
 * are mapped to the arguments throughout the execution, and the concrete semantics of this mapping
 * depends on the structural node the region is part of. A region argument is either linked
 * with a \ref StructuralInput or is a standalone argument.
 */
class RegionArgument : public Output
{
  util::IntrusiveListAnchor<RegionArgument> structural_input_anchor_{};

public:
  typedef util::IntrusiveListAccessor<RegionArgument, &RegionArgument::structural_input_anchor_>
      structural_input_accessor;

  ~RegionArgument() noexcept override;

  RegionArgument(
      rvsdg::Region * region,
      StructuralInput * input,
      std::shared_ptr<const rvsdg::Type> type);

  RegionArgument(const RegionArgument &) = delete;

  RegionArgument(RegionArgument &&) = delete;

  RegionArgument &
  operator=(const RegionArgument &) = delete;

  RegionArgument &
  operator=(RegionArgument &&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] StructuralInput *
  input() const noexcept
  {
    return input_;
  }

  /**
   * Creates a copy of the argument in \p region with the structural_input \p input.
   *
   * @param region The region where the copy of the argument is created in.
   * @param input  The structural_input to the argument, if any.
   *
   * @return A reference to the copied argument.
   */
  virtual RegionArgument &
  Copy(Region & region, StructuralInput * input);

  /**
   * \brief Creates region entry argument.
   *
   * \param region
   *   Region to create argument for.
   *
   * \param input
   *   (optional) input of parent node associated with this
   *   argument (deprecated, will be removed soon).
   *
   * \param type
   *   Result type.
   *
   * \returns
   *   Reference to the created argument.
   *
   * Creates an argument and registers it with the given region.
   */
  static RegionArgument &
  Create(rvsdg::Region & region, StructuralInput * input, std::shared_ptr<const rvsdg::Type> type);

private:
  StructuralInput * input_;
};

/**
 * \brief Represents the result of a region.
 *
 * Region results represent the final values of the region's acyclic graph. The result values
 * can be mapped back to the region arguments or the corresponding structural outputs
 * throughout the execution, but the concrete semantics of this mapping
 * depends on the structural node the region is part of. A region result is either linked
 * with a \ref StructuralOutput or is a standalone result.
 */
class RegionResult : public Input
{
  util::IntrusiveListAnchor<RegionResult> structural_output_anchor_{};

public:
  typedef util::IntrusiveListAccessor<RegionResult, &RegionResult::structural_output_anchor_>
      structural_output_accessor;

  ~RegionResult() noexcept override;

  RegionResult(
      rvsdg::Region * region,
      rvsdg::Output * origin,
      StructuralOutput * output,
      std::shared_ptr<const rvsdg::Type> type);

  RegionResult(const RegionResult &) = delete;

  RegionResult(RegionResult &&) = delete;

  RegionResult &
  operator=(const RegionResult &) = delete;

  RegionResult &
  operator=(RegionResult &&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] StructuralOutput *
  output() const noexcept
  {
    return output_;
  }

  /**
   * Creates a copy of the result with \p origin and structural_output \p output. The
   * result is created with the same type as \p origin and in the same region as \p origin.
   *
   * @param origin The origin for the result.
   * @param output The structural_output to the result, if any.
   *
   * @return A reference to the copied result.
   */
  virtual RegionResult &
  Copy(rvsdg::Output & origin, StructuralOutput * output);

  /**
   * \brief Create region exit result.
   *
   * \param region
   *   Region to create result for.
   *
   * \param origin
   *   Assigned result value.
   *
   * \param output
   *   (optional) output of parent node associated with this
   *   result (deprecated, will be removed soon).
   *
   * \param type
   *   Result type
   *
   * \returns
   *   Reference to the created result.
   *
   * Creates a result and registers it with the given region.
   */
  static RegionResult &
  Create(
      rvsdg::Region & region,
      rvsdg::Output & origin,
      StructuralOutput * output,
      std::shared_ptr<const rvsdg::Type> type);

private:
  StructuralOutput * output_;
};

/**
 * \brief Represent acyclic RVSDG subgraphs
 *
 * Regions represent acyclic RVSDG subgraphs and are instantiated with an index in \ref
 * StructuralNode%s. Each region has \ref RegionArgument%s and \ref RegionResult%s that represent
 * the values at the beginning and end of the acyclic graph, respectively. In addition, each region
 * keeps track of the following properties:
 *
 * 1. The nodes of the acyclic subgraph. They represent the computations performed in the region.
 * 2. The top nodes of the acyclic subgraph. These are all nodes of the region that have no inputs,
 * i.e., constants.
 * 3. The bottom nodes of the acyclic subgraph. These are all nodes of the region that have no
 * users, i.e. that are dead. See \ref Output::IsDead() for more information.
 */
class Region
{
  using RegionArgumentIterator = std::vector<RegionArgument *>::iterator;
  using RegionArgumentConstIterator = std::vector<RegionArgument *>::const_iterator;
  using RegionArgumentRange = util::IteratorRange<RegionArgumentIterator>;
  using RegionArgumentConstRange = util::IteratorRange<RegionArgumentConstIterator>;

  using RegionResultIterator = std::vector<RegionResult *>::iterator;
  using RegionResultConstIterator = std::vector<RegionResult *>::const_iterator;
  using RegionResultRange = util::IteratorRange<RegionResultIterator>;
  using RegionResultConstRange = util::IteratorRange<RegionResultConstIterator>;

  using region_nodes_list = util::IntrusiveList<Node, Node::region_node_list_accessor>;
  using region_top_node_list = util::IntrusiveList<Node, Node::region_top_node_list_accessor>;
  using region_bottom_node_list = util::IntrusiveList<Node, Node::region_bottom_node_list_accessor>;

  using TopNodeIterator = region_top_node_list::Iterator;
  using TopNodeConstIterator = region_top_node_list::ConstIterator;
  using TopNodeRange = util::IteratorRange<TopNodeIterator>;
  using TopNodeConstRange = util::IteratorRange<TopNodeConstIterator>;

  using NodeIterator = region_nodes_list::Iterator;
  using NodeConstIterator = region_nodes_list::ConstIterator;
  using NodeRange = util::IteratorRange<NodeIterator>;
  using NodeConstRange = util::IteratorRange<NodeConstIterator>;

  using BottomNodeIterator = region_bottom_node_list::Iterator;
  using BottomNodeConstIterator = region_bottom_node_list::ConstIterator;
  using BottomNodeRange = util::IteratorRange<BottomNodeIterator>;
  using BottomNodeConstRange = util::IteratorRange<BottomNodeConstIterator>;

public:
  ~Region() noexcept;

  Region(rvsdg::Region * parent, Graph * graph);

  Region(rvsdg::StructuralNode * node, size_t index);

  Region(const Region &) = delete;

  Region &
  operator=(const Region &) = delete;

  /**
   * @return Returns an iterator range for iterating through the arguments of the region.
   */
  [[nodiscard]] RegionArgumentRange
  Arguments() noexcept
  {
    return { arguments_.begin(), arguments_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the arguments of the region.
   */
  [[nodiscard]] RegionArgumentConstRange
  Arguments() const noexcept
  {
    return { arguments_.begin(), arguments_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the results of the region.
   */
  [[nodiscard]] RegionResultRange
  Results() noexcept
  {
    return { results_.begin(), results_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the results of the region.
   */
  [[nodiscard]] RegionResultConstRange
  Results() const noexcept
  {
    return { results_.begin(), results_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the top nodes of the region.
   * The top nodes are all nodes that do not have any inputs.
   */
  [[nodiscard]] TopNodeRange
  TopNodes() noexcept
  {
    return { topNodes_.begin(), topNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the top nodes of the region.
   * The top nodes are all nodes that do not have any inputs.
   */
  [[nodiscard]] TopNodeConstRange
  TopNodes() const noexcept
  {
    return { topNodes_.begin(), topNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeRange
  Nodes() noexcept
  {
    return { nodes_.begin(), nodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeConstRange
  Nodes() const noexcept
  {
    return { nodes_.begin(), nodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the bottom nodes of the region.
   * The bottom nodes are all nodes with only unused outputs, aka. dead nodes.
   */
  [[nodiscard]] BottomNodeRange
  BottomNodes() noexcept
  {
    return { bottomNodes_.begin(), bottomNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the bottom nodes of the region.
   * The bottom nodes are all nodes with only unused outputs, aka. dead nodes.
   */
  [[nodiscard]] BottomNodeConstRange
  BottomNodes() const noexcept
  {
    return { bottomNodes_.begin(), bottomNodes_.end() };
  }

  [[nodiscard]] Graph *
  graph() const noexcept
  {
    return graph_;
  }

  inline rvsdg::StructuralNode *
  node() const noexcept
  {
    return node_;
  }

  size_t
  index() const noexcept
  {
    return index_;
  }

  /**
   * Checks if the region is the RVSDG root region.
   *
   * @return Returns true if it is the root region, otherwise false.
   */
  [[nodiscard]] bool
  IsRootRegion() const noexcept;

  /**
   * Appends the given region \p argument to the list of region arguments.
   * Invalidates any existing iterators to the argument list.
   *
   * @param argument the argument to add
   * @return a reference to the added argument
   */
  RegionArgument &
  addArgument(std::unique_ptr<RegionArgument> argument);

  /**
   * Inserts the given region \p argument in the list of region arguments, at the given \p index.
   * Shifts any results with equal or greater index one to the right to make room.
   * Invalidates any existing iterators to the argument list.
   *
   * @param index the position to add the result
   * @param argument the argument to add
   * @return a reference to the added result
   * @see addArgument to add to the end
   */
  RegionArgument &
  insertArgument(size_t index, std::unique_ptr<RegionArgument> argument);

  /**
   * Removes an argument from the region given an arguments' index.
   *
   * An argument can only be removed, if it has no users. The removal of an argument invalidates the
   * region's existing argument iterators.
   *
   * @param index The arguments' index. It must be between [0, narguments()].
   *
   * \note The method must adjust the indices of the other arguments after the removal. The methods'
   * runtime is therefore O(n), where n is the region's number of arguments.
   *
   * \see narguments()
   * \see RegionArgument#index()
   * \see RegionArgument::nusers()
   */
  void
  RemoveArgument(size_t index);

  /**
   * Removes all arguments that have no users and match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const argument&)
   * @param match Defines the condition for the arguments to remove.
   */
  template<typename F>
  void
  RemoveArgumentsWhere(const F & match)
  {
    // iterate backwards to avoid the invalidation of 'n' by RemoveArgument()
    for (size_t n = narguments() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & argument = *this->argument(n);
      if (argument.nusers() == 0 && match(argument))
      {
        RemoveArgument(n);
      }
    }
  }

  inline size_t
  narguments() const noexcept
  {
    return arguments_.size();
  }

  inline RegionArgument *
  argument(size_t index) const noexcept
  {
    JLM_ASSERT(index < narguments());
    return arguments_[index];
  }

  /**
   * Appends the given region result to the list of region results.
   * Invalidates any existing iterators to the result list.
   *
   * @param result the result to add
   * @return a reference to the added result
   */
  RegionResult &
  addResult(std::unique_ptr<RegionResult> result);

  /**
   * Removes a result from the region given a results' index.
   *
   * The removal of a result invalidates the region's existing result iterators.
   *
   * @param index The results' index. It must be between [0, nresults()).
   *
   * \note The method must adjust the indices of the other results after the removal. The methods'
   * runtime is therefore O(n), where n is the region's number of results.
   *
   * \see nresults()
   * \see RegionResult#index()
   */
  void
  RemoveResult(size_t index);

  /**
   * Remove all results that match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const RegionResult&)
   * @param match Defines the condition for the results to remove.
   */
  template<typename F>
  void
  RemoveResultsWhere(const F & match)
  {
    // iterate backwards to avoid the invalidation of 'n' by RemoveResult()
    for (size_t n = nresults() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & result = *this->result(n);
      if (match(result))
      {
        RemoveResult(n);
      }
    }
  }

  /**
   * Remove all arguments that have no users.
   */
  void
  PruneArguments()
  {
    auto match = [](const RegionArgument &)
    {
      return true;
    };

    RemoveArgumentsWhere(match);
  }

  inline size_t
  nresults() const noexcept
  {
    return results_.size();
  }

  [[nodiscard]] RegionResult *
  result(size_t index) const noexcept
  {
    JLM_ASSERT(index < nresults());
    return results_[index];
  }

  /**
   * @return The number of nodes in the region.
   */
  [[nodiscard]] size_t
  numNodes() const noexcept
  {
    return numNodes_;
  }

  /**
   * @return The number of top nodes in the region.
   */
  [[nodiscard]] size_t
  numTopNodes() const noexcept
  {
    return numTopNodes_;
  }

  /**
   * @return The number of bottom nodes in the region.
   */
  [[nodiscard]] size_t
  numBottomNodes() const noexcept
  {
    return numBottomNodes_;
  }

  /**
   * Deletes the given node from the region.
   * The node must belong to this region, and be dead.
   * @param node the node to remove
   */
  void
  removeNode(Node * node);

  /**
    \brief Copy a region with substitutions
    \param target Target region to create nodes in
    \param smap Operand substitutions
    \param copy_arguments Copy region arguments
    \param copy_results Copy region results

    Copies all nodes of the specified region and its
    subregions into the target region. Substitutions
    will be performed as specified, and the substitution
    map will be updated as nodes are copied.
  */
  void
  copy(Region * target, SubstitutionMap & smap, bool copy_arguments, bool copy_results) const;

  /**
   * Removes all dead nodes from the region, including nodes that become dead during pruning.
   * @param recursive if true, any subregions are also pruned, as well as their subregions etc.
   */
  void
  prune(bool recursive);

  /**
   * @return the Node:Id that will be used for the next node created in the region.
   */
  [[nodiscard]] Node::Id
  getNextNodeId() const noexcept
  {
    return nextNodeId_;
  }

private:
  /**
   * \brief Adds \p node to the top nodes of the region.
   * @param node The node that is now a top node.
   * @see TopNodes
   *
   * @note This method is automatically invoked when a top node is created.
   */
  void
  onTopNodeAdded(Node & node);

  /**
   * Removes \p node from the list of top nodes in the region.
   * @param node The node that is no longer a top node.
   *
   * @note This method is automatically invoked when inputs are added to a node.
   */
  void
  onTopNodeRemoved(Node & node);

  /**
   * \brief Adds \p node to the set of bottom nodes in the region.
   * @param node The node that is now a bottom node.
   * @see BottomNodes
   *
   * @note This method is automatically invoked when a node is created or becomes dead.
   */
  void
  onBottomNodeAdded(Node & node);

  /**
   * Removes \p node from the list of bottom nodes in the region.
   * @param node The node that is no longer a bottom node.
   *
   * @note This method is automatically invoked when a node ceases to be dead.
   */
  void
  onBottomNodeRemoved(Node & node);

  /**
   * \brief Adds \p node to the list of nodes in the region.
   * @param node The node that has been created in the region.
   *
   * @note This method is automatically invoked when a node is created.
   */
  void
  onNodeAdded(Node & node);

  /**
   * Remove \p node from the region.
   * @param node The node that is removed.
   *
   * @note This method is automatically invoked when a node is deleted.
   */
  void
  onNodeRemoved(Node & node);

  /**
   * @return A unique identifier for a node within this region.
   *
   * @note This method is automatically invoked when a node is created.
   * The identifier is only unique within this region.
   */
  [[nodiscard]] Node::Id
  generateNodeId() noexcept
  {
    const auto nodeId = nextNodeId_;
    nextNodeId_++;
    return nodeId;
  }

  void
  notifyNodeCreate(Node * node);

  void
  notifyNodeDestroy(Node * node);

  void
  notifyInputCreate(Input * input);

  void
  notifyInputChange(Input * input, Output * old_origin, Output * new_origin);

  void
  notifyInputDestory(Input * input);

public:
  /**
   * Checks if an operation is contained within the given \p region. If \p checkSubregions is true,
   * then the subregions of all contained structural nodes are recursively checked as well.
   * @tparam Operation The operation to check for.
   * @param region The region to check.
   * @param checkSubregions If true, then the subregions of all contained structural nodes will be
   * checked as well.
   * @return True, if the operation is found. Otherwise, false.
   */
  template<class Operation>
  static inline bool
  ContainsOperation(const rvsdg::Region & region, bool checkSubregions);

  /**
   * Checks if a node type is contained within the given \p region. If \p checkSubregions is true,
   * then the subregions of all contained structural nodes are recursively checked as well.
   * @tparam Operation The operation to check for.
   * @param region The region to check.
   * @param checkSubregions If true, then the subregions of all contained structural nodes will be
   * checked as well.
   * @return True, if the operation is found. Otherwise, false.
   */
  template<class NodeType>
  static inline bool
  ContainsNodeType(const rvsdg::Region & region, bool checkSubregions);

  /**
   * Counts the number of (sub-)regions contained within \p region. The count includes \p region,
   * i.e., if \p region does not contain any structural nodes and therefore no subregions, then the
   * count is one.
   *
   * @param region The region for which to count the contained (sub-)regions.
   * @return The number of (sub-)regions.
   */
  [[nodiscard]] static size_t
  NumRegions(const rvsdg::Region & region) noexcept;

  /**
   * Converts \p region and all of its contained structural nodes with subregions to a tree in
   * ASCII format of the following form:
   *
   * RootRegion                              \n
   * -STRUCTURAL_TEST_NODE                   \n
   * --Region[0]                             \n
   * --Region[1]                             \n
   * ---STRUCTURAL_TEST_NODE                 \n
   * ----Region[0]                           \n
   * ----Region[1]                           \n
   * ----Region[2] NumNodes:0 NumArguments:0 \n
   *
   *
   * The above tree has a single structural node in the RVSDG's root region. This node has two
   * subregions, where the second subregion contains another structural node with three subregions.
   * For the third subregion, two annotations with label NumNodes and NumArguments was provided in
   * \p annotationMap.
   *
   * @param region The top-level region that is converted
   * @param annotationMap A map with annotations for instances of \ref Region%s or
   * StructuralNode%s.
   * @return A string containing the ASCII tree of \p region.
   */
  [[nodiscard]] static std::string
  ToTree(const rvsdg::Region & region, const util::AnnotationMap & annotationMap) noexcept;

  /**
   * Converts \p region and all of its contained structural nodes with subregions to a tree in
   * ASCII format of the following form:
   *
   * RootRegion              \n
   * -STRUCTURAL_TEST_NODE   \n
   * --Region[0]             \n
   * --Region[1]             \n
   * ---STRUCTURAL_TEST_NODE \n
   * ----Region[0]           \n
   * ----Region[1]           \n
   * ----Region[2]           \n
   *
   *
   * The above tree has a single structural node in the RVSDG's root region. This node has two
   * subregions, where the second subregion contains another structural node with three subregions.
   *
   * @param region The top-level region that is converted
   * @return A string containing the ASCII tree of \p region
   */
  [[nodiscard]] static std::string
  ToTree(const rvsdg::Region & region) noexcept;

  static std::unordered_map<const Node *, size_t>
  computeDepthMap(const Region & region);

private:
  static void
  ToTree(
      const rvsdg::Region & region,
      const util::AnnotationMap & annotationMap,
      size_t indentationDepth,
      std::stringstream & stream) noexcept;

  [[nodiscard]] static std::string
  GetAnnotationString(
      const void * key,
      const util::AnnotationMap & annotationMap,
      char annotationSeparator,
      char labelValueSeparator);

  [[nodiscard]] static std::string
  ToString(
      const std::vector<util::Annotation> & annotations,
      char annotationSeparator,
      char labelValueSeparator);

  [[nodiscard]] static std::string
  ToString(const util::Annotation & annotation, char labelValueSeparator);

  size_t index_;
  Graph * graph_;
  Node::Id nextNodeId_;
  rvsdg::StructuralNode * node_;

  // The region owns its results, arguments and nodes
  std::vector<RegionResult *> results_;
  std::vector<RegionArgument *> arguments_;
  region_top_node_list topNodes_;
  size_t numTopNodes_;
  region_bottom_node_list bottomNodes_;
  size_t numBottomNodes_;
  region_nodes_list nodes_;
  size_t numNodes_;

  // Allow RegionObservers to register themselves on const Regions
  mutable RegionObserver * observers_ = nullptr;

  friend class Node;
  friend class RegionObserver;
  friend class SimpleNode;
  friend class StructuralNode;
  friend class Input;
  friend class Output;
  friend class RegionResult;
};

/**
 * \brief Proxy object to observe changes to a region.
 *
 * Subscribers can implement and instantiate this interface for
 * a specific region to receive notifications about the region.
 *
 */
class RegionObserver
{
public:
  virtual ~RegionObserver() noexcept;

  explicit RegionObserver(const Region & region);

  RegionObserver(const RegionObserver &) = delete;

  RegionObserver &
  operator=(const RegionObserver &) = delete;

  /**
   * Called right after a node is added to the region,
   * after the node has its inputs and output added.
   * @param node the node being added
   */
  virtual void
  onNodeCreate(Node * node) = 0;

  /**
   * Called right before a node is removed from the region,
   * before the node has its inputs and outputs removed.
   * @param node the node being removed
   */
  virtual void
  onNodeDestroy(Node * node) = 0;

  /**
   * Called after a node gets a new input, or the region gets a new result.
   * This method is not called when creating new nodes, only modifying existing nodes.
   * @param input the new input
   */
  virtual void
  onInputCreate(Input * input) = 0;

  /**
   * Called right after the given input gets a new origin.
   * @param input the input.
   * @param old_origin the input's old origin.
   * @param new_origin the input's new origin.
   */
  virtual void
  onInputChange(Input * input, Output * old_origin, Output * new_origin) = 0;

  /**
   * Called right before a node input or region result is removed.
   * This method is not called when deleting nodes, only modifying existing nodes.
   * @param input the input that is removed
   */
  virtual void
  onInputDestroy(Input * input) = 0;

private:
  RegionObserver ** pprev_;
  RegionObserver * next_;

  friend class Region;
};

static inline void
remove(Node * node)
{
  return node->region()->removeNode(node);
}

size_t
nnodes(const rvsdg::Region * region) noexcept;

size_t
nstructnodes(const rvsdg::Region * region) noexcept;

size_t
nsimpnodes(const rvsdg::Region * region) noexcept;

size_t
ninputs(const rvsdg::Region * region) noexcept;

/**
 * \brief Checks if this is a result of a region inside a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param input
 *   Input to be checked.
 *
 * \returns
 *   Node of requested type to which the region belongs.
 *
 * Checks if the specified input is a region exit result belonging
 * to a node of specified type.
 * If this is the case, returns a pointer to the node of matched type.
 * If this is not the case (because either this is an input to a node
 * or or because the node owning the region is of a different kind,
 * or because this is the root region), returns nullptr.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType *
TryGetRegionParentNode(const rvsdg::Input & input) noexcept
{
  auto region = TryGetOwnerRegion(input);
  if (region)
  {
    return dynamic_cast<NodeType *>(region->node());
  }
  else
  {
    return nullptr;
  }
}

/**
 * \brief Checks if this is an argument of a region inside a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param output
 *   Output to be checked.
 *
 * \returns
 *   Node of requested type to which the region belongs.
 *
 * Checks if the specified input is a region entry argument belonging
 * to a node of specified type.
 * If this is the case, returns a pointer to the node of matched type.
 * If this is not the case (because either this is an input to a node
 * or or because the node owning the region is of a different kind,
 * or because this is the root region), returns nullptr.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType *
TryGetRegionParentNode(const rvsdg::Output & output) noexcept
{
  auto region = TryGetOwnerRegion(output);
  if (region)
  {
    return dynamic_cast<NodeType *>(region->node());
  }
  else
  {
    return nullptr;
  }
}

/**
 * \brief Asserts that this is a result of a region inside a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param input
 *   Input to be checked.
 *
 * \returns
 *   Node of requested type to which the region belongs.
 *
 * Checks if the specified input is a region exit result belonging
 * to a node of specified type.
 * If this is the case, returns a reference to the node of matched type,
 * otherwise throws an exception.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType &
AssertGetRegionParentNode(const rvsdg::Input & input)
{
  auto node = TryGetRegionParentNode<NodeType>(input);
  if (!node)
  {
    throw std::logic_error(std::string("expected node of type ") + typeid(NodeType).name());
  }
  return *node;
}

/**
 * \brief Asserts that this is an argument of a region inside a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param output
 *   Output to be checked.
 *
 * \returns
 *   Node of requested type to which the region belongs.
 *
 * Checks if the specified input is a region entry argument belonging
 * to a node of specified type.
 * If this is the case, returns a reference to the node of matched type,
 * otherwise throws an exception.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType &
AssertGetRegionParentNode(const rvsdg::Output & output)
{
  auto node = TryGetRegionParentNode<NodeType>(output);
  if (!node)
  {
    throw std::logic_error(std::string("expected node of type ") + typeid(NodeType).name());
  }
  return *node;
}

} // namespace

#endif
