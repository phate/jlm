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
class structural_op;
class StructuralOutput;
class SubstitutionMap;

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
  util::intrusive_list_anchor<RegionArgument> structural_input_anchor_;

public:
  typedef util::intrusive_list_accessor<RegionArgument, &RegionArgument::structural_input_anchor_>
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

  [[nodiscard]] std::variant<Node *, Region *>
  GetOwner() const noexcept override;

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
  util::intrusive_list_anchor<RegionResult> structural_output_anchor_;

public:
  typedef util::intrusive_list_accessor<RegionResult, &RegionResult::structural_output_anchor_>
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

  [[nodiscard]] std::variant<Node *, Region *>
  GetOwner() const noexcept override;

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
  typedef util::IntrusiveList<Node, Node::region_node_list_accessor> region_nodes_list;

  typedef util::IntrusiveList<Node, Node::region_top_node_list_accessor> region_top_node_list;

  typedef util::IntrusiveList<Node, Node::region_bottom_node_list_accessor> region_bottom_node_list;

  using RegionArgumentIterator = std::vector<RegionArgument *>::iterator;
  using RegionArgumentConstIterator = std::vector<RegionArgument *>::const_iterator;
  using RegionArgumentRange = util::IteratorRange<RegionArgumentIterator>;
  using RegionArgumentConstRange = util::IteratorRange<RegionArgumentConstIterator>;

  using RegionResultIterator = std::vector<RegionResult *>::iterator;
  using RegionResultConstIterator = std::vector<RegionResult *>::const_iterator;
  using RegionResultRange = util::IteratorRange<RegionResultIterator>;
  using RegionResultConstRange = util::IteratorRange<RegionResultConstIterator>;

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
   */
  [[nodiscard]] TopNodeRange
  TopNodes() noexcept
  {
    return { TopNodes_.begin(), TopNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the top nodes of the region.
   */
  [[nodiscard]] TopNodeConstRange
  TopNodes() const noexcept
  {
    return { TopNodes_.begin(), TopNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeRange
  Nodes() noexcept
  {
    return { Nodes_.begin(), Nodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeConstRange
  Nodes() const noexcept
  {
    return { Nodes_.begin(), Nodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the bottom nodes of the region.
   */
  [[nodiscard]] BottomNodeRange
  BottomNodes() noexcept
  {
    return { BottomNodes_.begin(), BottomNodes_.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the bottom nodes of the
   * region.
   */
  [[nodiscard]] BottomNodeConstRange
  BottomNodes() const noexcept
  {
    return { BottomNodes_.begin(), BottomNodes_.end() };
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

  /* \brief Append \p argument to the region
   *
   * Multiple invocations of append_argument for the same argument are undefined.
   */
  void
  append_argument(RegionArgument * argument);

  /* \brief Insert \p argument into argument list of the region
   *
   * Multiple invocations of append_argument for the same argument are undefined.
   */
  void
  insert_argument(size_t index, RegionArgument * argument);

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

  /* \brief Appends \p result to the region
   *
   * Multiple invocations of append_result for the same result are undefined.
   */
  void
  append_result(RegionResult * result);

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

  inline size_t
  nnodes() const noexcept
  {
    return Nodes_.size();
  }

  /**
   * @return The number of top nodes in the region.
   */
  [[nodiscard]] size_t
  NumTopNodes() const noexcept
  {
    return TopNodes_.size();
  }

  /**
   * @return The number of bottom nodes in the region.
   */
  [[nodiscard]] size_t
  NumBottomNodes() const noexcept
  {
    return BottomNodes_.size();
  }

  void
  remove_node(Node * node);

  /**
   * \brief Adds \p node to the top nodes of the region.
   *
   * The node \p node is only added to the top nodes of this region, iff:
   * 1. The node \p node belongs to the same region instance.
   * 2. The node \p node has no inputs.
   *
   * @param node The node that is added.
   * @return True, if \p node was added, otherwise false.
   *
   * @note This method is automatically invoked when a node is created. There is
   * no need to invoke it manually.
   */
  bool
  AddTopNode(Node & node);

  /**
   * \brief Adds \p node to the bottom nodes of the region.
   *
   * The node \p node is only added to the bottom nodes of this region, iff:
   * 1. The node \p node belongs to the same region instance.
   * 2. All the outputs of \p node are dead. See node::IsDead() for more details.
   *
   * @param node The node that is added.
   * @return True, if \p node was added, otherwise false.
   *
   * @note This method is automatically invoked when a node is created or becomes dead. There is
   * no need to invoke it manually.
   */
  bool
  AddBottomNode(Node & node);

  /**
   * \brief Adds \p node to the region.
   *
   * The node \p node is only added to this region, iff \p node belongs to the same region instance.
   *
   * @param node The node that is added.
   * @return True, if \p node was added, otherwise false.
   *
   * @note This method is automatically invoked when a node is created. There is no need to invoke
   * it manually.
   */
  bool
  AddNode(Node & node);

  /**
   * Removes \p node from the top nodes in the region.
   *
   * @param node The node that is removed.
   * @return True, if \p node was a top node and removed, otherwise false.
   *
   * @note This method is automatically invoked when inputs are added to a node. There is no need to
   * invoke it manually.
   */
  bool
  RemoveTopNode(Node & node);

  /**
   * Removes \p node from the bottom nodes in the region.
   *
   * @param node The node that is removed.
   * @return True, if \p node was a bottom node and removed, otherwise false.
   *
   * @note This method is automatically invoked when a node cedes to be dead. There is no need to
   * invoke it manually.
   */
  bool
  RemoveBottomNode(Node & node);

  /**
   * Remove \p node from the region.
   *
   * @param node The node that is removed.
   * @return True, if \p node was removed, otherwise false.
   *
   * @note This method is automatically invoked when a node is deleted. There is no need to invoke
   * it manually.
   */
  bool
  RemoveNode(Node & node);

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

  void
  prune(bool recursive);

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
  rvsdg::StructuralNode * node_;
  std::vector<RegionResult *> results_;
  std::vector<RegionArgument *> arguments_;
  region_bottom_node_list BottomNodes_;
  region_top_node_list TopNodes_;
  region_nodes_list Nodes_;
};

static inline void
remove(Node * node)
{
  return node->region()->remove_node(node);
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
