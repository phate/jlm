/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REGION_HPP
#define JLM_RVSDG_REGION_HPP

#include <stdbool.h>
#include <stddef.h>

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

class node;
class simple_node;
class simple_op;
class structural_input;
class structural_node;
class structural_op;
class structural_output;
class SubstitutionMap;

/**
 * \brief Represents the argument of a region.
 *
 * Region arguments represent the initial values of the region's acyclic graph. These values
 * are mapped to the arguments throughout the execution, and the concrete semantics of this mapping
 * depends on the structural node the region is part of. A region argument is either linked
 * with a \ref structural_input or is a standalone argument.
 */
class RegionArgument : public output
{
  util::intrusive_list_anchor<RegionArgument> structural_input_anchor_;

public:
  typedef util::intrusive_list_accessor<RegionArgument, &RegionArgument::structural_input_anchor_>
      structural_input_accessor;

  ~RegionArgument() noexcept override;

protected:
  RegionArgument(
      rvsdg::Region * region,
      structural_input * input,
      std::shared_ptr<const rvsdg::Type> type);

public:
  RegionArgument(const RegionArgument &) = delete;

  RegionArgument(RegionArgument &&) = delete;

  RegionArgument &
  operator=(const RegionArgument &) = delete;

  RegionArgument &
  operator=(RegionArgument &&) = delete;

  [[nodiscard]] structural_input *
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
  Copy(rvsdg::Region & region, structural_input * input) = 0;

private:
  structural_input * input_;
};

/**
 * \brief Represents the result of a region.
 *
 * Region results represent the final values of the region's acyclic graph. The result values
 * can be mapped back to the region arguments or the corresponding structural outputs
 * throughout the execution, but the concrete semantics of this mapping
 * depends on the structural node the region is part of. A region result is either linked
 * with a \ref structural_output or is a standalone result.
 */
class RegionResult : public input
{
  util::intrusive_list_anchor<RegionResult> structural_output_anchor_;

public:
  typedef util::intrusive_list_accessor<RegionResult, &RegionResult::structural_output_anchor_>
      structural_output_accessor;

  ~RegionResult() noexcept override;

protected:
  RegionResult(
      rvsdg::Region * region,
      rvsdg::output * origin,
      structural_output * output,
      std::shared_ptr<const rvsdg::Type> type);

public:
  RegionResult(const RegionResult &) = delete;

  RegionResult(RegionResult &&) = delete;

  RegionResult &
  operator=(const RegionResult &) = delete;

  RegionResult &
  operator=(RegionResult &&) = delete;

  [[nodiscard]] structural_output *
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
  Copy(rvsdg::output & origin, structural_output * output) = 0;

private:
  structural_output * output_;
};

/**
 * \brief Represent acyclic RVSDG subgraphs
 *
 * Regions represent acyclic RVSDG subgraphs and are instantiated with an index in \ref
 * structural_node%s. Each region has \ref RegionArgument%s and \ref RegionResult%s that represent
 * the values at the beginning and end of the acyclic graph, respectively. In addition, each region
 * keeps track of the following properties:
 *
 * 1. The nodes of the acyclic subgraph. They represent the computations performed in the region.
 * 2. The top nodes of the acyclic subgraph. These are all nodes of the region that have no inputs,
 * i.e., constants.
 * 3. The bottom nodes of the acyclic subgraph. These are all nodes of the region that have no
 * users, i.e. that are dead. See \ref output::IsDead() for more information.
 */
class Region
{
  typedef jlm::util::intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_node_list_accessor>
      region_nodes_list;

  typedef jlm::util::
      intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_top_node_list_accessor>
          region_top_node_list;

  typedef jlm::util::
      intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_bottom_node_list_accessor>
          region_bottom_node_list;

  using RegionArgumentIterator = std::vector<RegionArgument *>::iterator;
  using RegionArgumentConstIterator = std::vector<RegionArgument *>::const_iterator;
  using RegionArgumentRange = util::iterator_range<RegionArgumentIterator>;
  using RegionArgumentConstRange = util::iterator_range<RegionArgumentConstIterator>;

  using RegionResultIterator = std::vector<RegionResult *>::iterator;
  using RegionResultConstIterator = std::vector<RegionResult *>::const_iterator;
  using RegionResultRange = util::iterator_range<RegionResultIterator>;
  using RegionResultConstRange = util::iterator_range<RegionResultConstIterator>;

  using TopNodeIterator = region_top_node_list::iterator;
  using TopNodeConstIterator = region_top_node_list::const_iterator;
  using TopNodeRange = util::iterator_range<TopNodeIterator>;
  using TopNodeConstRange = util::iterator_range<TopNodeConstIterator>;

  using NodeIterator = region_nodes_list::iterator;
  using NodeConstIterator = region_nodes_list::const_iterator;
  using NodeRange = util::iterator_range<NodeIterator>;
  using NodeConstRange = util::iterator_range<NodeConstIterator>;

  using BottomNodeIterator = region_bottom_node_list::iterator;
  using BottomNodeConstIterator = region_bottom_node_list::const_iterator;
  using BottomNodeRange = util::iterator_range<BottomNodeIterator>;
  using BottomNodeConstRange = util::iterator_range<BottomNodeConstIterator>;

public:
  ~Region() noexcept;

  Region(rvsdg::Region * parent, jlm::rvsdg::graph * graph);

  Region(rvsdg::structural_node * node, size_t index);

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
    return { top_nodes.begin(), top_nodes.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the top nodes of the region.
   */
  [[nodiscard]] TopNodeConstRange
  TopNodes() const noexcept
  {
    return { top_nodes.begin(), top_nodes.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeRange
  Nodes() noexcept
  {
    return { nodes.begin(), nodes.end() };
  }

  /**
   * @return Returns an iterator range for iterating through the nodes of the region.
   */
  [[nodiscard]] NodeConstRange
  Nodes() const noexcept
  {
    return { nodes.begin(), nodes.end() };
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

  inline region_nodes_list::iterator
  begin()
  {
    return nodes.begin();
  }

  inline region_nodes_list::const_iterator
  begin() const
  {
    return nodes.begin();
  }

  inline region_nodes_list::iterator
  end()
  {
    return nodes.end();
  }

  inline region_nodes_list::const_iterator
  end() const
  {
    return nodes.end();
  }

  inline jlm::rvsdg::graph *
  graph() const noexcept
  {
    return graph_;
  }

  inline jlm::rvsdg::structural_node *
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
    return nodes.size();
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
  remove_node(jlm::rvsdg::node * node);

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
  AddBottomNode(rvsdg::node & node);

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
  RemoveBottomNode(rvsdg::node & node);

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

  void
  normalize(bool recursive);

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
  Contains(const rvsdg::Region & region, bool checkSubregions);

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
   * structural_node%s.
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

  region_nodes_list nodes;

  region_top_node_list top_nodes;

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
  jlm::rvsdg::graph * graph_;
  jlm::rvsdg::structural_node * node_;
  std::vector<RegionResult *> results_;
  std::vector<RegionArgument *> arguments_;
  region_bottom_node_list BottomNodes_;
};

static inline void
remove(jlm::rvsdg::node * node)
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

} // namespace

#endif
