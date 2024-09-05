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
class substitution_map;

/**
 * \brief Represents the argument of a region.
 *
 * Region arguments represent the initial values of the region's acyclic graph. These values
 * are mapped to the arguments throughout the execution, and the concrete semantics of this mapping
 * depends on the structural node the region is part of. A region argument is either linked
 * with a \ref structural_input or is a standalone argument.
 */
class argument : public output
{
  util::intrusive_list_anchor<argument> structural_input_anchor_;

public:
  typedef util::intrusive_list_accessor<argument, &argument::structural_input_anchor_>
      structural_input_accessor;

  ~argument() noexcept override;

protected:
  argument(
      rvsdg::region * region,
      structural_input * input,
      std::shared_ptr<const rvsdg::type> type);

public:
  argument(const argument &) = delete;

  argument(argument &&) = delete;

  argument &
  operator=(const argument &) = delete;

  argument &
  operator=(argument &&) = delete;

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
  virtual argument &
  Copy(rvsdg::region & region, structural_input * input) = 0;

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
      rvsdg::region * region,
      rvsdg::output * origin,
      structural_output * output,
      std::shared_ptr<const rvsdg::type> type);

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

class region
{
  typedef jlm::util::intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_node_list_accessor>
      region_nodes_list;

  typedef jlm::util::
      intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_top_node_list_accessor>
          region_top_node_list;

  typedef jlm::util::
      intrusive_list<jlm::rvsdg::node, jlm::rvsdg::node::region_bottom_node_list_accessor>
          region_bottom_node_list;

public:
  ~region();

  region(jlm::rvsdg::region * parent, jlm::rvsdg::graph * graph);

  region(jlm::rvsdg::structural_node * node, size_t index);

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
  append_argument(jlm::rvsdg::argument * argument);

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
   * \see argument#index()
   * \see argument::nusers()
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

  inline jlm::rvsdg::argument *
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
    auto match = [](const rvsdg::argument &)
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

  void
  remove_node(jlm::rvsdg::node * node);

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
  copy(region * target, substitution_map & smap, bool copy_arguments, bool copy_results) const;

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
  Contains(const jlm::rvsdg::region & region, bool checkSubregions);

  /**
   * Counts the number of (sub-)regions contained within \p region. The count includes \p region,
   * i.e., if \p region does not contain any structural nodes and therefore no subregions, then the
   * count is one.
   *
   * @param region The region for which to count the contained (sub-)regions.
   * @return The number of (sub-)regions.
   */
  [[nodiscard]] static size_t
  NumRegions(const jlm::rvsdg::region & region) noexcept;

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
   * @param annotationMap A map with annotations for instances of \ref region%s or
   * structural_node%s.
   * @return A string containing the ASCII tree of \p region.
   */
  [[nodiscard]] static std::string
  ToTree(const rvsdg::region & region, const util::AnnotationMap & annotationMap) noexcept;

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
  ToTree(const rvsdg::region & region) noexcept;

  region_nodes_list nodes;

  region_top_node_list top_nodes;

  region_bottom_node_list bottom_nodes;

private:
  static void
  ToTree(
      const rvsdg::region & region,
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
  std::vector<jlm::rvsdg::argument *> arguments_;
};

static inline void
remove(jlm::rvsdg::node * node)
{
  return node->region()->remove_node(node);
}

size_t
nnodes(const jlm::rvsdg::region * region) noexcept;

size_t
nstructnodes(const jlm::rvsdg::region * region) noexcept;

size_t
nsimpnodes(const jlm::rvsdg::region * region) noexcept;

size_t
ninputs(const jlm::rvsdg::region * region) noexcept;

} // namespace

#endif
