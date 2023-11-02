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

class argument : public output {
	jlm::util::intrusive_list_anchor<
		jlm::rvsdg::argument
	> structural_input_anchor_;

public:
	typedef jlm::util::intrusive_list_accessor<
		jlm::rvsdg::argument,
		&jlm::rvsdg::argument::structural_input_anchor_
	> structural_input_accessor;

	virtual
	~argument() noexcept;

protected:
	argument(
		jlm::rvsdg::region * region,
		jlm::rvsdg::structural_input * input,
		const jlm::rvsdg::port & port);

	argument(const argument &) = delete;

	argument(argument &&) = delete;

	argument &
	operator=(const argument &) = delete;

	argument &
	operator=(argument &&) = delete;

public:
	inline jlm::rvsdg::structural_input *
	input() const noexcept
	{
		return input_;
	}

	static jlm::rvsdg::argument *
	create(
		jlm::rvsdg::region * region,
		structural_input * input,
		const jlm::rvsdg::port & port);

private:
	jlm::rvsdg::structural_input * input_;
};

class result : public input {
	jlm::util::intrusive_list_anchor<
		jlm::rvsdg::result
	> structural_output_anchor_;

public:
	typedef jlm::util::intrusive_list_accessor<
		jlm::rvsdg::result,
		&jlm::rvsdg::result::structural_output_anchor_
	> structural_output_accessor;

	virtual
	~result() noexcept;

protected:
	result(
		jlm::rvsdg::region * region,
		jlm::rvsdg::output * origin,
		jlm::rvsdg::structural_output * output,
		const jlm::rvsdg::port & port);

	result(const result &) = delete;

	result(result &&) = delete;

	result &
	operator=(const result &) = delete;

	result &
	operator=(result &&) = delete;

public:
	inline jlm::rvsdg::structural_output *
	output() const noexcept
	{
		return output_;
	}

	static jlm::rvsdg::result *
	create(
		jlm::rvsdg::region * region,
		jlm::rvsdg::output * origin,
		jlm::rvsdg::structural_output * output,
		const jlm::rvsdg::port & port);

private:
	jlm::rvsdg::structural_output * output_;
};

class region {
	typedef jlm::util::intrusive_list<
		jlm::rvsdg::node,
		jlm::rvsdg::node::region_node_list_accessor
	> region_nodes_list;

	typedef jlm::util::intrusive_list<
		jlm::rvsdg::node,
		jlm::rvsdg::node::region_top_node_list_accessor
	> region_top_node_list;

	typedef jlm::util::intrusive_list<
		jlm::rvsdg::node,
		jlm::rvsdg::node::region_bottom_node_list_accessor
	> region_bottom_node_list;

public:
	~region();

	region(jlm::rvsdg::region * parent, jlm::rvsdg::graph * graph);

	region(
		jlm::rvsdg::structural_node * node,
		size_t index);

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

	void
	remove_argument(size_t index);

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
	append_result(jlm::rvsdg::result * result);

  /**
   * Removes a result from the region given a results' index.
   *
   * The removal of a result invalidates the region's existing result iterators.
   *
   * @param index The results' index. It must be between [0, nresults()).
   *
   * \note The method must adjust the indices of the other results after the removal. The methods' runtime is therefore
   * O(n), where n is the region's number of results.
   *
   * \see nresults()
   * \see result#index()
   */
  void
  RemoveResult(size_t index);

	inline size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	inline jlm::rvsdg::result *
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
	copy(
		region * target,
		substitution_map & smap,
		bool copy_arguments,
		bool copy_results) const;

	void
	prune(bool recursive);

	void
	normalize(bool recursive);

  /**
 * Checks if an operation is contained within the given \p region. If \p checkSubregions is true, then the subregions
 * of all contained structural nodes are recursively checked as well.
 * @tparam Operation The operation to check for.
 * @param region The region to check.
 * @param checkSubregions If true, then the subregions of all contained structural nodes will be checked as well.
 * @return True, if the operation is found. Otherwise, false.
 */
  template <class Operation> static inline bool
  Contains(const jlm::rvsdg::region & region, bool checkSubregions);

  /**
   * Counts the number of (sub-)regions contained within \p region. The count includes \p region, i.e., if \p region
   * does not contain any structural nodes and therefore no subregions, then the count is one.
   *
   * @param region The region for which to count the contained (sub-)regions.
   * @return The number of (sub-)regions.
   */
  [[nodiscard]] static size_t
  NumRegions(const jlm::rvsdg::region & region) noexcept;

	region_nodes_list nodes;

	region_top_node_list top_nodes;

	region_bottom_node_list bottom_nodes;

private:
	size_t index_;
	jlm::rvsdg::graph * graph_;
	jlm::rvsdg::structural_node * node_;
	std::vector<jlm::rvsdg::result*> results_;
	std::vector<jlm::rvsdg::argument*> arguments_;
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

} //namespace

#endif
