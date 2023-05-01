/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_CFG_STRUCTURE_HPP
#define JLM_LLVM_IR_CFG_STRUCTURE_HPP

#include <jlm/util/iterator_range.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

namespace jlm {

class cfg;
class cfg_edge;
class cfg_node;

/** \brief Strongly Connected Component
*/
class scc final {
	class constiterator;

public:
	scc(const std::unordered_set<cfg_node*> & nodes)
	: nodes_(nodes)
	{}

	constiterator
	begin() const;

	constiterator
	end() const;

	bool
	contains(cfg_node * node) const
	{
		return nodes_.find(node) != nodes_.end();
	}

	size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

private:
	std::unordered_set<cfg_node*> nodes_;
};

/** \brief Strongly Connected Component Iterator
*/
class scc::constiterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = cfg_node*;
  using difference_type = std::ptrdiff_t;
  using pointer = cfg_node**;
  using reference = cfg_node*&;

private:
  friend ::jlm::scc;

private:
	constiterator(const std::unordered_set<cfg_node*>::const_iterator & it)
	: it_(it)
	{}

public:
	cfg_node *
	operator->() const
	{
		return *it_;
	}

	cfg_node &
	operator*() const
	{
		return *operator->();
	}

	constiterator &
	operator++()
	{
		it_++;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	std::unordered_set<cfg_node*>::const_iterator it_;
};

/** \brief Strongly Connected Component Structure
*
* This class computes the structure of strongly connected components (SCCs). It detects the
* following entities:
*
* 1. Entry edges (eedges): All edges from a node outside the SCC pointing to a node in the SCC.
* 2. Entry nodes (enodes): All nodes that are the target of one or more entry edges.
* 3. Exit edges (xedges): All edges from a node inside the SCC pointing to a node outside the SCC.
* 4. Exit nodes (xnodes): All nodes that are the target of one or more exit edges.
* 5. Repetition edges (redges): All edges from a node inside the SCC to an entry node.
*/
class sccstructure final {
	using cfg_edge_constiterator = std::unordered_set<cfg_edge*>::const_iterator;
	using cfg_node_constiterator = std::unordered_set<cfg_node*>::const_iterator;

	using edge_iterator_range = iterator_range<cfg_edge_constiterator>;
	using node_iterator_range = iterator_range<cfg_node_constiterator>;

public:
	size_t
	nenodes() const noexcept
	{
		return enodes_.size();
	}

	size_t
	nxnodes() const noexcept
	{
		return xnodes_.size();
	}

	size_t
	needges() const noexcept
	{
		return eedges_.size();
	}

	size_t
	nredges() const noexcept
	{
		return redges_.size();
	}

	size_t
	nxedges() const noexcept
	{
		return xedges_.size();
	}

	node_iterator_range
	enodes() const
	{
		return node_iterator_range(enodes_);
	}

	node_iterator_range
	xnodes() const
	{
		return node_iterator_range(xnodes_);
	}

	edge_iterator_range
	eedges() const
	{
		return edge_iterator_range(eedges_);
	}

	edge_iterator_range
	redges() const
	{
		return edge_iterator_range(redges_);
	}

	edge_iterator_range
	xedges() const
	{
		return edge_iterator_range(xedges_);
	}

	/**
	* Creates a SCC structure from SCC \p scc.
	*/
	static std::unique_ptr<sccstructure>
	create(const jlm::scc & scc);

	/**
	* Checks if the SCC structure is a tail-controlled loop. A tail-controlled loop is defined as an
	* SSC with a single enttry node, as well as a single repetition and exit edge. Both these edges
	* must have the same CFG node as origin.
	*/
	bool
	is_tcloop() const;

private:
	std::unordered_set<cfg_node*> enodes_;
	std::unordered_set<cfg_node*> xnodes_;
	std::unordered_set<cfg_edge*> eedges_;
	std::unordered_set<cfg_edge*> redges_;
	std::unordered_set<cfg_edge*> xedges_;
};

bool
is_valid(const jlm::cfg & cfg);

bool
is_closed(const jlm::cfg & cfg);

bool
is_linear(const jlm::cfg & cfg);

/**
* Compute a Control Flow Graph's Strongly Connected Components.
*/
std::vector<scc>
find_sccs(const jlm::cfg & cfg);

/**
* Compute all Strongly Connected Components of a single-entry/single-exit region.
* The \p entry parameter must dominate the \p exit parameter.
*/
std::vector<scc>
find_sccs(cfg_node * entry, cfg_node * exit);

static inline bool
is_acyclic(const jlm::cfg & cfg)
{
	auto sccs = find_sccs(cfg);
	return sccs.size() == 0;
}

bool
is_structured(const jlm::cfg & cfg);

bool
is_proper_structured(const jlm::cfg & cfg);

bool
is_reducible(const jlm::cfg & cfg);

void
straighten(jlm::cfg & cfg);

/** \brief Remove all basic blocks without instructions
*/
void
purge(jlm::cfg & cfg);

void
prune(jlm::cfg & cfg);

}

#endif
