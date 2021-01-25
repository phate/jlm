/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CFG_STRUCTURE_HPP
#define JLM_IR_CFG_STRUCTURE_HPP

#include <unordered_set>
#include <vector>

namespace jlm {

class cfg;
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
class scc::constiterator final : public std::iterator<std::forward_iterator_tag, cfg_node*,
	ptrdiff_t> {
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

void
purge(jlm::cfg & cfg);

void
prune(jlm::cfg & cfg);

}

#endif
