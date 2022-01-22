/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_TRAVERSER_HPP
#define JIVE_RVSDG_TRAVERSER_HPP

#include <stdbool.h>
#include <stdlib.h>

namespace jive {
namespace detail {

template<typename T>
class traverser_iterator {
public:
	typedef std::input_iterator_tag iterator_category;
	typedef jive::node * value_type;
	typedef ssize_t difference_type;
	typedef value_type * pointer;
	typedef value_type & reference;

	constexpr
	traverser_iterator(T * traverser = nullptr, jive::node * node = nullptr) noexcept
		: traverser_(traverser)
		, node_(node)
	{
	}

	inline const traverser_iterator &
	operator++() noexcept
	{
		node_ = traverser_->next();
		return *this;
	}

	inline bool
	operator==(const traverser_iterator& other) const noexcept
	{
		return traverser_ == other.traverser_ && node_ == other.node_;
	}

	inline bool
	operator!=(const traverser_iterator& other) const noexcept
	{
		return !(*this == other);
	}

	inline value_type & operator*() noexcept { return node_; }

	inline value_type
	operator->() noexcept
	{
		return node_;
	}

private:
	T * traverser_;
	jive::node * node_;
};

}

enum class traversal_nodestate {
	ahead = -1,
	frontier = 0,
	behind = +1
};

/* support class to track traversal states of nodes */
class traversal_tracker final {
public:
	inline
	traversal_tracker(jive::graph * graph);
	
	inline traversal_nodestate
	get_nodestate(jive::node * node);
	
	inline void
	set_nodestate(jive::node * node, traversal_nodestate state);
	
	inline jive::node *
	peek_top();
	
	inline jive::node *
	peek_bottom();

private:
	tracker tracker_;
};

class topdown_traverser final {
public:
	~topdown_traverser() noexcept;

	explicit
	topdown_traverser(jive::region * region);

	jive::node *
	next();

	inline jive::region *
	region() const noexcept
	{
		return region_;
	}

	typedef detail::traverser_iterator<topdown_traverser> iterator;
	typedef jive::node * value_type;
	inline iterator begin() { return iterator(this, next()); }
	inline iterator end() { return iterator(this, nullptr); }

private:
	bool
	predecessors_visited(const jive::node * node) noexcept;

	void
	node_create(jive::node * node);

	void
	input_change(input * in, output * old_origin, output * new_origin);

	jive::region * region_;
	traversal_tracker tracker_;
	std::vector<callback> callbacks_;
};

class bottomup_traverser final {
public:
	~bottomup_traverser() noexcept;

	explicit
	bottomup_traverser(jive::region * region, bool revisit = false);

	jive::node *
	next();

	inline jive::region *
	region() const noexcept
	{
		return region_;
	}

	typedef detail::traverser_iterator<bottomup_traverser> iterator;
	typedef jive::node * value_type;
	inline iterator begin() { return iterator(this, next()); }
	inline iterator end() { return iterator(this, nullptr); }

private:
	void
	node_create(jive::node * node);

	void
	node_destroy(jive::node * node);

	void
	input_change(input * in, output * old_origin, output * new_origin);

	jive::region * region_;
	traversal_tracker tracker_;
	std::vector<callback> callbacks_;
	traversal_nodestate new_node_state_;
};

/* traversal tracker implementation */

traversal_tracker::traversal_tracker(jive::graph * graph)
	: tracker_(graph, 2)
{
}

traversal_nodestate
traversal_tracker::get_nodestate(jive::node * node)
{
	return static_cast<traversal_nodestate>(tracker_.get_nodestate(node));
}

void
traversal_tracker::set_nodestate(
	jive::node * node,
	traversal_nodestate state)
{
	tracker_.set_nodestate(node, static_cast<size_t>(state));
}

jive::node *
traversal_tracker::peek_top()
{
	return tracker_.peek_top(static_cast<size_t>(traversal_nodestate::frontier));
}

jive::node *
traversal_tracker::peek_bottom()
{
	return tracker_.peek_bottom(static_cast<size_t>(traversal_nodestate::frontier));
}

}

#endif
