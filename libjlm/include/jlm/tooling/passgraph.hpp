/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_PASSGRAPH_HPP
#define JLM_TOOLING_PASSGRAPH_HPP

#include <jlm/tooling/command.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

namespace jlm {

/* passgraph edge */

class passgraph_node;

class passgraph_edge final {
public:
	passgraph_edge(
		passgraph_node * source,
		passgraph_node * sink)
	: sink_(sink)
	, source_(source)
	{}

	inline passgraph_node *
	source() const noexcept
	{
		return source_;
	}

	inline passgraph_node *
	sink() const noexcept
	{
		return sink_;
	}

private:
	passgraph_node * sink_;
	passgraph_node * source_;
};

/* passgraph node */

class passgraph;

class passgraph_node final {
	typedef std::unordered_set<passgraph_edge*>::const_iterator const_inedge_iterator;

	class const_outedge_iterator final {
	public:
		inline
		const_outedge_iterator(
			const std::unordered_set<std::unique_ptr<passgraph_edge>>::const_iterator & it)
		: it_(it)
		{}

		inline const const_outedge_iterator &
		operator++() noexcept
		{
			++it_;
			return *this;
		}

		inline const_outedge_iterator
		operator++(int) noexcept
		{
			auto tmp = *this;
			++*this;
			return tmp;
		}

		inline bool
		operator==(const const_outedge_iterator & other) const noexcept
		{
			return it_ == other.it_;
		}

		inline bool
		operator!=(const const_outedge_iterator &  other) const noexcept
		{
			return !(other == *this);
		}

		inline passgraph_edge *
		operator->() const noexcept
		{
			return edge();
		}

		inline passgraph_edge &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline passgraph_edge *
		edge() const noexcept
		{
			return it_->get();
		}

	private:
		std::unordered_set<std::unique_ptr<passgraph_edge>>::const_iterator it_;
	};

public:
	~passgraph_node()
	{}

private:
	inline
	passgraph_node(passgraph * pgraph, std::unique_ptr<command> cmd)
	: pgraph_(pgraph)
	, cmd_(std::move(cmd))
	{}

public:
	passgraph_node(const passgraph_node&) = delete;

	passgraph_node(passgraph_node&&) = delete;

	passgraph_node &
	operator=(const passgraph_node&) = delete;

	passgraph_node &
	operator=(passgraph_node&&) = delete;

	passgraph *
	pgraph() const noexcept
	{
		return pgraph_;
	}

	command &
	cmd() const noexcept
	{
		return *cmd_;
	}

	inline size_t
	ninedges() const noexcept
	{
		return inedges_.size();
	}

	inline size_t
	noutedges() const noexcept
	{
		return outedges_.size();
	}

	const_outedge_iterator
	begin_outedges() const
	{
		return const_outedge_iterator(outedges_.begin());
	}

	const_outedge_iterator
	end_outedges() const
	{
		return const_outedge_iterator(outedges_.end());
	}

	const_inedge_iterator
	begin_inedges() const
	{
		return inedges_.begin();
	}

	const_inedge_iterator
	end_inedges() const
	{
		return inedges_.end();
	}

	const_outedge_iterator
	begin() const
	{
		return begin_outedges();
	}

	const_outedge_iterator
	end() const
	{
		return end_outedges();
	}

	void
	add_edge(passgraph_node * sink)
	{
		std::unique_ptr<passgraph_edge> outedge(new passgraph_edge(this, sink));
		auto ptr = outedge.get();
		outedges_.insert(std::move(outedge));
		sink->inedges_.insert(ptr);
	}

	static passgraph_node *
	create(passgraph * graph, std::unique_ptr<command> cmd);

private:
	passgraph * pgraph_;
	std::unique_ptr<command> cmd_;
	std::unordered_set<passgraph_edge*> inedges_;
	std::unordered_set<std::unique_ptr<passgraph_edge>> outedges_;
};

/* passgraph */

class passgraph final {
	typedef std::unordered_set<
		std::unique_ptr<passgraph_node>
	>::const_iterator const_iterator;

public:
	~passgraph()
	{}

	passgraph();

	passgraph(const passgraph&) = delete;

	passgraph(passgraph&&) = delete;

	passgraph &
	operator=(const passgraph&) = delete;

	passgraph &
	operator=(passgraph&&) = delete;

	inline passgraph_node *
	entry() const noexcept
	{
		return entry_;
	}

	inline passgraph_node *
	exit() const noexcept
	{
		return exit_;
	}

	const_iterator
	begin() const
	{
		return nodes_.begin();
	}

	const_iterator
	end() const
	{
		return nodes_.end();
	}

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline void
	add_node(std::unique_ptr<jlm::passgraph_node> node)
	{
		nodes_.insert(std::move(node));
	}

	void
	run() const;

private:
	passgraph_node * exit_;
	passgraph_node * entry_;
	std::unordered_set<std::unique_ptr<passgraph_node>> nodes_;
};

std::vector<passgraph_node*>
topsort(const passgraph * pgraph);

}

#endif
