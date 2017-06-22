/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CFG_NODE_H
#define JLM_IR_CFG_NODE_H

#include <jlm/common.hpp>

#include <list>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>


namespace jlm {

class attribute {
public:
	virtual
	~attribute();

	inline constexpr
	attribute()
	{}

	virtual std::string
	debug_string() const noexcept = 0;

	virtual std::unique_ptr<attribute>
	copy() const = 0;
};

class cfg;
class cfg_node;

class cfg_edge final {
public:
	~cfg_edge() noexcept {};

	cfg_edge(cfg_node * source, cfg_node * sink, size_t index) noexcept;

	void
	divert(cfg_node * new_sink);

	cfg_node *
	split();

	inline cfg_node * source() const noexcept { return source_; }
	inline cfg_node * sink() const noexcept { return sink_; }
	inline size_t index() const noexcept { return index_; }

	inline bool is_selfloop() const noexcept { return source_ == sink_; }

private:
	cfg_node * source_;
	cfg_node * sink_;
	size_t index_;

	friend cfg_node;
};

class cfg_node final {
	class const_outedge_iterator final {
	public:
		inline
		const_outedge_iterator(const std::vector<std::unique_ptr<cfg_edge>>::const_iterator & it)
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
		operator!=(const const_outedge_iterator & other) const noexcept
		{
			return !(other == *this);
		}

		inline cfg_edge *
		operator->() const noexcept
		{
			return edge();
		}

		inline cfg_edge *
		edge() const noexcept
		{
			return it_->get();
		}

	private:
		std::vector<std::unique_ptr<cfg_edge>>::const_iterator it_;
	};

public:
	inline
	cfg_node(jlm::cfg & cfg, const jlm::attribute & attr)
	: cfg_(&cfg)
	, attr_(std::move(attr.copy()))
	{}

protected:
	cfg_node(jlm::cfg & cfg) : cfg_(&cfg) {}

public:
	inline jlm::attribute &
	attribute() noexcept
	{
		return *attr_;
	}

	inline const jlm::attribute &
	attribute() const noexcept
	{
		return *attr_;
	}

	virtual std::string
	debug_string() const;

	inline jlm::cfg * cfg() const noexcept { return cfg_; }

	cfg_edge * add_outedge(cfg_node * successor, size_t index);

	void remove_outedge(cfg_edge * edge);

	void remove_outedges();

	inline cfg_edge *
	outedge(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < noutedges());
		auto it = outedges_.begin();
		std::advance(it, index);
		return it->get();
	}

	size_t noutedges() const noexcept;

	inline const_outedge_iterator
	begin_outedges() const
	{
		return const_outedge_iterator(outedges_.begin());
	}

	inline const_outedge_iterator
	end_outedges() const
	{
		return const_outedge_iterator(outedges_.end());
	}

	void divert_inedges(jlm::cfg_node * new_successor);

	void remove_inedges();

	size_t ninedges() const noexcept;

	std::list<cfg_edge*> inedges() const;

	bool no_predecessor() const noexcept;

	bool single_predecessor() const noexcept;

	bool no_successor() const noexcept;

	bool single_successor() const noexcept;

	inline bool is_branch() const noexcept { return noutedges() > 1; }

	bool has_selfloop_edge() const noexcept;

private:
	jlm::cfg * cfg_;
	std::unique_ptr<jlm::attribute> attr_;
	std::vector<std::unique_ptr<cfg_edge>> outedges_;
	std::list<cfg_edge*> inedges_;

	friend cfg_edge;
};

}

#endif
