/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CFG_NODE_H
#define JLM_IR_CFG_NODE_H

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
};

class cfg_node final {
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
		return outedges()[index];
	}

	size_t noutedges() const noexcept;

	std::vector<cfg_edge*> outedges() const;

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
	std::unordered_set<std::unique_ptr<cfg_edge>> outedges_;
	std::list<cfg_edge*> inedges_;

	friend cfg_edge;
};

}

#endif
