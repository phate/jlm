/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_CFG_NODE_H
#define JLM_FRONTEND_CFG_NODE_H

#include <list>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>


namespace jlm {
namespace frontend {

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

class cfg_node {
public:
	virtual ~cfg_node();

protected:
	cfg_node(jlm::frontend::cfg & cfg) : cfg_(&cfg) {}

public:
	virtual std::string debug_string() const = 0;

	inline jlm::frontend::cfg * cfg() const noexcept { return cfg_; }

	cfg_edge * add_outedge(cfg_node * successor, size_t index);

	void remove_outedge(cfg_edge * edge);

	void remove_outedges();

	size_t noutedges() const noexcept;

	std::vector<cfg_edge*> outedges() const;

	void divert_inedges(jlm::frontend::cfg_node * new_successor);

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
	std::unordered_set<std::unique_ptr<cfg_edge>> outedges_;
	std::list<cfg_edge*> inedges_;
	jlm::frontend::cfg * cfg_;

	friend cfg_edge;
};

}
}

#endif
