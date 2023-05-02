/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_IPGRAPH_MODULE_HPP
#define JLM_LLVM_IR_IPGRAPH_MODULE_HPP

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/tac.hpp>

#include <jlm/util/file.hpp>

namespace jlm {

/* global value */

class gblvalue final : public gblvariable {
public:
	virtual
	~gblvalue();

	inline
	gblvalue(data_node * node)
	: gblvariable(node->type(), node->name())
	, node_(node)
	{}

	gblvalue(const gblvalue &) = delete;

	gblvalue(gblvalue &&) = delete;

	gblvalue &
	operator=(const gblvalue &) = delete;

	gblvalue &
	operator=(gblvalue &&) = delete;

	inline data_node *
	node() const noexcept
	{
		return node_;
	}

private:
	data_node * node_;
};

static inline std::unique_ptr<jlm::gblvalue>
create_gblvalue(data_node * node)
{
	return std::make_unique<jlm::gblvalue>(node);
}

/* ipgraph module */

class ipgraph_module final {
	typedef std::unordered_set<const jlm::gblvalue*>::const_iterator const_iterator;

public:
	inline
	~ipgraph_module()
	{}

	inline
	ipgraph_module(
		const jlm::filepath & source_filename,
		const std::string & target_triple,
		const std::string & data_layout) noexcept
	: data_layout_(data_layout)
	, target_triple_(target_triple)
	, source_filename_(source_filename)
	{}

	inline jlm::ipgraph &
	ipgraph() noexcept
	{
		return clg_;
	}

	inline const jlm::ipgraph &
	ipgraph() const noexcept
	{
		return clg_;
	}

	const_iterator
	begin() const
	{
		return globals_.begin();
	}

	const_iterator
	end() const
	{
		return globals_.end();
	}

	inline jlm::gblvalue *
	create_global_value(data_node * node)
	{
		auto v = jlm::create_gblvalue(node);
		auto ptr = v.get();
		globals_.insert(ptr);
		functions_[node] = ptr;
		variables_.insert(std::move(v));
		return ptr;
	}

	inline jlm::variable *
	create_variable(const jive::type & type, const std::string & name)
	{
		auto v = std::make_unique<jlm::variable>(type, name);
		auto pv = v.get();
		variables_.insert(std::move(v));
		return pv;
	}

	inline jlm::variable *
	create_variable(const jive::type & type)
	{
		static uint64_t c = 0;
		auto v = std::make_unique<jlm::variable>(type, strfmt("v", c++));
		auto pv = v.get();
		variables_.insert(std::move(v));
		return pv;
	}

	inline jlm::variable *
	create_variable(function_node * node)
	{
		JLM_ASSERT(!variable(node));

		auto v = std::unique_ptr<jlm::variable>(new fctvariable(node));
		auto pv = v.get();
		functions_[node] = pv;
		variables_.insert(std::move(v));
		return pv;
	}

	const jlm::variable *
	variable(const ipgraph_node * node) const noexcept
	{
		auto it = functions_.find(node);
		return it != functions_.end() ? it->second : nullptr;
	}

	const jlm::filepath &
	source_filename() const noexcept
	{
		return source_filename_;
	}

	inline const std::string &
	target_triple() const noexcept
	{
		return target_triple_;
	}

	inline const std::string &
	data_layout() const noexcept
	{
		return data_layout_;
	}

	static std::unique_ptr<ipgraph_module>
	create(
		const jlm::filepath & source_filename,
		const std::string & target_triple,
		const std::string & data_layout)
	{
		return std::make_unique<ipgraph_module>(source_filename, target_triple, data_layout);
	}

private:
	jlm::ipgraph clg_;
	std::string data_layout_;
	std::string target_triple_;
	const jlm::filepath source_filename_;
	std::unordered_set<const jlm::gblvalue*> globals_;
	std::unordered_set<std::unique_ptr<jlm::variable>> variables_;
	std::unordered_map<const ipgraph_node*, const jlm::variable*> functions_;
};

static inline size_t
ntacs(const ipgraph_module & im)
{
	size_t ntacs = 0;
	for (const auto & n : im.ipgraph()) {
		auto f = dynamic_cast<const function_node*>(&n);
		if (!f) continue;

		auto cfg = f->cfg();
		if (!cfg) continue;

		for (const auto & node : *f->cfg()) {
			if (auto bb = dynamic_cast<const basic_block*>(&node))
				ntacs += bb->tacs().ntacs();
		}
	}

	return ntacs;
}

}

#endif
