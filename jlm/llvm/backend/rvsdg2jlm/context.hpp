/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDG2JLM_CONTEXT_HPP
#define JLM_LLVM_BACKEND_RVSDG2JLM_CONTEXT_HPP

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/rvsdg/node.hpp>

namespace jlm {

class cfg_node;
class ipgraph_module;
class variable;

namespace rvsdg2jlm {

class context final {
public:
	inline
	context(ipgraph_module & im)
	: cfg_(nullptr)
	, module_(im)
	, lpbb_(nullptr)
	{}

	context(const context&) = delete;

	context(context&&) = delete;

	context &
	operator=(const context&) = delete;

	context&
	operator=(context&&) = delete;

	inline ipgraph_module &
	module() const noexcept
	{
		return module_;
	}

	inline void
	insert(const jive::output * port, const jlm::variable * v)
	{
		JLM_ASSERT(ports_.find(port) == ports_.end());
		JLM_ASSERT(port->type() == v->type());
		ports_[port] = v;
	}

	inline const jlm::variable *
	variable(const jive::output * port)
	{
		auto it = ports_.find(port);
		JLM_ASSERT(it != ports_.end());
		return it->second;
	}

	inline basic_block *
	lpbb() const noexcept
	{
		return lpbb_;
	}

	inline void
	set_lpbb(basic_block * lpbb) noexcept
	{
		lpbb_ = lpbb;
	}

	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_;
	}

	inline void
	set_cfg(jlm::cfg * cfg) noexcept
	{
		cfg_ = cfg;
	}

private:
	jlm::cfg * cfg_;
	ipgraph_module & module_;
	basic_block * lpbb_;
	std::unordered_map<const jive::output*, const jlm::variable*> ports_;
};

}}

#endif
