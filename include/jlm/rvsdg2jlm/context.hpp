/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

namespace jlm {

class cfg_node;
class module;
class variable;

namespace rvsdg2jlm {

class context final {
public:
	inline
	context(jlm::module & module)
	: cfg_(nullptr)
	, module_(module)
	, lpbb_(nullptr)
	{}

	context(const context&) = delete;

	context(context&&) = delete;

	context &
	operator=(const context&) = delete;

	context&
	operator=(context&&) = delete;

	inline jlm::module &
	module() const noexcept
	{
		return module_;
	}

	inline void
	insert(const jive::oport * port, const jlm::variable * v)
	{
		ports_[port] = v;
	}

	inline const jlm::variable *
	variable(const jive::oport * port)
	{
		auto it = ports_.find(port);
		JLM_DEBUG_ASSERT(it != ports_.end());
		return it->second;
	}

	inline jlm::cfg_node *
	lpbb() const noexcept
	{
		return lpbb_;
	}

	inline void
	set_lpbb(jlm::cfg_node * lpbb) noexcept
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
	jlm::module & module_;
	jlm::cfg_node * lpbb_;
	std::unordered_map<const jive::oport*, const jlm::variable*> ports_;
};

}}
