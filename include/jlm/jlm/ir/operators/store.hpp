/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_STORE_HPP
#define JLM_IR_OPERATORS_STORE_HPP

#include <jive/arch/addresstype.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/simple-normal-form.h>
#include <jive/rvsdg/simple-node.h>

#include <jlm/jlm/ir/tac.hpp>
#include <jlm/jlm/ir/types.hpp>

namespace jlm {

/* store normal form */

class store_normal_form final : public jive::simple_normal_form {
public:
	virtual
	~store_normal_form() noexcept;

	store_normal_form(
		const std::type_info & opclass,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept;

	virtual bool
	normalize_node(jive::node * node) const override;

	virtual std::vector<jive::output*>
	normalized_create(
		jive::region * region,
		const jive::simple_op & op,
		const std::vector<jive::output*> & operands) const override;

	virtual void
	set_store_mux_reducible(bool enable);

	virtual void
	set_store_store_reducible(bool enable);

	virtual void
	set_store_alloca_reducible(bool enable);

	virtual void
	set_multiple_origin_reducible(bool enable);

	inline bool
	get_store_mux_reducible() const noexcept
	{
		return enable_store_mux_;
	}

	inline bool
	get_store_store_reducible() const noexcept
	{
		return enable_store_store_;
	}

	inline bool
	get_store_alloca_reducible() const noexcept
	{
		return enable_store_alloca_;
	}

	inline bool
	get_multiple_origin_reducible() const noexcept
	{
		return enable_multiple_origin_;
	}

private:
	bool enable_store_mux_;
	bool enable_store_store_;
	bool enable_store_alloca_;
	bool enable_multiple_origin_;
};

/* store operator */

class store_op final : public jive::simple_op {
public:
	virtual
	~store_op() noexcept;

	inline
	store_op(
		const jlm::ptrtype & ptype,
		size_t nstates,
		size_t alignment)
	: simple_op(create_srcports(ptype, nstates),
			std::vector<jive::port>(nstates, {jive::memtype::instance()}))
	, alignment_(alignment)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::valuetype &
	value_type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(&argument(1).type());
	}

	inline size_t
	nstates() const noexcept
	{
		return nresults();
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static jlm::store_normal_form *
	normal_form(jive::graph * graph) noexcept
	{
		return static_cast<jlm::store_normal_form*>(graph->node_normal_form(typeid(store_op)));
	}

private:
	static inline std::vector<jive::port>
	create_srcports(const ptrtype & ptype, size_t nstates)
	{
		std::vector<jive::port> ports({ptype, ptype.pointee_type()});
		std::vector<jive::port> states(nstates, {jive::memtype::instance()});
		ports.insert(ports.end(), states.begin(), states.end());
		return ports;
	}
	size_t alignment_;
};

static inline std::unique_ptr<jlm::tac>
create_store_tac(
	const variable * address,
	const variable * value,
	size_t alignment,
	jlm::variable * state)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw jlm::error("expected pointer type.");

	jlm::store_op op(*at, 1, alignment);
	return create_tac(op, {address, value, state}, {state});
}

static inline std::vector<jive::output*>
create_store(
	jive::output * address,
	jive::output * value,
	const std::vector<jive::output*> & states,
	size_t alignment)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw jlm::error("expected pointer type.");

	std::vector<jive::output*> operands({address, value});
	operands.insert(operands.end(), states.begin(), states.end());

	jlm::store_op op(*at, states.size(), alignment);
	return jive::simple_node::create_normalized(address->region(), op, operands);
}

}

#endif
