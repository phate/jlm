/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_STORE_HPP
#define JLM_IR_OPERATORS_STORE_HPP

#include <jive/arch/memorytype.h>
#include <jive/vsdg/simple-normal-form.h>
#include <jive/vsdg/simple_node.h>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

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
	: simple_op()
	, nstates_(nstates)
	, aport_(ptype)
	, vport_(ptype.pointee_type())
	, alignment_(alignment)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::value::type &
	value_type() const noexcept
	{
		return *static_cast<const jive::value::type*>(&vport_.type());
	}

	inline size_t
	nstates() const noexcept
	{
		return nstates_;
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

private:
	size_t nstates_;
	jive::port aport_;
	jive::port vport_;
	size_t alignment_;
};

static inline bool
is_store_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::store_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_store_tac(
	const variable * address,
	const variable * value,
	size_t alignment,
	jlm::variable * state)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw std::logic_error("Expected pointer type.");

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
	if (!at) throw std::logic_error("Expected pointer type.");

	std::vector<jive::output*> operands({address, value});
	operands.insert(operands.end(), states.begin(), states.end());

	jlm::store_op op(*at, states.size(), alignment);
	return jive::create_normalized(address->region(), op, operands);
}

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

	inline bool
	get_store_mux_reducible() const noexcept
	{
		return enable_store_mux_;
	}

private:
	bool enable_store_mux_;
};

}

#endif
