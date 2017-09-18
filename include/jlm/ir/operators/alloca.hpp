/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_ALLOCA_HPP
#define JLM_IR_OPERATORS_ALLOCA_HPP

#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/simple-normal-form.h>
#include <jive/vsdg/simple_node.h>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* alloca normal form */

class alloca_normal_form final : public jive::simple_normal_form {
public:
	virtual
	~alloca_normal_form() noexcept;

	alloca_normal_form(
		const std::type_info & opclass,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept;

	virtual bool
	normalize_node(jive::node * node) const override;

	virtual std::vector<jive::output*>
	normalized_create(
		jive::region * region,
		const jive::simple_op & op,
		const std::vector<jive::output*> & arguments) const override;

	virtual void
	set_alloca_mux_reducible(bool enable);

	virtual void
	set_alloca_alloca_reducible(bool enable);

	inline bool
	get_alloca_mux_reducible() const noexcept
	{
		return enable_alloca_mux_;
	}

	inline bool
	get_alloca_alloca_reducible() const noexcept
	{
		return enable_alloca_alloca_;
	}

private:
	bool enable_alloca_mux_;
	bool enable_alloca_alloca_;
};

/* alloca operator */

class alloca_op final : public jive::simple_op {
public:
	virtual
	~alloca_op() noexcept;

	inline
	alloca_op(
		const jlm::ptrtype & atype,
		const jive::bits::type & btype,
		size_t alignment)
	: simple_op()
	, aport_(atype)
	, bport_(btype)
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

	inline const jive::bits::type &
	size_type() const noexcept
	{
		return *static_cast<const jive::bits::type*>(&bport_.type());
	}

	inline const jive::value::type &
	value_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&aport_.type())->pointee_type();
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static jlm::alloca_normal_form *
	normal_form(jive::graph * graph) noexcept
	{
		return static_cast<jlm::alloca_normal_form*>(graph->node_normal_form(typeid(alloca_op)));
	}

private:
	jive::port aport_;
	jive::port bport_;
	size_t alignment_;
};

static inline bool
is_alloca_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::alloca_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_alloca_tac(
	const jive::base::type & vtype,
	const variable * size,
	size_t alignment,
	jlm::variable * state,
	jlm::variable * result)
{
	auto vt = dynamic_cast<const jive::value::type*>(&vtype);
	if (!vt) throw std::logic_error("Expected value type.");

	auto bt = dynamic_cast<const jive::bits::type*>(&size->type());
	if (!bt) throw std::logic_error("Expected bits type.");

	jlm::alloca_op op(jlm::ptrtype(*vt), *bt, alignment);
	return create_tac(op, {size, state}, {result, state});
}

static inline std::vector<jive::output*>
create_alloca(
	const jive::base::type & type,
	jive::output * size,
	jive::output * state,
	size_t alignment)
{
	auto vt = dynamic_cast<const jive::value::type*>(&type);
	if (!vt) throw std::logic_error("Expected value type.");

	auto bt = dynamic_cast<const jive::bits::type*>(&size->type());
	if (!bt) throw std::logic_error("Expected bits type.");

	jlm::alloca_op op(jlm::ptrtype(*vt), *bt, alignment);
	return jive::create_normalized(size->region(), op, {size, state});
}

}

#endif
