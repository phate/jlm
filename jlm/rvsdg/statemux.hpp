/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STATEMUX_HPP
#define JLM_RVSDG_STATEMUX_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jive {

/* mux normal form */

class mux_normal_form final : public simple_normal_form {
public:
	virtual
	~mux_normal_form() noexcept;

	mux_normal_form(
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
	set_mux_mux_reducible(bool enable);

	virtual void
	set_multiple_origin_reducible(bool enable);

	inline bool
	get_mux_mux_reducible() const noexcept
	{
		return enable_mux_mux_;
	}

	inline bool
	get_multiple_origin_reducible() const noexcept
	{
		return enable_multiple_origin_;
	}

private:
	bool enable_mux_mux_;
	bool enable_multiple_origin_;
};

/* mux operation */

class mux_op final : public simple_op {
public:
	virtual
	~mux_op() noexcept;

	inline
	mux_op(const statetype & type, size_t narguments, size_t nresults)
	: simple_op(std::vector<jive::port>(narguments, {type}),
			std::vector<jive::port>(nresults, {type}))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static jive::mux_normal_form *
	normal_form(jive::graph * graph) noexcept
	{
		return static_cast<jive::mux_normal_form*>(graph->node_normal_form(typeid(mux_op)));
	}
};

static inline bool
is_mux_op(const jive::operation & op)
{
	return dynamic_cast<const jive::mux_op*>(&op) != nullptr;
}

static inline std::vector<jive::output*>
create_state_mux(
	const jive::type & type,
	const std::vector<jive::output*> & operands,
	size_t nresults)
{
	if (operands.empty())
		throw jive::compiler_error("Insufficient number of operands.");

	auto st = dynamic_cast<const jive::statetype*>(&type);
	if (!st) throw jive::compiler_error("Expected state type.");

	auto region = operands.front()->region();
	jive::mux_op op(*st, operands.size(), nresults);
	return simple_node::create_normalized(region, op, operands);
}

static inline jive::output *
create_state_merge(
	const jive::type & type,
	const std::vector<jive::output*> & operands)
{
	return create_state_mux(type, operands, 1)[0];
}

static inline std::vector<jive::output*>
create_state_split(
	const jive::type & type,
	jive::output * operand,
	size_t nresults)
{
	return create_state_mux(type, {operand}, nresults);
}

}

#endif
