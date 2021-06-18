/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_LOAD_HPP
#define JLM_IR_OPERATORS_LOAD_HPP

#include <jive/arch/addresstype.hpp>
#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/simple-normal-form.hpp>
#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* load normal form */

class load_normal_form final : public jive::simple_normal_form {
public:
	virtual
	~load_normal_form();

	load_normal_form(
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

	inline void
	set_load_mux_reducible(bool enable) noexcept
	{
		enable_load_mux_ = enable;
	}

	inline bool
	get_load_mux_reducible() const noexcept
	{
		return enable_load_mux_;
	}

	inline void
	set_load_alloca_reducible(bool enable) noexcept
	{
		enable_load_alloca_ = enable;
	}

	inline bool
	get_load_alloca_reducible() const noexcept
	{
		return enable_load_alloca_;
	}

	inline void
	set_multiple_origin_reducible(bool enable) noexcept
	{
		enable_multiple_origin_ = enable;
	}

	inline bool
	get_multiple_origin_reducible() const noexcept
	{
		return enable_multiple_origin_;
	}

	inline void
	set_load_store_state_reducible(bool enable) noexcept
	{
		enable_load_store_state_ = enable;
	}

	inline bool
	get_load_store_state_reducible() const noexcept
	{
		return enable_load_store_state_;
	}

	inline void
	set_load_store_alloca_reducible(bool enable) noexcept
	{
		enable_load_store_alloca_ = enable;
	}

	inline bool
	get_load_store_alloca_reducible() const noexcept
	{
		return enable_load_store_alloca_;
	}

	inline void
	set_load_store_reducible(bool enable) noexcept
	{
		enable_load_store_ = enable;
	}

	inline bool
	get_load_store_reducible() const noexcept
	{
		return enable_load_store_;
	}

	void
	set_load_load_state_reducible(bool enable) noexcept
	{
		enable_load_load_state_ = enable;
	}

	bool
	get_load_load_state_reducible() const noexcept
	{
		return enable_load_load_state_;
	}

private:
	bool enable_load_mux_;
	bool enable_load_store_;
	bool enable_load_alloca_;
	bool enable_load_load_state_;
	bool enable_multiple_origin_;
	bool enable_load_store_state_;
	bool enable_load_store_alloca_;
};

/* load operator */

class load_op final : public jive::simple_op {
public:
	virtual
	~load_op() noexcept;

	inline
	load_op(
		const jlm::ptrtype & ptype,
		size_t nstates,
		size_t alignment)
	: simple_op(create_ports(ptype, nstates), create_ports(ptype.pointee_type(), nstates))
	, alignment_(alignment)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::valuetype &
	pointee_type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(&result(0).type());
	}

	inline size_t
	nstates() const noexcept
	{
		return narguments()-1;
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static jlm::load_normal_form *
	normal_form(jive::graph * graph) noexcept
	{
		return static_cast<jlm::load_normal_form*>(graph->node_normal_form(typeid(load_op)));
	}

	static std::unique_ptr<jlm::tac>
	create(
		const variable * address,
		const variable * instate,
		size_t alignment)
	{
		auto pt = check_address(address->type());

		jlm::load_op op(*pt, 1, alignment);
		return tac::create(op, {address, instate});
	}

	static std::vector<jive::output*>
	create(
		jive::output * address,
		const std::vector<jive::output*> & states,
		size_t alignment)
	{
		auto pt = check_address(address->type());

		if (states.size() == 0)
			throw jlm::error("Expected at least one memory state.");

		std::vector<jive::output*> operands({address});
		operands.insert(operands.end(), states.begin(), states.end());

		jlm::load_op op(*pt, states.size(), alignment);
		return jive::simple_node::create_normalized(address->region(), op, operands);
	}

private:
	static const jlm::ptrtype *
	check_address(const jive::type & type)
	{
		auto pt = dynamic_cast<const jlm::ptrtype*>(&type);
		if (!pt) throw jlm::error("Expected pointer type.");

		return pt;
	}

	static std::vector<jive::port>
	create_ports(const jive::valuetype & vtype, size_t nstates)
	{
		std::vector<jive::port> ports(1, {vtype});
		std::vector<jive::port> states(nstates, {jive::memtype::instance()});
		ports.insert(ports.end(), states.begin(), states.end());
		return ports;
	}

	size_t alignment_;
};

}

#endif
