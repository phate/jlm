/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_ALLOCA_HPP
#define JLM_IR_OPERATORS_ALLOCA_HPP

#include <jive/types/bitstring/type.hpp>
#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/simple-normal-form.hpp>
#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* alloca operator */

class alloca_op final : public jive::simple_op {
public:
	virtual
	~alloca_op() noexcept;

	inline
	alloca_op(
		const PointerType & atype,
		const jive::bittype & btype,
		size_t alignment)
	: simple_op({btype}, {atype, {MemoryStateType::Create()}})
	, alignment_(alignment)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::bittype &
	size_type() const noexcept
	{
		return *static_cast<const jive::bittype*>(&argument(0).type());
	}

	inline const jive::valuetype &
	value_type() const noexcept
	{
		return static_cast<const PointerType *>(&result(0).type())->GetElementType();
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static std::unique_ptr<jlm::tac>
	create(
		const jive::type & vtype,
		const variable * size,
		size_t alignment)
	{
		auto vt = dynamic_cast<const jive::valuetype*>(&vtype);
		if (!vt) throw jlm::error("expected value type.");

		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		alloca_op op(PointerType(*vt), *bt, alignment);
		return tac::create(op, {size});
	}

	static std::vector<jive::output*>
	create(
		const jive::type & type,
		jive::output * size,
		size_t alignment)
	{
		auto vt = dynamic_cast<const jive::valuetype*>(&type);
		if (!vt) throw jlm::error("expected value type.");

		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		alloca_op op(PointerType(*vt), *bt, alignment);
		return jive::simple_node::create_normalized(size->region(), op, {size});
	}

private:
	size_t alignment_;
};

}

#endif
