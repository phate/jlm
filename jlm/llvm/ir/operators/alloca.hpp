/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_ALLOCA_HPP
#define JLM_LLVM_IR_OPERATORS_ALLOCA_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm {

/* alloca operator */

class alloca_op final : public jive::simple_op {
public:
	virtual
	~alloca_op() noexcept;

  inline
  alloca_op(
    const jive::valuetype & allocatedType,
    const jive::bittype & btype,
    size_t alignment)
    : simple_op({btype}, {PointerType(), {MemoryStateType::Create()}})
    , alignment_(alignment)
    , AllocatedType_(allocatedType.copy())
  {}

  alloca_op(const alloca_op & other)
    : simple_op(other)
    , alignment_(other.alignment_)
    , AllocatedType_(other.AllocatedType_->copy())
  {}

  alloca_op(alloca_op && other) noexcept
    : simple_op(other)
    , alignment_(other.alignment_)
    , AllocatedType_(std::move(other.AllocatedType_))
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
		return *AssertedCast<const jive::valuetype>(AllocatedType_.get());
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static std::unique_ptr<jlm::tac>
	create(
		const jive::valuetype & allocatedType,
		const variable * size,
		size_t alignment)
	{
		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		alloca_op op(allocatedType, *bt, alignment);
		return tac::create(op, {size});
	}

	static std::vector<jive::output*>
	create(
		const jive::valuetype & allocatedType,
		jive::output * size,
		size_t alignment)
	{
		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		alloca_op op(allocatedType, *bt, alignment);
		return jive::simple_node::create_normalized(size->region(), op, {size});
	}

private:
	size_t alignment_;
  std::unique_ptr<jive::type> AllocatedType_;
};

}

#endif
