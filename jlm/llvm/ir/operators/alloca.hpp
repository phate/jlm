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

class alloca_op final : public rvsdg::simple_op {
public:
	virtual
	~alloca_op() noexcept;

  inline
  alloca_op(
    const rvsdg::valuetype & allocatedType,
    const rvsdg::bittype & btype,
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

	virtual std::unique_ptr<rvsdg::operation>
	copy() const override;

	inline const rvsdg::bittype &
	size_type() const noexcept
	{
		return *static_cast<const rvsdg::bittype*>(&argument(0).type());
	}

	inline const rvsdg::valuetype &
	value_type() const noexcept
	{
		return *util::AssertedCast<const rvsdg::valuetype>(AllocatedType_.get());
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

	static std::unique_ptr<jlm::tac>
	create(
		const rvsdg::valuetype & allocatedType,
		const variable * size,
		size_t alignment)
	{
		auto bt = dynamic_cast<const rvsdg::bittype*>(&size->type());
		if (!bt) throw util::error("expected bits type.");

		alloca_op op(allocatedType, *bt, alignment);
		return tac::create(op, {size});
	}

	static std::vector<rvsdg::output*>
	create(
		const rvsdg::valuetype & allocatedType,
		rvsdg::output * size,
		size_t alignment)
	{
		auto bt = dynamic_cast<const rvsdg::bittype*>(&size->type());
		if (!bt) throw util::error("expected bits type.");

		alloca_op op(allocatedType, *bt, alignment);
		return rvsdg::simple_node::create_normalized(size->region(), op, {size});
	}

private:
	size_t alignment_;
  std::unique_ptr<rvsdg::type> AllocatedType_;
};

}

#endif
