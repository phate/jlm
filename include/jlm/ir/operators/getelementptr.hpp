/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_GETELEMENTPTR_HPP
#define JLM_IR_OPERATORS_GETELEMENTPTR_HPP

#include <jive/types/bitstring/type.h>
#include <jive/rvsdg/simple-node.h>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* getelementptr operator */

class getelementptr_op final : public jive::simple_op {
public:
	virtual
	~getelementptr_op();

	inline
	getelementptr_op(
		const jlm::ptrtype & ptype,
		const std::vector<jive::bits::type> & btypes,
		const jlm::ptrtype & rtype)
	: jive::simple_op()
	, pport_(ptype)
	, rport_(rtype)
	{
		for (const auto & type : btypes)
			bports_.push_back(type);
	}

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

	inline size_t
	nindices() const noexcept
	{
		return bports_.size();
	}

	const jive::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&pport_.type())->pointee_type();
	}

private:
	jive::port pport_;
	jive::port rport_;
	std::vector<jive::port> bports_;
};

static inline bool
is_getelementptr_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::getelementptr_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_getelementptr_tac(
	const variable * address,
	const std::vector<const variable*> offsets,
	jlm::variable * result)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	std::vector<jive::bits::type> bts;
	for (const auto & v : offsets) {
		auto bt = dynamic_cast<const jive::bits::type*>(&v->type());
		if (!bt) throw std::logic_error("Expected bitstring type.");
		bts.push_back(*bt);
	}

	auto rt = dynamic_cast<const jlm::ptrtype*>(&result->type());
	if (!rt) throw std::logic_error("Expected pointer type.");

	jlm::getelementptr_op op(*at, bts, *rt);
	std::vector<const variable*> operands(1, address);
	operands.insert(operands.end(), offsets.begin(), offsets.end());
	return create_tac(op, operands, {result});
}

}

#endif
