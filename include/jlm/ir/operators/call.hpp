/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_CALL_HPP
#define JLM_IR_OPERATORS_CALL_HPP

#include <jive/rvsdg/simple-node.h>
#include <jive/types/function/fcttype.h>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* call operator */

class call_op final : public jive::simple_op {
public:
	virtual
	~call_op();

	inline
	call_op(const jive::fct::type & fcttype)
	: simple_op()
	{
		arguments_.push_back(ptrtype(fcttype));
		for (size_t n = 0; n < fcttype.narguments(); n++)
			arguments_.push_back(fcttype.argument_type(n));
		for (size_t n = 0; n < fcttype.nresults(); n++)
			results_.push_back(fcttype.result_type(n));
	}

	virtual bool
	operator==(const operation & other) const noexcept;

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

	inline const jive::fct::type &
	fcttype() const noexcept
	{
		auto at = static_cast<const ptrtype*>(&argument(0).type());
		return *static_cast<const jive::fct::type*>(&at->pointee_type());
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::vector<jive::output*>
	create(jive::output * function, const std::vector<jive::output*> & arguments)
	{
		auto at = dynamic_cast<const ptrtype*>(&function->type());
		if (!at) throw std::logic_error("Expected pointer type.");

		auto ft = dynamic_cast<const jive::fct::type*>(&at->pointee_type());
		if (!ft) throw std::logic_error("Expected function type.");

		call_op op(*ft);
		std::vector<jive::output*> operands({function});
		operands.insert(operands.end(), arguments.begin(), arguments.end());
		return create_normalized(function->region(), op, operands);
	}

private:
	std::vector<jive::port> results_;
	std::vector<jive::port> arguments_;
};

static inline bool
is_call_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const call_op*>(&op) != nullptr;
}

static inline bool
is_call_node(const jive::node * node) noexcept
{
	return jive::is_opnode<call_op>(node);
}

static inline std::unique_ptr<tac>
create_call_tac(
	const variable * function,
	const std::vector<const variable*> & arguments,
	const std::vector<const variable*> & results)
{
	auto at = dynamic_cast<const ptrtype*>(&function->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	auto ft = dynamic_cast<const jive::fct::type*>(&at->pointee_type());
	if (!ft) throw std::logic_error("Expected function type.");

	call_op op(*ft);
	std::vector<const variable*> operands({function});
	operands.insert(operands.end(), arguments.begin(), arguments.end());
	return create_tac(op, operands, results);
}

static inline std::vector<jive::output*>
create_call(
	jive::output * function,
	const std::vector<jive::output*> & arguments)
{
	auto at = dynamic_cast<const ptrtype*>(&function->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	auto ft = dynamic_cast<const jive::fct::type*>(&at->pointee_type());
	if (!ft) throw std::logic_error("Expected function type.");

	call_op op(*ft);
	std::vector<jive::output*> operands({function});
	operands.insert(operands.end(), arguments.begin(), arguments.end());
	return jive::create_normalized(function->region(), op, operands);
}

}

#endif
