/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_CALL_HPP
#define JLM_IR_OPERATORS_CALL_HPP

#include <jive/rvsdg/simple-node.hpp>
#include <jive/types/function.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

namespace lambda {
	class node;
}

/* call operator */

class call_op final : public jive::simple_op {
public:
	virtual
	~call_op();

	inline
	call_op(const jive::fcttype & fcttype)
	: simple_op(create_srcports(fcttype), create_dstports(fcttype))
	{}

 	virtual bool
 	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	inline const jive::fcttype &
	fcttype() const noexcept
	{
		auto at = static_cast<const ptrtype*>(&argument(0).type());
		return *static_cast<const jive::fcttype*>(&at->pointee_type());
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::vector<jive::output*>
	create(jive::output * function, const std::vector<jive::output*> & arguments)
	{
		auto at = dynamic_cast<const ptrtype*>(&function->type());
		if (!at) throw jlm::error("expected pointer type.");

		auto ft = dynamic_cast<const jive::fcttype*>(&at->pointee_type());
		if (!ft) throw jlm::error("expected function type.");

		call_op op(*ft);
		std::vector<jive::output*> operands({function});
		operands.insert(operands.end(), arguments.begin(), arguments.end());
		return jive::simple_node::create_normalized(function->region(), op, operands);
	}

	static std::unique_ptr<tac>
	create(
		const variable * function,
		const std::vector<const variable*> & arguments,
		const std::vector<const variable*> & results)
	{
		auto at = dynamic_cast<const ptrtype*>(&function->type());
		if (!at) throw jlm::error("Expected pointer type.");

		auto ft = dynamic_cast<const jive::fcttype*>(&at->pointee_type());
		if (!ft) throw jlm::error("Expected function type.");

		call_op op(*ft);
		std::vector<const variable*> operands({function});
		operands.insert(operands.end(), arguments.begin(), arguments.end());
		return tac::create(op, operands, results);
	}

private:
	static inline std::vector<jive::port>
	create_srcports(const jive::fcttype & fcttype)
	{
		std::vector<jive::port> ports(1, {ptrtype(fcttype)});
		for (size_t n = 0; n < fcttype.narguments(); n++)
			ports.push_back(fcttype.argument_type(n));

		return ports;
	}

	static inline std::vector<jive::port>
	create_dstports(const jive::fcttype & fcttype)
	{
		std::vector<jive::port> ports;
		for (size_t n = 0; n < fcttype.nresults(); n++)
			ports.push_back(fcttype.result_type(n));

		return ports;
	}
};

/**
* \brief Traces function input of call node
*
* Traces the function input of a call node upwards, trying to
* find the corresponding lambda output. The function can handle
* invariant gamma exit variables and invariant theta loop variables.
*
* \param node A call node.
*
* \return The traced output.
*/
jive::output *
trace_function_input(const jive::simple_node & node);

/**
* \brief Checks if a node is a direct call node.
*
* \param node A simple node
*
* \return The corresponding lambda node if its a direct call, otherwise NULL.
*/
lambda::node *
is_direct_call(const jive::simple_node & node);

}

#endif
