/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_CALL_HPP
#define JLM_IR_OPERATORS_CALL_HPP

#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

namespace lambda {
	class node;
}

/** \brief Call operation class
 *
 */
class CallOperation final : public jive::simple_op {
public:
	~CallOperation() override;

	explicit
	CallOperation(const FunctionType & functionType)
	: simple_op(create_srcports(functionType), create_dstports(functionType))
	{}

	bool
 	operator==(const operation & other) const noexcept override;

	std::string
	debug_string() const override;

	const FunctionType &
	GetFunctionType() const noexcept
	{
		auto at = static_cast<const ptrtype*>(&argument(0).type());
		return *static_cast<const FunctionType*>(&at->pointee_type());
	}

	std::unique_ptr<jive::operation>
	copy() const override;

	static std::vector<jive::output*>
	create(jive::output * function, const std::vector<jive::output*> & arguments)
	{
		auto at = dynamic_cast<const ptrtype*>(&function->type());
		if (!at) throw jlm::error("expected pointer type.");

		auto ft = dynamic_cast<const FunctionType*>(&at->pointee_type());
		if (!ft) throw jlm::error("expected function type.");

		CallOperation op(*ft);
		std::vector<jive::output*> operands({function});
		operands.insert(operands.end(), arguments.begin(), arguments.end());
		return jive::simple_node::create_normalized(function->region(), op, operands);
	}

	static std::unique_ptr<tac>
	create(
		const variable * function,
		const std::vector<const variable*> & arguments)
	{
		auto at = dynamic_cast<const ptrtype*>(&function->type());
		if (!at) throw jlm::error("Expected pointer type.");

		auto ft = dynamic_cast<const FunctionType*>(&at->pointee_type());
		if (!ft) throw jlm::error("Expected function type.");

		CallOperation op(*ft);
		std::vector<const variable*> operands({function});
		operands.insert(operands.end(), arguments.begin(), arguments.end());
		return tac::create(op, operands);
	}

private:
	static inline std::vector<jive::port>
	create_srcports(const FunctionType & functionType)
	{
		std::vector<jive::port> ports(1, {ptrtype(functionType)});
    for (auto & argumentType : functionType.Arguments())
			ports.emplace_back(argumentType);

		return ports;
	}

	static inline std::vector<jive::port>
	create_dstports(const FunctionType & functionType)
	{
		std::vector<jive::port> ports;
    for (auto & resultType : functionType.Results())
			ports.emplace_back(resultType);

		return ports;
	}
};

/** \brief Call node
 *
 */
class CallNode final : public jive::simple_node {
private:
  CallNode(
    jive::region * region,
    CallOperation & operation,
    const std::vector<jive::output*> & operands)
    : simple_node(region, operation, operands)
  {}

public:
  const CallOperation&
  GetOperation() const noexcept
  {
    return *static_cast<const CallOperation*>(&operation());
  }

  size_t
  NumArguments() const noexcept
  {
    return ninputs()-1;
  }

  size_t
  NumResults() const noexcept
  {
    return noutputs();
  }

  jive::input *
  GetFunctionInput() const noexcept
  {
    auto function = input(0);
    JLM_ASSERT(is<FunctionType>(function->type()));
    return function;
  }

  static std::vector<jive::output*>
  Create(
    jive::output * function,
    std::vector<jive::output*> arguments)
  {
    auto functionType = CheckAndExtractFunctionType(function);

    CallOperation callOperation(functionType);
    std::vector<jive::output*> operands({function});
    operands.insert(operands.end(), arguments.begin(), arguments.end());
    return jive::simple_node::create_normalized(function->region(), callOperation, operands);
  }

  static std::vector<jive::output*>
  Create(std::vector<jive::output*> operands)
  {
    JLM_ASSERT(!operands.empty());
    return Create(operands[0], {std::next(operands.begin()), operands.end()});
  }

private:
  static const FunctionType &
  CheckAndExtractFunctionType(const jive::output * function)
  {
    auto pointerType = dynamic_cast<const ptrtype*>(&function->type());
    if (!pointerType)
      throw jlm::error("Expected pointer type.");

    auto functionType = dynamic_cast<const FunctionType*>(&pointerType->pointee_type());
    if (!functionType)
      throw jlm::error("Expected function type.");

    return *functionType;
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
