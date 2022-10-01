/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_CALL_HPP
#define JLM_IR_OPERATORS_CALL_HPP

#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

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

	[[nodiscard]] std::string
	debug_string() const override;

	[[nodiscard]] const FunctionType &
	GetFunctionType() const noexcept
	{
    auto pointerType = AssertedCast<const PointerType>(&argument(0).type());
    return *AssertedCast<const FunctionType>(&pointerType->GetElementType());
	}

	[[nodiscard]] std::unique_ptr<jive::operation>
	copy() const override;

	static std::unique_ptr<tac>
	create(
		const variable * function,
		const std::vector<const variable*> & arguments)
	{
		auto at = dynamic_cast<const PointerType*>(&function->type());
		if (!at) throw jlm::error("Expected pointer type.");

		auto ft = dynamic_cast<const FunctionType*>(&at->GetElementType());
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
		std::vector<jive::port> ports(1, {PointerType(functionType)});
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

/** \brief Call node classifier
 *
 * The CallTypeClassifier class provides information about the call type of a call node. It currently distinguishes
 * between three different call types:
 * 1. Direct call - A call to a statically visible function within the module.
 * 2. External call - A call to an imported function, i.e., a function from outside of the module.
 * 3. Indirect call - A call to a statically not visible function.
 */
class CallTypeClassifier final {
public:
  enum class CallType {
    DirectCall,
    ExternalCall,
    IndirectCall
  };

  CallTypeClassifier(
    CallType callType,
    jive::output & output)
  : CallType_(callType)
  , Output_(&output)
  {}

  /** \brief Return call type.
   *
   */
  [[nodiscard]] CallType
  GetCallType() const noexcept
  {
    return CallType_;
  }

  /** \brief Determines whether call is a direct call.
   *
   * @return True if call is a direct call, otherwise false.
   */
  [[nodiscard]] bool
  IsDirectCall() const noexcept
  {
    return GetCallType() == CallType::DirectCall;
  }

  /** \brief Determines whether call is an external call.
   *
   * @return True if call is an external call, otherwise false.
   */
  [[nodiscard]] bool
  IsExternalCall() const noexcept
  {
    return GetCallType() == CallType::ExternalCall;
  }

  [[nodiscard]] bool
  IsIndirectCall() const noexcept
  {
    return GetCallType() == CallType::IndirectCall;
  }

  /** \brief Returns the called function.
   *
   * GetLambdaOutput() only returns a valid result if the call node is a direct call.
   *
   * @return The called function.
   */
  [[nodiscard]] lambda::output &
  GetLambdaOutput() const noexcept
  {
    JLM_ASSERT(GetCallType() == CallType::DirectCall);
    return *AssertedCast<lambda::output>(Output_);
  }

  /** \brief Returns the imported function.
   *
   * GetImport() only returns a valid result if the call is an external call.
   *
   * @return The imported function.
   */
  [[nodiscard]] jive::argument &
  GetImport() const noexcept
  {
    JLM_ASSERT(GetCallType() == CallType::ExternalCall);
    return *AssertedCast<jive::argument>(Output_);
  }

  /** \brief Return origin of a call node's function input.
   *
   * GetFunctionOrigin() returns the last output returned by CallNode::TraceFunctionInput().
   *
   * @return Traced origin of a call node's function input.
   *
   * @see CallNode::TraceFunctionInput(), CallNode::GetFunctionInput()
   */
  [[nodiscard]] jive::output &
  GetFunctionOrigin() const noexcept
  {
    return *Output_;
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateDirectCallClassifier(jive::output & output)
  {
    return std::make_unique<CallTypeClassifier>(CallType::DirectCall, output);
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateExternalCallClassifier(jive::output & output)
  {
    return std::make_unique<CallTypeClassifier>(CallType::ExternalCall, output);
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateIndirectCallClassifier(jive::output & output)
  {
    return std::make_unique<CallTypeClassifier>(CallType::IndirectCall, output);
  }

private:
  CallType CallType_;
  jive::output * Output_;
};

/** \brief Call node
 *
 */
class CallNode final : public jive::simple_node {
private:
  CallNode(
    jive::region & region,
    const CallOperation & operation,
    const std::vector<jive::output*> & operands)
    : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const CallOperation&
  GetOperation() const noexcept
  {
    return *AssertedCast<const CallOperation>(&operation());
  }

  [[nodiscard]] size_t
  NumArguments() const noexcept
  {
    return ninputs()-1;
  }

  [[nodiscard]] jive::input *
  Argument(size_t n) const
  {
    return input(n);
  }

  [[nodiscard]] size_t
  NumResults() const noexcept
  {
    return noutputs();
  }

  [[nodiscard]] jive::output *
  Result(size_t n) const noexcept
  {
    return output(n);
  }

  [[nodiscard]] jive::input *
  GetFunctionInput() const noexcept
  {
    auto function = input(0);
    auto pointerType = AssertedCast<const PointerType>(&function->type());
    JLM_ASSERT(is<FunctionType>(pointerType->GetElementType()));
    return function;
  }

  [[nodiscard]] jive::input *
  GetIoStateInput() const noexcept
  {
    auto iOState = input(ninputs()-3);
    JLM_ASSERT(is<iostatetype>(iOState->type()));
    return iOState;
  }

  [[nodiscard]] jive::input *
  GetMemoryStateInput() const noexcept
  {
    auto memoryState = input(ninputs()-2);
    JLM_ASSERT(is<MemoryStateType>(memoryState->type()));
    return memoryState;
  }

  [[nodiscard]] jive::input *
  GetLoopStateInput() const noexcept
  {
    auto loopState = input(ninputs()-1);
    JLM_ASSERT(is<loopstatetype>(loopState->type()));
    return loopState;
  }

  [[nodiscard]] jive::output *
  GetIoStateOutput() const noexcept
  {
    auto iOState = output(noutputs()-3);
    JLM_ASSERT(is<iostatetype>(iOState->type()));
    return iOState;
  }

  [[nodiscard]] jive::output *
  GetMemoryStateOutput() const noexcept
  {
    auto memoryState = output(noutputs()-2);
    JLM_ASSERT(is<MemoryStateType>(memoryState->type()));
    return memoryState;
  }

  [[nodiscard]] jive::output *
  GetLoopStateOutput() const noexcept
  {
    auto loopState = output(noutputs()-1);
    JLM_ASSERT(is<loopstatetype>(loopState->type()));
    return loopState;
  }

  static std::vector<jive::output*>
  Create(
    jive::output * function,
    const std::vector<jive::output*> & arguments)
  {
    auto functionType = ExtractFunctionType(function);
    CheckFunctionType(functionType);

    CallOperation callOperation(functionType);
    std::vector<jive::output*> operands({function});
    operands.insert(operands.end(), arguments.begin(), arguments.end());

    return jive::outputs(new CallNode(
      *function->region(),
      callOperation,
      operands));
  }

  static std::vector<jive::output*>
  Create(
    jive::region & region,
    const CallOperation & callOperation,
    const std::vector<jive::output*> & operands)
  {
    CheckFunctionType(callOperation.GetFunctionType());

    return jive::outputs(new CallNode(
      region,
      callOperation,
      operands));
  }

  /**
  * \brief Traces function input of call node
  *
  * Traces the function input of a call node upwards, trying to
  * find the corresponding lambda output. The function can handle
  * invariant gamma exit variables and invariant theta loop variables.
  *
  * \param callNode A call node.
  *
  * \return The traced output.
  */
  static jive::output *
  TraceFunctionInput(const CallNode & callNode);

  /** \brief Classifies a call node.
   *
   * Classifies a call node according to its call type.
   *
   * @param callNode A call node.
   * @return A CallTypeClassifier.
   */
  static std::unique_ptr<CallTypeClassifier>
  ClassifyCall(const CallNode & callNode);

private:
  [[nodiscard]] static const FunctionType &
  ExtractFunctionType(const jive::output * function)
  {
    auto pointerType = dynamic_cast<const PointerType*>(&function->type());
    if (!pointerType)
      throw jlm::error("Expected pointer type.");

    auto functionType = dynamic_cast<const FunctionType*>(&pointerType->GetElementType());
    if (!functionType)
      throw jlm::error("Expected function type.");

    return *functionType;
  }

  static void
  CheckFunctionType(const FunctionType & functionType)
  {
    auto CheckArgumentTypes = [](const FunctionType & functionType)
    {
      if (functionType.NumArguments() < 3)
        throw error("Expected at least three argument types.");

      auto loopStateArgumentIndex = functionType.NumArguments()-1;
      auto memoryStateArgumentIndex = functionType.NumArguments()-2;
      auto iOStateArgumentIndex = functionType.NumArguments()-3;

      if (!is<loopstatetype>(functionType.ArgumentType(loopStateArgumentIndex)))
        throw error("Expected loop state type.");

      if (!is<MemoryStateType>(functionType.ArgumentType(memoryStateArgumentIndex)))
        throw error("Expected memory state type.");

      if (!is<iostatetype>(functionType.ArgumentType(iOStateArgumentIndex)))
        throw error("Expected IO state type.");
    };

    auto CheckResultTypes = [](const FunctionType & functionType)
    {
      if (functionType.NumResults() < 3)
        throw error("Expected at least three result types.");

      auto loopStateResultIndex = functionType.NumResults()-1;
      auto memoryStateResultIndex = functionType.NumResults()-2;
      auto iOStateResultIndex = functionType.NumResults()-3;

      if (!is<loopstatetype>(functionType.ResultType(loopStateResultIndex)))
        throw error("Expected loop state type.");

      if (!is<MemoryStateType>(functionType.ResultType(memoryStateResultIndex)))
        throw error("Expected memory state type.");

      if (!is<iostatetype>(functionType.ResultType(iOStateResultIndex)))
        throw error("Expected IO state type.");
    };

    CheckArgumentTypes(functionType);
    CheckResultTypes(functionType);
  }
};

}

#endif
