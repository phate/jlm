/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_CALL_HPP
#define JLM_LLVM_IR_OPERATORS_CALL_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/** \brief Call operation class
 *
 */
class CallOperation final : public jlm::rvsdg::simple_op
{
public:
  ~CallOperation() override;

  explicit CallOperation(const FunctionType & functionType)
      : simple_op(create_srcports(functionType), create_dstports(functionType)),
        FunctionType_(functionType)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] const FunctionType &
  GetFunctionType() const noexcept
  {
    return FunctionType_;
  }

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<tac>
  create(
      const variable * function,
      const FunctionType & functionType,
      const std::vector<const variable *> & arguments)
  {
    CheckFunctionInputType(function->type());

    CallOperation op(functionType);
    std::vector<const variable *> operands({ function });
    operands.insert(operands.end(), arguments.begin(), arguments.end());
    return tac::create(op, operands);
  }

private:
  static inline std::vector<jlm::rvsdg::port>
  create_srcports(const FunctionType & functionType)
  {
    std::vector<jlm::rvsdg::port> ports(1, { PointerType() });
    for (auto & argumentType : functionType.Arguments())
      ports.emplace_back(argumentType);

    return ports;
  }

  static inline std::vector<jlm::rvsdg::port>
  create_dstports(const FunctionType & functionType)
  {
    std::vector<jlm::rvsdg::port> ports;
    for (auto & resultType : functionType.Results())
      ports.emplace_back(resultType);

    return ports;
  }

  static void
  CheckFunctionInputType(const jlm::rvsdg::type & type)
  {
    if (!is<PointerType>(type))
      throw jlm::util::error("Expected pointer type.");
  }

  FunctionType FunctionType_;
};

/** \brief Call node classifier
 *
 * The CallTypeClassifier class provides information about the call type of a call node.
 */
class CallTypeClassifier final
{
public:
  enum class CallType
  {
    /**
     * A call to a statically visible function within the module that is not part of a mutual
     * recursive call chain.
     */
    NonRecursiveDirectCall,

    /**
     * A call to a statically visible function within the module that is part of a mutual recursive
     * call chain.
     */
    RecursiveDirectCall,

    /**
     * A call to an imported function, i.e., a function from outside of the module.
     */
    ExternalCall,

    /**
     * A call to a statically not visible function.
     */
    IndirectCall
  };

  CallTypeClassifier(CallType callType, jlm::rvsdg::output & output)
      : CallType_(callType),
        Output_(&output)
  {}

  /** \brief Return call type.
   *
   */
  [[nodiscard]] CallType
  GetCallType() const noexcept
  {
    return CallType_;
  }

  /** \brief Determines whether call is a non-recursive direct call.
   *
   * @return True if call is a non-recursive direct call, otherwise false.
   */
  [[nodiscard]] bool
  IsNonRecursiveDirectCall() const noexcept
  {
    return GetCallType() == CallType::NonRecursiveDirectCall;
  }

  /** \brief Determines whether call is a recursive direct call.
   *
   * @return True if call is a recursive direct call, otherwise false.
   */
  [[nodiscard]] bool
  IsRecursiveDirectCall() const noexcept
  {
    return GetCallType() == CallType::RecursiveDirectCall;
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

  /** \brief Determines whether call is an indirect call.
   *
   * @return True if call is an indirect call, otherwise false.
   */
  [[nodiscard]] bool
  IsIndirectCall() const noexcept
  {
    return GetCallType() == CallType::IndirectCall;
  }

  /** \brief Returns the called function.
   *
   * GetLambdaOutput() only returns a valid result if the call node is a (non-)recursive direct
   * call.
   *
   * @return The called function.
   */
  [[nodiscard]] lambda::output &
  GetLambdaOutput() const noexcept
  {
    if (GetCallType() == CallType::NonRecursiveDirectCall)
    {
      return *jlm::util::AssertedCast<lambda::output>(Output_);
    }

    JLM_ASSERT(GetCallType() == CallType::RecursiveDirectCall);
    auto argument = jlm::util::AssertedCast<jlm::rvsdg::argument>(Output_);
    /*
     * FIXME: This assumes that all recursion variables where added before the dependencies. It
     * would be better if we did not use the index for retrieving the result, but instead
     * explicitly encoded it in an phi_argument.
     */
    return *jlm::util::AssertedCast<lambda::output>(
        argument->region()->result(argument->index())->origin());
  }

  /** \brief Returns the imported function.
   *
   * GetImport() only returns a valid result if the call is an external call.
   *
   * @return The imported function.
   */
  [[nodiscard]] jlm::rvsdg::argument &
  GetImport() const noexcept
  {
    JLM_ASSERT(GetCallType() == CallType::ExternalCall);
    return *jlm::util::AssertedCast<jlm::rvsdg::argument>(Output_);
  }

  /** \brief Return origin of a call node's function input.
   *
   * GetFunctionOrigin() returns the last output returned by CallNode::TraceFunctionInput().
   *
   * @return Traced origin of a call node's function input.
   *
   * @see CallNode::TraceFunctionInput(), CallNode::GetFunctionInput()
   */
  [[nodiscard]] jlm::rvsdg::output &
  GetFunctionOrigin() const noexcept
  {
    return *Output_;
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateNonRecursiveDirectCallClassifier(lambda::output & output)
  {
    return std::make_unique<CallTypeClassifier>(CallType::NonRecursiveDirectCall, output);
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateRecursiveDirectCallClassifier(jlm::rvsdg::argument & output)
  {
    JLM_ASSERT(is_phi_recvar_argument(&output));
    return std::make_unique<CallTypeClassifier>(CallType::RecursiveDirectCall, output);
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateExternalCallClassifier(jlm::rvsdg::argument & argument)
  {
    JLM_ASSERT(argument.region() == argument.region()->graph()->root());
    return std::make_unique<CallTypeClassifier>(CallType::ExternalCall, argument);
  }

  static std::unique_ptr<CallTypeClassifier>
  CreateIndirectCallClassifier(jlm::rvsdg::output & output)
  {
    return std::make_unique<CallTypeClassifier>(CallType::IndirectCall, output);
  }

private:
  CallType CallType_;
  jlm::rvsdg::output * Output_;
};

/** \brief Call node
 *
 */
class CallNode final : public jlm::rvsdg::simple_node
{
private:
  CallNode(
      jlm::rvsdg::region & region,
      const CallOperation & operation,
      const std::vector<jlm::rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const CallOperation &
  GetOperation() const noexcept
  {
    return *jlm::util::AssertedCast<const CallOperation>(&operation());
  }

  /**
   * @return The number of arguments to the call.
   *
   * \note This is equivalent to ninputs() - 1 as NumArguments() ignores the function input.
   */
  [[nodiscard]] size_t
  NumArguments() const noexcept
  {
    return ninputs() - 1;
  }

  /**
   * @param n The index of the function argument.
   * @return The input for the given index \p n.
   */
  [[nodiscard]] jlm::rvsdg::input *
  Argument(size_t n) const
  {
    JLM_ASSERT(n < NumArguments());
    return input(n + 1);
  }

  /**
   * @return The number of results from the call.
   */
  [[nodiscard]] size_t
  NumResults() const noexcept
  {
    return noutputs();
  }

  /**
   * @param n The index of the function result.
   * @return The output for the given index \p n.
   */
  [[nodiscard]] jlm::rvsdg::output *
  Result(size_t n) const noexcept
  {
    JLM_ASSERT(n < NumResults());
    return output(n);
  }

  [[nodiscard]] std::vector<rvsdg::output *>
  Results() const noexcept
  {
    return rvsdg::outputs(this);
  }

  /**
   * @return The call node's function input.
   */
  [[nodiscard]] jlm::rvsdg::input *
  GetFunctionInput() const noexcept
  {
    auto functionInput = input(0);
    JLM_ASSERT(is<PointerType>(functionInput->type()));
    return functionInput;
  }

  /**
   * @return The call node's input/output state input.
   */
  [[nodiscard]] jlm::rvsdg::input *
  GetIoStateInput() const noexcept
  {
    auto iOState = input(ninputs() - 2);
    JLM_ASSERT(is<iostatetype>(iOState->type()));
    return iOState;
  }

  /**
   * @return The call node's memory state input.
   */
  [[nodiscard]] jlm::rvsdg::input *
  GetMemoryStateInput() const noexcept
  {
    auto memoryState = input(ninputs() - 1);
    JLM_ASSERT(is<MemoryStateType>(memoryState->type()));
    return memoryState;
  }

  /**
   * @return The call node's input/output state output.
   */
  [[nodiscard]] jlm::rvsdg::output *
  GetIoStateOutput() const noexcept
  {
    auto iOState = output(noutputs() - 2);
    JLM_ASSERT(is<iostatetype>(iOState->type()));
    return iOState;
  }

  /**
   * @return The call node's memory state output.
   */
  [[nodiscard]] jlm::rvsdg::output *
  GetMemoryStateOutput() const noexcept
  {
    auto memoryState = output(noutputs() - 1);
    JLM_ASSERT(is<MemoryStateType>(memoryState->type()));
    return memoryState;
  }

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<jlm::rvsdg::output *>
  Create(
      rvsdg::output * function,
      const FunctionType & functionType,
      const std::vector<rvsdg::output *> & arguments)
  {
    return CreateNode(function, functionType, arguments).Results();
  }

  static std::vector<jlm::rvsdg::output *>
  Create(
      rvsdg::region & region,
      const CallOperation & callOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return CreateNode(region, callOperation, operands).Results();
  }

  static CallNode &
  CreateNode(
      rvsdg::region & region,
      const CallOperation & callOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    CheckFunctionType(callOperation.GetFunctionType());

    return *(new CallNode(region, callOperation, operands));
  }

  static CallNode &
  CreateNode(
      rvsdg::output * function,
      const FunctionType & functionType,
      const std::vector<rvsdg::output *> & arguments)
  {
    CheckFunctionInputType(function->type());

    CallOperation callOperation(functionType);
    std::vector<rvsdg::output *> operands({ function });
    operands.insert(operands.end(), arguments.begin(), arguments.end());

    return CreateNode(*function->region(), callOperation, operands);
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
  static jlm::rvsdg::output *
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
  static void
  CheckFunctionInputType(const jlm::rvsdg::type & type)
  {
    if (!is<PointerType>(type))
      throw jlm::util::error("Expected pointer type.");
  }

  static void
  CheckFunctionType(const FunctionType & functionType)
  {
    auto CheckArgumentTypes = [](const FunctionType & functionType)
    {
      if (functionType.NumArguments() < 2)
        throw jlm::util::error("Expected at least three argument types.");

      auto memoryStateArgumentIndex = functionType.NumArguments() - 1;
      auto iOStateArgumentIndex = functionType.NumArguments() - 2;

      if (!is<MemoryStateType>(functionType.ArgumentType(memoryStateArgumentIndex)))
        throw jlm::util::error("Expected memory state type.");

      if (!is<iostatetype>(functionType.ArgumentType(iOStateArgumentIndex)))
        throw jlm::util::error("Expected IO state type.");
    };

    auto CheckResultTypes = [](const FunctionType & functionType)
    {
      if (functionType.NumResults() < 2)
        throw jlm::util::error("Expected at least three result types.");

      auto memoryStateResultIndex = functionType.NumResults() - 1;
      auto iOStateResultIndex = functionType.NumResults() - 2;

      if (!is<MemoryStateType>(functionType.ResultType(memoryStateResultIndex)))
        throw jlm::util::error("Expected memory state type.");

      if (!is<iostatetype>(functionType.ResultType(iOStateResultIndex)))
        throw jlm::util::error("Expected IO state type.");
    };

    CheckArgumentTypes(functionType);
    CheckResultTypes(functionType);
  }
};

}

#endif
