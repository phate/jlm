/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_CALL_HPP
#define JLM_LLVM_IR_OPERATORS_CALL_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/** \brief Call operation class
 *
 */
class CallOperation final : public jlm::rvsdg::SimpleOperation
{
public:
  ~CallOperation() override;

  explicit CallOperation(std::shared_ptr<const FunctionType> functionType)
      : SimpleOperation(create_srctypes(functionType), functionType->Results()),
        FunctionType_(std::move(functionType))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] const std::shared_ptr<const FunctionType> &
  GetFunctionType() const noexcept
  {
    return FunctionType_;
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<tac>
  create(
      const variable * function,
      std::shared_ptr<const FunctionType> functionType,
      const std::vector<const variable *> & arguments)
  {
    CheckFunctionInputType(function->type());

    CallOperation op(std::move(functionType));
    std::vector<const variable *> operands({ function });
    operands.insert(operands.end(), arguments.begin(), arguments.end());
    return tac::create(op, operands);
  }

private:
  static inline std::vector<std::shared_ptr<const rvsdg::Type>>
  create_srctypes(const std::shared_ptr<const FunctionType> & functionType)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ functionType });
    for (auto & argumentType : functionType->Arguments())
      types.emplace_back(argumentType);

    return types;
  }

  static void
  CheckFunctionInputType(const jlm::rvsdg::Type & type)
  {
    if (!is<FunctionType>(type))
      throw jlm::util::error("Expected function type.");
  }

  std::shared_ptr<const FunctionType> FunctionType_;
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
  [[nodiscard]] rvsdg::output &
  GetLambdaOutput() const noexcept
  {
    if (GetCallType() == CallType::NonRecursiveDirectCall)
    {
      return *Output_;
    }

    JLM_ASSERT(GetCallType() == CallType::RecursiveDirectCall);
    auto argument = jlm::util::AssertedCast<jlm::rvsdg::RegionArgument>(Output_);
    /*
     * FIXME: This assumes that all recursion variables where added before the dependencies. It
     * would be better if we did not use the index for retrieving the result, but instead
     * explicitly encoded it in an phi_argument.
     */
    return *argument->region()->result(argument->index())->origin();
  }

  /** \brief Returns the imported function.
   *
   * GetImport() only returns a valid result if the call is an external call.
   *
   * @return The imported function.
   */
  [[nodiscard]] rvsdg::RegionArgument &
  GetImport() const noexcept
  {
    JLM_ASSERT(GetCallType() == CallType::ExternalCall);
    return *jlm::util::AssertedCast<rvsdg::RegionArgument>(Output_);
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

  /**
    \brief Classify callee as non-recursive.

    \param output
      Output representing the function called (must be a lambda).

    \pre
      The given output must belong to a lambda node.
  */
  static std::unique_ptr<CallTypeClassifier>
  CreateNonRecursiveDirectCallClassifier(rvsdg::output & output)
  {
    rvsdg::AssertGetOwnerNode<lambda::node>(output);
    return std::make_unique<CallTypeClassifier>(CallType::NonRecursiveDirectCall, output);
  }

  /**
    \brief Classify callee as recursive.

    \param output
      Output representing the function called (must be phi argument).

    \pre
      The given output must belong to a phi node.
  */
  static std::unique_ptr<CallTypeClassifier>
  CreateRecursiveDirectCallClassifier(rvsdg::RegionArgument & output)
  {
    JLM_ASSERT(is<phi::rvargument>(&output));
    return std::make_unique<CallTypeClassifier>(CallType::RecursiveDirectCall, output);
  }

  /**
    \brief Classify callee as external.

    \param argument
      Output representing the function called (must be graph argument).

    \pre
      The given output must be an argument to the root region of the graph.
  */
  static std::unique_ptr<CallTypeClassifier>
  CreateExternalCallClassifier(rvsdg::RegionArgument & argument)
  {
    JLM_ASSERT(argument.region() == &argument.region()->graph()->GetRootRegion());
    return std::make_unique<CallTypeClassifier>(CallType::ExternalCall, argument);
  }

  /**
    \brief Classify callee as inderict.

    \param output
      Output representing the function called (supposed to be pointer).
  */
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
class CallNode final : public jlm::rvsdg::SimpleNode
{
private:
  CallNode(
      rvsdg::Region & region,
      std::unique_ptr<CallOperation> operation,
      const std::vector<jlm::rvsdg::output *> & operands)
      : SimpleNode(region, std::move(operation), operands)
  {}

public:
  [[nodiscard]] const CallOperation &
  GetOperation() const noexcept override
  {
    return *jlm::util::AssertedCast<const CallOperation>(&SimpleNode::GetOperation());
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

  /**
   * @return The outputs of the call node.
   */
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
    JLM_ASSERT(is<FunctionType>(functionInput->type()));
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

  /**
   *
   * @param callNode The call node for which to retrieve the CallEntryMemoryStateMergeOperation
   * node.
   * @return The CallEntryMemoryStateMergeOperation node connected to the memory state input if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateInput()
   * @see GetMemoryStateExitSplit()
   */
  [[nodiscard]] static rvsdg::SimpleNode *
  GetMemoryStateEntryMerge(const CallNode & callNode) noexcept
  {
    auto node = rvsdg::output::GetNode(*callNode.GetMemoryStateInput()->origin());
    return is<CallEntryMemoryStateMergeOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                        : nullptr;
  }

  /**
   *
   * @param callNode The call node for which to retrieve the CallExitMemoryStateSplitOperation node.
   * @return The CallExitMemoryStateSplitOperation node connected to the memory state output if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateOutput()
   * @see GetMemoryStateEntryMerge()
   */
  [[nodiscard]] static rvsdg::SimpleNode *
  GetMemoryStateExitSplit(const CallNode & callNode) noexcept
  {
    // If a memory state exit split node is present, then we would expect the node to be the only
    // user of the memory state output.
    if (callNode.GetMemoryStateOutput()->nusers() != 1)
      return nullptr;

    auto node = rvsdg::node_input::GetNode(**callNode.GetMemoryStateOutput()->begin());
    return is<CallExitMemoryStateSplitOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                       : nullptr;
  }

  Node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<jlm::rvsdg::output *>
  Create(
      rvsdg::output * function,
      std::shared_ptr<const FunctionType> functionType,
      const std::vector<rvsdg::output *> & arguments)
  {
    return CreateNode(function, std::move(functionType), arguments).Results();
  }

  static std::vector<jlm::rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<CallOperation> callOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return CreateNode(region, std::move(callOperation), operands).Results();
  }

  static CallNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<CallOperation> callOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    CheckFunctionType(*callOperation->GetFunctionType());

    return *(new CallNode(region, std::move(callOperation), operands));
  }

  static CallNode &
  CreateNode(
      rvsdg::output * function,
      std::shared_ptr<const FunctionType> functionType,
      const std::vector<rvsdg::output *> & arguments)
  {
    CheckFunctionInputType(function->type());

    auto callOperation = std::make_unique<CallOperation>(std::move(functionType));
    std::vector<rvsdg::output *> operands({ function });
    operands.insert(operands.end(), arguments.begin(), arguments.end());

    return CreateNode(*function->region(), std::move(callOperation), operands);
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
  CheckFunctionInputType(const jlm::rvsdg::Type & type)
  {
    if (!is<FunctionType>(type))
      throw jlm::util::error("Expected function type.");
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
