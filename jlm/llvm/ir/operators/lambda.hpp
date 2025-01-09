/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LAMBDA_HPP
#define JLM_LLVM_IR_OPERATORS_LAMBDA_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/iterator_range.hpp>

#include <optional>
#include <utility>

namespace jlm::llvm
{

class CallNode;

namespace lambda
{

/** \brief Lambda operation
 *
 * A lambda operation determines a lambda's name and \ref FunctionType "function type".
 */
class operation final : public rvsdg::StructuralOperation
{
public:
  ~operation() override;

  operation(
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      std::string name,
      const jlm::llvm::linkage & linkage,
      jlm::llvm::attributeset attributes)
      : type_(std::move(type)),
        name_(std::move(name)),
        linkage_(linkage),
        attributes_(std::move(attributes))
  {}

  operation(const operation & other) = default;

  operation(operation && other) noexcept = default;

  operation &
  operator=(const operation & other) = default;

  operation &
  operator=(operation && other) noexcept = default;

  [[nodiscard]] const jlm::llvm::FunctionType &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const jlm::llvm::FunctionType> &
  Type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

private:
  std::shared_ptr<const jlm::llvm::FunctionType> type_;
  std::string name_;
  jlm::llvm::linkage linkage_;
  jlm::llvm::attributeset attributes_;
};

/** \brief Lambda node
 *
 * A lambda node represents a lambda expression in the RVSDG. Its creation requires the invocation
 * of two functions: \ref create() and \ref finalize(). First, a node with only the function
 * arguments is created by invoking \ref create(). The free variables of the lambda expression can
 * then be added to the lambda node using the \ref AddContextVar() method, and the body of the
 * lambda node can be created. Finally, the lambda node can be finalized by invoking \ref
 * finalize().
 *
 * The following snippet illustrates the creation of lambda nodes:
 *
 * \code{.cpp}
 *   auto lambda = lambda::node::create(...);
 *   ...
 *   auto cv1 = lambda->AddContextVar(...);
 *   auto cv2 = lambda->AddContextVar(...);
 *   ...
 *   // generate lambda body
 *   ...
 *   auto output = lambda->finalize(...);
 * \endcode
 */
class node final : public rvsdg::StructuralNode
{
public:
  class CallSummary;

  ~node() override;

private:
  node(rvsdg::Region & parent, lambda::operation op);

public:
  /**
   * \brief Bound context variable
   *
   * Context variables may be bound at the point of creation of a
   * lambda abstraction. These are represented as inputs to the
   * lambda node itself, and made accessible to the body of the
   * lambda in the form of an initial argument to the subregion.
   */
  struct ContextVar
  {
    /**
     * \brief Input variable bound into lambda node
     *
     * The input port into the lambda node that supplies the value
     * of the context variable bound into the lambda at the
     * time the lambda abstraction is built.
     */
    rvsdg::input * input;

    /**
     * \brief Access to bound object in subregion.
     *
     * Supplies access to the value bound into the lambda abstraction
     * from inside the region contained in the lambda node. This
     * evaluates to the value bound into the lambda.
     */
    rvsdg::output * inner;
  };

  [[nodiscard]] std::vector<rvsdg::output *>
  GetFunctionArguments() const;

  [[nodiscard]] std::vector<rvsdg::input *>
  GetFunctionResults() const;

  [[nodiscard]] const jlm::llvm::attributeset &
  GetArgumentAttributes(const rvsdg::output & argument) const noexcept;

  void
  SetArgumentAttributes(rvsdg::output & argument, const jlm::llvm::attributeset & attributes);

  [[nodiscard]] rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] const lambda::operation &
  GetOperation() const noexcept override;

  [[nodiscard]] const jlm::llvm::FunctionType &
  type() const noexcept
  {
    return GetOperation().type();
  }

  [[nodiscard]] const std::shared_ptr<const jlm::llvm::FunctionType> &
  Type() const noexcept
  {
    return GetOperation().Type();
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return GetOperation().name();
  }

  [[nodiscard]] const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return GetOperation().linkage();
  }

  [[nodiscard]] const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return GetOperation().attributes();
  }

  /**
   * \brief Adds a context/free variable to the lambda node.
   *
   * \param origin
   *   The value to be bound into the lambda node.
   *
   * \pre
   *   \p origin must be from the same region as the lambda node.
   *
   * \return The context variable argument of the lambda abstraction.
   */
  ContextVar
  AddContextVar(jlm::rvsdg::output & origin);

  /**
   * \brief Maps input to context variable.
   *
   * \param input
   *   Input to the lambda node.
   *
   * \returns
   *   The context variable description corresponding to the input.
   *
   * \pre
   *   \p input must be input to this node.
   *
   * Returns the context variable description corresponding
   * to this input of the lambda node. All inputs to the lambda
   * node are by definition bound context variables that are
   * accessible in the subregion through the corresponding
   * argument.
   */
  [[nodiscard]] ContextVar
  MapInputContextVar(const rvsdg::input & input) const noexcept;

  /**
   * \brief Maps bound variable reference to context variable
   *
   * \param output
   *   Region argument to lambda subregion
   *
   * \returns
   *   The context variable description corresponding to the argument
   *
   * \pre
   *   \p output must be an argument to the subregion of this node
   *
   * Returns the context variable description corresponding
   * to this bound variable reference in the lambda node region.
   * Note that some arguments of the region are formal call arguments
   * and do not have an associated context variable description.
   */
  [[nodiscard]] std::optional<ContextVar>
  MapBinderContextVar(const rvsdg::output & output) const noexcept;

  /**
   * \brief Gets all bound context variables
   *
   * \returns
   *   The context variable descriptions.
   *
   * Returns all context variable descriptions.
   */
  [[nodiscard]] std::vector<ContextVar>
  GetContextVars() const noexcept;

  /**
   * Remove lambda inputs and their respective arguments.
   *
   * An input must match the condition specified by \p match and its argument must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const rvsdg::input&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed inputs.
   */
  template<typename F>
  size_t
  RemoveLambdaInputsWhere(const F & match);

  /**
   * Remove all dead inputs.
   *
   * @return The number of removed inputs.
   *
   * \see RemoveLambdaInputsWhere()
   */
  size_t
  PruneLambdaInputs()
  {
    auto match = [](const rvsdg::input &)
    {
      return true;
    };

    return RemoveLambdaInputsWhere(match);
  }

  [[nodiscard]] rvsdg::output *
  output() const noexcept;

  lambda::node *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const override;

  lambda::node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * @return The memory state argument of the lambda subregion.
   */
  [[nodiscard]] rvsdg::output &
  GetMemoryStateRegionArgument() const noexcept;

  /**
   * @return The memory state result of the lambda subregion.
   */
  [[nodiscard]] rvsdg::input &
  GetMemoryStateRegionResult() const noexcept;

  /**
   *
   * @param lambdaNode The lambda node for which to retrieve the
   * LambdaEntryMemoryStateSplitOperation node.
   * @return The LambdaEntryMemoryStateSplitOperation node connected to the memory state input if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateExitMerge()
   */
  static rvsdg::SimpleNode *
  GetMemoryStateEntrySplit(const lambda::node & lambdaNode) noexcept;

  /**
   *
   * @param lambdaNode The lambda node for which to retrieve the
   * LambdaExitMemoryStateMergeOperation node.
   * @return The LambdaExitMemoryStateMergeOperation node connected to the memory state output if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateEntrySplit()
   */
  [[nodiscard]] static rvsdg::SimpleNode *
  GetMemoryStateExitMerge(const lambda::node & lambdaNode) noexcept;

  /**
   * Creates a lambda node in the region \p parent with the function type \p type and name \p name.
   * After the invocation of \ref create(), the lambda node only features the function arguments.
   * Free variables can be added to the function node using \ref AddContextVar(). The generation of
   * the node can be finished using the \ref finalize() method.
   *
   * \param parent The region where the lambda node is created.
   * \param type The lambda node's type.
   * \param name The lambda node's name.
   * \param linkage The lambda node's linkage.
   * \param attributes The lambda node's attributes.
   *
   * \return A lambda node featuring only function arguments.
   */
  static node *
  create(
      rvsdg::Region * parent,
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      const std::string & name,
      const jlm::llvm::linkage & linkage,
      const jlm::llvm::attributeset & attributes);

  /**
   * See \ref create().
   */
  static node *
  create(
      rvsdg::Region * parent,
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      const std::string & name,
      const jlm::llvm::linkage & linkage)
  {
    return create(parent, type, name, linkage, {});
  }

  /**
   * Finalizes the creation of a lambda node.
   *
   * \param results The result values of the lambda expression, originating from the lambda region.
   *
   * \return The output of the lambda node.
   */
  rvsdg::output *
  finalize(const std::vector<jlm::rvsdg::output *> & results);

  /**
   * Compute the \ref CallSummary of the lambda.
   *
   * @return A new CallSummary instance.
   */
  [[nodiscard]] std::unique_ptr<CallSummary>
  ComputeCallSummary() const;

  /**
   * Determines whether \p lambdaNode is exported from the module.
   *
   * @param lambdaNode The lambda node to be checked.
   * @return True if \p lambdaNode is exported, otherwise false.
   *
   * \note This method is equivalent to invoking CallSummary::IsExported().
   */
  [[nodiscard]] static bool
  IsExported(const lambda::node & lambdaNode);

private:
  std::vector<jlm::llvm::attributeset> ArgumentAttributes_;
};

/**
 * The CallSummary of a lambda summarizes all call usages of the lambda. It distinguishes between
 * three call usages:
 *
 * 1. The export of the lambda, which is null if the lambda is not exported.
 * 2. All direct calls of the lambda.
 * 3. All other usages, e.g., indirect calls.
 */
class node::CallSummary final
{
  using DirectCallsConstRange = util::IteratorRange<std::vector<CallNode *>::const_iterator>;
  using OtherUsersConstRange = util::IteratorRange<std::vector<rvsdg::input *>::const_iterator>;

public:
  CallSummary(
      GraphExport * rvsdgExport,
      std::vector<CallNode *> directCalls,
      std::vector<rvsdg::input *> otherUsers)
      : RvsdgExport_(rvsdgExport),
        DirectCalls_(std::move(directCalls)),
        OtherUsers_(std::move(otherUsers))
  {}

  /**
   * Determines whether the lambda is dead.
   *
   * @return True if the lambda is dead, otherwise false.
   */
  [[nodiscard]] bool
  IsDead() const noexcept
  {
    return RvsdgExport_ == nullptr && DirectCalls_.empty() && OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda is exported from the RVSDG
   *
   * @return True if the lambda is exported, otherwise false.
   */
  [[nodiscard]] bool
  IsExported() const noexcept
  {
    return RvsdgExport_ != nullptr;
  }

  /**
   * Determines whether the lambda is only(!) exported from the RVSDG.
   *
   * @return True if the lambda is only exported, otherwise false.
   */
  [[nodiscard]] bool
  IsOnlyExported() const noexcept
  {
    return RvsdgExport_ != nullptr && DirectCalls_.empty() && OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda has only direct calls.
   *
   * @return True if the lambda has only direct calls, otherwise false.
   */
  [[nodiscard]] bool
  HasOnlyDirectCalls() const noexcept
  {
    return RvsdgExport_ == nullptr && OtherUsers_.empty() && !DirectCalls_.empty();
  }

  /**
   * Determines whether the lambda has no other usages, i.e., it can only be exported and/or have
   * direct calls.
   *
   * @return True if the lambda has no other usages, otherwise false.
   */
  [[nodiscard]] bool
  HasNoOtherUsages() const noexcept
  {
    return OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda has only(!) other usages.
   *
   * @return True if the lambda has only other usages, otherwise false.
   */
  [[nodiscard]] bool
  HasOnlyOtherUsages() const noexcept
  {
    return RvsdgExport_ == nullptr && DirectCalls_.empty() && !OtherUsers_.empty();
  }

  /**
   * Returns the number of direct call sites invoking the lambda.
   *
   * @return The number of direct call sites.
   */
  [[nodiscard]] size_t
  NumDirectCalls() const noexcept
  {
    return DirectCalls_.size();
  }

  /**
   * Returns the number of all other users that are not direct calls.
   *
   * @return The number of usages that are not direct calls.
   */
  [[nodiscard]] size_t
  NumOtherUsers() const noexcept
  {
    return OtherUsers_.size();
  }

  /**
   * Returns the export of the lambda.
   *
   * @return The export of the lambda from the RVSDG root region.
   */
  [[nodiscard]] GraphExport *
  GetRvsdgExport() const noexcept
  {
    return RvsdgExport_;
  }

  /**
   * Returns an \ref util::iterator_range for iterating through all direct call sites.
   *
   * @return An \ref util::iterator_range of all direct call sites.
   */
  [[nodiscard]] DirectCallsConstRange
  DirectCalls() const noexcept
  {
    return { DirectCalls_.begin(), DirectCalls_.end() };
  }

  /**
   * Returns an \ref util::iterator_range for iterating through all other usages.
   *
   * @return An \ref util::iterator_range of all other usages.
   */
  [[nodiscard]] OtherUsersConstRange
  OtherUsers() const noexcept
  {
    return { OtherUsers_.begin(), OtherUsers_.end() };
  }

  /**
   * Creates a new CallSummary.
   *
   * @param rvsdgExport The lambda export.
   * @param directCalls The direct call sites of a lambda.
   * @param otherUsers All other usages of a lambda.
   *
   * @return A new CallSummary instance.
   *
   * @see ComputeCallSummary()
   */
  static std::unique_ptr<CallSummary>
  Create(
      GraphExport * rvsdgExport,
      std::vector<CallNode *> directCalls,
      std::vector<rvsdg::input *> otherUsers)
  {
    return std::make_unique<CallSummary>(
        rvsdgExport,
        std::move(directCalls),
        std::move(otherUsers));
  }

private:
  GraphExport * RvsdgExport_;
  std::vector<CallNode *> DirectCalls_;
  std::vector<rvsdg::input *> OtherUsers_;
};

template<typename F>
size_t
lambda::node::RemoveLambdaInputsWhere(const F & match)
{
  size_t numRemovedInputs = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto lambdaInput = input(n);
    auto & argument = *MapInputContextVar(*lambdaInput).inner;

    if (argument.IsDead() && match(*lambdaInput))
    {
      subregion()->RemoveArgument(argument.index());
      RemoveInput(n);
      numRemovedInputs++;
    }
  }

  return numRemovedInputs;
}

}
}

#endif
