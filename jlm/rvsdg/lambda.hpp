/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_LAMBDA_HPP
#define JLM_RVSDG_LAMBDA_HPP

#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/iterator_range.hpp>

#include <optional>
#include <utility>

namespace jlm::rvsdg
{

/** \brief Lambda operation
 *
 * A lambda operation determines a lambda's \ref FunctionType "function type".
 */
class LambdaOperation : public rvsdg::StructuralOperation
{
public:
  ~LambdaOperation() override;

  explicit LambdaOperation(std::shared_ptr<const FunctionType> type);

  [[nodiscard]] const FunctionType &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const FunctionType> &
  Type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

private:
  std::shared_ptr<const FunctionType> type_;
};

/** \brief Lambda node
 *
 * A lambda node represents a lambda expression in the RVSDG. Its creation requires the invocation
 * of two functions: \ref Create() and \ref finalize(). First, a node with only the function
 * arguments is created by invoking \ref Create(). The free variables of the lambda expression can
 * then be added to the lambda node using the \ref AddContextVar() method, and the body of the
 * lambda node can be created. Finally, the lambda node can be finalized by invoking \ref
 * finalize().
 *
 * The following snippet illustrates the creation of lambda nodes:
 *
 * \code{.cpp}
 *   auto lambda = LambdaNode::create(...);
 *   ...
 *   auto cv1 = lambda->AddContextVar(...);
 *   auto cv2 = lambda->AddContextVar(...);
 *   ...
 *   // generate lambda body
 *   ...
 *   auto output = lambda->finalize(...);
 * \endcode
 */
class LambdaNode final : public rvsdg::StructuralNode
{
public:
  ~LambdaNode() override;

private:
  LambdaNode(rvsdg::Region & parent, std::unique_ptr<LambdaOperation> op);

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
    rvsdg::Input * input;

    /**
     * \brief Access to bound object in subregion.
     *
     * Supplies access to the value bound into the lambda abstraction
     * from inside the region contained in the lambda node. This
     * evaluates to the value bound into the lambda.
     */
    rvsdg::Output * inner;
  };

  [[nodiscard]] std::vector<rvsdg::Output *>
  GetFunctionArguments() const;

  [[nodiscard]] std::vector<rvsdg::Input *>
  GetFunctionResults() const;

  [[nodiscard]] rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] LambdaOperation &
  GetOperation() const noexcept override;

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
  AddContextVar(jlm::rvsdg::Output & origin);

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
  MapInputContextVar(const rvsdg::Input & input) const noexcept;

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
  MapBinderContextVar(const rvsdg::Output & output) const noexcept;

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
    auto match = [](const rvsdg::Input &)
    {
      return true;
    };

    return RemoveLambdaInputsWhere(match);
  }

  [[nodiscard]] rvsdg::Output *
  output() const noexcept;

  LambdaNode *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const override;

  LambdaNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * Creates a lambda node in the region \p parent with the function type \p type and name \p name.
   * After the invocation of \ref Create(), the lambda node only features the function arguments.
   * Free variables can be added to the function node using \ref AddContextVar(). The generation of
   * the node can be finished using the \ref finalize() method.
   *
   * \param parent The region where the lambda node is created.
   * \param operation Operational details for lambda node (including function signature).
   *
   * \return A lambda node featuring only function arguments.
   */
  static LambdaNode *
  Create(rvsdg::Region & parent, std::unique_ptr<LambdaOperation> operation);

  /**
   * Finalizes the creation of a lambda node.
   *
   * \param results The result values of the lambda expression, originating from the lambda region.
   *
   * \return The output of the lambda node.
   */
  rvsdg::Output *
  finalize(const std::vector<jlm::rvsdg::Output *> & results);

private:
  std::unique_ptr<LambdaOperation> Operation_;
};

template<typename F>
size_t
LambdaNode::RemoveLambdaInputsWhere(const F & match)
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

#endif
