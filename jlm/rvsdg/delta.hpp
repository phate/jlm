/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_DELTA_HPP
#define JLM_RVSDG_DELTA_HPP

#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jlm::rvsdg
{

/** \brief Delta operation
 */
class DeltaOperation : public rvsdg::StructuralOperation
{
public:
  ~DeltaOperation() noexcept override;

  DeltaOperation(
      std::shared_ptr<const rvsdg::ValueType> type,
      bool constant,
      std::shared_ptr<const rvsdg::ValueType> reftype)
      : constant_(constant),
        type_(std::move(type)),
        reftype_(std::move(reftype))
  {}

  DeltaOperation(const DeltaOperation & other) = default;

  DeltaOperation(DeltaOperation && other) noexcept = default;

  DeltaOperation &
  operator=(const DeltaOperation &) = delete;

  DeltaOperation &
  operator=(DeltaOperation &&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] bool
  operator==(const Operation & other) const noexcept override;

  bool
  constant() const noexcept
  {
    return constant_;
  }

  [[nodiscard]] const rvsdg::ValueType &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  Type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  ReferenceType() const noexcept
  {
    return reftype_;
  }

  /**
   * \brief Creates parameterized delta operation.
   *
   * \param type
   *   The type of the value bound in the definition.
   *
   * \param constant
   *   Whether the data element is to be considered runtime constant.
   *
   * \param reftype
   *   The type used to reference the bound object (typically a poniter type).
   *
   * \returns
   *   The created operation structure.
   *
   * Creates a parameterized delta operation that binds an
   * (initial) value into a program definition.
   */
  static inline std::unique_ptr<DeltaOperation>
  Create(
      std::shared_ptr<const rvsdg::ValueType> type,
      bool constant,
      std::shared_ptr<const rvsdg::ValueType> reftype)
  {
    return std::make_unique<DeltaOperation>(std::move(type), constant, std::move(reftype));
  }

private:
  bool constant_;
  std::shared_ptr<const rvsdg::ValueType> type_;
  std::shared_ptr<const rvsdg::ValueType> reftype_;
};

/** \brief Delta node
 *
 * A delta node represents a global variable in the RVSDG. Its creation requires the invocation
 * of two functions: \ref Create() and \ref finalize(). First, a delta node is created by invoking
 * \ref Create(). The delta's dependencies can then be added using the \ref AddContextVar() method,
 * and the body of the delta node can be created. Finally, the delta node can be finalized by
 * invoking \ref finalize().
 *
 * The following snippet illustrates the creation of delta nodes:
 *
 * \code{.cpp}
 *   auto delta = DeltaNode::create(...);
 *   ...
 *   auto cv1 = delta->AddContextVar(...);
 *   auto cv2 = delta->AddContextVar(...);
 *   ...
 *   // generate delta body
 *   ...
 *   auto output = delta->finalize(...);
 * \endcode
 */
class DeltaNode final : public rvsdg::StructuralNode
{
public:
  ~DeltaNode() noexcept override;

private:
  DeltaNode(rvsdg::Region * parent, std::unique_ptr<DeltaOperation> op)
      : StructuralNode(parent, 1),
        Operation_(std::move(op))
  {}

public:
  /**
   * \brief Bound context variable
   *
   * Context variables may be bound at the point of creation of a
   * delta abstraction. These are represented as inputs to the
   * delta node itself, and made accessible to the body of the
   * delta in the form of an initial argument to the subregion.
   */
  struct ContextVar
  {
    /**
     * \brief Input variable bound into delta node
     *
     * The input port into the delta node that supplies the value
     * of the context variable bound into the delta at the
     * time the delta abstraction is built.
     */
    rvsdg::Input * input;

    /**
     * \brief Access to bound object in subregion.
     *
     * Supplies access to the value bound into the delta abstraction
     * from inside the region contained in the delta node. This
     * evaluates to the value bound into the delta.
     */
    rvsdg::Output * inner;
  };

  /**
   * \brief Adds a context/free variable to the delta node.
   *
   * \param origin
   *   The value to be bound into the delta node.
   *
   * \pre
   *   \p origin must be from the same region as the delta node.
   *
   * \return The context variable argument of the delta abstraction.
   */
  ContextVar
  AddContextVar(jlm::rvsdg::Output & origin);

  /**
   * \brief Maps input to context variable.
   *
   * \param input
   *   Input to the delta node.
   *
   * \returns
   *   The context variable description corresponding to the input.
   *
   * \pre
   *   \p input must be input to this node.
   *
   * Returns the context variable description corresponding
   * to this input of the delta node. All inputs to the delta
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
   *   Region argument to delta subregion
   *
   * \returns
   *   The context variable description corresponding to the argument
   *
   * \pre
   *   \p output must be an argument to the subregion of this node
   *
   * Returns the context variable description corresponding
   * to this bound variable reference in the delta node region.
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

  rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] const DeltaOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  Type() const noexcept
  {
    return GetOperation().Type();
  }

  bool
  constant() const noexcept
  {
    return GetOperation().constant();
  }

  /**
   * Remove delta inputs and their respective arguments.
   *
   * An input must match the condition specified by \p match and its argument must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const cvinput&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed inputs.
   *
   * \see cvargument#IsDead()
   */
  template<typename F>
  size_t
  RemoveDeltaInputsWhere(const F & match);

  /**
   * Remove all dead inputs.
   *
   * @return The number of removed inputs.
   *
   * \see RemoveDeltaInputsWhere()
   */
  size_t
  PruneDeltaInputs()
  {
    auto match = [](const rvsdg::Input &)
    {
      return true;
    };

    return RemoveDeltaInputsWhere(match);
  }

  [[nodiscard]] rvsdg::Output &
  output() const noexcept;

  [[nodiscard]] rvsdg::Input &
  result() const noexcept;

  DeltaNode *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const override;

  DeltaNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * Creates a delta node in the region \p parent with the detail information
   * given in the operation struct.
   *
   * After the invocation of \ref Create(), the delta node has no inputs or outputs.
   * Free variables can be added to the delta node using \ref AddContextVar(). The generation of the
   * node can be finished using the \ref finalize() method.
   *
   * \param parent The region where the delta node is created.
   * \param op The delta node operation
   *
   * \return A delta node without inputs or outputs.
   */
  static DeltaNode *
  Create(rvsdg::Region * parent, std::unique_ptr<DeltaOperation> op)
  {
    return new DeltaNode(parent, std::move(op));
  }

  /**
   * Finalizes the creation of a delta node.
   *
   * \param result The result values of the delta expression, originating from the delta region.
   *
   * \return The output of the delta node.
   */
  rvsdg::Output &
  finalize(rvsdg::Output * result);

private:
  std::unique_ptr<DeltaOperation> Operation_;
};

template<typename F>
size_t
DeltaNode::RemoveDeltaInputsWhere(const F & match)
{
  size_t numRemovedInputs = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & deltaInput = *input(n);
    auto & argument = *deltaInput.arguments.first();

    if (argument.IsDead() && match(deltaInput))
    {
      subregion()->RemoveArgument(argument.index());
      RemoveInput(deltaInput.index());
      numRemovedInputs++;
    }
  }

  return numRemovedInputs;
}

}

#endif
