/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_DELTA_HPP
#define JLM_LLVM_IR_OPERATORS_DELTA_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/iterator_range.hpp>

namespace jlm::llvm
{

/** \brief Delta operation
 */
class DeltaOperation final : public rvsdg::StructuralOperation
{
public:
  ~DeltaOperation() noexcept override;

  DeltaOperation(
      std::shared_ptr<const rvsdg::ValueType> type,
      const std::string & name,
      const llvm::linkage & linkage,
      std::string section,
      bool constant)
      : constant_(constant),
        name_(name),
        Section_(std::move(section)),
        linkage_(linkage),
        type_(std::move(type))
  {}

  DeltaOperation(const DeltaOperation & other) = default;

  DeltaOperation(DeltaOperation && other) noexcept = default;

  DeltaOperation &
  operator=(const DeltaOperation &) = delete;

  DeltaOperation &
  operator=(DeltaOperation &&) = delete;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  virtual bool
  operator==(const Operation & other) const noexcept override;

  const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const std::string &
  Section() const noexcept
  {
    return Section_;
  }

  const llvm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

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

private:
  bool constant_;
  std::string name_;
  std::string Section_;
  llvm::linkage linkage_;
  std::shared_ptr<const rvsdg::ValueType> type_;
};

namespace delta
{

class cvargument;
class cvinput;

}

/** \brief Delta node
 *
 * A delta node represents a global variable in the RVSDG. Its creation requires the invocation
 * of two functions: \ref Create() and \ref finalize(). First, a delta node is create by invoking
 * \ref Create(). The delta's dependencies can then be added using the \ref add_ctxvar() method,
 * and the body of the delta node can be created. Finally, the delta node can be finalized by
 * invoking \ref finalize().
 *
 * The following snippet illustrates the creation of delta nodes:
 *
 * \code{.cpp}
 *   auto delta = DeltaNode::create(...);
 *   ...
 *   auto cv1 = delta->add_ctxvar(...);
 *   auto cv2 = delta->add_ctxvar(...);
 *   ...
 *   // generate delta body
 *   ...
 *   auto output = delta->finalize(...);
 * \endcode
 */
class DeltaNode final : public rvsdg::StructuralNode
{
  class cviterator;
  class cvconstiterator;

  using ctxvar_range = util::IteratorRange<cviterator>;
  using ctxvar_constrange = util::IteratorRange<cvconstiterator>;

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

  ctxvar_range
  ctxvars();

  ctxvar_constrange
  ctxvars() const;

  rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] const DeltaOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] const rvsdg::ValueType &
  type() const noexcept
  {
    return GetOperation().type();
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  Type() const noexcept
  {
    return GetOperation().Type();
  }

  const std::string &
  name() const noexcept
  {
    return GetOperation().name();
  }

  [[nodiscard]] const std::string &
  Section() const noexcept
  {
    return GetOperation().Section();
  }

  const llvm::linkage &
  linkage() const noexcept
  {
    return GetOperation().linkage();
  }

  bool
  constant() const noexcept
  {
    return GetOperation().constant();
  }

  size_t
  ncvarguments() const noexcept
  {
    return ninputs();
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
    auto match = [](const delta::cvinput &)
    {
      return true;
    };

    return RemoveDeltaInputsWhere(match);
  }

  delta::cvinput *
  input(size_t n) const noexcept;

  delta::cvargument *
  cvargument(size_t n) const noexcept;

  [[nodiscard]] rvsdg::Output &
  output() const noexcept;

  [[nodiscard]] rvsdg::Input &
  result() const noexcept;

  virtual DeltaNode *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const override;

  virtual DeltaNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * Creates a delta node in the region \p parent with the pointer type \p type and name \p name.
   * After the invocation of \ref Create(), the delta node has no inputs or outputs.
   * Free variables can be added to the delta node using \ref add_ctxvar(). The generation of the
   * node can be finished using the \ref finalize() method.
   *
   * \param parent The region where the delta node is created.
   * \param type The delta node's type.
   * \param name The delta node's name.
   * \param linkage The delta node's linkage.
   * \param section The delta node's section.
   * \param constant True, if the delta node is constant, otherwise false.
   *
   * \return A delta node without inputs or outputs.
   */
  static DeltaNode *
  Create(
      rvsdg::Region * parent,
      std::shared_ptr<const rvsdg::ValueType> type,
      const std::string & name,
      const llvm::linkage & linkage,
      std::string section,
      bool constant)
  {
    auto op = std::make_unique<DeltaOperation>(
        std::move(type),
        name,
        linkage,
        std::move(section),
        constant);
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

namespace delta
{

/** \brief Delta context variable input
 */
class cvinput final : public rvsdg::StructuralInput
{
  friend ::jlm::llvm::DeltaNode;

public:
  ~cvinput() override;

private:
  cvinput(DeltaNode * node, rvsdg::Output * origin)
      : StructuralInput(node, origin, origin->Type())
  {}

  static cvinput *
  create(DeltaNode * node, rvsdg::Output * origin)
  {
    auto input = std::unique_ptr<cvinput>(new cvinput(node, origin));
    return static_cast<cvinput *>(node->append_input(std::move(input)));
  }

public:
  cvargument *
  argument() const noexcept;

  DeltaNode *
  node() const noexcept
  {
    return static_cast<DeltaNode *>(StructuralInput::node());
  }
};

}

/** \brief Delta context variable iterator
 */
class DeltaNode::cviterator final : public rvsdg::Input::iterator<delta::cvinput>
{
  friend ::jlm::llvm::DeltaNode;

  constexpr cviterator(delta::cvinput * input)
      : rvsdg::Input::iterator<delta::cvinput>(input)
  {}

  virtual delta::cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

/** \brief Delta context variable const iterator
 */
class DeltaNode::cvconstiterator final : public rvsdg::Input::constiterator<delta::cvinput>
{
  friend ::jlm::llvm::DeltaNode;

  constexpr cvconstiterator(const delta::cvinput * input)
      : rvsdg::Input::constiterator<delta::cvinput>(input)
  {}

  virtual const delta::cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

namespace delta
{

/** \brief Delta context variable argument
 */
class cvargument final : public rvsdg::RegionArgument
{
  friend ::jlm::llvm::DeltaNode;

public:
  ~cvargument() override;

  cvargument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

private:
  cvargument(rvsdg::Region * region, cvinput * input)
      : rvsdg::RegionArgument(region, input, input->Type())
  {}

  static cvargument *
  create(rvsdg::Region * region, delta::cvinput * input)
  {
    auto argument = new cvargument(region, input);
    region->append_argument(argument);
    return argument;
  }

public:
  cvinput *
  input() const noexcept
  {
    return static_cast<cvinput *>(rvsdg::RegionArgument::input());
  }
};

}

template<typename F>
size_t
DeltaNode::RemoveDeltaInputsWhere(const F & match)
{
  size_t numRemovedInputs = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & deltaInput = *input(n);
    auto & argument = *deltaInput.argument();

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
