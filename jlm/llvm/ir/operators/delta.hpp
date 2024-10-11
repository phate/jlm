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

namespace delta
{

/** \brief Delta operation
 */
class operation final : public rvsdg::structural_op
{
public:
  ~operation() override;

  operation(
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

  operation(const operation & other) = default;

  operation(operation && other) noexcept = default;

  operation &
  operator=(const operation &) = delete;

  operation &
  operator=(operation &&) = delete;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  virtual bool
  operator==(const rvsdg::operation & other) const noexcept override;

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

class cvargument;
class cvinput;
class output;
class result;

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
 *   auto delta = delta::node::create(...);
 *   ...
 *   auto cv1 = delta->add_ctxvar(...);
 *   auto cv2 = delta->add_ctxvar(...);
 *   ...
 *   // generate delta body
 *   ...
 *   auto output = delta->finalize(...);
 * \endcode
 */
class node final : public rvsdg::structural_node
{
  class cviterator;
  class cvconstiterator;

  using ctxvar_range = jlm::util::iterator_range<cviterator>;
  using ctxvar_constrange = jlm::util::iterator_range<cvconstiterator>;

public:
  ~node() override;

private:
  node(rvsdg::Region * parent, delta::operation && op)
      : structural_node(op, parent, 1)
  {}

public:
  ctxvar_range
  ctxvars();

  ctxvar_constrange
  ctxvars() const;

  rvsdg::Region *
  subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  const delta::operation &
  operation() const noexcept
  {
    return *static_cast<const delta::operation *>(&structural_node::operation());
  }

  [[nodiscard]] const rvsdg::ValueType &
  type() const noexcept
  {
    return operation().type();
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  Type() const noexcept
  {
    return operation().Type();
  }

  const std::string &
  name() const noexcept
  {
    return operation().name();
  }

  [[nodiscard]] const std::string &
  Section() const noexcept
  {
    return operation().Section();
  }

  const llvm::linkage &
  linkage() const noexcept
  {
    return operation().linkage();
  }

  bool
  constant() const noexcept
  {
    return operation().constant();
  }

  size_t
  ncvarguments() const noexcept
  {
    return ninputs();
  }

  /**
   * Adds a context/free variable to the delta node. The \p origin must be from the same region
   * as the delta node.
   *
   * \return The context variable argument from the delta region.
   */
  delta::cvargument *
  add_ctxvar(rvsdg::output * origin);

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
    auto match = [](const cvinput &)
    {
      return true;
    };

    return RemoveDeltaInputsWhere(match);
  }

  cvinput *
  input(size_t n) const noexcept;

  delta::cvargument *
  cvargument(size_t n) const noexcept;

  delta::output *
  output() const noexcept;

  delta::result *
  result() const noexcept;

  virtual delta::node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  virtual delta::node *
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
  static node *
  Create(
      rvsdg::Region * parent,
      std::shared_ptr<const rvsdg::ValueType> type,
      const std::string & name,
      const llvm::linkage & linkage,
      std::string section,
      bool constant)
  {
    delta::operation op(std::move(type), name, linkage, std::move(section), constant);
    return new delta::node(parent, std::move(op));
  }

  /**
   * Finalizes the creation of a delta node.
   *
   * \param result The result values of the delta expression, originating from the delta region.
   *
   * \return The output of the delta node.
   */
  delta::output *
  finalize(rvsdg::output * result);
};

/** \brief Delta context variable input
 */
class cvinput final : public rvsdg::structural_input
{
  friend ::jlm::llvm::delta::node;

public:
  ~cvinput() override;

private:
  cvinput(delta::node * node, rvsdg::output * origin)
      : structural_input(node, origin, origin->Type())
  {}

  static cvinput *
  create(delta::node * node, rvsdg::output * origin)
  {
    auto input = std::unique_ptr<cvinput>(new cvinput(node, origin));
    return static_cast<cvinput *>(node->append_input(std::move(input)));
  }

public:
  cvargument *
  argument() const noexcept;

  delta::node *
  node() const noexcept
  {
    return static_cast<delta::node *>(structural_input::node());
  }
};

/** \brief Delta context variable iterator
 */
class node::cviterator final : public rvsdg::input::iterator<cvinput>
{
  friend ::jlm::llvm::delta::node;

  constexpr cviterator(cvinput * input)
      : rvsdg::input::iterator<cvinput>(input)
  {}

  virtual cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

/** \brief Delta context variable const iterator
 */
class node::cvconstiterator final : public rvsdg::input::constiterator<cvinput>
{
  friend ::jlm::llvm::delta::node;

  constexpr cvconstiterator(const cvinput * input)
      : rvsdg::input::constiterator<cvinput>(input)
  {}

  virtual const cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

/** \brief Delta output
 */
class output final : public rvsdg::structural_output
{
  friend ::jlm::llvm::delta::node;

public:
  ~output() override;

  output(delta::node * node, std::shared_ptr<const rvsdg::Type> type)
      : structural_output(node, std::move(type))
  {}

private:
  static output *
  create(delta::node * node, std::shared_ptr<const rvsdg::Type> type)
  {
    auto output = std::make_unique<delta::output>(node, std::move(type));
    return static_cast<delta::output *>(node->append_output(std::move(output)));
  }

public:
  delta::node *
  node() const noexcept
  {
    return static_cast<delta::node *>(structural_output::node());
  }
};

/** \brief Delta context variable argument
 */
class cvargument final : public rvsdg::RegionArgument
{
  friend ::jlm::llvm::delta::node;

public:
  ~cvargument() override;

  cvargument &
  Copy(rvsdg::Region & region, jlm::rvsdg::structural_input * input) override;

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

/** \brief Delta result
 */
class result final : public rvsdg::RegionResult
{
  friend ::jlm::llvm::delta::node;

public:
  ~result() override;

  result &
  Copy(rvsdg::output & origin, jlm::rvsdg::structural_output * output) override;

private:
  explicit result(rvsdg::output * origin)
      : rvsdg::RegionResult(origin->region(), origin, nullptr, origin->Type())
  {}

  static result *
  create(rvsdg::output * origin)
  {
    auto result = new delta::result(origin);
    origin->region()->append_result(result);
    return result;
  }

public:
  delta::output *
  output() const noexcept
  {
    return static_cast<delta::output *>(rvsdg::RegionResult::output());
  }
};

template<typename F>
size_t
delta::node::RemoveDeltaInputsWhere(const F & match)
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
}

#endif
