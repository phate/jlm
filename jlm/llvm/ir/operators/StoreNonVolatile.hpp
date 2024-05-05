/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STORENONVOLATILE_HPP
#define JLM_LLVM_IR_OPERATORS_STORENONVOLATILE_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::llvm
{

/* store normal form */

class store_normal_form final : public jlm::rvsdg::simple_normal_form
{
public:
  virtual ~store_normal_form();

  store_normal_form(
      const std::type_info & opclass,
      jlm::rvsdg::node_normal_form * parent,
      jlm::rvsdg::graph * graph) noexcept;

  virtual bool
  normalize_node(jlm::rvsdg::node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      jlm::rvsdg::region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & operands) const override;

  virtual void
  set_store_mux_reducible(bool enable);

  virtual void
  set_store_store_reducible(bool enable);

  virtual void
  set_store_alloca_reducible(bool enable);

  virtual void
  set_multiple_origin_reducible(bool enable);

  inline bool
  get_store_mux_reducible() const noexcept
  {
    return enable_store_mux_;
  }

  inline bool
  get_store_store_reducible() const noexcept
  {
    return enable_store_store_;
  }

  inline bool
  get_store_alloca_reducible() const noexcept
  {
    return enable_store_alloca_;
  }

  inline bool
  get_multiple_origin_reducible() const noexcept
  {
    return enable_multiple_origin_;
  }

private:
  bool enable_store_mux_;
  bool enable_store_store_;
  bool enable_store_alloca_;
  bool enable_multiple_origin_;
};

/**
 * Represents an LLVM store instruction.
 *
 * @see StoreNonVolatileOperation
 */
class StoreNonVolatileOperation final : public jlm::rvsdg::simple_op
{
public:
  ~StoreNonVolatileOperation() noexcept override;

  StoreNonVolatileOperation(
      const jlm::rvsdg::valuetype & storedType,
      size_t numStates,
      size_t alignment)
      : simple_op(
          CreateOperandPorts(storedType, numStates),
          std::vector<jlm::rvsdg::port>(numStates, { MemoryStateType::Create() })),
        Alignment_(alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  [[nodiscard]] const PointerType &
  GetPointerType() const noexcept
  {
    return *jlm::util::AssertedCast<const PointerType>(&argument(0).type());
  }

  [[nodiscard]] const jlm::rvsdg::valuetype &
  GetStoredType() const noexcept
  {
    return *jlm::util::AssertedCast<const jlm::rvsdg::valuetype>(&argument(1).type());
  }

  [[nodiscard]] size_t
  NumStates() const noexcept
  {
    return nresults();
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return Alignment_;
  }

  static store_normal_form *
  GetNormalForm(jlm::rvsdg::graph * graph) noexcept
  {
    return jlm::util::AssertedCast<store_normal_form>(
        graph->node_normal_form(typeid(StoreNonVolatileOperation)));
  }

  static std::unique_ptr<llvm::tac>
  Create(const variable * address, const variable * value, const variable * state, size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value->type());

    StoreNonVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, state });
  }

private:
  static const jlm::rvsdg::valuetype &
  CheckAndExtractStoredType(const jlm::rvsdg::type & type)
  {
    if (auto storedType = dynamic_cast<const jlm::rvsdg::valuetype *>(&type))
      return *storedType;

    throw jlm::util::error("Expected ValueType");
  }

  static std::vector<jlm::rvsdg::port>
  CreateOperandPorts(const jlm::rvsdg::valuetype & storedType, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports({ PointerType(), storedType });
    std::vector<jlm::rvsdg::port> states(numStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t Alignment_;
};

/** \brief StoreNonVolatileNode class
 *
 */
class StoreNonVolatileNode final : public jlm::rvsdg::simple_node
{
private:
  class MemoryStateInputIterator final
      : public jlm::rvsdg::input::iterator<jlm::rvsdg::simple_input>
  {
    friend StoreNonVolatileNode;

    constexpr explicit MemoryStateInputIterator(jlm::rvsdg::simple_input * input)
        : jlm::rvsdg::input::iterator<jlm::rvsdg::simple_input>(input)
    {}

    [[nodiscard]] jlm::rvsdg::simple_input *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
    }
  };

  class MemoryStateOutputIterator final
      : public jlm::rvsdg::output::iterator<jlm::rvsdg::simple_output>
  {
    friend StoreNonVolatileNode;

    constexpr explicit MemoryStateOutputIterator(jlm::rvsdg::simple_output * output)
        : jlm::rvsdg::output::iterator<jlm::rvsdg::simple_output>(output)
    {}

    [[nodiscard]] jlm::rvsdg::simple_output *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->noutputs() > index + 1 ? node->output(index + 1) : nullptr;
    }
  };

  using MemoryStateInputRange = jlm::util::iterator_range<MemoryStateInputIterator>;
  using MemoryStateOutputRange = jlm::util::iterator_range<MemoryStateOutputIterator>;

  StoreNonVolatileNode(
      jlm::rvsdg::region & region,
      const StoreNonVolatileOperation & operation,
      const std::vector<jlm::rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const StoreNonVolatileOperation &
  GetOperation() const noexcept
  {
    return *jlm::util::AssertedCast<const StoreNonVolatileOperation>(&operation());
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept
  {
    JLM_ASSERT(ninputs() > 2);
    return { MemoryStateInputIterator(input(2)), MemoryStateInputIterator(nullptr) };
  }

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept
  {
    JLM_ASSERT(noutputs() > 1);
    return { MemoryStateOutputIterator(output(0)), MemoryStateOutputIterator(nullptr) };
  }

  [[nodiscard]] size_t
  NumStates() const noexcept
  {
    return GetOperation().NumStates();
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return GetOperation().GetAlignment();
  }

  [[nodiscard]] jlm::rvsdg::input *
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return addressInput;
  }

  [[nodiscard]] jlm::rvsdg::input *
  GetValueInput() const noexcept
  {
    auto valueInput = input(1);
    JLM_ASSERT(is<jlm::rvsdg::valuetype>(valueInput->type()));
    return valueInput;
  }

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<jlm::rvsdg::output *>
  Create(
      jlm::rvsdg::output * address,
      jlm::rvsdg::output * value,
      const std::vector<jlm::rvsdg::output *> & states,
      size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value->type());

    std::vector<jlm::rvsdg::output *> operands({ address, value });
    operands.insert(operands.end(), states.begin(), states.end());

    StoreNonVolatileOperation storeOperation(storedType, states.size(), alignment);
    return Create(*address->region(), storeOperation, operands);
  }

  static std::vector<jlm::rvsdg::output *>
  Create(
      jlm::rvsdg::region & region,
      const StoreNonVolatileOperation & storeOperation,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, storeOperation, operands));
  }

  static StoreNonVolatileNode &
  CreateNode(
      jlm::rvsdg::region & region,
      const StoreNonVolatileOperation & storeOperation,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return *(new StoreNonVolatileNode(region, storeOperation, operands));
  }

private:
  static const jlm::rvsdg::valuetype &
  CheckAndExtractStoredType(const jlm::rvsdg::type & type)
  {
    if (auto storedType = dynamic_cast<const jlm::rvsdg::valuetype *>(&type))
      return *storedType;

    throw jlm::util::error("Expected ValueType.");
  }
};

}

#endif
