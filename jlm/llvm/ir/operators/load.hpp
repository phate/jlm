/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LOAD_HPP
#define JLM_LLVM_IR_OPERATORS_LOAD_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::llvm
{

/* load normal form */

class load_normal_form final : public rvsdg::simple_normal_form
{
public:
  virtual ~load_normal_form();

  load_normal_form(
      const std::type_info & opclass,
      rvsdg::node_normal_form * parent,
      rvsdg::graph * graph) noexcept;

  virtual bool
  normalize_node(rvsdg::node * node) const override;

  virtual std::vector<rvsdg::output *>
  normalized_create(
      rvsdg::region * region,
      const rvsdg::simple_op & op,
      const std::vector<rvsdg::output *> & operands) const override;

  inline void
  set_load_mux_reducible(bool enable) noexcept
  {
    enable_load_mux_ = enable;
  }

  inline bool
  get_load_mux_reducible() const noexcept
  {
    return enable_load_mux_;
  }

  inline void
  set_load_alloca_reducible(bool enable) noexcept
  {
    enable_load_alloca_ = enable;
  }

  inline bool
  get_load_alloca_reducible() const noexcept
  {
    return enable_load_alloca_;
  }

  inline void
  set_multiple_origin_reducible(bool enable) noexcept
  {
    enable_multiple_origin_ = enable;
  }

  inline bool
  get_multiple_origin_reducible() const noexcept
  {
    return enable_multiple_origin_;
  }

  inline void
  set_load_store_state_reducible(bool enable) noexcept
  {
    enable_load_store_state_ = enable;
  }

  inline bool
  get_load_store_state_reducible() const noexcept
  {
    return enable_load_store_state_;
  }

  inline void
  set_load_store_reducible(bool enable) noexcept
  {
    enable_load_store_ = enable;
  }

  inline bool
  get_load_store_reducible() const noexcept
  {
    return enable_load_store_;
  }

  void
  set_load_load_state_reducible(bool enable) noexcept
  {
    enable_load_load_state_ = enable;
  }

  bool
  get_load_load_state_reducible() const noexcept
  {
    return enable_load_load_state_;
  }

private:
  bool enable_load_mux_;
  bool enable_load_store_;
  bool enable_load_alloca_;
  bool enable_load_load_state_;
  bool enable_multiple_origin_;
  bool enable_load_store_state_;
};

/** \brief LoadOperation class
 *
 * This operator is the Jlm equivalent of LLVM's load instruction.
 */
class LoadOperation final : public rvsdg::simple_op
{
public:
  ~LoadOperation() noexcept override;

  LoadOperation(const rvsdg::valuetype & loadedType, size_t numStates, size_t alignment)
      : simple_op(CreateOperandPorts(numStates), CreateResultPorts(loadedType, numStates)),
        alignment_(alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] const PointerType &
  GetPointerType() const noexcept
  {
    return *jlm::util::AssertedCast<const PointerType>(&argument(0).type());
  }

  [[nodiscard]] const rvsdg::valuetype &
  GetLoadedType() const noexcept
  {
    return *jlm::util::AssertedCast<const rvsdg::valuetype>(&result(0).type());
  }

  [[nodiscard]] size_t
  NumStates() const noexcept
  {
    return narguments() - 1;
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return alignment_;
  }

  static load_normal_form *
  GetNormalForm(rvsdg::graph * graph) noexcept
  {
    return jlm::util::AssertedCast<load_normal_form>(
        graph->node_normal_form(typeid(LoadOperation)));
  }

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * state,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    CheckAddressType(address->type());

    LoadOperation operation(loadedType, 1, alignment);
    return tac::create(operation, { address, state });
  }

private:
  static void
  CheckAddressType(const rvsdg::type & addressType)
  {
    if (!is<PointerType>(addressType))
      throw jlm::util::error("Expected pointer type.");
  }

  static std::vector<rvsdg::port>
  CreateOperandPorts(size_t numStates)
  {
    std::vector<rvsdg::port> ports(1, { PointerType() });
    std::vector<rvsdg::port> states(numStates, { MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(const rvsdg::valuetype & loadedType, size_t numStates)
  {
    std::vector<rvsdg::port> ports(1, { loadedType });
    std::vector<rvsdg::port> states(numStates, { MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t alignment_;
};

class LoadNode final : public rvsdg::simple_node
{
private:
  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::simple_input>
  {
    friend LoadNode;

    constexpr explicit MemoryStateInputIterator(rvsdg::simple_input * input)
        : rvsdg::input::iterator<rvsdg::simple_input>(input)
    {}

    [[nodiscard]] rvsdg::simple_input *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
    }
  };

  class MemoryStateOutputIterator final : public rvsdg::output::iterator<rvsdg::simple_output>
  {
    friend LoadNode;

    constexpr explicit MemoryStateOutputIterator(rvsdg::simple_output * output)
        : rvsdg::output::iterator<rvsdg::simple_output>(output)
    {}

    [[nodiscard]] rvsdg::simple_output *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->noutputs() > index + 1 ? node->output(index + 1) : nullptr;
    }
  };

  using MemoryStateInputRange = jlm::util::iterator_range<MemoryStateInputIterator>;
  using MemoryStateOutputRange = jlm::util::iterator_range<MemoryStateOutputIterator>;

  LoadNode(
      rvsdg::region & region,
      const LoadOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const LoadOperation &
  GetOperation() const noexcept
  {
    return *jlm::util::AssertedCast<const LoadOperation>(&operation());
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept
  {
    JLM_ASSERT(ninputs() > 1);
    return { MemoryStateInputIterator(input(1)), MemoryStateInputIterator(nullptr) };
  }

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept
  {
    JLM_ASSERT(noutputs() > 1);
    return { MemoryStateOutputIterator(output(1)), MemoryStateOutputIterator(nullptr) };
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

  [[nodiscard]] rvsdg::input *
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return addressInput;
  }

  [[nodiscard]] rvsdg::output *
  GetValueOutput() const noexcept
  {
    auto valueOutput = output(0);
    JLM_ASSERT(is<rvsdg::valuetype>(valueOutput->type()));
    return valueOutput;
  }

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      const std::vector<rvsdg::output *> & states,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    CheckAddressType(address->type());

    std::vector<rvsdg::output *> operands({ address });
    operands.insert(operands.end(), states.begin(), states.end());

    LoadOperation loadOperation(loadedType, states.size(), alignment);
    return Create(*address->region(), loadOperation, operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::region & region,
      const LoadOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, loadOperation, operands));
  }

  static LoadNode &
  CreateNode(
      rvsdg::region & region,
      const LoadOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadNode(region, loadOperation, operands));
  }

private:
  static void
  CheckAddressType(const rvsdg::type & addressType)
  {
    if (!is<PointerType>(addressType))
      throw jlm::util::error("Expected pointer type.");
  }
};

}

#endif
