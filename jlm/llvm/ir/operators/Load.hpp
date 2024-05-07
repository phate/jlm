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

/**
 * Abstract base class for load operations.
 *
 * @see LoadVolatileOperation
 * @see LoadNonVolatileOperation
 */
class LoadOperation : public rvsdg::simple_op
{
protected:
  LoadOperation(
      const std::vector<rvsdg::port> & operandPorts,
      const std::vector<rvsdg::port> & resultPorts,
      size_t alignment)
      : simple_op(operandPorts, resultPorts),
        Alignment_(alignment)
  {
    JLM_ASSERT(!operandPorts.empty() && !resultPorts.empty());

    auto & addressType = operandPorts[0].type();
    JLM_ASSERT(is<PointerType>(addressType));

    auto & loadedType = resultPorts[0].type();
    JLM_ASSERT(is<rvsdg::valuetype>(loadedType));

    JLM_ASSERT(operandPorts.size() == resultPorts.size());
    for (size_t n = 1; n < operandPorts.size(); n++)
    {
      auto & operandType = operandPorts[n].type();
      auto & resultType = resultPorts[n].type();
      JLM_ASSERT(operandType == resultType);
      JLM_ASSERT(is<rvsdg::statetype>(operandType));
    }
  }

public:
  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return Alignment_;
  }

  [[nodiscard]] const rvsdg::valuetype &
  GetLoadedType() const noexcept
  {
    return *util::AssertedCast<const rvsdg::valuetype>(&result(0).type());
  }

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;

private:
  size_t Alignment_;
};

/**
 * Represents a volatile LLVM load instruction.
 *
 * In contrast to LLVM, a volatile load requires in an RVSDG setting an I/O state as it incorporates
 * externally visible side-effects. This I/O state allows the volatile load operation to be
 * sequentialized with respect to other volatile memory accesses and I/O operations. This additional
 * I/O state is the main reason why volatile loads are modeled as its own operation and volatile is
 * not just a flag at the normal \ref LoadNonVolatileOperation.
 *
 * @see StoreVolatileOperation
 */
class LoadVolatileOperation final : public LoadOperation
{
public:
  ~LoadVolatileOperation() noexcept override;

  LoadVolatileOperation(
      const rvsdg::valuetype & loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
          CreateOperandPorts(numMemoryStates),
          CreateResultPorts(loadedType, numMemoryStates),
          alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * iOState,
      const variable * memoryState,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(loadedType, 1, alignment);
    return tac::create(operation, { address, iOState, memoryState });
  }

private:
  static std::vector<rvsdg::port>
  CreateOperandPorts(size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports({ PointerType(), iostatetype() });
    std::vector<rvsdg::port> states(numMemoryStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(const rvsdg::valuetype & loadedType, size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports({ loadedType, iostatetype() });
    std::vector<rvsdg::port> states(numMemoryStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }
};

/**
 * Abstract base class for load nodes.
 *
 * @see LoadVolatileNode
 * @see LoadNonVolatileNode
 */
class LoadNode : public rvsdg::simple_node
{
protected:
  LoadNode(
      rvsdg::region & region,
      const LoadOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::simple_input>
  {
  public:
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
  public:
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

  using MemoryStateInputRange = util::iterator_range<MemoryStateInputIterator>;
  using MemoryStateOutputRange = util::iterator_range<MemoryStateOutputIterator>;

public:
  [[nodiscard]] virtual const LoadOperation &
  GetOperation() const noexcept = 0;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return GetOperation().NumMemoryStates();
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return GetOperation().GetAlignment();
  }

  [[nodiscard]] rvsdg::input &
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return *addressInput;
  }

  [[nodiscard]] rvsdg::output &
  GetLoadedValueOutput() const noexcept
  {
    auto valueOutput = output(0);
    JLM_ASSERT(is<rvsdg::valuetype>(valueOutput->type()));
    return *valueOutput;
  }

  [[nodiscard]] virtual MemoryStateInputRange
  MemoryStateInputs() const noexcept = 0;

  [[nodiscard]] virtual MemoryStateOutputRange
  MemoryStateOutputs() const noexcept = 0;
};

/**
 * Represents a LoadVolatileOperation in an RVSDG.
 */
class LoadVolatileNode final : public LoadNode
{
private:
  LoadVolatileNode(
      rvsdg::region & region,
      const LoadVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : LoadNode(region, operation, operands)
  {}

public:
  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  [[nodiscard]] const LoadVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioInput = input(1);
    JLM_ASSERT(is<iostatetype>(ioInput->type()));
    return *ioInput;
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  static LoadVolatileNode &
  CreateNode(
      rvsdg::region & region,
      const LoadVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadVolatileNode(region, loadOperation, operands));
  }

  static LoadVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & iOState,
      const std::vector<rvsdg::output *> & memoryStates,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    LoadVolatileOperation operation(loadedType, memoryStates.size(), alignment);
    return CreateNode(*address.region(), operation, operands);
  }
};

/**
 * Represents an LLVM load instruction.
 *
 * @see LoadVolatileOperation
 */
class LoadNonVolatileOperation final : public LoadOperation
{
public:
  ~LoadNonVolatileOperation() noexcept override;

  LoadNonVolatileOperation(
      const rvsdg::valuetype & loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
          CreateOperandPorts(numMemoryStates),
          CreateResultPorts(loadedType, numMemoryStates),
          alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static load_normal_form *
  GetNormalForm(rvsdg::graph * graph) noexcept
  {
    return jlm::util::AssertedCast<load_normal_form>(
        graph->node_normal_form(typeid(LoadNonVolatileOperation)));
  }

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * state,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    LoadNonVolatileOperation operation(loadedType, 1, alignment);
    return tac::create(operation, { address, state });
  }

private:
  static std::vector<rvsdg::port>
  CreateOperandPorts(size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports(1, { PointerType() });
    std::vector<rvsdg::port> states(numMemoryStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(const rvsdg::valuetype & loadedType, size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports(1, { loadedType });
    std::vector<rvsdg::port> states(numMemoryStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }
};

/**
 * Represents a LoadNonVolatileOperation in an RVSDG.
 */
class LoadNonVolatileNode final : public LoadNode
{
private:
  LoadNonVolatileNode(
      rvsdg::region & region,
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : LoadNode(region, operation, operands)
  {}

public:
  [[nodiscard]] const LoadNonVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      const std::vector<rvsdg::output *> & states,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ address });
    operands.insert(operands.end(), states.begin(), states.end());

    LoadNonVolatileOperation loadOperation(loadedType, states.size(), alignment);
    return Create(*address->region(), loadOperation, operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::region & region,
      const LoadNonVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, loadOperation, operands));
  }

  static LoadNonVolatileNode &
  CreateNode(
      rvsdg::region & region,
      const LoadNonVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadNonVolatileNode(region, loadOperation, operands));
  }
};

}

#endif
