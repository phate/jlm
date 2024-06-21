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
      const std::vector<std::shared_ptr<const rvsdg::type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::type>> & resultTypes,
      size_t alignment)
      : simple_op(operandTypes, resultTypes),
        Alignment_(alignment)
  {
    JLM_ASSERT(!operandTypes.empty() && !resultTypes.empty());

    auto & addressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(addressType));

    auto & loadedType = *resultTypes[0];
    JLM_ASSERT(is<rvsdg::valuetype>(loadedType));

    JLM_ASSERT(operandTypes.size() == resultTypes.size());
    for (size_t n = 1; n < operandTypes.size(); n++)
    {
      auto & operandType = *operandTypes[n];
      auto & resultType = *resultTypes[n];
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

  [[nodiscard]] std::shared_ptr<const rvsdg::valuetype>
  GetLoadedType() const noexcept
  {
    auto type = std::dynamic_pointer_cast<const rvsdg::valuetype>(result(0).Type());
    JLM_ASSERT(type);
    return type;
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
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
          CreateOperandTypes(numMemoryStates),
          CreateResultTypes(std::move(loadedType), numMemoryStates),
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
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, iOState, memoryState });
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types(
        { PointerType::Create(), iostatetype::Create() });
    std::vector<std::shared_ptr<const rvsdg::type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateResultTypes(std::shared_ptr<const rvsdg::valuetype> loadedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types(
        { std::move(loadedType), iostatetype::Create() });
    std::vector<std::shared_ptr<const rvsdg::type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
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

public:
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

  /**
   * Create a new copy of this LoadNode that consumes the provided \p memoryStates.
   *
   * @param memoryStates The memory states the newly created copy should consume.
   * @return A newly created LoadNode.
   */
  [[nodiscard]] virtual LoadNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const = 0;
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

  [[nodiscard]] rvsdg::output &
  GetIoStateOutput() const noexcept
  {
    auto ioOutput = output(1);
    JLM_ASSERT(is<iostatetype>(ioOutput->type()));
    return *ioOutput;
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] LoadVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

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
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    LoadVolatileOperation operation(std::move(loadedType), memoryStates.size(), alignment);
    return CreateNode(*address.region(), operation, operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::region & region,
      const LoadVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, loadOperation, operands));
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
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
          CreateOperandTypes(numMemoryStates),
          CreateResultTypes(std::move(loadedType), numMemoryStates),
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
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t alignment)
  {
    LoadNonVolatileOperation operation(std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, state });
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types(1, PointerType::Create());
    std::vector<std::shared_ptr<const rvsdg::type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateResultTypes(std::shared_ptr<const rvsdg::valuetype> loadedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types(1, std::move(loadedType));
    std::vector<std::shared_ptr<const rvsdg::type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
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

  [[nodiscard]] LoadNonVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(*address, memoryStates, std::move(loadedType), alignment));
  }

  static LoadNonVolatileNode &
  CreateNode(
      rvsdg::output & address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::valuetype> loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    LoadNonVolatileOperation loadOperation(std::move(loadedType), memoryStates.size(), alignment);
    return CreateNode(*address.region(), loadOperation, operands);
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
