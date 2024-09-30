/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STORE_HPP
#define JLM_LLVM_IR_OPERATORS_STORE_HPP

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
      rvsdg::Region * region,
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
 * Abstract base class for store operations.
 *
 * @see StoreVolatileOperation
 * @see StoreNonVolatileOperation
 */
class StoreOperation : public rvsdg::simple_op
{
protected:
  StoreOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      size_t alignment)
      : simple_op(operandTypes, resultTypes),
        Alignment_(alignment)
  {
    JLM_ASSERT(operandTypes.size() >= 2);

    auto & addressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(addressType));

    auto & storedType = *operandTypes[1];
    JLM_ASSERT(is<rvsdg::valuetype>(storedType));

    JLM_ASSERT(operandTypes.size() == resultTypes.size() + 2);
    for (size_t n = 0; n < resultTypes.size(); n++)
    {
      auto & operandType = *operandTypes[n + 2];
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

  [[nodiscard]] const rvsdg::valuetype &
  GetStoredType() const noexcept
  {
    return *util::AssertedCast<const rvsdg::valuetype>(argument(1).get());
  }

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;

private:
  size_t Alignment_;
};

/**
 * Represents an LLVM store instruction.
 *
 * @see StoreVolatileOperation
 */
class StoreNonVolatileOperation final : public StoreOperation
{
public:
  ~StoreNonVolatileOperation() noexcept override;

  StoreNonVolatileOperation(
      std::shared_ptr<const rvsdg::valuetype> storedType,
      size_t numMemoryStates,
      size_t alignment)
      : StoreOperation(
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            { numMemoryStates, MemoryStateType::Create() },
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

  static store_normal_form *
  GetNormalForm(rvsdg::graph * graph) noexcept
  {
    return util::AssertedCast<store_normal_form>(
        graph->node_normal_form(typeid(StoreNonVolatileOperation)));
  }

  static std::unique_ptr<llvm::tac>
  Create(const variable * address, const variable * value, const variable * state, size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreNonVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, state });
  }

private:
  static const std::shared_ptr<const jlm::rvsdg::valuetype>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::valuetype>(type))
    {
      return storedType;
    }

    throw util::error("Expected value type");
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::valuetype> storedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { PointerType::Create(), std::move(storedType) });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }
};

/**
 * Abstract base class for store nodes
 *
 * @see StoreVolatileNode
 * @see StoreNonVolatileNode
 */
class StoreNode : public rvsdg::simple_node
{
protected:
  StoreNode(
      rvsdg::Region & region,
      const StoreOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::simple_input>
  {
  public:
    constexpr explicit MemoryStateInputIterator(rvsdg::simple_input * input)
        : rvsdg::input::iterator<jlm::rvsdg::simple_input>(input)
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

  [[nodiscard]] virtual const StoreOperation &
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

  [[nodiscard]] rvsdg::input &
  GetStoredValueInput() const noexcept
  {
    auto valueInput = input(1);
    JLM_ASSERT(is<rvsdg::valuetype>(valueInput->type()));
    return *valueInput;
  }

  [[nodiscard]] virtual MemoryStateInputRange
  MemoryStateInputs() const noexcept = 0;

  [[nodiscard]] virtual MemoryStateOutputRange
  MemoryStateOutputs() const noexcept = 0;

  /**
   * Create a new copy of this StoreNode that consumes the provided \p memoryStates.
   *
   * @param memoryStates The memory states the newly created copy should consume.
   * @return A newly created StoreNode.
   */
  [[nodiscard]] virtual StoreNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const = 0;
};

/**
 * Represents a StoreNonVolatileOperation in an RVSDG.
 */
class StoreNonVolatileNode final : public StoreNode
{
private:
  StoreNonVolatileNode(
      rvsdg::Region & region,
      const StoreNonVolatileOperation & operation,
      const std::vector<jlm::rvsdg::output *> & operands)
      : StoreNode(region, operation, operands)
  {}

public:
  [[nodiscard]] const StoreNonVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] StoreNonVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  rvsdg::node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      rvsdg::output * value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(*address, *value, memoryStates, alignment));
  }

  static StoreNonVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value.Type());

    std::vector<rvsdg::output *> operands({ &address, &value });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    StoreNonVolatileOperation storeOperation(std::move(storedType), memoryStates.size(), alignment);
    return CreateNode(*address.region(), storeOperation, operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      const StoreNonVolatileOperation & storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, storeOperation, operands));
  }

  static StoreNonVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      const StoreNonVolatileOperation & storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new StoreNonVolatileNode(region, storeOperation, operands));
  }

private:
  static std::shared_ptr<const rvsdg::valuetype>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::valuetype>(type))
    {
      return storedType;
    }

    throw util::error("Expected value type.");
  }
};

/**
 * Represents a volatile LLVM store instruction.
 *
 * In contrast to LLVM, a volatile store requires in an RVSDG setting an I/O state as it
 * incorporates externally visible side-effects. This I/O state allows the volatile store operation
 * to be sequentialized with respect to other volatile memory accesses and I/O operations. This
 * additional I/O state is the main reason why volatile stores are modeled as its own operation and
 * volatile is not just a flag at the normal \ref StoreNonVolatileOperation.
 *
 * @see StoreNonVolatileOperation
 */
class StoreVolatileOperation final : public StoreOperation
{
public:
  ~StoreVolatileOperation() noexcept override;

  StoreVolatileOperation(
      std::shared_ptr<const rvsdg::valuetype> storedType,
      size_t numMemoryStates,
      size_t alignment)
      : StoreOperation(
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            CreateResultTypes(numMemoryStates),
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
      const variable * value,
      const variable * ioState,
      const variable * memoryState,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, ioState, memoryState });
  }

private:
  static std::shared_ptr<const rvsdg::valuetype>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::valuetype>(type))
      return storedType;

    throw jlm::util::error("Expected value type");
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::valuetype> storedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { PointerType::Create(), std::move(storedType), iostatetype::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ iostatetype::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> memoryStates(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), memoryStates.begin(), memoryStates.end());
    return types;
  }
};

/**
 * Represents a StoreVolatileOperation in an RVSDG.
 */
class StoreVolatileNode final : public StoreNode
{
  StoreVolatileNode(
      rvsdg::Region & region,
      const StoreVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : StoreNode(region, operation, operands)
  {}

public:
  [[nodiscard]] const StoreVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] StoreVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioStateInput = input(2);
    JLM_ASSERT(is<iostatetype>(ioStateInput->type()));
    return *ioStateInput;
  }

  [[nodiscard]] rvsdg::output &
  GetIoStateOutput() const noexcept
  {
    auto ioStateOutput = output(0);
    JLM_ASSERT(is<iostatetype>(ioStateOutput->type()));
    return *ioStateOutput;
  }

  rvsdg::node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static StoreVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      const StoreVolatileOperation & storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new StoreVolatileNode(region, storeOperation, operands));
  }

  static StoreVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & value,
      rvsdg::output & ioState,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value.Type());

    std::vector<rvsdg::output *> operands({ &address, &value, &ioState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    StoreVolatileOperation storeOperation(storedType, memoryStates.size(), alignment);
    return CreateNode(*address.region(), storeOperation, operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      const StoreVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, loadOperation, operands));
  }

private:
  static std::shared_ptr<const rvsdg::valuetype>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::valuetype>(type))
      return storedType;

    throw jlm::util::error("Expected value type.");
  }
};

}

#endif
