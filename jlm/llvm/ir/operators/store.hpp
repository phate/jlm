/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STORE_HPP
#define JLM_LLVM_IR_OPERATORS_STORE_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm {

/* store normal form */

class store_normal_form final : public jive::simple_normal_form {
public:
	virtual
	~store_normal_form();

	store_normal_form(
		const std::type_info & opclass,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept;

	virtual bool
	normalize_node(jive::node * node) const override;

	virtual std::vector<jive::output*>
	normalized_create(
		jive::region * region,
		const jive::simple_op & op,
		const std::vector<jive::output*> & operands) const override;

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

/** \brief Store Operation
 *
 * This operator is the Jlm equivalent of LLVM's store instruction.
 */
class StoreOperation final : public jive::simple_op {
public:
  ~StoreOperation() noexcept override;

  StoreOperation(
    const jive::valuetype & storedType,
    size_t numStates,
    size_t alignment)
    : simple_op(
      CreateOperandPorts(storedType, numStates),
      std::vector<jive::port>(numStates, {MemoryStateType::Create()}))
    , Alignment_(alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

  [[nodiscard]] const PointerType &
  GetPointerType() const noexcept
  {
    return *AssertedCast<const PointerType>(&argument(0).type());
  }

  [[nodiscard]] const jive::valuetype &
  GetStoredType() const noexcept
  {
    return *AssertedCast<const jive::valuetype>(&argument(1).type());
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

  static jlm::store_normal_form *
  GetNormalForm(jive::graph * graph) noexcept
  {
    return AssertedCast<jlm::store_normal_form>(graph->node_normal_form(typeid(StoreOperation)));
  }

  static std::unique_ptr<jlm::tac>
  Create(
    const variable * address,
    const variable * value,
    const variable * state,
    size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value->type());

    StoreOperation op(storedType, 1, alignment);
    return tac::create(op, {address, value, state});
  }

private:
  static const jive::valuetype &
  CheckAndExtractStoredType(const jive::type & type)
  {
    if (auto storedType = dynamic_cast<const jive::valuetype*>(&type))
      return *storedType;

    throw error("Expected ValueType");
  }

  static std::vector<jive::port>
  CreateOperandPorts(
    const jive::valuetype & storedType,
    size_t numStates)
  {
    std::vector<jive::port> ports({PointerType(), storedType});
    std::vector<jive::port> states(numStates, {MemoryStateType()});
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t Alignment_;
};

/** \brief StoreNode class
 *
 */
class StoreNode final : public jive::simple_node {
private:
  class MemoryStateInputIterator final : public jive::input::iterator<jive::simple_input> {
    friend StoreNode;

    constexpr explicit
    MemoryStateInputIterator(jive::simple_input * input)
      : jive::input::iterator<jive::simple_input>(input)
    {}

    [[nodiscard]] jive::simple_input *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->ninputs() > index+1
             ? node->input(index+1)
             : nullptr;
    }
  };

  class MemoryStateOutputIterator final : public jive::output::iterator<jive::simple_output> {
    friend StoreNode;

    constexpr explicit
    MemoryStateOutputIterator(jive::simple_output * output)
      : jive::output::iterator<jive::simple_output>(output)
    {}

    [[nodiscard]] jive::simple_output *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->noutputs() > index+1
             ? node->output(index+1)
             : nullptr;
    }
  };

  using MemoryStateInputRange = iterator_range<MemoryStateInputIterator>;
  using MemoryStateOutputRange = iterator_range<MemoryStateOutputIterator>;

  StoreNode(
    jive::region & region,
    const StoreOperation & operation,
    const std::vector<jive::output*> & operands)
    : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const StoreOperation&
  GetOperation() const noexcept
  {
    return *AssertedCast<const StoreOperation>(&operation());
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept
  {
    JLM_ASSERT(ninputs() > 2);
    return {MemoryStateInputIterator(input(2)), MemoryStateInputIterator(nullptr)};
  }

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept
  {
    JLM_ASSERT(noutputs() > 1);
    return {MemoryStateOutputIterator(output(0)), MemoryStateOutputIterator(nullptr)};
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

  [[nodiscard]] jive::input *
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return addressInput;
  }

  [[nodiscard]] jive::input *
  GetValueInput() const noexcept
  {
    auto valueInput = input(1);
    JLM_ASSERT(is<jive::valuetype>(valueInput->type()));
    return valueInput;
  }

  static std::vector<jive::output*>
  Create(
    jive::output * address,
    jive::output * value,
    const std::vector<jive::output*> & states,
    size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value->type());

    std::vector<jive::output*> operands({address, value});
    operands.insert(operands.end(), states.begin(), states.end());

    StoreOperation storeOperation(storedType, states.size(), alignment);
    return jive::outputs(new StoreNode(
      *address->region(),
      storeOperation,
      operands));
  }

  static std::vector<jive::output*>
  Create(
    jive::region & region,
    const StoreOperation & storeOperation,
    const std::vector<jive::output*> & operands)
  {
    return jive::outputs(new StoreNode(
      region,
      storeOperation,
      operands));
  }

private:
  static const jive::valuetype &
  CheckAndExtractStoredType(const jive::type & type)
  {
    if (auto storedType = dynamic_cast<const jive::valuetype*>(&type))
      return *storedType;

    throw error("Expected ValueType.");
  }
};

}

#endif
