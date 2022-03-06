/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_LOAD_HPP
#define JLM_IR_OPERATORS_LOAD_HPP

#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/simple-normal-form.hpp>
#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

/* load normal form */

class load_normal_form final : public jive::simple_normal_form {
public:
	virtual
	~load_normal_form();

	load_normal_form(
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
	set_load_store_alloca_reducible(bool enable) noexcept
	{
		enable_load_store_alloca_ = enable;
	}

	inline bool
	get_load_store_alloca_reducible() const noexcept
	{
		return enable_load_store_alloca_;
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
	bool enable_load_store_alloca_;
};

/** \brief LoadOperation class
 *
 * This operator is the Jlm equivalent of LLVM's load instruction.
 */
class LoadOperation final : public jive::simple_op {
public:
  ~LoadOperation() noexcept override;

  LoadOperation(
    const ptrtype & pointerType,
    size_t numStates,
    size_t alignment)
    : simple_op(CreatePorts(pointerType, numStates), CreatePorts(pointerType.pointee_type(), numStates))
    , alignment_(alignment)
  {
    if (numStates == 0)
      throw error("Expected at least one state.");
  }

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

  [[nodiscard]] const ptrtype &
  GetPointerType() const noexcept
  {
    return *AssertedCast<const ptrtype>(&argument(0).type());
  }

  [[nodiscard]] size_t
  NumStates() const noexcept
  {
    return narguments()-1;
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return alignment_;
  }

  static jlm::load_normal_form *
  GetNormalForm(jive::graph * graph) noexcept
  {
    return AssertedCast<load_normal_form>(graph->node_normal_form(typeid(LoadOperation)));
  }

  static std::unique_ptr<jlm::tac>
  Create(
    const variable * address,
    const variable * state,
    size_t alignment)
  {
    auto & pointerType = CheckAndConvertType(address->type());

    LoadOperation operation(pointerType, 1, alignment);
    return tac::create(operation, {address, state});
  }

  static std::vector<jive::output*>
  Create(
    jive::output * address,
    const std::vector<jive::output*> & states,
    size_t alignment)
  {
    auto & pointerType = CheckAndConvertType(address->type());

    if (states.empty())
      throw error("Expected at least one memory state.");

    std::vector<jive::output*> operands({address});
    operands.insert(operands.end(), states.begin(), states.end());

    LoadOperation operation(pointerType, states.size(), alignment);
    return jive::simple_node::create_normalized(address->region(), operation, operands);
  }

private:
  static const ptrtype &
  CheckAndConvertType(const jive::type & type)
  {
    if (auto pointerType = dynamic_cast<const ptrtype*>(&type))
      return *pointerType;

    throw error("Expected pointer type.");
  }

  static std::vector<jive::port>
  CreatePorts(const jive::valuetype & valueType, size_t numStates)
  {
    std::vector<jive::port> ports(1, {valueType});
    std::vector<jive::port> states(numStates, {MemoryStateType::Create()});
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t alignment_;
};

class LoadNode final : public jive::simple_node {
private:
  class MemoryStateInputIterator final : public jive::input::iterator<jive::simple_input> {
    friend LoadNode;

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
    friend LoadNode;

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

  LoadNode(
    jive::region & region,
    const LoadOperation & operation,
    const std::vector<jive::output*> & operands)
    : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const LoadOperation&
  GetOperation() const noexcept
  {
    return *AssertedCast<const LoadOperation>(&operation());
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept
  {
    JLM_ASSERT(ninputs() > 1);
    return {MemoryStateInputIterator(input(1)), MemoryStateInputIterator(nullptr)};
  }

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept
  {
    JLM_ASSERT(noutputs() > 1);
    return {MemoryStateOutputIterator(output(1)), MemoryStateOutputIterator(nullptr)};
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
    JLM_ASSERT(is<ptrtype>(addressInput->type()));
    return addressInput;
  }

  [[nodiscard]] jive::output *
  GetValueOutput() const noexcept
  {
    auto valueOutput = output(0);
    JLM_ASSERT(is<jive::valuetype>(valueOutput->type()));
    return valueOutput;
  }

  static std::vector<jive::output*>
  Create(
    jive::output * address,
    const std::vector<jive::output*> & states,
    size_t alignment)
  {
    auto & pointerType = CheckAndConvertType(address->type());

    std::vector<jive::output*> operands({address});
    operands.insert(operands.end(), states.begin(), states.end());

    LoadOperation loadOperation(pointerType, states.size(), alignment);
    return jive::outputs(new LoadNode(
      *address->region(),
      loadOperation,
      operands));
  }

  static std::vector<jive::output*>
  Create(
    jive::region & region,
    const LoadOperation & loadOperation,
    const std::vector<jive::output*> & operands)
  {
    return jive::outputs(new LoadNode(
      region,
      loadOperation,
      operands));
  }

private:
  static const ptrtype &
  CheckAndConvertType(const jive::type & type)
  {
    if (auto pointerType = dynamic_cast<const ptrtype*>(&type))
      return *pointerType;

    throw error("Expected pointer type.");
  }
};

}

#endif
