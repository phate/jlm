/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_STORE_HPP
#define JLM_IR_OPERATORS_STORE_HPP

#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/simple-normal-form.hpp>
#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

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
    const ptrtype & pointerType,
    size_t numStates,
    size_t alignment)
    : simple_op(
    CreateArgumentPorts(pointerType, numStates),
    std::vector<jive::port>(numStates, {MemoryStateType::Create()}))
    , Alignment_(alignment)
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

  [[nodiscard]] const jive::valuetype &
  GetValueType() const noexcept
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
    auto pointerType = dynamic_cast<const ptrtype*>(&address->type());
    if (!pointerType)
      throw jlm::error("expected pointer type.");

    StoreOperation op(*pointerType, 1, alignment);
    return tac::create(op, {address, value, state});
  }

  static std::vector<jive::output*>
  Create(
    jive::output * address,
    jive::output * value,
    const std::vector<jive::output*> & states,
    size_t alignment)
  {
    auto pointerType = dynamic_cast<const ptrtype*>(&address->type());
    if (!pointerType)
      throw jlm::error("expected pointer type.");

    if (states.empty())
      throw jlm::error("Expected at least one memory state.");

    std::vector<jive::output*> operands({address, value});
    operands.insert(operands.end(), states.begin(), states.end());

    StoreOperation op(*pointerType, states.size(), alignment);
    return jive::simple_node::create_normalized(address->region(), op, operands);
  }

private:
  static std::vector<jive::port>
  CreateArgumentPorts(
    const ptrtype & pointerType,
    size_t numStates)
  {
    MemoryStateType memoryStateType;
    std::vector<jive::port> ports({pointerType, pointerType.pointee_type()});
    std::vector<jive::port> states(numStates, {memoryStateType});
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t Alignment_;
};

}

#endif
