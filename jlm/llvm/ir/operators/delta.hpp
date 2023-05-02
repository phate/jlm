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

namespace jlm {
namespace delta {

/** \brief Delta operation
*/
class operation final : public jive::structural_op {
public:
	~operation() override;

	operation(
		const jive::valuetype & type,
		const std::string & name,
		const jlm::linkage & linkage,
    std::string section,
		bool constant)
	: constant_(constant)
	, name_(name)
  , Section_(std::move(section))
	, linkage_(linkage)
	, type_(type.copy())
	{}

	operation(const operation & other)
	: constant_(other.constant_)
	, name_(other.name_)
  , Section_(other.Section_)
	, linkage_(other.linkage_)
	, type_(other.type_->copy())
	{}

	operation(operation && other) noexcept
	: constant_(other.constant_)
	, name_(std::move(other.name_))
  , Section_(std::move(other.Section_))
	, linkage_(other.linkage_)
	, type_(std::move(other.type_))
	{}

	operation &
	operator=(const operation&) = delete;

	operation &
	operator=(operation&&) = delete;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual bool
	operator==(const jive::operation & other) const noexcept override;

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

	const jlm::linkage &
	linkage() const noexcept
	{
		return linkage_;
	}

	bool
	constant() const noexcept
	{
		return constant_;
	}

	[[nodiscard]] const jive::valuetype &
	type() const noexcept
	{
    return *AssertedCast<jive::valuetype>(type_.get());
	}

private:
	bool constant_;
	std::string name_;
  std::string Section_;
	jlm::linkage linkage_;
	std::unique_ptr<jive::type> type_;
};

class cvargument;
class cvinput;
class output;
class result;

/** \brief Delta node
*
* A delta node represents a global variable in the RVSDG. Its creation requires the invocation
* of two functions: \ref create() and \ref finalize(). First, a delta node is create by invoking
* \ref create(). The delta's dependencies can then be added using the \ref add_ctxvar() method,
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
class node final : public jive::structural_node {
	class cviterator;
	class cvconstiterator;

	using ctxvar_range = iterator_range<cviterator>;
	using ctxvar_constrange = iterator_range<cvconstiterator>;

public:
	~node() override;

private:
	node(
		jive::region * parent,
		delta::operation && op)
	: structural_node(op, parent, 1)
	{}

public:
	ctxvar_range
	ctxvars();

	ctxvar_constrange
	ctxvars() const;

	jive::region *
	subregion() const noexcept
	{
		return structural_node::subregion(0);
	}

	const delta::operation &
	operation() const noexcept
	{
		return *static_cast<const delta::operation*>(&structural_node::operation());
	}

	[[nodiscard]] const jive::valuetype &
	type() const noexcept
	{
		return operation().type();
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

	const jlm::linkage &
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
	add_ctxvar(jive::output * origin);

	cvinput *
	input(size_t n) const noexcept;

	delta::cvargument *
	cvargument(size_t n) const noexcept;

	delta::output *
	output() const noexcept;

	delta::result *
	result() const noexcept;

	virtual delta::node *
	copy(
		jive::region * region,
		const std::vector<jive::output*> & operands) const override;

	virtual delta::node *
	copy(
		jive::region * region,
		jive::substitution_map & smap) const override;

	/**
	* Creates a delta node in the region \p parent with the pointer type \p type and name \p name.
	* After the invocation of \ref create(), the delta node has no inputs or outputs.
	* Free variables can be added to the delta node using \ref add_ctxvar(). The generation of the
	* node can be finished using the \ref finalize() method.
	*
	* \param parent The region where the delta node is created.
	* \param type The delta node's type.
	* \param name The delta node's name.
	* \param linkage The delta node's linkage.
	* \param Section The delta node's section.
	* \param constant True, if the delta node is constant, otherwise false.
	*
	* \return A delta node without inputs or outputs.
	*/
	static node *
	Create(
		jive::region * parent,
    const jive::valuetype & type,
		const std::string & name,
		const jlm::linkage & linkage,
    std::string section,
		bool constant)
	{
		delta::operation op(type, name, linkage, std::move(section), constant);
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
	finalize(jive::output * result);
};

/** \brief Delta context variable input
*/
class cvinput final : public jive::structural_input {
	friend ::jlm::delta::node;

public:
	~cvinput() override;

private:
	cvinput(
		delta::node * node,
		jive::output * origin)
	: structural_input(node, origin, origin->port())
	{}

	static cvinput *
	create(
		delta::node * node,
		jive::output * origin)
	{
		auto input = std::unique_ptr<cvinput>(new cvinput(node, origin));
		return static_cast<cvinput*>(node->append_input(std::move(input)));
	}

public:
	cvargument *
	argument() const noexcept;

	delta::node *
	node() const noexcept
	{
		return static_cast<delta::node*>(structural_input::node());
	}
};

/** \brief Delta context variable iterator
*/
class node::cviterator final : public jive::input::iterator<cvinput> {
	friend ::jlm::delta::node;

	constexpr
	cviterator(cvinput * input)
	: jive::input::iterator<cvinput>(input)
	{}

	virtual cvinput *
	next() const override
	{
		auto node = value()->node();
		auto index = value()->index();

		return node->ninputs() > index+1 ? node->input(index+1) : nullptr;
	}
};

/** \brief Delta context variable const iterator
*/
class node::cvconstiterator final : public jive::input::constiterator<cvinput> {
	friend ::jlm::delta::node;

	constexpr
	cvconstiterator(const cvinput * input)
	: jive::input::constiterator<cvinput>(input)
	{}

	virtual const cvinput *
	next() const override
	{
		auto node = value()->node();
		auto index = value()->index();

		return node->ninputs() > index+1 ? node->input(index+1) : nullptr;
	}
};

/** \brief Delta output
*/
class output final : public jive::structural_output {
	friend ::jlm::delta::node;

public:
	~output() override;

private:
	output(
		delta::node * node,
		const jive::port & port)
	: structural_output(node, port)
	{}

	static output *
	create(
		delta::node * node,
		const jive::port & port)
	{
		auto output = std::unique_ptr<delta::output>(new delta::output(node, port));
		return static_cast<delta::output*>(node->append_output(std::move(output)));
	}

public:
	delta::node *
	node() const noexcept
	{
		return static_cast<delta::node*>(structural_output::node());
	}
};

/** \brief Delta context variable argument
*/
class cvargument final : public jive::argument {
	friend ::jlm::delta::node;

public:
	~cvargument() override;

private:
	cvargument(
		jive::region * region,
		cvinput * input)
	: jive::argument(region, input, input->port())
	{}

	static cvargument *
	create(
		jive::region * region,
		delta::cvinput * input)
	{
		auto argument = new cvargument(region, input);
		region->append_argument(argument);
		return argument;
	}

public:
	cvinput *
	input() const noexcept
	{
		return static_cast<cvinput*>(jive::argument::input());
	}
};

/** \brief Delta result
*/
class result final : public jive::result {
	friend ::jlm::delta::node;

public:
	~result() override;

private:
	result(jive::output * origin)
	: jive::result(origin->region(), origin, nullptr, origin->port())
	{}

	static result *
	create(jive::output * origin)
	{
		auto result = new delta::result(origin);
		origin->region()->append_result(result);
		return result;
	}

public:
	delta::output *
	output() const noexcept
	{
		return static_cast<delta::output*>(jive::result::output());
	}
};

}}

#endif
