/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_H
#define JLM_IR_OPERATORS_H

#include <jive/types/function/fcttype.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/operators/nullary.h>

namespace jlm {

/* phi operator */

class phi_op final : public jive::operation {
public:
	virtual
	~phi_op() noexcept;

	/* FIXME: check that number of arguments is not zero */
	inline
	phi_op(size_t narguments, const jive::base::type & type)
	: narguments_(narguments)
	, type_(type.copy())
	{}

	inline
	phi_op(const phi_op & other)
	: narguments_(other.narguments_)
	, type_(other.type_->copy())
	{}

	inline
	phi_op(phi_op && other) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::base::type &
	type() const noexcept
	{
		return *type_;
	}

private:
	size_t narguments_;
	std::unique_ptr<jive::base::type> type_;
};

/* assignment operator */

class assignment_op final : public jive::operation {
public:
	virtual
	~assignment_op() noexcept;

	inline
	assignment_op(const jive::base::type & type)
	: type_(type.copy())
	{}

	inline
	assignment_op(const assignment_op & other)
	: type_(other.type_->copy())
	{}

	inline
	assignment_op(assignment_op && other) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	std::unique_ptr<jive::base::type> type_;
};

/* apply operator */

/*
	We require an apply operator that does not expect a function as first argument. This is
	due to the call and control flow graph being separate representations: one represents
	functions and the other the body of a function. Thus, we have no way of handing the
	function as first operand to the "normal" apply operator. We use in this operator a
	clg_node as function identifier instead, resolving the issue in the RVSDG construction
	phase.
*/

class clg_node;

class apply_op final : public jive::operation {
public:
	virtual
	~apply_op() noexcept;

	inline
	apply_op(const clg_node * function)
	: function_(function)
	{}

	inline
	apply_op(const apply_op & other) = default;

	inline
	apply_op(apply_op && other) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const clg_node *
	function() const noexcept
	{
		return function_;
	}

private:
	const clg_node * function_;
};

/* select operator */

class select_op final : public jive::operation {
public:
	virtual
	~select_op() noexcept;

	inline
	select_op(const jive::base::type & type)
	: type_(type.copy())
	{}

	inline
	select_op(const select_op & other)
	: type_(other.type_->copy())
	{}

	inline
	select_op(select_op && other) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::base::type &
	type() const noexcept
	{
		return *type_;
	}

private:
	std::unique_ptr<jive::base::type> type_;
};

}

#endif
