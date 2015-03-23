/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/clg.hpp>
#include <jlm/IR/operators.hpp>

namespace jlm {

/* phi operator */

phi_op::~phi_op() noexcept
{}

bool
phi_op::operator==(const operation & other) const noexcept
{
	const phi_op * op = dynamic_cast<const phi_op*>(&other);
	return op && op->narguments_ == narguments_ && *op->type_ == *type_;
}

size_t
phi_op::narguments() const noexcept
{
	return narguments_;
}

const jive::base::type &
phi_op::argument_type(size_t index) const noexcept
{
	return *type_;
}

size_t
phi_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
phi_op::result_type(size_t index) const noexcept
{
	return *type_;
}

std::string
phi_op::debug_string() const
{
	return "PHI";
}

std::unique_ptr<jive::operation>
phi_op::copy() const
{
	return std::unique_ptr<jive::operation>(new phi_op(*this));
}

/* assignment operator */

assignment_op::~assignment_op() noexcept
{}

bool
assignment_op::operator==(const operation & other) const noexcept
{
	const assignment_op * op = dynamic_cast<const assignment_op*>(&other);
	return op && *op->type_ == *type_;
}

size_t
assignment_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
assignment_op::argument_type(size_t index) const noexcept
{
	return *type_;
}

size_t
assignment_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
assignment_op::result_type(size_t index) const noexcept
{
	return *type_;
}

std::string
assignment_op::debug_string() const
{
	return "ASSIGN";
}

std::unique_ptr<jive::operation>
assignment_op::copy() const
{
	return std::unique_ptr<jive::operation>(new assignment_op(*this));
}

/* apply operator */

apply_op::~apply_op() noexcept
{}

bool
apply_op::operator==(const operation & other) const noexcept
{
	const apply_op * op = dynamic_cast<const apply_op*>(&other);
	return op && op->function_ == function_;
}

size_t
apply_op::narguments() const noexcept
{
	return function_->type().narguments();
}

const jive::base::type &
apply_op::argument_type(size_t index) const noexcept
{
	return *function_->type().argument_type(index);
}

size_t
apply_op::nresults() const noexcept
{
	return function_->type().nreturns();
}

const jive::base::type &
apply_op::result_type(size_t index) const noexcept
{
	return *function_->type().return_type(index);
}

std::string
apply_op::debug_string() const
{
	return std::string("APPLY ").append(function_->name());
}

std::unique_ptr<jive::operation>
apply_op::copy() const
{
	return std::unique_ptr<jive::operation>(new apply_op(*this));
}

}
