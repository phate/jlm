/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/clg.hpp>
#include <jlm/IR/operators.hpp>

#include <jive/arch/addresstype.h>
#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>

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

/* select operator */

select_op::~select_op() noexcept
{}

bool
select_op::operator==(const operation & other) const noexcept
{
	const select_op * op = dynamic_cast<const select_op*>(&other);
	return op && *op->type_ == *type_;
}

size_t
select_op::narguments() const noexcept
{
	return 3;
}

const jive::base::type &
select_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());

	if (index == 0) {
		static const jive::bits::type bit1(1);
		return bit1;
	}

	return *type_;
}

size_t
select_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
select_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return *type_;
}

std::string
select_op::debug_string() const
{
	return "SELECT";
}

std::unique_ptr<jive::operation>
select_op::copy() const
{
	return std::unique_ptr<jive::operation>(new select_op(*this));
}

/* alloca operator */

alloca_op::~alloca_op() noexcept
{}

bool
alloca_op::operator==(const operation & other) const noexcept
{
	return false;
}

size_t
alloca_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
alloca_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return jive::mem::memtype;
}

size_t
alloca_op::nresults() const noexcept
{
	return 2;
}

const jive::base::type &
alloca_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());

	if (index == 0)
		return jive::addr::addrtype;

	return jive::mem::memtype;
}

std::string
alloca_op::debug_string() const
{
	return "ALLOCA";
}

std::unique_ptr<jive::operation>
alloca_op::copy() const
{
	return std::unique_ptr<jive::operation>(new alloca_op(*this));
}

}
