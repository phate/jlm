/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/clg.hpp>
#include <jlm/IR/operators.hpp>

#include <jive/arch/addresstype.h>
#include <jive/arch/memorytype.h>
#include <jive/types/float/flttype.h>

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
	return jive::mem::type::instance();
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

	if (index == 0) {
		static const jive::addr::type addrtype;
		return addrtype;
	}

	return jive::mem::type::instance();
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

/* bits2flt operator */

bits2flt_op::~bits2flt_op() noexcept
{}

bool
bits2flt_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const bits2flt_op*>(&other);
	return op && itype_ == op->itype_;
}

size_t
bits2flt_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
bits2flt_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return itype_;
}

size_t
bits2flt_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
bits2flt_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static jive::flt::type flttype;
	return flttype;
}

std::string
bits2flt_op::debug_string() const
{
	return "BITS2FLT";
}

std::unique_ptr<jive::operation>
bits2flt_op::copy() const
{
	return std::unique_ptr<jive::operation>(new bits2flt_op(*this));
}

/* flt2bits operator */

flt2bits_op::~flt2bits_op() noexcept
{}

bool
flt2bits_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const flt2bits_op*>(&other);
	return op && otype_ == op->otype_;
}

size_t
flt2bits_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
flt2bits_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	static jive::flt::type flttype;
	return flttype;
}

size_t
flt2bits_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
flt2bits_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return otype_;
}

std::string
flt2bits_op::debug_string() const
{
	return "FLT2BITS";
}

std::unique_ptr<jive::operation>
flt2bits_op::copy() const
{
	return std::unique_ptr<jive::operation>(new flt2bits_op(*this));
}

/* branch operator */

branch_op::~branch_op() noexcept
{}

bool
branch_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const branch_op*>(&other);
	return op && type_ == op->type_;
}

size_t
branch_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
branch_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return type_;
}

size_t
branch_op::nresults() const noexcept
{
	return 0;
}

const jive::base::type &
branch_op::result_type(size_t index) const noexcept
{
	JLM_ASSERT(0 && "Branch operator has no return types.");
}

std::string
branch_op::debug_string() const
{
	return "BRANCH";
}

std::unique_ptr<jive::operation>
branch_op::copy() const
{
	return std::unique_ptr<jive::operation>(new branch_op(*this));
}

}
