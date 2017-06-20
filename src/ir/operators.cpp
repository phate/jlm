/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/clg.hpp>
#include <jlm/ir/operators.hpp>

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
	auto op = dynamic_cast<const jlm::alloca_op*>(&other);
	return op && op->atype_ == atype_ && op->btype_ == btype_;
}

size_t
alloca_op::narguments() const noexcept
{
	return 2;
}

const jive::base::type &
alloca_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return btype_;

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
	if (index == 0)
		return atype_;

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

/* ptr_constant_null operator */

ptr_constant_null_op::~ptr_constant_null_op() noexcept
{}

bool
ptr_constant_null_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const ptr_constant_null_op*>(&other);
	return op && op->ptype_ == ptype_;
}

size_t
ptr_constant_null_op::narguments() const noexcept
{
	return 0;
}

const jive::base::type &
ptr_constant_null_op::argument_type(size_t index) const noexcept
{
	JLM_ASSERT(0);
}

size_t
ptr_constant_null_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
ptr_constant_null_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return ptype_;
}

std::string
ptr_constant_null_op::debug_string() const
{
	return "NULLPTR";
}

std::unique_ptr<jive::operation>
ptr_constant_null_op::copy() const
{
	return std::unique_ptr<jive::operation>(new ptr_constant_null_op(*this));
}

/* load operator */

load_op::~load_op() noexcept
{}

bool
load_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const load_op*>(&other);
	return op && op->nstates_ == nstates_ && op->ptype_ == ptype_;
}

size_t
load_op::narguments() const noexcept
{
	return 1 + nstates();
}

const jive::base::type &
load_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return ptype_;

	return jive::mem::type::instance();
}

size_t
load_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
load_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return pointee_type();
}

std::string
load_op::debug_string() const
{
	return "LOAD";
}

std::unique_ptr<jive::operation>
load_op::copy() const
{
	return std::unique_ptr<jive::operation>(new load_op(*this));
}

/* store operator */

store_op::~store_op() noexcept
{}

bool
store_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const store_op*>(&other);
	return op && op->nstates_ == nstates_ && op->ptype_ == ptype_;
}

size_t
store_op::narguments() const noexcept
{
	return 2 + nstates();
}

const jive::base::type &
store_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return ptype_;

	if (index == 1)
		return value_type();

	return jive::mem::type::instance();
}

size_t
store_op::nresults() const noexcept
{
	return nstates();
}

const jive::base::type &
store_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return jive::mem::type::instance();
}

std::string
store_op::debug_string() const
{
	return "STORE";
}

std::unique_ptr<jive::operation>
store_op::copy() const
{
	return std::unique_ptr<jive::operation>(new store_op(*this));
}

/* bits2ptr operator */

bits2ptr_op::~bits2ptr_op()
{}

bool
bits2ptr_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::bits2ptr_op*>(&other);
	return op && op->btype_ == btype_ && op->ptype_ == ptype_;
}

size_t
bits2ptr_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
bits2ptr_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return btype_;
}

size_t
bits2ptr_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
bits2ptr_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return ptype_;
}

std::string
bits2ptr_op::debug_string() const
{
	return "BITS2PTR";
}

std::unique_ptr<jive::operation>
bits2ptr_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::bits2ptr_op(*this));
}

/* ptr2bits operator */

ptr2bits_op::~ptr2bits_op()
{}

bool
ptr2bits_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::ptr2bits_op*>(&other);
	return op && op->btype_ == btype_ && op->ptype_ == ptype_;
}

size_t
ptr2bits_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
ptr2bits_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return ptype_;
}

size_t
ptr2bits_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
ptr2bits_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return btype_;
}

std::string
ptr2bits_op::debug_string() const
{
	return "PTR2BITS";
}

std::unique_ptr<jive::operation>
ptr2bits_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::ptr2bits_op(*this));
}

/* ptroffset operator */

ptroffset_op::~ptroffset_op()
{}

bool
ptroffset_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::ptroffset_op*>(&other);
	return op && op->ptype_ == ptype_ && op->btypes_ == btypes_ && op->rtype_ == rtype_;
}

size_t
ptroffset_op::narguments() const noexcept
{
	return 1 + nindices();
}

const jive::base::type &
ptroffset_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return ptype_;

	return btypes_[index-1];
}

size_t
ptroffset_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
ptroffset_op::result_type(size_t index) const noexcept
{
	return rtype_;
}

std::string
ptroffset_op::debug_string() const
{
	return "PTROFFSET";
}

std::unique_ptr<jive::operation>
ptroffset_op::copy() const
{
	return std::unique_ptr<jive::operation>(new ptroffset_op(*this));
}

/* data array constant operator */

data_array_constant_op::~data_array_constant_op()
{}

bool
data_array_constant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const data_array_constant_op*>(&other);
	return op && op->type() == type() && op->size() == size();
}

size_t
data_array_constant_op::narguments() const noexcept
{
	return size();
}

const jive::base::type &
data_array_constant_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return type();
}

size_t
data_array_constant_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
data_array_constant_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return type_;
}

std::string
data_array_constant_op::debug_string() const
{
	return "ARRAYCONSTANT";
}

std::unique_ptr<jive::operation>
data_array_constant_op::copy() const
{
	return std::unique_ptr<jive::operation>(new data_array_constant_op(*this));
}

}
