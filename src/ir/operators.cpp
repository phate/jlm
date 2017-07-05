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
	return op && op->nodes_ == nodes_ && *op->type_ == *type_;
}

size_t
phi_op::narguments() const noexcept
{
	return nodes_.size();
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
	std::string str("[");
	for (size_t n = 0; n < narguments(); n++) {
		str += strfmt(node(n));
		if (n != narguments()-1)
			str += ", ";
	}
	str += "]";

	return "PHI" + str;
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

/* pointer compare operator */

ptrcmp_op::~ptrcmp_op()
{}

bool
ptrcmp_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::ptrcmp_op*>(&other);
	return op && op->ptype_ == ptype_ && op->cmp_ == cmp_;
}

size_t
ptrcmp_op::narguments() const noexcept
{
	return 2;
}

const jive::base::type &
ptrcmp_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return ptype_;
}

size_t
ptrcmp_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
ptrcmp_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());

	static jive::bits::type bits1(1);
	return bits1;
}

std::string
ptrcmp_op::debug_string() const
{
	static std::unordered_map<jlm::cmp, std::string> map({
		{cmp::eq, "eq"}, {cmp::ne, "ne"}, {cmp::gt, "gt"}
	, {cmp::ge, "ge"}, {cmp::lt, "lt"}, {cmp::le, "le"}
	});

	JLM_DEBUG_ASSERT(map.find(cmp()) != map.end());
	return "PTRCMP " + map[cmp()];
}

std::unique_ptr<jive::operation>
ptrcmp_op::copy() const
{
	return std::unique_ptr<jive::operation>(new ptrcmp_op(*this));
}

/* zext operator */

zext_op::~zext_op()
{}

bool
zext_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::zext_op*>(&other);
	return op && op->nsrcbits() == nsrcbits() && op->ndstbits() == ndstbits();
}

size_t
zext_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
zext_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srctype_;
}

size_t
zext_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
zext_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dsttype_;
}

std::string
zext_op::debug_string() const
{
	return "ZEXT";
}

std::unique_ptr<jive::operation>
zext_op::copy() const
{
	return std::unique_ptr<jive::operation>(new zext_op(*this));
}

/* floating point constant operator */

fpconstant_op::~fpconstant_op()
{}

bool
fpconstant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::fpconstant_op*>(&other);
	return op && op->constant() == constant();
}

size_t
fpconstant_op::narguments() const noexcept
{
	return 0;
}

const jive::base::type &
fpconstant_op::argument_type(size_t) const noexcept
{
	JLM_ASSERT(0);
}

size_t
fpconstant_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
fpconstant_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return type_;
}

std::string
fpconstant_op::debug_string() const
{
	return strfmt(constant());
}

std::unique_ptr<jive::operation>
fpconstant_op::copy() const
{
	return std::unique_ptr<jive::operation>(new fpconstant_op(*this));
}

/* floating point comparison operator */

fpcmp_op::~fpcmp_op()
{}

bool
fpcmp_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::fpcmp_op*>(&other);
	return op && op->cmp() == cmp() && op->size() == size();
}

size_t
fpcmp_op::narguments() const noexcept
{
	return 2;
}

const jive::base::type &
fpcmp_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return type_;
}

size_t
fpcmp_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
fpcmp_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static jive::bits::type bits1(1);
	return bits1;
}

std::string
fpcmp_op::debug_string() const
{
	static std::unordered_map<fpcmp, std::string> map({
	  {fpcmp::oeq, "oeq"}, {fpcmp::ogt, "ogt"}, {fpcmp::oge, "oge"}, {fpcmp::olt, "olt"}
	, {fpcmp::ole, "ole"}, {fpcmp::one, "one"}, {fpcmp::ord, "ord"}, {fpcmp::ueq, "ueq"}
	, {fpcmp::ugt, "ugt"}, {fpcmp::uge, "uge"}, {fpcmp::ult, "ult"}, {fpcmp::ule, "ule"}
	, {fpcmp::une, "une"}, {fpcmp::uno, "uno"}
	});

	JLM_DEBUG_ASSERT(map.find(cmp()) != map.end());
	return "FPCMP " + map[cmp()];
}

std::unique_ptr<jive::operation>
fpcmp_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::fpcmp_op(*this));
}

/* undef constant operator */

undef_constant_op::~undef_constant_op()
{}

bool
undef_constant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::undef_constant_op*>(&other);
	return op && op->type() == type();
}

size_t
undef_constant_op::narguments() const noexcept
{
	return 0;
}

const jive::base::type &
undef_constant_op::argument_type(size_t index) const noexcept
{
	JLM_ASSERT(0);
}

size_t
undef_constant_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
undef_constant_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return *type_;
}

std::string
undef_constant_op::debug_string() const
{
	return "undef";
}

std::unique_ptr<jive::operation>
undef_constant_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::undef_constant_op(*this));
}

/* floating point arithmetic operator */

fpbin_op::~fpbin_op()
{}

bool
fpbin_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::fpbin_op*>(&other);
	return op && op->fpop() == fpop() && op->size() == size();
}

size_t
fpbin_op::narguments() const noexcept
{
	return 2;
}

const jive::base::type &
fpbin_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return type_;
}

size_t
fpbin_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
fpbin_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return type_;
}

std::string
fpbin_op::debug_string() const
{
	static std::unordered_map<jlm::fpop, std::string> map({
		{fpop::add, "add"}, {fpop::sub, "sub"}, {fpop::mul, "mul"}
	, {fpop::div, "div"}, {fpop::mod, "mod"}
	});

	JLM_DEBUG_ASSERT(map.find(fpop()) != map.end());
	return "FPOP " + map[fpop()];
}

std::unique_ptr<jive::operation>
fpbin_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::fpbin_op(*this));
}

/* fpext operator */

fpext_op::~fpext_op()
{}

bool
fpext_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::fpext_op*>(&other);
	return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

size_t
fpext_op::narguments() const noexcept
{
	return 1;
}

const jive::base::type &
fpext_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srctype_;
}

size_t
fpext_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
fpext_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dsttype_;
}

std::string
fpext_op::debug_string() const
{
	return "fpext";
}

std::unique_ptr<jive::operation>
fpext_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::fpext_op(*this));
}

/* valist operator */

valist_op::~valist_op()
{}

bool
valist_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::valist_op*>(&other);
	return op && op->arguments_ == arguments_;
}

size_t
valist_op::narguments() const noexcept
{
	return arguments_.size();
}

const jive::base::type &
valist_op::argument_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return *arguments_[index];
}

size_t
valist_op::nresults() const noexcept
{
	return 1;
}

const jive::base::type &
valist_op::result_type(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static varargtype vatype;
	return vatype;
}

std::string
valist_op::debug_string() const
{
	return "valist";
}

std::unique_ptr<jive::operation>
valist_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::valist_op(*this));
}

}
