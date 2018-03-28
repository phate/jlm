/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/callgraph.hpp>
#include <jlm/ir/operators/operators.hpp>

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/float/flttype.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

namespace jlm {

/* phi operator */

phi_op::~phi_op() noexcept
{}

bool
phi_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const phi_op*>(&other);
	return op && op->nodes_ == nodes_ && op->port_ == port_;
}

size_t
phi_op::narguments() const noexcept
{
	return nodes_.size();
}

const jive::port &
phi_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
phi_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
phi_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
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
	auto op = dynamic_cast<const assignment_op*>(&other);
	return op && op->port_ == port_;
}

size_t
assignment_op::narguments() const noexcept
{
	return 2;
}

const jive::port &
assignment_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
assignment_op::nresults() const noexcept
{
	return 0;
}

const jive::port &
assignment_op::result(size_t index) const noexcept
{
	JLM_ASSERT(0);
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
	auto op = dynamic_cast<const select_op*>(&other);
	return op && op->port_ == port_;
}

size_t
select_op::narguments() const noexcept
{
	return 3;
}

const jive::port &
select_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());

	if (index == 0) {
		static const jive::port p(jive::bits::type(1));
		return p;
	}

	return port_;
}

size_t
select_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
select_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
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

/* bits2flt operator */

bits2flt_op::~bits2flt_op() noexcept
{}

bool
bits2flt_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const bits2flt_op*>(&other);
	return op && port_ == op->port_;
}

size_t
bits2flt_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
bits2flt_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
bits2flt_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
bits2flt_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static const jive::flt::type t;
	static const jive::port p(t);
	return p;
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

/* fp2ui operator */

fp2ui_op::~fp2ui_op() noexcept
{}

bool
fp2ui_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const fp2ui_op*>(&other);
	return op
	    && op->srcport_ == srcport_
	    && op->dstport_ == dstport_;
}

size_t
fp2ui_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
fp2ui_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
fp2ui_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fp2ui_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
fp2ui_op::debug_string() const
{
	return "FP2UI";
}

std::unique_ptr<jive::operation>
fp2ui_op::copy() const
{
	return std::unique_ptr<jive::operation>(new fp2ui_op(*this));
}

/* fp2si operator */

fp2si_op::~fp2si_op() noexcept
{}

bool
fp2si_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const fp2si_op*>(&other);
	return op
	    && op->srcport_ == srcport_
	    && op->dstport_ == dstport_;
}

size_t
fp2si_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
fp2si_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
fp2si_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fp2si_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
fp2si_op::debug_string() const
{
	return "FP2UI";
}

std::unique_ptr<jive::operation>
fp2si_op::copy() const
{
	return std::unique_ptr<jive::operation>(new fp2si_op(*this));
}

/* branch operator */

branch_op::~branch_op() noexcept
{}

bool
branch_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const branch_op*>(&other);
	return op && port_ == op->port_;
}

size_t
branch_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
branch_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
branch_op::nresults() const noexcept
{
	return 0;
}

const jive::port &
branch_op::result(size_t index) const noexcept
{
	JLM_ASSERT(0 && "Branch operator has no return port.");
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
	return op && op->port_ == port_;
}

size_t
ptr_constant_null_op::narguments() const noexcept
{
	return 0;
}

const jive::port &
ptr_constant_null_op::argument(size_t index) const noexcept
{
	JLM_ASSERT(0);
}

size_t
ptr_constant_null_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
ptr_constant_null_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
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

/* bits2ptr operator */

bits2ptr_op::~bits2ptr_op()
{}

bool
bits2ptr_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::bits2ptr_op*>(&other);
	return op && op->bport_ == bport_ && op->pport_ == pport_;
}

size_t
bits2ptr_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
bits2ptr_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return bport_;
}

size_t
bits2ptr_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
bits2ptr_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return pport_;
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
	return op && op->bport_ == bport_ && op->pport_ == pport_;
}

size_t
ptr2bits_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
ptr2bits_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return pport_;
}

size_t
ptr2bits_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
ptr2bits_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return bport_;
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

/* data array constant operator */

data_array_constant_op::~data_array_constant_op()
{}

bool
data_array_constant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const data_array_constant_op*>(&other);
	return op && op->aport_ == aport_ && op->eport_ == eport_;
}

size_t
data_array_constant_op::narguments() const noexcept
{
	return size();
}

const jive::port &
data_array_constant_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return eport_;
}

size_t
data_array_constant_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
data_array_constant_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return aport_;
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
	return op && op->port_ == port_ && op->cmp_ == cmp_;
}

size_t
ptrcmp_op::narguments() const noexcept
{
	return 2;
}

const jive::port &
ptrcmp_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
ptrcmp_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
ptrcmp_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());

	static const jive::port p(jive::bits::type(1));
	return p;
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
	return op && op->srcport_ == srcport_ && op->dstport_ == dstport_;
}

size_t
zext_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
zext_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
zext_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
zext_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
zext_op::debug_string() const
{
	return strfmt("ZEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<jive::operation>
zext_op::copy() const
{
	return std::unique_ptr<jive::operation>(new zext_op(*this));
}

jive_unop_reduction_path_t
zext_op::can_reduce_operand(const jive::output * operand) const noexcept
{
	if (jive::is_bitconstant_node(producer(operand)))
		return jive_unop_reduction_constant;

	return jive_unop_reduction_none;
}

jive::output *
zext_op::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * operand) const
{
	if (path == jive_unop_reduction_constant) {
		auto c = static_cast<const jive::bits::constant_op*>(&producer(operand)->operation());
		return create_bitconstant(operand->node()->region(), c->value().zext(ndstbits()-nsrcbits()));
	}

	return nullptr;
}

/* floating point constant operator */

fpconstant_op::~fpconstant_op()
{}

bool
fpconstant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::fpconstant_op*>(&other);
	return op && op->constant().compare(constant()) == llvm::APFloatBase::cmpEqual;
}

size_t
fpconstant_op::narguments() const noexcept
{
	return 0;
}

const jive::port &
fpconstant_op::argument(size_t) const noexcept
{
	JLM_ASSERT(0);
}

size_t
fpconstant_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fpconstant_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
}

std::string
fpconstant_op::debug_string() const
{
	llvm::SmallVector<char, 32> v;
	constant().toString(v, 32, 0);

	std::string s("FP(");
	for (const auto & c : v)
		s += c;
	s += ")";

	return s;
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
	return op && op->cmp_ == cmp_ && op->port_ == port_;
}

size_t
fpcmp_op::narguments() const noexcept
{
	return 2;
}

const jive::port &
fpcmp_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
fpcmp_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fpcmp_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static const jive::port p(jive::bits::type(1));
	return p;
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
	return op && op->port_ == port_;
}

size_t
undef_constant_op::narguments() const noexcept
{
	return 0;
}

const jive::port &
undef_constant_op::argument(size_t index) const noexcept
{
	JLM_ASSERT(0);
}

size_t
undef_constant_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
undef_constant_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
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

const jive::port &
fpbin_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return port_;
}

size_t
fpbin_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fpbin_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return port_;
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

const jive::port &
fpext_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
fpext_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fpext_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
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

/* fptrunc operator */

fptrunc_op::~fptrunc_op()
{}

bool
fptrunc_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const fptrunc_op*>(&other);
	return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

size_t
fptrunc_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
fptrunc_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
fptrunc_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
fptrunc_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
fptrunc_op::debug_string() const
{
	return "fptrunc";
}

std::unique_ptr<jive::operation>
fptrunc_op::copy() const
{
	return std::unique_ptr<jive::operation>(new fptrunc_op(*this));
}

/* valist operator */

valist_op::~valist_op()
{}

bool
valist_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::valist_op*>(&other);
	return op && op->ports_ == ports_;
}

size_t
valist_op::narguments() const noexcept
{
	return ports_.size();
}

const jive::port &
valist_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return ports_[index];
}

size_t
valist_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
valist_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static const varargtype vatype;
	static const jive::port p(vatype);
	return p;
}

std::string
valist_op::debug_string() const
{
	return "VALIST";
}

std::unique_ptr<jive::operation>
valist_op::copy() const
{
	return std::unique_ptr<jive::operation>(new jlm::valist_op(*this));
}

/* bitcast operator */

bitcast_op::~bitcast_op()
{}

bool
bitcast_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const bitcast_op*>(&other);
	return op && op->srcport_ == srcport_ && op->dstport_ == dstport_;
}

size_t
bitcast_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
bitcast_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
bitcast_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
bitcast_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
bitcast_op::debug_string() const
{
	return strfmt("BITCAST[", srcport_.type().debug_string(),
		" -> ", dstport_.type().debug_string(), "]");
}

std::unique_ptr<jive::operation>
bitcast_op::copy() const
{
	return std::unique_ptr<jive::operation>(new bitcast_op(*this));
}

/* struct constant operator */

struct_constant_op::~struct_constant_op()
{}

bool
struct_constant_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const struct_constant_op*>(&other);
	return op && op->result_ == result_ && op->arguments_ == arguments_;
}

size_t
struct_constant_op::narguments() const noexcept
{
	return type().declaration()->nelements();
}

const jive::port &
struct_constant_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return arguments_[index];
}

size_t
struct_constant_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
struct_constant_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return result_;
}

std::string
struct_constant_op::debug_string() const
{
	return "STRUCTCONSTANT";
}

std::unique_ptr<jive::operation>
struct_constant_op::copy() const
{
	return std::unique_ptr<jive::operation>(new struct_constant_op(*this));
}

/* trunc operator */

trunc_op::~trunc_op()
{}

bool
trunc_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const trunc_op*>(&other);
	return op && op->oport_ == oport_ && op->rport_ == rport_;
}

size_t
trunc_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
trunc_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return oport_;
}

size_t
trunc_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
trunc_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return rport_;
}

std::string
trunc_op::debug_string() const
{
	return strfmt("TRUNC[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<jive::operation>
trunc_op::copy() const
{
	return std::unique_ptr<jive::operation>(new trunc_op(*this));
}


/* uitofp operator */

uitofp_op::~uitofp_op()
{}

bool
uitofp_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const uitofp_op*>(&other);
	return op && op->srcport_ == srcport_ && op->dstport_ == dstport_;
}

size_t
uitofp_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
uitofp_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
uitofp_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
uitofp_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
uitofp_op::debug_string() const
{
	return "UITOFP";
}

std::unique_ptr<jive::operation>
uitofp_op::copy() const
{
	return std::make_unique<uitofp_op>(*this);
}

/* sitofp operator */

sitofp_op::~sitofp_op()
{}

bool
sitofp_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const sitofp_op*>(&other);
	return op && op->srcport_ == srcport_ && op->dstport_ == dstport_;
}

size_t
sitofp_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
sitofp_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return srcport_;
}

size_t
sitofp_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
sitofp_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
sitofp_op::debug_string() const
{
	return "SITOFP";
}

std::unique_ptr<jive::operation>
sitofp_op::copy() const
{
	return std::make_unique<sitofp_op>(*this);
}

/* constant array operator */

constant_array_op::~constant_array_op()
{}

bool
constant_array_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const constant_array_op*>(&other);
	return op && op->aport_ == aport_ && op->eport_ == eport_;
}

size_t
constant_array_op::narguments() const noexcept
{
	return size();
}

const jive::port &
constant_array_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return eport_;
}

size_t
constant_array_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
constant_array_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return aport_;
}

std::string
constant_array_op::debug_string() const
{
	return "CONSTANTARRAY";
}

std::unique_ptr<jive::operation>
constant_array_op::copy() const
{
	return std::unique_ptr<jive::operation>(new constant_array_op(*this));
}

/* constant aggregate zero operator */

constant_aggregate_zero_op::~constant_aggregate_zero_op()
{}

bool
constant_aggregate_zero_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const constant_aggregate_zero_op*>(&other);
	return op && op->dstport_ == dstport_;
}

size_t
constant_aggregate_zero_op::narguments() const noexcept
{
	return 0;
}

const jive::port &
constant_aggregate_zero_op::argument(size_t index) const noexcept
{
	JLM_ASSERT(0);
}

size_t
constant_aggregate_zero_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
constant_aggregate_zero_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return dstport_;
}

std::string
constant_aggregate_zero_op::debug_string() const
{
	return "CONSTANT_AGGREGATE_ZERO";
}

std::unique_ptr<jive::operation>
constant_aggregate_zero_op::copy() const
{
	return std::unique_ptr<jive::operation>(new constant_aggregate_zero_op(*this));
}

}
