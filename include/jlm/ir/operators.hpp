/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_H
#define JLM_IR_OPERATORS_H

#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>
#include <jive/types/record/rcdtype.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/controltype.h>
#include <jive/vsdg/nullary.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm {

class cfg_node;

/* phi operator */

class phi_op final : public jive::simple_op {
public:
	virtual
	~phi_op() noexcept;

	inline
	phi_op(const std::vector<jlm::cfg_node*> & nodes, const jive::base::type & type)
	: port_(type)
	, nodes_(nodes)
	{
		if (nodes.size() < 2)
			throw std::logic_error("Expected at least two arguments.");
	}

	phi_op(const phi_op &) = default;

	phi_op &
	operator=(const phi_op &) = delete;

	phi_op &
	operator=(phi_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::base::type &
	type() const noexcept
	{
		return port_.type();
	}

	inline cfg_node *
	node(size_t n) const noexcept
	{
		JLM_DEBUG_ASSERT(n < narguments());
		return nodes_[n];
	}

private:
	jive::port port_;
	std::vector<cfg_node*> nodes_;
};

static inline bool
is_phi_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::phi_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_phi_tac(
	const std::vector<std::pair<const variable*, cfg_node*>> & arguments,
	jlm::variable * result)
{
	std::vector<cfg_node*> nodes;
	std::vector<const variable*> variables;
	for (const auto & p : arguments) {
		nodes.push_back(p.second);
		variables.push_back(p.first);
	}

	phi_op phi(nodes, result->type());
	return create_tac(phi, variables, {result});
}

/* assignment operator */

class assignment_op final : public jive::simple_op {
public:
	virtual
	~assignment_op() noexcept;

	inline
	assignment_op(const jive::base::type & type)
	: port_(type)
	{}

	assignment_op(const assignment_op &) = default;

	assignment_op(assignment_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port port_;
};

static inline std::unique_ptr<jlm::tac>
create_assignment(
	const jive::base::type & type,
	const variable * arg,
	const variable * r)
{
	return create_tac(assignment_op(type), {r, arg}, {});
}

static inline bool
is_assignment_op(const jive::operation & op)
{
	return dynamic_cast<const assignment_op*>(&op) != nullptr;
}

/* select operator */

class select_op final : public jive::simple_op {
public:
	virtual
	~select_op() noexcept;

	inline
	select_op(const jive::base::type & type)
	: port_(type)
	{}

	select_op(const select_op &) = default;

	select_op(select_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::base::type &
	type() const noexcept
	{
		return port_.type();
	}

private:
	jive::port port_;
};

static inline bool
is_select_op(const jive::operation & op)
{
	return dynamic_cast<const select_op*>(&op) != nullptr;
}

/* alloca operator */

class alloca_op final : public jive::simple_op {
public:
	virtual
	~alloca_op() noexcept;

	inline
	alloca_op(const jlm::ptrtype & atype, const jive::bits::type & btype)
	: simple_op()
	, aport_(atype)
	, bport_(btype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::bits::type &
	size_type() const noexcept
	{
		return *static_cast<const jive::bits::type*>(&bport_.type());
	}

	inline const jive::value::type &
	value_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&aport_.type())->pointee_type();
	}

private:
	jive::port aport_;
	jive::port bport_;
};

static inline bool
is_alloca_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::alloca_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_alloca_tac(
	const jive::base::type & vtype,
	const variable * size,
	jlm::variable * state,
	jlm::variable * result)
{
	auto vt = dynamic_cast<const jive::value::type*>(&vtype);
	if (!vt) throw std::logic_error("Expected value type.");

	auto bt = dynamic_cast<const jive::bits::type*>(&size->type());
	if (!bt) throw std::logic_error("Expected bits type.");

	jlm::alloca_op op(jlm::ptrtype(*vt), *bt);
	return create_tac(op, {size, state}, {result, state});
}

/* bits2flt operator */

class bits2flt_op final : public jive::simple_op {
public:
	virtual
	~bits2flt_op() noexcept;

	inline
	bits2flt_op(const jive::bits::type & type)
	: port_(type)
	{}

	bits2flt_op(const bits2flt_op &) = default;

	bits2flt_op(bits2flt_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port port_;
};

/* flt2bits operator */

class flt2bits_op final : public jive::simple_op {
public:
	virtual
	~flt2bits_op() noexcept;

	inline
	flt2bits_op(const jive::bits::type & type)
	: port_(type)
	{}

	flt2bits_op(const flt2bits_op &) = default;

	flt2bits_op(flt2bits_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port port_;
};

/* branch operator */

class branch_op final : public jive::simple_op {
public:
	virtual
	~branch_op() noexcept;

	inline
	branch_op(const jive::ctl::type & type)
	: port_(type)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nalternatives() const noexcept
	{
		return static_cast<const jive::ctl::type*>(&port_.type())->nalternatives();
	}

private:
	jive::port port_;
};

static inline bool
is_branch_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::branch_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_branch_tac(size_t nalternatives, const variable * operand)
{
	jive::ctl::type type(nalternatives);
	branch_op op(type);
	return create_tac(op, {operand}, {});
}

/* ptr constant */

class ptr_constant_null_op final : public jive::simple_op {
public:
	virtual
	~ptr_constant_null_op() noexcept;

	inline
	ptr_constant_null_op(const jlm::ptrtype & ptype)
	: simple_op()
	, port_(ptype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&port_.type())->pointee_type();
	}

private:
	jive::port port_;
};

static inline bool
is_ptr_constant_null_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptr_constant_null_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptr_constant_null_tac(const jive::base::type & ptype, jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&ptype);
	if (!pt) throw std::logic_error("Expected pointer type.");

	jlm::ptr_constant_null_op op(*pt);
	return create_tac(op, {}, {result});
}

/* load operator */

class load_op final : public jive::simple_op {
public:
	virtual
	~load_op() noexcept;

	inline
	load_op(
		const jlm::ptrtype & ptype,
		size_t nstates,
		size_t alignment)
	: simple_op()
	, nstates_(nstates)
	, aport_(ptype)
	, vport_(ptype.pointee_type())
	, alignment_(alignment)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return *static_cast<const jive::value::type*>(&vport_.type());
	}

	inline size_t
	nstates() const noexcept
	{
		return nstates_;
	}

	inline size_t
	alignment() const noexcept
	{
		return alignment_;
	}

private:
	size_t nstates_;
	jive::port aport_;
	jive::port vport_;
	size_t alignment_;
};

static inline bool
is_load_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::load_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_load_tac(const variable * address, const variable * state, jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!pt) throw std::logic_error("Expected pointer type.");

	jlm::load_op op(*pt, 1, 0);
	return create_tac(op, {address, state}, {result});
}

/* store operator */

class store_op final : public jive::simple_op {
public:
	virtual
	~store_op() noexcept;

	inline
	store_op(const jlm::ptrtype & ptype, size_t nstates)
	: simple_op()
	, nstates_(nstates)
	, aport_(ptype)
	, vport_(ptype.pointee_type())
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::value::type &
	value_type() const noexcept
	{
		return *static_cast<const jive::value::type*>(&vport_.type());
	}

	inline size_t
	nstates() const noexcept
	{
		return nstates_;
	}

private:
	size_t nstates_;
	jive::port aport_;
	jive::port vport_;
};

static inline bool
is_store_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::store_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_store_tac(const variable * address, const variable * value, jlm::variable * state)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	jlm::store_op op(*at, 1);
	return create_tac(op, {address, value, state}, {state});
}

/* bits2ptr operator */

class bits2ptr_op final : public jive::simple_op {
public:
	virtual
	~bits2ptr_op() noexcept;

	inline
	bits2ptr_op(const jive::bits::type & btype, const jlm::ptrtype & ptype)
	: simple_op()
	, pport_(ptype)
	, bport_(btype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nbits() const noexcept
	{
		return static_cast<const jive::bits::type*>(&bport_.type())->nbits();
	}

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&pport_.type())->pointee_type();
	}

private:
	jive::port pport_;
	jive::port bport_;
};

static inline bool
is_bits2ptr(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::bits2ptr_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_bits2ptr_tac(const variable * argument, jlm::variable * result)
{
	auto at = dynamic_cast<const jive::bits::type*>(&argument->type());
	if (!at) throw std::logic_error("Expected bitstring type.");

	auto pt = dynamic_cast<const jlm::ptrtype*>(&result->type());
	if (!pt) throw std::logic_error("Expected pointer type.");

	jlm::bits2ptr_op op(*at, *pt);
	return create_tac(op, {argument}, {result});
}

/* ptr2bits operator */

class ptr2bits_op final : public jive::simple_op {
public:
	virtual
	~ptr2bits_op() noexcept;

	inline
	ptr2bits_op(const jlm::ptrtype & ptype, const jive::bits::type & btype)
	: simple_op()
	, pport_(ptype)
	, bport_(btype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nbits() const noexcept
	{
		return static_cast<const jive::bits::type*>(&bport_.type())->nbits();
	}

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&pport_.type())->pointee_type();
	}

private:
	jive::port pport_;
	jive::port bport_;
};

static inline bool
is_ptr2bits(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::ptr2bits_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptr2bits_tac(const variable * argument, jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&argument->type());
	if (!pt) throw std::logic_error("Expected pointer type.");

	auto bt = dynamic_cast<const jive::bits::type*>(&result->type());
	if (!bt) throw std::logic_error("Expected bitstring type.");

	jlm::ptr2bits_op op(*pt, *bt);
	return create_tac(op, {argument}, {result});
}

/* ptroffset operator */

class ptroffset_op final : public jive::simple_op {
public:
	virtual
	~ptroffset_op();

	inline
	ptroffset_op(
		const jlm::ptrtype & ptype,
		const std::vector<jive::bits::type> & btypes,
		const jlm::ptrtype & rtype)
	: jive::simple_op()
	, pport_(ptype)
	, rport_(rtype)
	{
		for (const auto & type : btypes)
			bports_.push_back(type);
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nindices() const noexcept
	{
		return bports_.size();
	}

	const jive::base::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&pport_.type())->pointee_type();
	}

private:
	jive::port pport_;
	jive::port rport_;
	std::vector<jive::port> bports_;
};

static inline bool
is_ptroffset_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptroffset_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptroffset_tac(
	const variable * address,
	const std::vector<const variable*> offsets,
	jlm::variable * result)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	std::vector<jive::bits::type> bts;
	for (const auto & v : offsets) {
		auto bt = dynamic_cast<const jive::bits::type*>(&v->type());
		if (!bt) throw std::logic_error("Expected bitstring type.");
		bts.push_back(*bt);
	}

	auto rt = dynamic_cast<const jlm::ptrtype*>(&result->type());
	if (!rt) throw std::logic_error("Expected pointer type.");

	jlm::ptroffset_op op(*at, bts, *rt);
	std::vector<const variable*> operands(1, address);
	operands.insert(operands.end(), offsets.begin(), offsets.end());
	return create_tac(op, operands, {result});
}

/* data array constant operator */

class data_array_constant_op final : public jive::simple_op {
public:
	virtual
	~data_array_constant_op();

	inline
	data_array_constant_op(const jive::value::type & type, size_t size)
	: jive::simple_op()
	, aport_(jlm::arraytype(type, size))
	, eport_(type)
	{
		if (size == 0)
			throw std::logic_error("Size equals zero.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	size() const noexcept
	{
		return static_cast<const jlm::arraytype*>(&aport_.type())->nelements();
	}

	inline const jive::value::type &
	type() const noexcept
	{
		return *static_cast<const jive::value::type*>(&eport_.type());
	}

private:
	jive::port aport_;
	jive::port eport_;
};

static inline bool
is_data_array_constant_op(const jive::operation & op)
{
	return dynamic_cast<const data_array_constant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_data_array_constant_tac(
	const std::vector<const variable*> & elements,
	jlm::variable * result)
{
	if (elements.size() == 0)
		throw std::logic_error("Expected at least one element.");

	auto vt = dynamic_cast<const jive::value::type*>(&elements[0]->type());
	if (!vt) throw std::logic_error("Expected value type.");

	data_array_constant_op op(*vt, elements.size());
	return create_tac(op, elements, {result});
}

/* pointer compare operator */

enum class cmp {eq, ne, gt, ge, lt, le};

class ptrcmp_op final : public jive::simple_op {
public:
	virtual
	~ptrcmp_op() noexcept;

	inline
	ptrcmp_op(const jlm::ptrtype & ptype, const jlm::cmp & cmp)
	: jive::simple_op()
	, cmp_(cmp)
	, port_(ptype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline jlm::cmp
	cmp() const noexcept
	{
		return cmp_;
	}

	const jive::base::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&port_.type())->pointee_type();
	}

private:
	jlm::cmp cmp_;
	jive::port port_;
};

static inline bool
is_ptrcmp_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptrcmp_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptrcmp_tac(
	const jlm::cmp & cmp,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&op1->type());
	if (!pt) throw std::logic_error("Expected pointer type.");

	jlm::ptrcmp_op op(*pt, cmp);
	return create_tac(op, {op1, op2}, {result});
}

/* zext operator */

class zext_op final : public jive::simple_op {
public:
	virtual
	~zext_op() noexcept;

	inline
	zext_op(size_t nsrcbits, size_t ndstbits)
	: jive::simple_op()
	, srcport_(jive::bits::type(nsrcbits))
	, dstport_(jive::bits::type(ndstbits))
	{
		if (ndstbits < nsrcbits)
			throw std::logic_error("# destination bits must be greater than # source bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nsrcbits() const noexcept
	{
		return static_cast<const jive::bits::type*>(&srcport_.type())->nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return static_cast<const jive::bits::type*>(&dstport_.type())->nbits();
	}

private:
	jive::port srcport_;
	jive::port dstport_;
};

static inline bool
is_zext_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::zext_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_zext_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!st) throw std::logic_error("Expected bitstring type.");

	auto dt = dynamic_cast<const jive::bits::type*>(&result->type());
	if (!dt) throw std::logic_error("Expected bitstring type.");

	jlm::zext_op op(st->nbits(), dt->nbits());
	return create_tac(op, {operand}, {result});
}

/* floating point constant operator */

class fpconstant_op final : public jive::simple_op {
public:
	virtual
	~fpconstant_op();

	inline
	fpconstant_op(const jlm::fpsize & size, double constant)
	: jive::simple_op()
	, constant_(constant)
	, port_(jlm::fptype(size))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline double
	constant() const noexcept
	{
		return constant_;
	}

	inline const fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&port_.type())->size();
	}

private:
	double constant_;
	jive::port port_;
};

static inline bool
is_fpconstant_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::fpconstant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_fpconstant_tac(double constant, jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!ft) throw std::logic_error("Expected floating point type.");

	jlm::fpconstant_op op(ft->size(), constant);
	return create_tac(op, {}, {result});
}

/* floating point comparison operator */

enum class fpcmp {oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno};

class fpcmp_op final : public jive::simple_op {
public:
	virtual
	~fpcmp_op() noexcept;

	inline
	fpcmp_op(const jlm::fpcmp & cmp, const jlm::fpsize & size)
	: jive::simple_op()
	, cmp_(cmp)
	, port_(jlm::fptype(size))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jlm::fpcmp &
	cmp() const noexcept
	{
		return cmp_;
	}

	inline const jlm::fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&port_.type())->size();
	}

private:
	jlm::fpcmp cmp_;
	jive::port port_;
};

static inline bool
is_fpcmp_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::fpcmp_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_fpcmp_tac(
	const jlm::fpcmp & cmp,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
	if (!ft) throw std::logic_error("Expected floating point type.");

	jlm::fpcmp_op op(cmp, ft->size());
	return create_tac(op, {op1, op2}, {result});
}

/* undef constant operator */

class undef_constant_op final : public jive::simple_op {
public:
	virtual
	~undef_constant_op() noexcept;

	inline
	undef_constant_op(const jive::value::type & type)
	: jive::simple_op()
	, port_(type)
	{}

	undef_constant_op(const undef_constant_op &) = default;

	undef_constant_op &
	operator=(const undef_constant_op &) = delete;

	undef_constant_op &
	operator=(undef_constant_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::value::type &
	type() const noexcept
	{
		return *static_cast<const jive::value::type*>(&port_.type());
	}

private:
	jive::port port_;
};

static inline bool
is_undef_constant_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::undef_constant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_undef_constant_tac(jlm::variable * result)
{
	auto vt = dynamic_cast<const jive::value::type*>(&result->type());
	if (!vt) throw std::logic_error("Expected value type.");

	jlm::undef_constant_op op(*vt);
	return create_tac(op, {}, {result});
}

/* floating point arithmetic operator */

enum class fpop {add, sub, mul, div, mod};

class fpbin_op final : public jive::simple_op {
public:
	virtual
	~fpbin_op() noexcept;

	inline
	fpbin_op(const jlm::fpop & op, const jlm::fpsize & size)
	: jive::simple_op()
	, op_(op)
	, port_(jlm::fptype(size))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jlm::fpop &
	fpop() const noexcept
	{
		return op_;
	}

	inline const jlm::fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&port_.type())->size();
	}

private:
	jlm::fpop op_;
	jive::port port_;
};

static inline bool
is_fpbin_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::fpbin_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_fpbin_tac(
	const jlm::fpop & fpop,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
	if (!ft) throw std::logic_error("Expected floating point type.");

	jlm::fpbin_op op(fpop, ft->size());
	return create_tac(op, {op1, op2}, {result});
}

/* fpext operator */

class fpext_op final : public jive::simple_op {
public:
	virtual
	~fpext_op() noexcept;

	inline
	fpext_op(const jlm::fpsize & srcsize, const jlm::fpsize & dstsize)
	: jive::simple_op()
	, srcport_(jlm::fptype(srcsize))
	, dstport_(jlm::fptype(dstsize))
	{
		if (srcsize == fpsize::flt && dstsize == fpsize::half)
			throw std::logic_error("Destination type size must be bigger than source type size.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jlm::fpsize &
	srcsize() const noexcept
	{
		return static_cast<const jlm::fptype*>(&srcport_.type())->size();
	}

	inline const jlm::fpsize &
	dstsize() const noexcept
	{
		return static_cast<const jlm::fptype*>(&dstport_.type())->size();
	}

private:
	jive::port srcport_;
	jive::port dstport_;
};

static inline bool
is_fpext_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::fpext_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_fpext_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jlm::fptype*>(&operand->type());
	if (!st) throw std::logic_error("Expected floating point type.");

	auto dt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!dt) throw std::logic_error("Expected floating point type.");

	jlm::fpext_op op(st->size(), dt->size());
	return create_tac(op, {operand}, {result});
}

/* valist operator */

class valist_op final : public jive::simple_op {
public:
	virtual
	~valist_op() noexcept;

	inline
	valist_op(std::vector<std::unique_ptr<jive::base::type>> types)
	: jive::simple_op()
	{
		for (const auto & type : types)
			ports_.push_back({type->copy()});
	}

	valist_op(const valist_op &) = default;

	valist_op &
	operator=(const valist_op &) = delete;

	valist_op &
	operator=(valist_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	std::vector<jive::port> ports_;
};

static inline bool
is_valist_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::valist_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_valist_tac(
	const std::vector<const variable*> & arguments,
	jlm::module & m)
{
	std::vector<std::unique_ptr<jive::base::type>> operands;
	for (const auto & argument : arguments)
		operands.push_back(argument->type().copy());

	varargtype t;
	auto result = m.create_tacvariable(t);

	jlm::valist_op op(std::move(operands));
	auto tac = create_tac(op, arguments, {result});
	result->set_tac(tac.get());

	return tac;
}

/* bitcast operator */

class bitcast_op final : public jive::simple_op {
public:
	virtual
	~bitcast_op();

	inline
	bitcast_op(const jive::value::type & srctype, const jive::value::type & dsttype)
	: jive::simple_op()
	, srcport_(srctype)
	, dstport_(dsttype)
	{}

	bitcast_op(const bitcast_op &) = default;

	bitcast_op(jive::operation &&) = delete;

	bitcast_op &
	operator=(const jive::operation &) = delete;

	bitcast_op &
	operator=(jive::operation &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port srcport_;
	jive::port dstport_;
};

static inline bool
is_bitcast_op(const jive::operation & op)
{
	return dynamic_cast<const bitcast_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_bitcast_tac(const variable * argument, variable * result)
{
	auto at = dynamic_cast<const jive::value::type*>(&argument->type());
	if (!at) throw std::logic_error("Expected value type.");

	auto rt = dynamic_cast<const jive::value::type*>(&result->type());
	if (!rt) throw std::logic_error("Expected value type.");

	bitcast_op op(*at, *rt);
	return create_tac(op, {argument}, {result});
}

/* struct constant operator */

class struct_constant_op final : public jive::simple_op {
public:
	virtual
	~struct_constant_op();

	inline
	struct_constant_op(const structtype & type)
	: jive::simple_op()
	, result_(type)
	{
		for (size_t n = 0; n < type.declaration()->nelements(); n++)
			arguments_.push_back(type.declaration()->element(n));
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const structtype &
	type() const noexcept
	{
		return *static_cast<const structtype*>(&result_.type());
	}

private:
	jive::port result_;
	std::vector<jive::port> arguments_;
};

static inline bool
is_struct_constant_op(const jive::operation & op)
{
	return dynamic_cast<const struct_constant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_struct_constant_tac(
	const std::vector<const variable*> & elements,
	jlm::variable * result)
{
	auto rt = dynamic_cast<const structtype*>(&result->type());
	if (!rt) throw std::logic_error("Expected struct type.");

	struct_constant_op op(*rt);
	return create_tac(op, elements, {result});
}

/* trunc operator */

class trunc_op final : public jive::simple_op {
public:
	virtual
	~trunc_op();

	inline
	trunc_op(const jive::bits::type & otype, const jive::bits::type & rtype)
	: jive::simple_op()
	, oport_(otype)
	, rport_(rtype)
	{
		if (otype.nbits() < rtype.nbits())
			throw std::logic_error("Expected operand's #bits to be larger than results' #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port oport_;
	jive::port rport_;
};

static inline bool
is_trunc_op(const jive::operation & op)
{
	return dynamic_cast<const trunc_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_trunc_tac(const variable * operand, jlm::variable * result)
{
	auto ot = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	auto rt = dynamic_cast<const jive::bits::type*>(&result->type());
	if (!rt) throw std::logic_error("Expected bits type.");

	trunc_op op(*ot, *rt);
	return create_tac(op, {operand}, {result});
}

/* sext operator */

class sext_op final : public jive::simple_op {
public:
	virtual
	~sext_op();

	inline
	sext_op(const jive::bits::type & otype, const jive::bits::type & rtype)
	: jive::simple_op()
	, oport_(otype)
	, rport_(rtype)
	{
		if (otype.nbits() >= rtype.nbits())
			throw std::logic_error("Expected operand's #bits to be smaller than results's #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port oport_;
	jive::port rport_;
};

static inline bool
is_sext_op(const jive::operation & op)
{
	return dynamic_cast<const sext_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_sext_tac(const variable * operand, jlm::variable * result)
{
	auto ot = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!ot) throw std::logic_error("Expected bits type.");

	auto rt = dynamic_cast<const jive::bits::type*>(&result->type());
	if (!rt) throw std::logic_error("Expected bits type.");

	sext_op op(*ot, *rt);
	return create_tac(op, {operand}, {result});
}

/* sitofp operator */

class sitofp_op final : public jive::simple_op {
public:
	virtual
	~sitofp_op();

	inline
	sitofp_op(const jive::bits::type & srctype, const jlm::fptype & dsttype)
	: jive::simple_op()
	, dstport_(dsttype)
	, srcport_(srctype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::port &
	argument(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::port &
	result(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::port dstport_;
	jive::port srcport_;
};

static inline bool
is_sitofp_op(const jive::operation & op)
{
	return dynamic_cast<const sitofp_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_sitofp_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bits::type*>(&operand->type());
	if (!st) throw std::logic_error("Expected bits type.");

	auto rt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!rt) throw std::logic_error("Expected floating point type.");

	sitofp_op op(*st, *rt);
	return create_tac(op, {operand}, {result});
}

}

#endif
