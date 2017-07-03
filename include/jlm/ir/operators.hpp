/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_H
#define JLM_IR_OPERATORS_H

#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/controltype.h>
#include <jive/vsdg/operators/nullary.h>

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
	: nodes_(nodes)
	, type_(type.copy())
	{
		if (nodes.size() < 2)
			throw std::logic_error("Expected at least two arguments.");
	}

	inline
	phi_op(const phi_op & other)
	: nodes_(other.nodes_)
	, type_(other.type_->copy())
	{}

	inline
	phi_op(phi_op && other)
	: nodes_(other.nodes_)
	, type_(std::move(other.type_))
	{}

	phi_op &
	operator=(const phi_op &) = delete;

	phi_op &
	operator=(phi_op &&) = delete;

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

	inline cfg_node *
	node(size_t n) const noexcept
	{
		JLM_DEBUG_ASSERT(n < narguments());
		return nodes_[n];
	}

private:
	std::vector<cfg_node*> nodes_;
	std::unique_ptr<jive::base::type> type_;
};

static inline bool
is_phi_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::phi_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_phi_tac(
	const std::vector<std::pair<const variable*, cfg_node*>> & arguments,
	const variable * result)
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

static inline std::unique_ptr<jlm::tac>
create_assignment(
	const jive::base::type & type,
	const variable * arg,
	const variable * r)
{
	return create_tac(assignment_op(type), {arg}, {r});
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
	, atype_(atype)
	, btype_(btype)
	{}

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

	inline const jive::bits::type &
	size_type() const noexcept
	{
		return btype_;
	}

	inline const jive::value::type &
	value_type() const noexcept
	{
		return atype_.pointee_type();
	}

private:
	jlm::ptrtype atype_;
	jive::bits::type btype_;
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
	const variable * state,
	const variable * result)
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
		: itype_(type)
	{}

	inline
	bits2flt_op(const bits2flt_op & other) = default;

	inline
	bits2flt_op(bits2flt_op && other) = default;

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
	jive::bits::type itype_;
};

/* flt2bits operator */

class flt2bits_op final : public jive::simple_op {
public:
	virtual
	~flt2bits_op() noexcept;

	inline
	flt2bits_op(const jive::bits::type & type)
		: otype_(type)
	{}

	inline
	flt2bits_op(const flt2bits_op & other) = default;

	inline
	flt2bits_op(flt2bits_op && other) = default;

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
	jive::bits::type otype_;
};

/* branch operator */

class branch_op final : public jive::simple_op {
public:
	virtual
	~branch_op() noexcept;

	inline
	branch_op(const jive::ctl::type & type)
	: type_(type)
	{}

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

	inline size_t
	nalternatives() const noexcept
	{
		return type_.nalternatives();
	}

private:
	jive::ctl::type type_;
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
	, ptype_(ptype)
	{}

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

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

private:
	jlm::ptrtype ptype_;
};

static inline bool
is_ptr_constant_null_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptr_constant_null_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptr_constant_null_tac(const jive::base::type & ptype, const variable * result)
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
	load_op(const jlm::ptrtype & ptype, size_t nstates)
	: simple_op()
	, nstates_(nstates)
	, ptype_(ptype)
	{}

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

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

	inline size_t
	nstates() const noexcept
	{
		return nstates_;
	}

private:
	size_t nstates_;
	jlm::ptrtype ptype_;
};

static inline bool
is_load_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::load_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_load_tac(const variable * address, const variable * state, const variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!pt) throw std::logic_error("Expected pointer type.");

	jlm::load_op op(*pt, 1);
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
	, ptype_(ptype)
	{}

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

	inline const jive::value::type &
	value_type() const noexcept
	{
		return ptype_.pointee_type();
	}

	inline size_t
	nstates() const noexcept
	{
		return nstates_;
	}

private:
	size_t nstates_;
	jlm::ptrtype ptype_;
};

static inline bool
is_store_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::store_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_store_tac(const variable * address, const variable * value, const variable * state)
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
	, ptype_(ptype)
	, btype_(btype)
	{}

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

	inline size_t
	nbits() const noexcept
	{
		return btype_.nbits();
	}

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

private:
	jlm::ptrtype ptype_;
	jive::bits::type btype_;
};

static inline bool
is_bits2ptr(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::bits2ptr_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_bits2ptr_tac(const variable * argument, const variable * result)
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
	, ptype_(ptype)
	, btype_(btype)
	{}

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

	inline size_t
	nbits() const noexcept
	{
		return btype_.nbits();
	}

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

private:
	jlm::ptrtype ptype_;
	jive::bits::type btype_;
};

static inline bool
is_ptr2bits(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::ptr2bits_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptr2bits_tac(const variable * argument, const variable * result)
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
	, ptype_(ptype)
	, rtype_(rtype)
	, btypes_(btypes)
	{}

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

	inline size_t
	nindices() const noexcept
	{
		return btypes_.size();
	}

	const jive::base::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

private:
	jlm::ptrtype ptype_;
	jlm::ptrtype rtype_;
	std::vector<jive::bits::type> btypes_;
};

static inline bool
is_ptroffset_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptroffset_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptroffset_tac(
	const variable * address,
	const std::vector<const variable*> offsets, const variable * result)
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
	, type_(type, size)
	{
		if (size == 0)
			throw std::logic_error("Size equals zero.");
	}

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

	std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	size() const noexcept
	{
		return type_.nelements();
	}

	inline const jive::value::type &
	type() const noexcept
	{
		return type_.element_type();
	}

private:
	jlm::arraytype type_;
};

static inline bool
is_data_array_constant_op(const jive::operation & op)
{
	return dynamic_cast<const data_array_constant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_data_array_constant_tac(
	const std::vector<const variable*> & elements,
	const variable * result)
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
	, ptype_(ptype)
	{}

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

	inline jlm::cmp
	cmp() const noexcept
	{
		return cmp_;
	}

	const jive::base::type &
	pointee_type() const noexcept
	{
		return ptype_.pointee_type();
	}

private:
	jlm::cmp cmp_;
	jlm::ptrtype ptype_;
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
	const variable * result)
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
	, srctype_(nsrcbits)
	, dsttype_(ndstbits)
	{
		if (ndstbits < nsrcbits)
			throw std::logic_error("# destination bits must be greater than # source bits.");
	}

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

	inline size_t
	nsrcbits() const noexcept
	{
		return srctype_.nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return dsttype_.nbits();
	}

private:
	jive::bits::type srctype_;
	jive::bits::type dsttype_;
};

static inline bool
is_zext_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::zext_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_zext_tac(const variable * operand, const variable * result)
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
	, type_(size)
	{}

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

	inline double
	constant() const noexcept
	{
		return constant_;
	}

	inline const fpsize &
	size() const noexcept
	{
		return type_.size();
	}

private:
	double constant_;
	jlm::fptype type_;
};

static inline bool
is_fpconstant_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::fpconstant_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_fpconstant_tac(double constant, const variable * result)
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
	, type_(size)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

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
		return type_.size();
	}

private:
	jlm::fpcmp cmp_;
	jlm::fptype type_;
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
	const variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
	if (!ft) throw std::logic_error("Expected floating point type.");

	jlm::fpcmp_op op(cmp, ft->size());
	return create_tac(op, {op1, op2}, {result});
}

}

#endif
