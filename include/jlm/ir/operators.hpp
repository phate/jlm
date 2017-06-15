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

/* phi operator */

class phi_op final : public jive::simple_op {
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

static inline bool
is_phi_op(const jive::operation & op)
{
	return dynamic_cast<const jlm::phi_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_phi_tac(const std::vector<const variable*> & arguments, const variable * result)
{
	phi_op phi(arguments.size(), result->type());
	return create_tac(phi, arguments, {result});
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
	alloca_op(size_t nbytes)
		: nbytes_(nbytes)
	{}

	inline
	alloca_op(const alloca_op & other) = default;

	inline
	alloca_op(alloca_op && other) = default;

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
	size_t nbytes_;
};

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
		const jive::bits::type & btype,
		const jlm::ptrtype & rtype)
	: jive::simple_op()
	, ptype_(ptype)
	, rtype_(rtype)
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

private:
	jlm::ptrtype ptype_;
	jlm::ptrtype rtype_;
	jive::bits::type btype_;
};

static inline bool
is_ptroffset(const jive::operation & op)
{
	return dynamic_cast<const jlm::ptroffset_op*>(&op) != nullptr;
}

static inline std::unique_ptr<jlm::tac>
create_ptroffset_tac(const variable * address, const variable * offset, const variable * result)
{
	auto at = dynamic_cast<const jlm::ptrtype*>(&address->type());
	if (!at) throw std::logic_error("Expected pointer type.");

	auto bt = dynamic_cast<const jive::bits::type*>(&offset->type());
	if (!bt) throw std::logic_error("Expected bitstring type.");

	auto rt = dynamic_cast<const jlm::ptrtype*>(&result->type());
	if (!rt) throw std::logic_error("Expected pointer type.");

	jlm::ptroffset_op op(*at, *bt, *rt);
	return create_tac(op, {address, offset}, {result});
}

}

#endif
