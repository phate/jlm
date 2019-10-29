/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_OPERATORS_HPP
#define JLM_IR_OPERATORS_OPERATORS_HPP

#include <jive/types/bitstring/type.h>
#include <jive/types/function.h>
#include <jive/types/record.h>
#include <jive/rvsdg/binary.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/nullary.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/type.h>
#include <jive/rvsdg/unary.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

#include <llvm/ADT/APFloat.h>

namespace jlm {

class cfg_node;

/* phi operator */

class phi_op final : public jive::simple_op {
public:
	virtual
	~phi_op() noexcept;

	inline
	phi_op(const std::vector<jlm::cfg_node*> & nodes, const jive::type & type)
	: jive::simple_op(std::vector<jive::port>(nodes.size(), {type}), {type})
	, nodes_(nodes)
	{
		if (nodes.size() < 2)
			throw jlm::error("expected at least two arguments.");
	}

	phi_op(const phi_op &) = default;

	phi_op &
	operator=(const phi_op &) = delete;

	phi_op &
	operator=(phi_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::type &
	type() const noexcept
	{
		return result(0).type();
	}

	inline cfg_node *
	node(size_t n) const noexcept
	{
		JLM_DEBUG_ASSERT(n < narguments());
		return nodes_[n];
	}

private:
	std::vector<cfg_node*> nodes_;
};

static inline std::unique_ptr<jlm::tac>
create_phi_tac(
	const std::vector<std::pair<const variable*, cfg_node*>> & arguments,
	const jlm::variable * result)
{
	std::vector<cfg_node*> nodes;
	std::vector<const variable*> variables;
	for (const auto & p : arguments) {
		nodes.push_back(p.second);
		variables.push_back(p.first);
	}

	phi_op phi(nodes, result->type());
	return tac::create(phi, variables, {result});
}

/* assignment operator */

class assignment_op final : public jive::simple_op {
public:
	virtual
	~assignment_op() noexcept;

	inline
	assignment_op(const jive::type & type)
	: simple_op({type, type}, {})
	{}

	assignment_op(const assignment_op &) = default;

	assignment_op(assignment_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_assignment(
	const jive::type & type,
	const variable * arg,
	const variable * r)
{
	return tac::create(assignment_op(type), {r, arg}, {});
}

/* select operator */

class select_op final : public jive::simple_op {
public:
	virtual
	~select_op() noexcept;

	inline
	select_op(const jive::type & type)
	: jive::simple_op({jive::bit1, type, type}, {type})
	{}

	select_op(const select_op &) = default;

	select_op(select_op &&) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::type &
	type() const noexcept
	{
		return result(0).type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_select_tac(
	const jlm::variable * p,
	const jlm::variable * t,
	const jlm::variable * f,
	jlm::variable * result)
{
	select_op op(t->type());
	return tac::create(op, {p, t, f}, {result});
}

/* fp2ui operator */

class fp2ui_op final : public jive::unary_op {
public:
	virtual
	~fp2ui_op() noexcept;

	inline
	fp2ui_op(const fpsize & size, const jive::bittype & type)
	: jive::unary_op(fptype(size), type)
	{}

	inline
	fp2ui_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto st = dynamic_cast<const fptype*>(srctype.get());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jive::bittype*>(dsttype.get());
		if (!dt) throw jlm::error("expected bitstring type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;
};

static inline std::unique_ptr<jlm::tac>
create_fp2ui_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	fp2ui_op op(st->size(), *dt);
	return tac::create(op, {operand}, {result});
}

/* fp2si operator */

class fp2si_op final : public jive::unary_op {
public:
	virtual
	~fp2si_op() noexcept;

	inline
	fp2si_op(const fpsize & size, const jive::bittype & type)
	: jive::unary_op({fptype(size)}, {type})
	{}

	inline
	fp2si_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: jive::unary_op({*srctype}, {*dsttype})
	{
		auto st = dynamic_cast<const fptype*>(srctype.get());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jive::bittype*>(dsttype.get());
		if (!dt) throw jlm::error("expected bitstring type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;
};

static inline std::unique_ptr<jlm::tac>
create_fp2si_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	fp2si_op op(st->size(), *dt);
	return tac::create(op, {operand}, {result});
}

/* ctl2bits operator */

class ctl2bits_op final : public jive::simple_op {
public:
	virtual
	~ctl2bits_op() noexcept;

	inline
	ctl2bits_op(const jive::ctltype & srctype, const jive::bittype & dsttype)
	: jive::simple_op({srctype}, {dsttype})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_ctl2bits_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::ctltype*>(&operand->type());
	if (!st) throw jlm::error("expected control type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	ctl2bits_op op(*st, *dt);
	return tac::create(op, {operand}, {result});
}

/* branch operator */

class branch_op final : public jive::simple_op {
public:
	virtual
	~branch_op() noexcept;

	inline
	branch_op(const jive::ctltype & type)
	: jive::simple_op({type}, {})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	nalternatives() const noexcept
	{
		return static_cast<const jive::ctltype*>(&argument(0).type())->nalternatives();
	}
};

static inline std::unique_ptr<jlm::tac>
create_branch_tac(size_t nalternatives, const variable * operand)
{
	jive::ctltype type(nalternatives);
	branch_op op(type);
	return tac::create(op, {operand}, {});
}

/* ptr constant */

class ptr_constant_null_op final : public jive::simple_op {
public:
	virtual
	~ptr_constant_null_op() noexcept;

	inline
	ptr_constant_null_op(const jlm::ptrtype & type)
	: simple_op({}, {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::valuetype &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&result(0).type())->pointee_type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_ptr_constant_null_tac(const jive::type & ptype, jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&ptype);
	if (!pt) throw jlm::error("expected pointer type.");

	jlm::ptr_constant_null_op op(*pt);
	return tac::create(op, {}, {result});
}

/* bits2ptr operator */

class bits2ptr_op final : public jive::unary_op {
public:
	virtual
	~bits2ptr_op() noexcept;

	inline
	bits2ptr_op(const jive::bittype & btype, const jlm::ptrtype & ptype)
	: unary_op(btype, ptype)
	{}

	inline
	bits2ptr_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto at = dynamic_cast<const jive::bittype*>(srctype.get());
		if (!at) throw jlm::error("expected bitstring type.");

		auto pt = dynamic_cast<const jlm::ptrtype*>(dsttype.get());
		if (!pt) throw jlm::error("expected pointer type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;

	inline size_t
	nbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&argument(0).type())->nbits();
	}

	inline const jive::valuetype &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&result(0).type())->pointee_type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_bits2ptr_tac(const variable * argument, jlm::variable * result)
{
	auto at = dynamic_cast<const jive::bittype*>(&argument->type());
	if (!at) throw jlm::error("expected bitstring type.");

	auto pt = dynamic_cast<const jlm::ptrtype*>(&result->type());
	if (!pt) throw jlm::error("expected pointer type.");

	jlm::bits2ptr_op op(*at, *pt);
	return tac::create(op, {argument}, {result});
}

/* ptr2bits operator */

class ptr2bits_op final : public jive::unary_op {
public:
	virtual
	~ptr2bits_op() noexcept;

	inline
	ptr2bits_op(const jlm::ptrtype & ptype, const jive::bittype & btype)
	: unary_op(ptype, btype)
	{}

	inline
	ptr2bits_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto pt = dynamic_cast<const jlm::ptrtype*>(srctype.get());
		if (!pt) throw jlm::error("expected pointer type.");

		auto bt = dynamic_cast<const jive::bittype*>(dsttype.get());
		if (!bt) throw jlm::error("expected bitstring type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;

	inline size_t
	nbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&result(0).type())->nbits();
	}

	inline const jive::valuetype &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&argument(0).type())->pointee_type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_ptr2bits_tac(const variable * argument, jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&argument->type());
	if (!pt) throw jlm::error("expected pointer type.");

	auto bt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!bt) throw jlm::error("expected bitstring type.");

	jlm::ptr2bits_op op(*pt, *bt);
	return tac::create(op, {argument}, {result});
}

/* data array constant operator */

class data_array_constant_op final : public jive::simple_op {
public:
	virtual
	~data_array_constant_op();

	inline
	data_array_constant_op(const jive::valuetype & type, size_t size)
	: simple_op(std::vector<jive::port>(size, type), {jlm::arraytype(type, size)})
	{
		if (size == 0)
			throw jlm::error("size equals zero.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	size() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->nelements();
	}

	inline const jive::valuetype &
	type() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->element_type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_data_array_constant_tac(
	const std::vector<const variable*> & elements,
	jlm::variable * result)
{
	if (elements.size() == 0)
		throw jlm::error("expected at least one element.");

	auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
	if (!vt) throw jlm::error("expected value type.");

	data_array_constant_op op(*vt, elements.size());
	return tac::create(op, elements, {result});
}

/* pointer compare operator */

enum class cmp {eq, ne, gt, ge, lt, le};

class ptrcmp_op final : public jive::binary_op {
public:
	virtual
	~ptrcmp_op() noexcept;

	inline
	ptrcmp_op(const jlm::ptrtype & ptype, const jlm::cmp & cmp)
	: binary_op({ptype, ptype}, {jive::bit1})
	, cmp_(cmp)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual jive_binop_reduction_path_t
	can_reduce_operand_pair(
		const jive::output * op1,
		const jive::output * op2) const noexcept override;

	virtual jive::output *
	reduce_operand_pair(
		jive_binop_reduction_path_t path,
		jive::output * op1,
		jive::output * op2) const override;

	inline jlm::cmp
	cmp() const noexcept
	{
		return cmp_;
	}

	const jive::type &
	pointee_type() const noexcept
	{
		return static_cast<const jlm::ptrtype*>(&argument(0).type())->pointee_type();
	}

private:
	jlm::cmp cmp_;
};

static inline std::unique_ptr<jlm::tac>
create_ptrcmp_tac(
	const jlm::cmp & cmp,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto pt = dynamic_cast<const jlm::ptrtype*>(&op1->type());
	if (!pt) throw jlm::error("expected pointer type.");

	jlm::ptrcmp_op op(*pt, cmp);
	return tac::create(op, {op1, op2}, {result});
}

/* zext operator */

class zext_op final : public jive::unary_op {
public:
	virtual
	~zext_op() noexcept;

	inline
	zext_op(size_t nsrcbits, size_t ndstbits)
	: unary_op({jive::bittype(nsrcbits)}, {jive::bittype(ndstbits)})
	{
		if (ndstbits < nsrcbits)
			throw jlm::error("# destination bits must be greater than # source bits.");
	}

	inline
	zext_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto st = dynamic_cast<const jive::bittype*>(srctype.get());
		if (!st) throw jlm::error("expected bitstring type.");

		auto dt = dynamic_cast<const jive::bittype*>(dsttype.get());
		if (!dt) throw jlm::error("expected bitstring type.");

		if (dt->nbits() < st->nbits())
			throw jlm::error("# destination bits must be greater than # source bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(const jive::output * operand) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * operand) const override;

	inline size_t
	nsrcbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&argument(0).type())->nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&result(0).type())->nbits();
	}
};

static inline std::unique_ptr<jlm::tac>
create_zext_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!st) throw jlm::error("expected bitstring type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	jlm::zext_op op(st->nbits(), dt->nbits());
	return tac::create(op, {operand}, {result});
}

/* floating point constant operator */

class fpconstant_op final : public jive::simple_op {
public:
	virtual
	~fpconstant_op();

	inline
	fpconstant_op(const jlm::fpsize & size, const llvm::APFloat & constant)
	: simple_op({}, {fptype(size)})
	, constant_(constant)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const llvm::APFloat &
	constant() const noexcept
	{
		return constant_;
	}

	inline const fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&result(0).type())->size();
	}

private:
	/* FIXME: I would not like to use the APFloat here,
	   but I don't have a replacement right now. */
	llvm::APFloat constant_;
};

static inline std::unique_ptr<jlm::tac>
create_fpconstant_tac(const llvm::APFloat & constant, jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!ft) throw jlm::error("expected floating point type.");

	jlm::fpconstant_op op(ft->size(), constant);
	return tac::create(op, {}, {result});
}

/* floating point comparison operator */

enum class fpcmp {
	TRUE, FALSE, oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
};

class fpcmp_op final : public jive::binary_op {
public:
	virtual
	~fpcmp_op() noexcept;

	inline
	fpcmp_op(const jlm::fpcmp & cmp, const jlm::fpsize & size)
	: binary_op({fptype(size), fptype(size)}, {jive::bit1})
	, cmp_(cmp)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_binop_reduction_path_t
	can_reduce_operand_pair(
		const jive::output * op1,
		const jive::output * op2) const noexcept override;

	jive::output *
	reduce_operand_pair(
		jive_binop_reduction_path_t path,
		jive::output * op1,
		jive::output * op2) const override;

	inline const jlm::fpcmp &
	cmp() const noexcept
	{
		return cmp_;
	}

	inline const jlm::fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&argument(0).type())->size();
	}

private:
	jlm::fpcmp cmp_;
};

static inline std::unique_ptr<jlm::tac>
create_fpcmp_tac(
	const jlm::fpcmp & cmp,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
	if (!ft) throw jlm::error("expected floating point type.");

	jlm::fpcmp_op op(cmp, ft->size());
	return tac::create(op, {op1, op2}, {result});
}

/* undef constant operator */

class undef_constant_op final : public jive::simple_op {
public:
	virtual
	~undef_constant_op() noexcept;

	inline
	undef_constant_op(const jive::valuetype & type)
	: simple_op({}, {type})
	{}

	undef_constant_op(const undef_constant_op &) = default;

	undef_constant_op &
	operator=(const undef_constant_op &) = delete;

	undef_constant_op &
	operator=(undef_constant_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const jive::valuetype &
	type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(&result(0).type());
	}

	static inline jive::output *
	create(jive::region * region, const jive::type & type)
	{
		auto vt = dynamic_cast<const jive::valuetype*>(&type);
		if (!vt) throw jlm::error("expected value type.");

		jlm::undef_constant_op op(*vt);
		return jive::simple_node::create_normalized(region, op, {})[0];
	}
};

static inline std::unique_ptr<jlm::tac>
create_undef_constant_tac(const jlm::variable * result)
{
	auto vt = dynamic_cast<const jive::valuetype*>(&result->type());
	if (!vt) throw jlm::error("expected value type.");

	jlm::undef_constant_op op(*vt);
	return tac::create(op, {}, {result});
}

/* floating point arithmetic operator */

enum class fpop {add, sub, mul, div, mod};

class fpbin_op final : public jive::binary_op {
public:
	virtual
	~fpbin_op() noexcept;

	inline
	fpbin_op(const jlm::fpop & op, const jlm::fpsize & size)
	: binary_op({fptype(size), fptype(size)}, {fptype(size)})
	, op_(op)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_binop_reduction_path_t
	can_reduce_operand_pair(
		const jive::output * op1,
		const jive::output * op2) const noexcept override;

	jive::output *
	reduce_operand_pair(
		jive_binop_reduction_path_t path,
		jive::output * op1,
		jive::output * op2) const override;

	inline const jlm::fpop &
	fpop() const noexcept
	{
		return op_;
	}

	inline const jlm::fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&result(0).type())->size();
	}

private:
	jlm::fpop op_;
};

static inline std::unique_ptr<jlm::tac>
create_fpbin_tac(
	const jlm::fpop & fpop,
	const variable * op1,
	const variable * op2,
	jlm::variable * result)
{
	auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
	if (!ft) throw jlm::error("expected floating point type.");

	jlm::fpbin_op op(fpop, ft->size());
	return tac::create(op, {op1, op2}, {result});
}

/* fpext operator */

class fpext_op final : public jive::unary_op {
public:
	virtual
	~fpext_op() noexcept;

	inline
	fpext_op(const jlm::fpsize & srcsize, const jlm::fpsize & dstsize)
	: unary_op(fptype(srcsize), fptype(dstsize))
	{
		if (srcsize == fpsize::flt && dstsize == fpsize::half)
			throw jlm::error("destination type size must be bigger than source type size.");
	}

	inline
	fpext_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto st = dynamic_cast<const jlm::fptype*>(srctype.get());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jlm::fptype*>(dsttype.get());
		if (!dt) throw jlm::error("expected floating point type.");

		if (st->size() == fpsize::flt && dt->size() == fpsize::half)
			throw jlm::error("destination type size must be bigger than source type size.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;

	inline const jlm::fpsize &
	srcsize() const noexcept
	{
		return static_cast<const jlm::fptype*>(&argument(0).type())->size();
	}

	inline const jlm::fpsize &
	dstsize() const noexcept
	{
		return static_cast<const jlm::fptype*>(&result(0).type())->size();
	}
};

static inline std::unique_ptr<jlm::tac>
create_fpext_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jlm::fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!dt) throw jlm::error("expected floating point type.");

	jlm::fpext_op op(st->size(), dt->size());
	return tac::create(op, {operand}, {result});
}

/* fptrunc operator */

class fptrunc_op final : public jive::unary_op {
public:
	virtual
	~fptrunc_op() noexcept;

	inline
	fptrunc_op(const fpsize & srcsize, const fpsize & dstsize)
	: unary_op(fptype(srcsize), fptype(dstsize))
	{
		if (srcsize == fpsize::half
		|| (srcsize == fpsize::flt && dstsize != fpsize::half)
		|| (srcsize == fpsize::dbl && dstsize == fpsize::dbl))
			throw jlm::error("destination tpye size must be smaller than source size type.");
	}

	inline
	fptrunc_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto st = dynamic_cast<const fptype*>(srctype.get());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const fptype*>(dsttype.get());
		if (!dt) throw jlm::error("expected floating point type.");

		if (st->size() == fpsize::half
		|| (st->size() == fpsize::flt && dt->size() != fpsize::half)
		|| (st->size() == fpsize::dbl && dt->size() == fpsize::dbl))
			throw jlm::error("destination tpye size must be smaller than source size type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;

	inline const fpsize &
	srcsize() const noexcept
	{
		return static_cast<const fptype*>(&argument(0).type())->size();
	}

	inline const fpsize &
	dstsize() const noexcept
	{
		return static_cast<const fptype*>(&result(0).type())->size();
	}
};

static inline std::unique_ptr<jlm::tac>
create_fptrunc_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const fptype*>(&result->type());
	if (!dt) throw jlm::error("expected floating point type.");

	fptrunc_op op(st->size(), dt->size());
	return tac::create(op, {operand}, {result});
}

/* valist operator */

class valist_op final : public jive::simple_op {
public:
	virtual
	~valist_op() noexcept;

	inline
	valist_op(std::vector<std::unique_ptr<jive::type>> types)
	: simple_op(create_srcports(std::move(types)), {varargtype()})
	{}

	valist_op(const valist_op &) = default;

	valist_op &
	operator=(const valist_op &) = delete;

	valist_op &
	operator=(valist_op &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	static inline std::vector<jive::port>
	create_srcports(std::vector<std::unique_ptr<jive::type>> types)
	{
		std::vector<jive::port> ports;
		for (const auto & type : types)
			ports.push_back(jive::port(*type));

		return ports;
	}
};

static inline std::unique_ptr<jlm::tac>
create_valist_tac(
	const std::vector<const variable*> & arguments,
	jlm::module & m)
{
	std::vector<std::unique_ptr<jive::type>> operands;
	for (const auto & argument : arguments)
		operands.push_back(argument->type().copy());

	varargtype t;
	auto result = m.create_tacvariable(t);

	jlm::valist_op op(std::move(operands));
	auto tac = tac::create(op, arguments, {result});
	result->set_tac(tac.get());

	return tac;
}

/* bitcast operator */

class bitcast_op final : public jive::unary_op {
public:
	virtual
	~bitcast_op();

	inline
	bitcast_op(const jive::valuetype & srctype, const jive::valuetype & dsttype)
	: unary_op(srctype, dsttype)
	{}

	inline
	bitcast_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto at = dynamic_cast<const jive::valuetype*>(srctype.get());
		if (!at) throw jlm::error("expected value type.");

		auto rt = dynamic_cast<const jive::valuetype*>(dsttype.get());
		if (!rt) throw jlm::error("expected value type.");
	}

	bitcast_op(const bitcast_op &) = default;

	bitcast_op(jive::operation &&) = delete;

	bitcast_op &
	operator=(const jive::operation &) = delete;

	bitcast_op &
	operator=(jive::operation &&) = delete;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;
};

static inline std::unique_ptr<jlm::tac>
create_bitcast_tac(const variable * argument, variable * result)
{
	auto at = dynamic_cast<const jive::valuetype*>(&argument->type());
	if (!at) throw jlm::error("expected value type.");

	auto rt = dynamic_cast<const jive::valuetype*>(&result->type());
	if (!rt) throw jlm::error("expected value type.");

	bitcast_op op(*at, *rt);
	return tac::create(op, {argument}, {result});
}

/* struct constant operator */

class struct_constant_op final : public jive::simple_op {
public:
	virtual
	~struct_constant_op();

	inline
	struct_constant_op(const structtype & type)
	: simple_op(create_srcports(type), {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline const structtype &
	type() const noexcept
	{
		return *static_cast<const structtype*>(&result(0).type());
	}

private:
	static inline std::vector<jive::port>
	create_srcports(const structtype & type)
	{
		std::vector<jive::port> ports;
		for (size_t n = 0; n < type.declaration()->nelements(); n++)
			ports.push_back(type.declaration()->element(n));

		return ports;
	}
};

static inline std::unique_ptr<jlm::tac>
create_struct_constant_tac(
	const std::vector<const variable*> & elements,
	jlm::variable * result)
{
	auto rt = dynamic_cast<const structtype*>(&result->type());
	if (!rt) throw jlm::error("expected struct type.");

	struct_constant_op op(*rt);
	return tac::create(op, elements, {result});
}

/* trunc operator */

class trunc_op final : public jive::unary_op {
public:
	virtual
	~trunc_op();

	inline
	trunc_op(const jive::bittype & otype, const jive::bittype & rtype)
	: unary_op(otype, rtype)
	{
		if (otype.nbits() < rtype.nbits())
			throw jlm::error("expected operand's #bits to be larger than results' #bits.");
	}

	inline
	trunc_op(
		std::unique_ptr<jive::type> optype,
		std::unique_ptr<jive::type> restype)
	: unary_op(*optype, *restype)
	{
		auto ot = dynamic_cast<const jive::bittype*>(optype.get());
		if (!ot) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jive::bittype*>(restype.get());
		if (!rt) throw jlm::error("expected bits type.");

		if (ot->nbits() < rt->nbits())
			throw jlm::error("expected operand's #bits to be larger than results' #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(const jive::output * operand) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * operand) const override;

	inline size_t
	nsrcbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&argument(0).type())->nbits();
	}

	inline size_t
	ndstbits() const noexcept
	{
		return static_cast<const jive::bittype*>(&result(0).type())->nbits();
	}
};

static inline std::unique_ptr<jlm::tac>
create_trunc_tac(const variable * operand, jlm::variable * result)
{
	auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!ot) throw jlm::error("expected bits type.");

	auto rt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!rt) throw jlm::error("expected bits type.");

	trunc_op op(*ot, *rt);
	return tac::create(op, {operand}, {result});
}

static inline jive::output *
create_trunc(size_t ndstbits, jive::output * operand)
{
	auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!ot) throw jlm::error("expected bits type.");

	trunc_op op(*ot, jive::bittype(ndstbits));
	return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
}

/* uitofp operator */

class uitofp_op final : public jive::unary_op {
public:
	virtual
	~uitofp_op();

	inline
	uitofp_op(const jive::bittype & srctype, const jlm::fptype & dsttype)
	: unary_op(srctype, dsttype)
	{}

	inline
	uitofp_op(
		std::unique_ptr<jive::type> optype,
		std::unique_ptr<jive::type> restype)
	: unary_op(*optype, *restype)
	{
		auto st = dynamic_cast<const jive::bittype*>(optype.get());
		if (!st) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jlm::fptype*>(restype.get());
		if (!rt) throw jlm::error("expected floating point type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(const jive::output * operand) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * operand) const override;
};

static inline std::unique_ptr<jlm::tac>
create_uitofp_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!st) throw jlm::error("expected bits type.");

	auto rt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!rt) throw jlm::error("expected floating point type.");

	uitofp_op op(*st, *rt);
	return tac::create(op, {operand}, {result});
}

/* sitofp operator */

class sitofp_op final : public jive::unary_op {
public:
	virtual
	~sitofp_op();

	inline
	sitofp_op(const jive::bittype & srctype, const jlm::fptype & dsttype)
	: unary_op(srctype, dsttype)
	{}

	inline
	sitofp_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto st = dynamic_cast<const jive::bittype*>(srctype.get());
		if (!st) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jlm::fptype*>(dsttype.get());
		if (!rt) throw jlm::error("expected floating point type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * output) const noexcept override;

	jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * output) const override;
};

static inline std::unique_ptr<jlm::tac>
create_sitofp_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!st) throw jlm::error("expected bits type.");

	auto rt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!rt) throw jlm::error("expected floating point type.");

	sitofp_op op(*st, *rt);
	return tac::create(op, {operand}, {result});
}

/* constant array operator */

class constant_array_op final : public jive::simple_op {
public:
	virtual
	~constant_array_op() noexcept;

	inline
	constant_array_op(const jive::valuetype & type, size_t size)
	: jive::simple_op(std::vector<jive::port>(size, type), {arraytype(type, size)})
	{
		if (size == 0)
			throw jlm::error("size equals zero.\n");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	size() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->nelements();
	}

	inline const jive::valuetype &
	type() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->element_type();
	}
};

static inline std::unique_ptr<jlm::tac>
create_constant_array_tac(
	const std::vector<const variable*> & elements,
	jlm::variable * result)
{
	if (elements.size() == 0)
		throw jlm::error("expected at least one element.\n");

	auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
	if (!vt) throw jlm::error("expected value type.\n");

	constant_array_op op(*vt, elements.size());
	return tac::create(op, elements, {result});
}

/* constant aggregate zero */

class constant_aggregate_zero_op final : public jive::simple_op {
public:
	virtual
	~constant_aggregate_zero_op();

	inline
	constant_aggregate_zero_op(const jive::type & type)
	: simple_op({}, {type})
	{
		auto st = dynamic_cast<const structtype*>(&type);
		auto at = dynamic_cast<const arraytype*>(&type);
		auto vt = dynamic_cast<const vectortype*>(&type);
		if (!st && !at && !vt)
			throw jlm::error("expected array, struct, or vector type.\n");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_constant_aggregate_zero_tac(jlm::variable * result)
{
	constant_aggregate_zero_op op(result->type());
	return tac::create(op, {}, {result});
}

/* extractelement operator */

class extractelement_op final : public jive::simple_op {
public:
	virtual
	~extractelement_op();

	inline
	extractelement_op(
		const vectortype & vtype,
		const jive::bittype & btype)
	: simple_op({vtype, btype}, {vtype.type()})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * vector,
		const jlm::variable * index,
		jlm::variable * result)
	{
		auto vt = dynamic_cast<const vectortype*>(&vector->type());
		if (!vt) throw jlm::error("expected vector type.");

		auto bt = dynamic_cast<const jive::bittype*>(&index->type());
		if (!bt) throw jlm::error("expected bit type.");

		extractelement_op op(*vt, *bt);
		return tac::create(op, {vector, index}, {result});
	}
};

/* shufflevector operator */

class shufflevector_op final : public jive::simple_op {
public:
	virtual
	~shufflevector_op();

	inline
	shufflevector_op(
		const vectortype & v1,
		const vectortype & v2,
		const vectortype & mask)
	: simple_op({v1, v2, mask}, {vectortype(v1.type(), mask.size())})
	{
		if (v1 != v2)
			throw jlm::error("expected the same vector type.");

		auto bt = dynamic_cast<const jive::bittype*>(&mask.type());
		if (!bt || bt->nbits() != 32)
			throw jlm::error("expected bit32 type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * v1,
		const jlm::variable * v2,
		const jlm::variable * mask,
		jlm::variable * result)
	{
		auto vt1 = dynamic_cast<const vectortype*>(&v1->type());
		auto vt2 = dynamic_cast<const vectortype*>(&v2->type());
		auto vt3 = dynamic_cast<const vectortype*>(&mask->type());
		if (!vt1 || !vt2 || !vt3) throw jlm::error("expected vector type.");

		shufflevector_op op(*vt1, *vt2, *vt3);
		return tac::create(op, {v1, v2, mask}, {result});
	}
};

/* constantvector operator */

class constantvector_op final : public jive::simple_op {
public:
	virtual
	~constantvector_op();

	inline
	constantvector_op(
		const vectortype & vt)
	: simple_op(std::vector<jive::port>(vt.size(), {vt.type()}), {vt})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const std::vector<const variable*> & operands,
		jlm::variable * result)
	{
		auto vt = dynamic_cast<const vectortype*>(&result->type());
		if (!vt) throw jlm::error("expected vector type.");

		constantvector_op op(*vt);
		return tac::create(op, operands, {result});
	}
};

/* insertelement operator */

class insertelement_op final : public jive::simple_op {
public:
	virtual
	~insertelement_op();

	inline
	insertelement_op(
		const vectortype & vectype,
		const jive::valuetype & vtype,
		const jive::bittype & btype)
	: simple_op({vectype, vtype, btype}, {vectype})
	{
		if (vectype.type() != vtype) {
			auto received = vtype.debug_string();
			auto expected = vectype.type().debug_string();
			throw jlm::error(strfmt("expected ", expected, ", got ", received));
		}
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * vector,
		const jlm::variable * value,
		const jlm::variable * index,
		jlm::variable * result)
	{
		auto vct = dynamic_cast<const vectortype*>(&vector->type());
		if (!vct) throw jlm::error("expected vector type.");

		auto vt = dynamic_cast<const jive::valuetype*>(&value->type());
		if (!vt) throw jlm::error("expected value type.");

		auto bt = dynamic_cast<const jive::bittype*>(&index->type());
		if (!bt) throw jlm::error("expected bit type.");

		insertelement_op op(*vct, *vt, *bt);
		return tac::create(op, {vector, value, index}, {result});
	}
};

/* vectorunary operator */

class vectorunary_op final : public jive::simple_op {
public:
	virtual
	~vectorunary_op();

	inline
	vectorunary_op(
		const jive::unary_op & op,
		const vectortype & operand,
		const vectortype & result)
	: simple_op({operand}, {result})
	, op_(std::move(op.copy()))
	{
		if (operand.type() != op.argument(0).type()) {
			auto received = operand.type().debug_string();
			auto expected = op.argument(0).type().debug_string();
			throw jlm::error(strfmt("expected ", expected, ", got ", received));
		}

		if (result.type() != op.result(0).type()) {
			auto received = result.type().debug_string();
			auto expected = op.result(0).type().debug_string();
			throw jlm::error(strfmt("expected ", expected, ", got ", received));
		}
	}

	inline
	vectorunary_op(const vectorunary_op & other)
	: simple_op(other)
	, op_(std::move(other.op_->copy()))
	{}

	inline
	vectorunary_op(vectorunary_op && other)
	: simple_op(other)
	, op_(std::move(other.op_))
	{}

	inline vectorunary_op &
	operator=(const vectorunary_op & other)
	{
		if (this != &other)
			op_ = std::move(other.op_->copy());

		return *this;
	}

	inline vectorunary_op &
	operator=(vectorunary_op && other)
	{
		if (this != &other)
			op_ = std::move(other.op_);

		return *this;
	}

	inline const jive::unary_op &
	operation() const noexcept
	{
		return *static_cast<const jive::unary_op*>(op_.get());
	}

	virtual bool
	operator==(const jive::operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const jive::unary_op & unop,
		const jlm::variable * operand,
		jlm::variable * result)
	{
		auto vct1 = dynamic_cast<const vectortype*>(&operand->type());
		auto vct2 = dynamic_cast<const vectortype*>(&result->type());
		if (!vct1 || !vct2) throw jlm::error("expected vector type.");

		vectorunary_op op(unop, *vct1, *vct2);
		return tac::create(op, {operand}, {result});
	}

private:
	std::unique_ptr<jive::operation> op_;
};

/* vectorbinary operator */

class vectorbinary_op final : public jive::simple_op {
public:
	virtual
	~vectorbinary_op();

	inline
	vectorbinary_op(
		const jive::binary_op & binop,
		const vectortype & op1,
		const vectortype & op2,
		const vectortype & result)
	: simple_op({op1, op2}, {result})
	, op_(std::move(binop.copy()))
	{
		if (op1 != op2)
			throw jlm::error("expected the same vector types.");

		if (op1.type() != binop.argument(0).type()) {
			auto received = op1.type().debug_string();
			auto expected = binop.argument(0).type().debug_string();
			throw jlm::error(strfmt("expected ", expected, ", got ", received));
		}

		if (result.type() != binop.result(0).type()) {
			auto received = result.type().debug_string();
			auto expected = binop.result(0).type().debug_string();
			throw jlm::error(strfmt("expected ", expected, ", got ", received));
		}
	}

	inline
	vectorbinary_op(const vectorbinary_op & other)
	: simple_op(other)
	, op_(std::move(other.op_->copy()))
	{}

	inline
	vectorbinary_op(vectorbinary_op && other)
	: simple_op(other)
	, op_(std::move(other.op_))
	{}

	inline vectorbinary_op &
	operator=(const vectorbinary_op & other)
	{
		if (this != &other)
			op_ = std::move(other.op_->copy());

		return *this;
	}

	inline vectorbinary_op &
	operator=(vectorbinary_op && other)
	{
		if (this != &other)
			op_ = std::move(other.op_);

		return *this;
	}

	inline const jive::binary_op &
	operation() const noexcept
	{
		return *static_cast<const jive::binary_op*>(op_.get());
	}

	virtual bool
	operator==(const jive::operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline std::unique_ptr<jlm::tac>
	create(
		const jive::binary_op & binop,
		const jlm::variable * op1,
		const jlm::variable * op2,
		jlm::variable * result)
	{
		auto vct1 = dynamic_cast<const vectortype*>(&op1->type());
		auto vct2 = dynamic_cast<const vectortype*>(&op2->type());
		auto vct3 = dynamic_cast<const vectortype*>(&result->type());
		if (!vct1 || !vct2 || !vct3) throw jlm::error("expected vector type.");

		vectorbinary_op op(binop, *vct1, *vct2, *vct3);
		return tac::create(op, {op1, op2}, {result});
	}

private:
	std::unique_ptr<jive::operation> op_;
};

/* constant data vector operator */

class constant_data_vector_op final : public jive::simple_op {
public:
	virtual
	~constant_data_vector_op() noexcept;

	inline
	constant_data_vector_op(const jive::valuetype & type, size_t size)
	: simple_op(std::vector<jive::port>(size, type), {vectortype(type, size)})
	{
		if (size == 0)
			throw jlm::error("size equals zero");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	inline size_t
	size() const noexcept
	{
		return static_cast<const vectortype*>(&result(0).type())->size();
	}

	inline const jive::valuetype &
	type() const noexcept
	{
		return static_cast<const vectortype*>(&result(0).type())->type();
	}

	static inline std::unique_ptr<jlm::tac>
	create(
		const std::vector<const variable*> & elements,
		jlm::variable * result)
	{
		if (elements.size() == 0)
			throw jlm::error("expected at least one element");

		auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
		if (!vt) throw jlm::error("expected value type");

		constant_data_vector_op op(*vt, elements.size());
		return tac::create(op, elements, {result});
	}
};

/* extractvalue operator */

class extractvalue_op final : public jive::simple_op {
	typedef std::vector<unsigned>::const_iterator const_iterator;
public:
	virtual
	~extractvalue_op() noexcept;

	inline
	extractvalue_op(
		const jive::type & aggtype,
		const std::vector<unsigned> & indices)
	: simple_op({aggtype}, {dsttype(aggtype, indices)})
	, indices_(indices)
	{
		if (indices.empty())
			throw jlm::error("expected at least one index.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const_iterator
	begin() const
	{
		return indices_.begin();
	}

	const_iterator
	end() const
	{
		return indices_.end();
	}

	static inline std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * aggregate,
		const std::vector<unsigned> & indices,
		jlm::variable * result)
	{
		extractvalue_op op(aggregate->type(), indices);
		return tac::create(op, {aggregate}, {result});
	}

private:
	static inline jive::port
	dsttype(
		const jive::type & aggtype,
		const std::vector<unsigned> & indices)
	{
		const jive::type * type = &aggtype;
		for (const auto & index : indices) {
			if (auto st = dynamic_cast<const structtype*>(type)) {
				if (index >= st->declaration()->nelements())
					throw jlm::error("extractvalue index out of bound.");

				type = &st->declaration()->element(index);
			} else if (auto at = dynamic_cast<const arraytype*>(type)) {
				if (index >= at->nelements())
					throw jlm::error("extractvalue index out of bound.");

				type = &at->element_type();
			} else
				throw jlm::error("expected struct or array type.");
		}

		return {*type};
	}

	std::vector<unsigned> indices_;
};

/* loop state mux operator */

class loopstatemux_op final : public jive::simple_op {
public:
	virtual
	~loopstatemux_op() noexcept;

	loopstatemux_op(size_t noperands, size_t nresults)
	: simple_op(create_portvector(noperands), create_portvector(nresults))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static std::vector<jive::output*>
	create(
		const std::vector<jive::output*> & operands,
		size_t nresults)
	{
		if (operands.empty())
			throw jlm::error("Insufficient number of operands.");

		auto region = operands.front()->region();
		loopstatemux_op op(operands.size(), nresults);
		return jive::simple_node::create_normalized(region, op, operands);
	}

	static std::vector<jive::output*>
	create_split(jive::output * operand, size_t nresults)
	{
		loopstatemux_op op(1, nresults);
		return jive::simple_node::create_normalized(operand->region(), op, {operand});
	}

	static jive::output *
	create_merge(const std::vector<jive::output*> & operands)
	{
		if (operands.empty())
			throw jlm::error("Insufficient number of operands.");

		loopstatemux_op op(operands.size(), 1);
		auto region = operands.front()->region();
		return jive::simple_node::create_normalized(region, op, operands)[0];
	}

private:
	static std::vector<jive::port>
	create_portvector(size_t size)
	{
		return std::vector<jive::port>(size, loopstatetype());
	}
};

}

#endif
