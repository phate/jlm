/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_OPERATORS_HPP
#define JLM_IR_OPERATORS_OPERATORS_HPP

#include <jive/types/bitstring/type.h>
#include <jive/types/function.h>
#include <jive/types/record.h>
#include <jive/rvsdg/control.h>
#include <jive/rvsdg/nullary.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/type.h>
#include <jive/rvsdg/unary.h>

#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/tac.hpp>
#include <jlm/jlm/ir/types.hpp>

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
	return create_tac(assignment_op(type), {r, arg}, {});
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
	return create_tac(op, {p, t, f}, {result});
}

/* fp2ui operator */

class fp2ui_op final : public jive::simple_op {
public:
	virtual
	~fp2ui_op() noexcept;

	inline
	fp2ui_op(const fpsize & size, const jive::bittype & type)
	: jive::simple_op({fptype(size)}, {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_fp2ui_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	fp2ui_op op(st->size(), *dt);
	return create_tac(op, {operand}, {result});
}

/* fp2si operator */

class fp2si_op final : public jive::simple_op {
public:
	virtual
	~fp2si_op() noexcept;

	inline
	fp2si_op(const fpsize & size, const jive::bittype & type)
	: jive::simple_op({fptype(size)}, {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_fp2si_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const fptype*>(&operand->type());
	if (!st) throw jlm::error("expected floating point type.");

	auto dt = dynamic_cast<const jive::bittype*>(&result->type());
	if (!dt) throw jlm::error("expected bitstring type.");

	fp2si_op op(st->size(), *dt);
	return create_tac(op, {operand}, {result});
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
	return create_tac(op, {operand}, {result});
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
	return create_tac(op, {operand}, {});
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
	return create_tac(op, {}, {result});
}

/* bits2ptr operator */

class bits2ptr_op final : public jive::simple_op {
public:
	virtual
	~bits2ptr_op() noexcept;

	inline
	bits2ptr_op(const jive::bittype & btype, const jlm::ptrtype & ptype)
	: simple_op({btype}, {ptype})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {argument}, {result});
}

/* ptr2bits operator */

class ptr2bits_op final : public jive::simple_op {
public:
	virtual
	~ptr2bits_op() noexcept;

	inline
	ptr2bits_op(const jlm::ptrtype & ptype, const jive::bittype & btype)
	: simple_op({ptype}, {btype})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {argument}, {result});
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
	: simple_op({ptype, ptype}, {jive::bit1})
	, cmp_(cmp)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {op1, op2}, {result});
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
	return create_tac(op, {operand}, {result});
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
	: simple_op({fptype(size), fptype(size)}, {jive::bit1})
	, cmp_(cmp)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

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
	return create_tac(op, {op1, op2}, {result});
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
create_undef_constant_tac(jlm::variable * result)
{
	auto vt = dynamic_cast<const jive::valuetype*>(&result->type());
	if (!vt) throw jlm::error("expected value type.");

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
	: simple_op({fptype(size), fptype(size)}, {fptype(size)})
	, op_(op)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

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
	return create_tac(op, {op1, op2}, {result});
}

/* fpext operator */

class fpext_op final : public jive::simple_op {
public:
	virtual
	~fpext_op() noexcept;

	inline
	fpext_op(const jlm::fpsize & srcsize, const jlm::fpsize & dstsize)
	: simple_op({fptype(srcsize)}, {fptype(dstsize)})
	{
		if (srcsize == fpsize::flt && dstsize == fpsize::half)
			throw jlm::error("destination type size must be bigger than source type size.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {operand}, {result});
}

/* fptrunc operator */

class fptrunc_op final : public jive::simple_op {
public:
	virtual
	~fptrunc_op() noexcept;

	inline
	fptrunc_op(const fpsize & srcsize, const fpsize & dstsize)
	: simple_op({fptype(srcsize)}, {fptype(dstsize)})
	{
		if (srcsize == fpsize::half
		|| (srcsize == fpsize::flt && dstsize != fpsize::half)
		|| (srcsize == fpsize::dbl && dstsize == fpsize::dbl))
			throw jlm::error("destination tpye size must be smaller than source size type.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {operand}, {result});
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
	bitcast_op(const jive::valuetype & srctype, const jive::valuetype & dsttype)
	: simple_op({srctype}, {dsttype})
	{}

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
};

static inline std::unique_ptr<jlm::tac>
create_bitcast_tac(const variable * argument, variable * result)
{
	auto at = dynamic_cast<const jive::valuetype*>(&argument->type());
	if (!at) throw jlm::error("expected value type.");

	auto rt = dynamic_cast<const jive::valuetype*>(&result->type());
	if (!rt) throw jlm::error("expected value type.");

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
	return create_tac(op, elements, {result});
}

/* trunc operator */

class trunc_op final : public jive::simple_op {
public:
	virtual
	~trunc_op();

	inline
	trunc_op(const jive::bittype & otype, const jive::bittype & rtype)
	: simple_op({otype}, {rtype})
	{
		if (otype.nbits() < rtype.nbits())
			throw jlm::error("expected operand's #bits to be larger than results' #bits.");
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

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
	return create_tac(op, {operand}, {result});
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

class uitofp_op final : public jive::simple_op {
public:
	virtual
	~uitofp_op();

	inline
	uitofp_op(const jive::bittype & srctype, const jlm::fptype & dsttype)
	: simple_op({srctype}, {dsttype})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_uitofp_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!st) throw jlm::error("expected bits type.");

	auto rt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!rt) throw jlm::error("expected floating point type.");

	uitofp_op op(*st, *rt);
	return create_tac(op, {operand}, {result});
}

/* sitofp operator */

class sitofp_op final : public jive::simple_op {
public:
	virtual
	~sitofp_op();

	inline
	sitofp_op(const jive::bittype & srctype, const jlm::fptype & dsttype)
	: simple_op({srctype}, {dsttype})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

static inline std::unique_ptr<jlm::tac>
create_sitofp_tac(const variable * operand, jlm::variable * result)
{
	auto st = dynamic_cast<const jive::bittype*>(&operand->type());
	if (!st) throw jlm::error("expected bits type.");

	auto rt = dynamic_cast<const jlm::fptype*>(&result->type());
	if (!rt) throw jlm::error("expected floating point type.");

	sitofp_op op(*st, *rt);
	return create_tac(op, {operand}, {result});
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
	return create_tac(op, elements, {result});
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
		/* FIXME: add support for vector type */
		auto st = dynamic_cast<const structtype*>(&type);
		auto at = dynamic_cast<const arraytype*>(&type);
		if (!st && !at)
			throw jlm::error("expected array or struct type.\n");
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
	return create_tac(op, {}, {result});
}

}

#endif
