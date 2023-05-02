/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_OPERATORS_HPP
#define JLM_LLVM_IR_OPERATORS_OPERATORS_HPP

#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/record.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

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
	{}

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
		JLM_ASSERT(n < narguments());
		return nodes_[n];
	}

	static std::unique_ptr<jlm::tac>
	create(
		const std::vector<std::pair<const variable*,cfg_node*>> & arguments,
		const jive::type & type)
	{
		std::vector<cfg_node*> nodes;
		std::vector<const variable*> operands;
		for (const auto & argument : arguments) {
			nodes.push_back(argument.second);
			operands.push_back(argument.first);
		}

		phi_op phi(nodes, type);
		return tac::create(phi, operands);
	}

private:
	std::vector<cfg_node*> nodes_;
};

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

	static std::unique_ptr<jlm::tac>
	create(const variable * rhs, const variable * lhs)
	{
		if (rhs->type() != lhs->type())
			throw jlm::error("LHS and RHS of assignment must have same type.");

		return tac::create(assignment_op(rhs->type()), {lhs, rhs});
	}
};

/* select operator */

class select_op final : public jive::simple_op {
public:
	virtual
	~select_op() noexcept;

	select_op(const jive::type & type)
	: jive::simple_op({jive::bit1, type, type}, {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const jive::type &
	type() const noexcept
	{
		return result(0).type();
	}

	static std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * p,
		const jlm::variable * t,
		const jlm::variable * f)
	{
		select_op op(t->type());
		return tac::create(op, {p, t, f});
	}
};

/* vector select operator */

class vectorselect_op final : public jive::simple_op {
public:
	virtual
	~vectorselect_op() noexcept;

private:
	vectorselect_op(
		const vectortype & pt,
		const vectortype & vt)
	: jive::simple_op({pt, vt, vt}, {vt})
	{}

public:
	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const jive::type &
	type() const noexcept
	{
		return result(0).type();
	}

	size_t
	size() const noexcept
	{
		return dynamic_cast<const vectortype*>(&type())->size();
	}

	static std::unique_ptr<jlm::tac>
	create(
		const variable * p,
		const variable * t,
		const variable * f)
	{
        if (is<fixedvectortype>(p->type()) && is<fixedvectortype>(t->type()))
            return createVectorSelectTac<fixedvectortype>(p, t, f);

        if (is<scalablevectortype>(p->type()) && is<scalablevectortype>(t->type()))
            return createVectorSelectTac<scalablevectortype>(p, t, f);

        throw error("Expected vector types as operands.");
	}

private:
    template<typename T> static std::unique_ptr<tac>
    createVectorSelectTac(
        const variable * p,
        const variable * t,
        const variable * f)
    {
        auto fvt = static_cast<const T*>(&t->type());
        T pt(jive::bit1, fvt->size());
        T vt(fvt->type(), fvt->size());
        vectorselect_op op(pt, vt);
        return tac::create(op, {p, t, f});
    }
};

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const fptype*>(&operand->type());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jive::bittype*>(&type);
		if (!dt) throw jlm::error("expected bitstring type.");

		fp2ui_op op(st->size(), *dt);
		return tac::create(op, {operand});
	}
};

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const fptype*>(&operand->type());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jive::bittype*>(&type);
		if (!dt) throw jlm::error("expected bitstring type.");

		fp2si_op op(st->size(), *dt);
		return tac::create(op, {operand});
	}
};

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const jive::ctltype*>(&operand->type());
		if (!st) throw jlm::error("expected control type.");

		auto dt = dynamic_cast<const jive::bittype*>(&type);
		if (!dt) throw jlm::error("expected bitstring type.");

		ctl2bits_op op(*st, *dt);
		return tac::create(op, {operand});
	}
};

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

	static std::unique_ptr<jlm::tac>
	create(size_t nalternatives, const variable * operand)
	{
		jive::ctltype type(nalternatives);
		branch_op op(type);
		return tac::create(op, {operand});
	}
};

/** \brief ConstantPointerNullOperation class
 *
 * This operator is the Jlm equivalent of LLVM's ConstantPointerNull constant.
 */
class ConstantPointerNullOperation final : public jive::simple_op {
public:
  ~ConstantPointerNullOperation() noexcept override;

  explicit
  ConstantPointerNullOperation(const PointerType & pointerType)
    : simple_op({}, {pointerType})
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

  [[nodiscard]] const PointerType &
  GetPointerType() const noexcept
  {
    return *AssertedCast<const PointerType>(&result(0).type());
  }

  static std::unique_ptr<jlm::tac>
  Create(const jive::type & type)
  {
    auto & pointerType = CheckAndExtractType(type);

    ConstantPointerNullOperation operation(pointerType);
    return tac::create(operation, {});
  }

  static jive::output *
  Create(
    jive::region * region,
    const jive::type & type)
  {
    auto & pointerType = CheckAndExtractType(type);

    ConstantPointerNullOperation operation(pointerType);
    return jive::simple_node::create_normalized(region, operation, {})[0];
  }

private:
  static const PointerType &
  CheckAndExtractType(const jive::type & type)
  {
    if (auto pointerType = dynamic_cast<const PointerType*>(&type))
      return *pointerType;

    throw jlm::error("expected pointer type.");
  }
};

/* bits2ptr operator */

class bits2ptr_op final : public jive::unary_op {
public:
	virtual
	~bits2ptr_op();

	inline
	bits2ptr_op(const jive::bittype & btype, const PointerType & ptype)
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

		auto pt = dynamic_cast<const PointerType*>(dsttype.get());
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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * argument,
		const jive::type & type)
	{
		auto at = dynamic_cast<const jive::bittype*>(&argument->type());
		if (!at) throw jlm::error("expected bitstring type.");

		auto pt = dynamic_cast<const PointerType*>(&type);
		if (!pt) throw jlm::error("expected pointer type.");

		jlm::bits2ptr_op op(*at, *pt);
		return tac::create(op, {argument});
	}

	static jive::output *
	create(
		jive::output * operand,
		const jive::type & type)
	{
		auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!ot) throw jlm::error("expected bitstring type.");

		auto pt = dynamic_cast<const PointerType*>(&type);
		if (!pt) throw jlm::error("expected pointer type.");

		jlm::bits2ptr_op op(*ot, *pt);
		return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}
};

/* ptr2bits operator */

class ptr2bits_op final : public jive::unary_op {
public:
	virtual
	~ptr2bits_op();

	inline
	ptr2bits_op(const PointerType & ptype, const jive::bittype & btype)
	: unary_op(ptype, btype)
	{}

	inline
	ptr2bits_op(
		std::unique_ptr<jive::type> srctype,
		std::unique_ptr<jive::type> dsttype)
	: unary_op(*srctype, *dsttype)
	{
		auto pt = dynamic_cast<const PointerType*>(srctype.get());
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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * argument,
		const jive::type & type)
	{
		auto pt = dynamic_cast<const PointerType*>(&argument->type());
		if (!pt) throw jlm::error("expected pointer type.");

		auto bt = dynamic_cast<const jive::bittype*>(&type);
		if (!bt) throw jlm::error("expected bitstring type.");

		jlm::ptr2bits_op op(*pt, *bt);
		return tac::create(op, {argument});
	}
};

/* Constant Data Array operator */

class ConstantDataArray final : public jive::simple_op {
public:
  virtual
  ~ConstantDataArray();

  ConstantDataArray(const jive::valuetype & type, size_t size)
    : simple_op(std::vector<jive::port>(size, type), {jlm::arraytype(type, size)})
  {
    if (size == 0)
      throw jlm::error("size equals zero.");
  }

  virtual bool
  operator==(const jive::operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jive::operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return static_cast<const arraytype*>(&result(0).type())->nelements();
  }

  const jive::valuetype &
  type() const noexcept
  {
    return static_cast<const arraytype*>(&result(0).type())->element_type();
  }

  static std::unique_ptr<jlm::tac>
  create(const std::vector<const variable*> & elements)
  {
    if (elements.size() == 0)
      throw jlm::error("expected at least one element.");

    auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
    if (!vt) throw jlm::error("expected value type.");

    ConstantDataArray op(*vt, elements.size());
    return tac::create(op, elements);
  }

  static jive::output *
  Create(const std::vector<jive::output*> & elements)
  {
    if (elements.empty())
      throw error("Expected at least one element.");

    auto valueType = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
    if (!valueType)
    {
      throw error("Expected value type.");
    }

    ConstantDataArray operation(*valueType, elements.size());
    return jive::simple_node::create_normalized(elements[0]->region(), operation, elements)[0];
  }
};

/* pointer compare operator */

enum class cmp {eq, ne, gt, ge, lt, le};

class ptrcmp_op final : public jive::binary_op {
public:
	virtual
	~ptrcmp_op();

	inline
	ptrcmp_op(const PointerType & ptype, const jlm::cmp & cmp)
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

	static std::unique_ptr<jlm::tac>
	create(
		const jlm::cmp & cmp,
		const variable * op1,
		const variable * op2)
	{
		auto pt = dynamic_cast<const PointerType*>(&op1->type());
		if (!pt) throw jlm::error("expected pointer type.");

		jlm::ptrcmp_op op(*pt, cmp);
		return tac::create(op, {op1, op2});
	}

private:
	jlm::cmp cmp_;
};

/* zext operator */

class zext_op final : public jive::unary_op {
public:
	virtual
	~zext_op();

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!st) throw jlm::error("expected bitstring type.");

		auto dt = dynamic_cast<const jive::bittype*>(&type);
		if (!dt) throw jlm::error("expected bitstring type.");

		jlm::zext_op op(st->nbits(), dt->nbits());
		return tac::create(op, {operand});
	}
};

/* floating point constant operator */

class ConstantFP final : public jive::simple_op {
public:
	virtual
	~ConstantFP();

	inline
	ConstantFP(const jlm::fpsize & size, const llvm::APFloat & constant)
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

	static std::unique_ptr<jlm::tac>
	create(
		const llvm::APFloat & constant,
		const jive::type & type)
	{
		auto ft = dynamic_cast<const jlm::fptype*>(&type);
		if (!ft) throw jlm::error("expected floating point type.");

		jlm::ConstantFP op(ft->size(), constant);
		return tac::create(op, {});
	}

private:
	/* FIXME: I would not like to use the APFloat here,
	   but I don't have a replacement right now. */
	llvm::APFloat constant_;
};

/* floating point comparison operator */

enum class fpcmp {
	TRUE, FALSE, oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
};

class fpcmp_op final : public jive::binary_op {
public:
	virtual
	~fpcmp_op();

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

	static std::unique_ptr<jlm::tac>
	create(
		const jlm::fpcmp & cmp,
		const variable * op1,
		const variable * op2)
	{
		auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
		if (!ft) throw jlm::error("expected floating point type.");

		jlm::fpcmp_op op(cmp, ft->size());
		return tac::create(op, {op1, op2});
	}

private:
	jlm::fpcmp cmp_;
};

/** \brief UndefValueOperation class
 *
 * This operator is the Jlm equivalent of LLVM's UndefValue constant.
 */
class UndefValueOperation final : public jive::simple_op {
public:
  ~UndefValueOperation() noexcept override;

  explicit
  UndefValueOperation(const jive::type & type)
    : simple_op({}, {type})
  {}

  UndefValueOperation(const UndefValueOperation &) = default;

  UndefValueOperation &
  operator=(const UndefValueOperation &) = delete;

  UndefValueOperation &
  operator=(UndefValueOperation &&) = delete;

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

  [[nodiscard]] const jive::type &
  GetType() const noexcept
  {
    return result(0).type();
  }

  static jive::output *
  Create(
    jive::region & region,
    const jive::type & type)
  {
    UndefValueOperation operation(type);
    return jive::simple_node::create_normalized(&region, operation, {})[0];
  }

  static std::unique_ptr<jlm::tac>
  Create(const jive::type & type)
  {
    UndefValueOperation operation(type);
    return tac::create(operation, {});
  }

  static std::unique_ptr<jlm::tac>
  Create(
    const jive::type & type,
    const std::string & name)
  {
    UndefValueOperation operation(type);
    return tac::create(operation, {}, {name});
  }

  static std::unique_ptr<jlm::tac>
  Create(std::unique_ptr<tacvariable> result)
  {
    auto & type = result->type();

    std::vector<std::unique_ptr<tacvariable>> results;
    results.push_back(std::move(result));

    UndefValueOperation operation(type);
    return tac::create(operation, {}, std::move(results));
  }
};

/** \brief PoisonValueOperation class
 *
 * This operator is the Jlm equivalent of LLVM's PoisonValue constant.
 */
class PoisonValueOperation final : public jive::simple_op {
public:
  ~PoisonValueOperation() noexcept override;

  explicit
  PoisonValueOperation(const jive::valuetype & type)
  : jive::simple_op({}, {type})
  {}

  PoisonValueOperation(const PoisonValueOperation&)
  = default;

  PoisonValueOperation(PoisonValueOperation&&)  = delete;

  PoisonValueOperation &
  operator=(const PoisonValueOperation&) = delete;

  PoisonValueOperation &
  operator=(PoisonValueOperation&&)  = delete;

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jive::operation>
  copy() const override;

  const jive::valuetype &
  GetType() const noexcept
  {
    auto & type = result(0).type();
    JLM_ASSERT(dynamic_cast<const jive::valuetype*>(&type));
    return *static_cast<const jive::valuetype*>(&type);
  }

  static std::unique_ptr<jlm::tac>
  Create(const jive::type & type)
  {
    auto & valueType = CheckAndConvertType(type);

    PoisonValueOperation operation(valueType);
    return tac::create(operation, {});
  }

  static jive::output *
  Create(jive::region * region, const jive::type & type)
  {
    auto & valueType = CheckAndConvertType(type);

    PoisonValueOperation operation(valueType);
    return jive::simple_node::create_normalized(region, operation, {})[0];
  }

private:
  static const jive::valuetype &
  CheckAndConvertType(const jive::type & type)
  {
    if (auto valueType = dynamic_cast<const jive::valuetype*>(&type))
      return *valueType;

    throw jlm::error("Expected value type.");
  }
};

/* floating point arithmetic operator */

enum class fpop {add, sub, mul, div, mod};

class fpbin_op final : public jive::binary_op {
public:
	virtual
	~fpbin_op();

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

	static std::unique_ptr<jlm::tac>
	create(
		const jlm::fpop & fpop,
		const variable * op1,
		const variable * op2)
	{
		auto ft = dynamic_cast<const jlm::fptype*>(&op1->type());
		if (!ft) throw jlm::error("expected floating point type.");

		jlm::fpbin_op op(fpop, ft->size());
		return tac::create(op, {op1, op2});
	}

private:
	jlm::fpop op_;
};

/* fpext operator */

class fpext_op final : public jive::unary_op {
public:
	virtual
	~fpext_op();

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const jlm::fptype*>(&operand->type());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const jlm::fptype*>(&type);
		if (!dt) throw jlm::error("expected floating point type.");

		jlm::fpext_op op(st->size(), dt->size());
		return tac::create(op, {operand});
	}
};

/* fpneg operator */

class fpneg_op final : public jive::unary_op {
public:
	~fpneg_op() override;

	fpneg_op(const jlm::fpsize & size)
	: unary_op(fptype(size), fptype(size))
	{}

	fpneg_op(const jive::type & type)
	: unary_op(type, type)
	{
		auto st = dynamic_cast<const jlm::fptype*>(&type);
		if (!st) throw jlm::error("expected floating point type.");
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

	const jlm::fpsize &
	size() const noexcept
	{
		return static_cast<const jlm::fptype*>(&argument(0).type())->size();
	}

	static std::unique_ptr<jlm::tac>
	create(const variable * operand)
	{
		auto type = dynamic_cast<const jlm::fptype*>(&operand->type());
		if (!type) throw jlm::error("expected floating point type.");

		jlm::fpneg_op op(type->size());
		return tac::create(op, {operand});
	}
};

/* fptrunc operator */

class fptrunc_op final : public jive::unary_op {
public:
	virtual
	~fptrunc_op();

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const fptype*>(&operand->type());
		if (!st) throw jlm::error("expected floating point type.");

		auto dt = dynamic_cast<const fptype*>(&type);
		if (!dt) throw jlm::error("expected floating point type.");

		fptrunc_op op(st->size(), dt->size());
		return tac::create(op, {operand});
	}
};

/* valist operator */

class valist_op final : public jive::simple_op {
public:
	virtual
	~valist_op();

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

	static std::unique_ptr<jlm::tac>
	create(const std::vector<const variable*> & arguments)
	{
		std::vector<std::unique_ptr<jive::type>> operands;
		for (const auto & argument : arguments)
			operands.push_back(argument->type().copy());

		jlm::valist_op op(std::move(operands));
		return tac::create(op, arguments);
	}

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
		check_types(*srctype, *dsttype);
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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto pair = check_types(operand->type(), type);

		bitcast_op op(*pair.first, *pair.second);
		return tac::create(op, {operand});
	}

	static jive::output *
	create(jive::output * operand, const jive::type & rtype)
	{
		auto pair = check_types(operand->type(), rtype);

		bitcast_op op(*pair.first, *pair.second);
		return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}

private:
	static std::pair<const jive::valuetype*, const jive::valuetype*>
	check_types(const jive::type & otype, const jive::type & rtype)
	{
		auto ot = dynamic_cast<const jive::valuetype*>(&otype);
		if (!ot) throw jlm::error("expected value type.");

		auto rt = dynamic_cast<const jive::valuetype*>(&rtype);
		if (!rt) throw jlm::error("expected value type.");

		return std::make_pair(ot, rt);
	}
};

/* ConstantStruct operator */

class ConstantStruct final : public jive::simple_op {
public:
	virtual
	~ConstantStruct();

	inline
	ConstantStruct(const StructType & type)
	: simple_op(create_srcports(type), {type})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const StructType &
	type() const noexcept
	{
		return *static_cast<const StructType*>(&result(0).type());
	}

	static std::unique_ptr<jlm::tac>
	create(
		const std::vector<const variable*> & elements,
		const jive::type & type)
	{
		auto rt = dynamic_cast<const StructType*>(&type);
		if (!rt) throw jlm::error("expected struct type.");

		ConstantStruct op(*rt);
		return tac::create(op, elements);
	}

private:
	static inline std::vector<jive::port>
	create_srcports(const StructType & type)
	{
		std::vector<jive::port> ports;
		for (size_t n = 0; n < type.GetDeclaration().nelements(); n++)
			ports.push_back(type.GetDeclaration().element(n));

		return ports;
	}
};

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!ot) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jive::bittype*>(&type);
		if (!rt) throw jlm::error("expected bits type.");

		trunc_op op(*ot, *rt);
		return tac::create(op, {operand});
	}

	static jive::output *
	create(
		size_t ndstbits,
		jive::output * operand)
	{
		auto ot = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!ot) throw jlm::error("expected bits type.");

		trunc_op op(*ot, jive::bittype(ndstbits));
		return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}
};


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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!st) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jlm::fptype*>(&type);
		if (!rt) throw jlm::error("expected floating point type.");

		uitofp_op op(*st, *rt);
		return tac::create(op, {operand});
	}
};

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

	static std::unique_ptr<jlm::tac>
	create(
		const variable * operand,
		const jive::type & type)
	{
		auto st = dynamic_cast<const jive::bittype*>(&operand->type());
		if (!st) throw jlm::error("expected bits type.");

		auto rt = dynamic_cast<const jlm::fptype*>(&type);
		if (!rt) throw jlm::error("expected floating point type.");

		sitofp_op op(*st, *rt);
		return tac::create(op, {operand});
	}
};

/* ConstantArray */

class ConstantArray final : public jive::simple_op {
public:
	virtual
	~ConstantArray();

	ConstantArray(const jive::valuetype & type, size_t size)
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

	size_t
	size() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->nelements();
	}

	const jive::valuetype &
	type() const noexcept
	{
		return static_cast<const arraytype*>(&result(0).type())->element_type();
	}

	static std::unique_ptr<jlm::tac>
	create(const std::vector<const variable*> & elements)
	{
		if (elements.size() == 0)
			throw jlm::error("expected at least one element.\n");

		auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
		if (!vt) throw jlm::error("expected value type.\n");

		ConstantArray op(*vt, elements.size());
		return tac::create(op, elements);
	}
};

/* ConstantAggregateZero operator */

class ConstantAggregateZero final : public jive::simple_op {
public:
  virtual
  ~ConstantAggregateZero();

  ConstantAggregateZero(const jive::type & type)
    : simple_op({}, {type})
  {
    auto st = dynamic_cast<const StructType*>(&type);
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

  static std::unique_ptr<jlm::tac>
  create(const jive::type & type)
  {
    ConstantAggregateZero op(type);
    return tac::create(op, {});
  }

  static jive::output *
  Create(
    jive::region & region,
    const jive::type & type)
  {
    ConstantAggregateZero operation(type);
    return jive::simple_node::create_normalized(&region, operation, {})[0];
  }
};

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
		const jlm::variable * index)
	{
		auto vt = dynamic_cast<const vectortype*>(&vector->type());
		if (!vt) throw jlm::error("expected vector type.");

		auto bt = dynamic_cast<const jive::bittype*>(&index->type());
		if (!bt) throw jlm::error("expected bit type.");

		extractelement_op op(*vt, *bt);
		return tac::create(op, {vector, index});
	}
};

/* shufflevector operator */

class shufflevector_op final : public jive::simple_op {
public:
	~shufflevector_op() override;

	shufflevector_op(
        const fixedvectortype & v,
        const std::vector<int> & mask)
    : simple_op({v, v}, {v})
    , Mask_(mask)
    {}

    shufflevector_op(
        const scalablevectortype & v,
        const std::vector<int> & mask)
    : simple_op({v, v}, {v})
    , Mask_(mask)
    {}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

    const llvm::ArrayRef<int>
    Mask() const
    {
        return Mask_;
    }

	static std::unique_ptr<jlm::tac>
	create(
            const variable * v1,
            const variable * v2,
            const std::vector<int> & mask)
	{
        if (is<fixedvectortype>(v1->type())
        && is<fixedvectortype>(v2->type()))
            return CreateShuffleVectorTac<fixedvectortype>(v1, v2, mask);

        if (is<scalablevectortype>(v1->type())
        && is<scalablevectortype>(v2->type()))
            return CreateShuffleVectorTac<scalablevectortype>(v1, v2, mask);

        throw error("Expected vector types as operands.");
	}

private:
    template<typename T> static std::unique_ptr<tac>
    CreateShuffleVectorTac(
            const variable * v1,
            const variable * v2,
            const std::vector<int> & mask)
    {
        auto vt = static_cast<const T*>(&v1->type());
        shufflevector_op op(*vt, mask);
        return tac::create(op, {v1, v2});
    }

    std::vector<int> Mask_;
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
		const jive::type & type)
	{
		auto vt = dynamic_cast<const vectortype*>(&type);
		if (!vt) throw jlm::error("expected vector type.");

		constantvector_op op(*vt);
		return tac::create(op, operands);
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
		const jlm::variable * index)
	{
		auto vct = dynamic_cast<const vectortype*>(&vector->type());
		if (!vct) throw jlm::error("expected vector type.");

		auto vt = dynamic_cast<const jive::valuetype*>(&value->type());
		if (!vt) throw jlm::error("expected value type.");

		auto bt = dynamic_cast<const jive::bittype*>(&index->type());
		if (!bt) throw jlm::error("expected bit type.");

		insertelement_op op(*vct, *vt, *bt);
		return tac::create(op, {vector, value, index});
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
	, op_(op.copy())
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
	, op_(other.op_->copy())
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
			op_ = other.op_->copy();

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
		const jive::type & type)
	{
		auto vct1 = dynamic_cast<const vectortype*>(&operand->type());
		auto vct2 = dynamic_cast<const vectortype*>(&type);
		if (!vct1 || !vct2) throw jlm::error("expected vector type.");

		vectorunary_op op(unop, *vct1, *vct2);
		return tac::create(op, {operand});
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
	, op_(binop.copy())
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
	, op_(other.op_->copy())
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
			op_ = other.op_->copy();

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
		const jive::type & type)
	{
		auto vct1 = dynamic_cast<const vectortype*>(&op1->type());
		auto vct2 = dynamic_cast<const vectortype*>(&op2->type());
		auto vct3 = dynamic_cast<const vectortype*>(&type);
		if (!vct1 || !vct2 || !vct3) throw jlm::error("expected vector type.");

		vectorbinary_op op(binop, *vct1, *vct2, *vct3);
		return tac::create(op, {op1, op2});
	}

private:
	std::unique_ptr<jive::operation> op_;
};

/* constant data vector operator */

class constant_data_vector_op final : public jive::simple_op {
public:
	~constant_data_vector_op() override;

private:
	constant_data_vector_op(const vectortype & vt)
	: simple_op(std::vector<jive::port>(vt.size(), vt.type()), {vt})
	{}

public:
	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	size_t
	size() const noexcept
	{
		return static_cast<const vectortype*>(&result(0).type())->size();
	}

	const jive::valuetype &
	type() const noexcept
	{
		return static_cast<const vectortype*>(&result(0).type())->type();
	}

	static std::unique_ptr<tac>
	Create(const std::vector<const variable*> & elements)
	{
        if (elements.empty())
            throw error("Expected at least one element.");

        auto vt = dynamic_cast<const jive::valuetype*>(&elements[0]->type());
        if (!vt) throw error("Expected value type.");

        constant_data_vector_op op(fixedvectortype(*vt, elements.size()));
        return tac::create(op, elements);
	}
};

/* ExtractValue operator */

class ExtractValue final : public jive::simple_op {
	typedef std::vector<unsigned>::const_iterator const_iterator;
public:
	virtual
	~ExtractValue();

	inline
	ExtractValue(
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

	const jive::valuetype &
	type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(&argument(0).type());
	}

	static inline std::unique_ptr<jlm::tac>
	create(
		const jlm::variable * aggregate,
		const std::vector<unsigned> & indices)
	{
		ExtractValue op(aggregate->type(), indices);
		return tac::create(op, {aggregate});
	}

private:
	static inline jive::port
	dsttype(
		const jive::type & aggtype,
		const std::vector<unsigned> & indices)
	{
		const jive::type * type = &aggtype;
		for (const auto & index : indices) {
			if (auto st = dynamic_cast<const StructType*>(type)) {
				if (index >= st->GetDeclaration().nelements())
					throw jlm::error("extractvalue index out of bound.");

				type = &st->GetDeclaration().element(index);
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
	~loopstatemux_op();

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

/* MemState operator */

class MemStateOperator : public jive::simple_op {
public:
	MemStateOperator(size_t noperands, size_t nresults)
	: simple_op(create_portvector(noperands), create_portvector(nresults))
	{}

private:
	static std::vector<jive::port>
	create_portvector(size_t size)
	{
		return {size, jive::port(MemoryStateType::Create())};
	}
};

/** \brief MemStateMerge operator
*/
class MemStateMergeOperator final : public MemStateOperator {
public:
	~MemStateMergeOperator() override;

	MemStateMergeOperator(size_t noperands)
	: MemStateOperator(noperands, 1)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static jive::output *
	Create(const std::vector<jive::output*> & operands)
	{
		if (operands.empty())
			throw error("Insufficient number of operands.");

		MemStateMergeOperator op(operands.size());
		auto region = operands.front()->region();
		return jive::simple_node::create_normalized(region, op, operands)[0];
	}

	static std::unique_ptr<tac>
	Create(const std::vector<const variable*> & operands)
	{
		if (operands.empty())
			throw error("Insufficient number of operands.");

		MemStateMergeOperator op(operands.size());
		return tac::create(op, operands);
	}
};

/** \brief MemStateSplit operator
*/
class MemStateSplitOperator final : public MemStateOperator {
public:
	~MemStateSplitOperator() override;

	MemStateSplitOperator(size_t nresults)
	: MemStateOperator(1, nresults)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static std::vector<jive::output*>
	Create(jive::output * operand, size_t nresults)
	{
		if (nresults == 0)
			throw error("Insufficient number of results.");

		MemStateSplitOperator op(nresults);
		return jive::simple_node::create_normalized(operand->region(), op, {operand});
	}
};

/* malloc operator */

class malloc_op final : public jive::simple_op {
public:
	virtual
	~malloc_op();

	malloc_op(const jive::bittype & btype)
	: simple_op({btype}, {PointerType(), {MemoryStateType::Create()}})
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const jive::bittype &
	size_type() const noexcept
	{
		return *static_cast<const jive::bittype*>(&argument(0).type());
	}

	FunctionType
	fcttype() const
	{
		JLM_ASSERT(narguments() == 1 && nresults() == 2);
		return FunctionType({&argument(0).type()}, {&result(0).type(), &result(1).type()});
	}

	static std::unique_ptr<jlm::tac>
	create(const variable * size)
	{
		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		jlm::malloc_op op(*bt);
		return tac::create(op, {size});
	}

	static std::vector<jive::output*>
	create(jive::output * size)
	{
		auto bt = dynamic_cast<const jive::bittype*>(&size->type());
		if (!bt) throw jlm::error("expected bits type.");

		jlm::malloc_op op(*bt);
		return jive::simple_node::create_normalized(size->region(), op, {size});
	}
};

/* free operator */

class free_op final : public jive::simple_op {
public:
	virtual
	~free_op();

	free_op(size_t nmemstates)
	: simple_op(create_operand_portvector(nmemstates), create_result_portvector(nmemstates))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	const FunctionType
	fcttype() const
	{
		JLM_ASSERT(narguments() == 3 && nresults() == 2);
		return FunctionType({&argument(0).type()}, {&result(0).type(), &result(1).type()});
	}

	static std::unique_ptr<jlm::tac>
	create(
		const variable * pointer,
		const std::vector<const variable*> & memstates,
		const variable * iostate)
	{
		if (memstates.empty())
			throw jlm::error("Number of memory states cannot be zero.");

		std::vector<const variable*> operands;
		operands.push_back(pointer);
		operands.insert(operands.end(), memstates.begin(), memstates.end());
		operands.push_back(iostate);

		free_op op(memstates.size());
		return tac::create(op, operands);
	}

	static std::vector<jive::output*>
	create(
		jive::output * pointer,
		const std::vector<jive::output*> & memstates,
		jive::output * iostate)
	{
		if (memstates.empty())
			throw jlm::error("Number of memory states cannot be zero.");

		std::vector<jive::output*> operands;
		operands.push_back(pointer);
		operands.insert(operands.end(), memstates.begin(), memstates.end());
		operands.push_back(iostate);

		free_op op(memstates.size());
		return jive::simple_node::create_normalized(pointer->region(), op, operands);
	}

private:
	static std::vector<jive::port>
	create_operand_portvector(size_t nmemstates)
	{
		std::vector<jive::port> memstates(nmemstates, {MemoryStateType::Create()});

		std::vector<jive::port> ports({PointerType()});
		ports.insert(ports.end(), memstates.begin(), memstates.end());
		ports.push_back(iostatetype::instance());

		return ports;
	}

	static std::vector<jive::port>
	create_result_portvector(size_t nmemstates)
	{
		std::vector<jive::port> ports(nmemstates, {MemoryStateType::Create()});
		ports.push_back(iostatetype::instance());

		return ports;
	}
};

/* memcpy operation */

class Memcpy final : public jive::simple_op {
public:
	virtual
	~Memcpy();

	Memcpy(
		const std::vector<jive::port> & operandPorts,
		const std::vector<jive::port> & resultPorts)
	: simple_op(operandPorts, resultPorts)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static std::unique_ptr<jlm::tac>
	create(
		const variable * destination,
		const variable * source,
		const variable * length,
		const variable * isVolatile,
		const std::vector<const variable*> & memoryStates)
	{
		auto operandPorts = CheckAndCreateOperandPorts(length->type(), memoryStates.size());
		auto resultPorts = CreateResultPorts(memoryStates.size());

		std::vector<const variable*> operands = {destination, source, length, isVolatile};
		operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

		Memcpy op(operandPorts, resultPorts);
		return tac::create(op, operands);
	}

	static std::vector<jive::output*>
	create(
		jive::output * destination,
		jive::output * source,
		jive::output * length,
		jive::output * isVolatile,
		const std::vector<jive::output*> & memoryStates)
	{
		auto operandPorts = CheckAndCreateOperandPorts(length->type(), memoryStates.size());
		auto resultPorts = CreateResultPorts(memoryStates.size());

		std::vector<jive::output*> operands = {destination, source, length, isVolatile};
		operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

		Memcpy op(operandPorts, resultPorts);
		return jive::simple_node::create_normalized(destination->region(), op, operands);
	}

private:
	static std::vector<jive::port>
	CheckAndCreateOperandPorts(
		const jive::type & length,
		size_t nMemoryStates)
	{
		if (length != jive::bit32
		&& length != jive::bit64)
			throw jlm::error("Expected 32 bit or 64 bit integer type.");

		if (nMemoryStates == 0)
			throw jlm::error("Number of memory states cannot be zero.");

    PointerType pointerType;
		std::vector<jive::port> ports = {pointerType, pointerType, length, jive::bit1};
		ports.insert(ports.end(), nMemoryStates, {MemoryStateType::Create()});

		return ports;
	}

	static std::vector<jive::port>
	CreateResultPorts(size_t nMemoryStates)
	{
		return std::vector<jive::port>(nMemoryStates, {MemoryStateType::Create()});
	}
};

/*
	FIXME: This function should be in jive and not in jlm.
*/
static inline jive::node *
input_node(const jive::input * input)
{
	auto ni = dynamic_cast<const jive::node_input*>(input);
	return ni != nullptr ? ni->node() : nullptr;
}

}

#endif
