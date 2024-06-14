/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_OPERATORS_HPP
#define JLM_LLVM_IR_OPERATORS_OPERATORS_HPP

#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

#include <llvm/ADT/APFloat.h>

namespace jlm::llvm
{

class cfg_node;

/* phi operator */

class phi_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~phi_op() noexcept;

  inline phi_op(const std::vector<llvm::cfg_node *> & nodes, const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ nodes.size(), type.copy() }, { type.copy() }),
        nodes_(nodes)
  {}

  inline phi_op(
      const std::vector<llvm::cfg_node *> & nodes,
      std::shared_ptr<const jlm::rvsdg::type> type)
      : jlm::rvsdg::simple_op(
          std::vector<std::shared_ptr<const jlm::rvsdg::type>>(nodes.size(), type),
          { type }),
        nodes_(nodes)
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

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  inline const jlm::rvsdg::type &
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

  static std::unique_ptr<llvm::tac>
  create(
      const std::vector<std::pair<const variable *, cfg_node *>> & arguments,
      const jlm::rvsdg::type & type)
  {
    std::vector<cfg_node *> nodes;
    std::vector<const variable *> operands;
    for (const auto & argument : arguments)
    {
      nodes.push_back(argument.second);
      operands.push_back(argument.first);
    }

    phi_op phi(nodes, type);
    return tac::create(phi, operands);
  }

  static std::unique_ptr<llvm::tac>
  create(
      const std::vector<std::pair<const variable *, cfg_node *>> & arguments,
      std::shared_ptr<const jlm::rvsdg::type> type)
  {
    std::vector<cfg_node *> nodes;
    std::vector<const variable *> operands;
    for (const auto & argument : arguments)
    {
      nodes.push_back(argument.second);
      operands.push_back(argument.first);
    }

    phi_op phi(nodes, std::move(type));
    return tac::create(phi, operands);
  }

private:
  std::vector<cfg_node *> nodes_;
};

/* assignment operator */

class assignment_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~assignment_op() noexcept;

  inline assignment_op(const jlm::rvsdg::type & type)
      : simple_op({ type.copy(), type.copy() }, {})
  {}

  assignment_op(const assignment_op &) = default;

  assignment_op(assignment_op &&) = default;

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * rhs, const variable * lhs)
  {
    if (rhs->type() != lhs->type())
      throw jlm::util::error("LHS and RHS of assignment must have same type.");

    return tac::create(assignment_op(rhs->type()), { lhs, rhs });
  }
};

/* select operator */

class select_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~select_op() noexcept;

  select_op(const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op(
          { jlm::rvsdg::bittype::Create(1), type.copy(), type.copy() },
          { type.copy() })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const jlm::rvsdg::type &
  type() const noexcept
  {
    return result(0).type();
  }

  static std::unique_ptr<llvm::tac>
  create(const llvm::variable * p, const llvm::variable * t, const llvm::variable * f)
  {
    select_op op(t->type());
    return tac::create(op, { p, t, f });
  }
};

/* vector select operator */

class vectorselect_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~vectorselect_op() noexcept;

private:
  vectorselect_op(const vectortype & pt, const vectortype & vt)
      : jlm::rvsdg::simple_op({ pt.copy(), vt.copy(), vt.copy() }, { vt.copy() })
  {}

public:
  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const jlm::rvsdg::type &
  type() const noexcept
  {
    return result(0).type();
  }

  size_t
  size() const noexcept
  {
    return dynamic_cast<const vectortype *>(&type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * p, const variable * t, const variable * f)
  {
    if (is<fixedvectortype>(p->type()) && is<fixedvectortype>(t->type()))
      return createVectorSelectTac<fixedvectortype>(p, t, f);

    if (is<scalablevectortype>(p->type()) && is<scalablevectortype>(t->type()))
      return createVectorSelectTac<scalablevectortype>(p, t, f);

    throw jlm::util::error("Expected vector types as operands.");
  }

private:
  template<typename T>
  static std::unique_ptr<tac>
  createVectorSelectTac(const variable * p, const variable * t, const variable * f)
  {
    auto fvt = static_cast<const T *>(&t->type());
    T pt(jlm::rvsdg::bittype::Create(1), fvt->size());
    T vt(fvt->Type(), fvt->size());
    vectorselect_op op(pt, vt);
    return tac::create(op, { p, t, f });
  }
};

/* fp2ui operator */

class fp2ui_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~fp2ui_op() noexcept;

  inline fp2ui_op(const fpsize & size, const jlm::rvsdg::bittype & type)
      : jlm::rvsdg::unary_op(fptype::Create(size), type.copy())
  {}

  inline fp2ui_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const fptype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(dsttype.get());
    if (!dt)
      throw jlm::util::error("expected bitstring type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const fptype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(&type);
    if (!dt)
      throw jlm::util::error("expected bitstring type.");

    fp2ui_op op(st->size(), *dt);
    return tac::create(op, { operand });
  }
};

/* fp2si operator */

class fp2si_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~fp2si_op() noexcept;

  inline fp2si_op(const fpsize & size, const jlm::rvsdg::bittype & type)
      : jlm::rvsdg::unary_op(fptype::Create(size), type.copy())
  {}

  inline fp2si_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : jlm::rvsdg::unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const fptype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(dsttype.get());
    if (!dt)
      throw jlm::util::error("expected bitstring type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const fptype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(&type);
    if (!dt)
      throw jlm::util::error("expected bitstring type.");

    fp2si_op op(st->size(), *dt);
    return tac::create(op, { operand });
  }
};

/* ctl2bits operator */

class ctl2bits_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ctl2bits_op() noexcept;

  inline ctl2bits_op(const jlm::rvsdg::ctltype & srctype, const jlm::rvsdg::bittype & dsttype)
      : jlm::rvsdg::simple_op({ srctype.copy() }, { dsttype.copy() })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const jlm::rvsdg::ctltype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected control type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(&type);
    if (!dt)
      throw jlm::util::error("expected bitstring type.");

    ctl2bits_op op(*st, *dt);
    return tac::create(op, { operand });
  }
};

/* branch operator */

class branch_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~branch_op() noexcept;

  inline branch_op(const jlm::rvsdg::ctltype & type)
      : jlm::rvsdg::simple_op({ type.copy() }, {})
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  inline size_t
  nalternatives() const noexcept
  {
    return static_cast<const jlm::rvsdg::ctltype *>(&argument(0).type())->nalternatives();
  }

  static std::unique_ptr<llvm::tac>
  create(size_t nalternatives, const variable * operand)
  {
    jlm::rvsdg::ctltype type(nalternatives);
    branch_op op(type);
    return tac::create(op, { operand });
  }
};

/** \brief ConstantPointerNullOperation class
 *
 * This operator is the Jlm equivalent of LLVM's ConstantPointerNull constant.
 */
class ConstantPointerNullOperation final : public jlm::rvsdg::simple_op
{
public:
  ~ConstantPointerNullOperation() noexcept override;

  explicit ConstantPointerNullOperation(const PointerType & pointerType)
      : simple_op({}, { pointerType.copy() })
  {}

  explicit ConstantPointerNullOperation(std::shared_ptr<const PointerType> pointerType)
      : simple_op({}, { std::move(pointerType) })
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  [[nodiscard]] const PointerType &
  GetPointerType() const noexcept
  {
    return *jlm::util::AssertedCast<const PointerType>(&result(0).type());
  }

  static std::unique_ptr<llvm::tac>
  Create(const jlm::rvsdg::type & type)
  {
    auto & pointerType = CheckAndExtractType(type.copy());

    ConstantPointerNullOperation operation(pointerType);
    return tac::create(operation, {});
  }

  static std::unique_ptr<llvm::tac>
  Create(std::shared_ptr<const rvsdg::type> type)
  {
    ConstantPointerNullOperation operation(CheckAndExtractType(type));
    return tac::create(operation, {});
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region * region, const jlm::rvsdg::type & type)
  {
    auto & pointerType = CheckAndExtractType(type.copy());

    ConstantPointerNullOperation operation(pointerType);
    return jlm::rvsdg::simple_node::create_normalized(region, operation, {})[0];
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region * region, std::shared_ptr<const rvsdg::type> type)
  {
    ConstantPointerNullOperation operation(CheckAndExtractType(type));
    return jlm::rvsdg::simple_node::create_normalized(region, operation, {})[0];
  }

private:
  static const std::shared_ptr<const PointerType>
  CheckAndExtractType(std::shared_ptr<const jlm::rvsdg::type> type)
  {
    if (auto pointerType = std::dynamic_pointer_cast<const PointerType>(type))
      return pointerType;

    throw jlm::util::error("expected pointer type.");
  }
};

/* bits2ptr operator */

class bits2ptr_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~bits2ptr_op();

  inline bits2ptr_op(const jlm::rvsdg::bittype & btype, const PointerType & ptype)
      : unary_op(btype.copy(), ptype.copy())
  {}

  inline bits2ptr_op(
      std::shared_ptr<const jlm::rvsdg::bittype> btype,
      std::shared_ptr<const PointerType> ptype)
      : unary_op(std::move(btype), std::move(ptype))
  {}

  inline bits2ptr_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto at = dynamic_cast<const jlm::rvsdg::bittype *>(srctype.get());
    if (!at)
      throw jlm::util::error("expected bitstring type.");

    auto pt = dynamic_cast<const PointerType *>(dsttype.get());
    if (!pt)
      throw jlm::util::error("expected pointer type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  inline size_t
  nbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&argument(0).type())->nbits();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * argument, const jlm::rvsdg::type & type)
  {
    auto at = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(argument->Type());
    if (!at)
      throw jlm::util::error("expected bitstring type.");

    auto pt = std::dynamic_pointer_cast<const PointerType>(type.copy());
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    bits2ptr_op op(at, pt);
    return tac::create(op, { argument });
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * argument, std::shared_ptr<const jlm::rvsdg::type> type)
  {
    auto at = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(argument->Type());
    if (!at)
      throw jlm::util::error("expected bitstring type.");

    auto pt = std::dynamic_pointer_cast<const PointerType>(type);
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    bits2ptr_op op(at, pt);
    return tac::create(op, { argument });
  }

  static jlm::rvsdg::output *
  create(jlm::rvsdg::output * operand, const jlm::rvsdg::type & type)
  {
    auto ot = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(operand->Type());
    if (!ot)
      throw jlm::util::error("expected bitstring type.");

    auto pt = std::dynamic_pointer_cast<const PointerType>(type.copy());
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    bits2ptr_op op(ot, pt);
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }

  static jlm::rvsdg::output *
  create(jlm::rvsdg::output * operand, std::shared_ptr<const jlm::rvsdg::type> type)
  {
    auto ot = std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(operand->Type());
    if (!ot)
      throw jlm::util::error("expected bitstring type.");

    auto pt = std::dynamic_pointer_cast<const PointerType>(type);
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    bits2ptr_op op(ot, pt);
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }
};

/* ptr2bits operator */

class ptr2bits_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~ptr2bits_op();

  inline ptr2bits_op(const PointerType & ptype, const jlm::rvsdg::bittype & btype)
      : unary_op(ptype.copy(), btype.copy())
  {}

  inline ptr2bits_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto pt = dynamic_cast<const PointerType *>(srctype.get());
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(dsttype.get());
    if (!bt)
      throw jlm::util::error("expected bitstring type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  inline size_t
  nbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&result(0).type())->nbits();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * argument, const jlm::rvsdg::type & type)
  {
    auto pt = dynamic_cast<const PointerType *>(&argument->type());
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&type);
    if (!bt)
      throw jlm::util::error("expected bitstring type.");

    ptr2bits_op op(*pt, *bt);
    return tac::create(op, { argument });
  }
};

/* Constant Data Array operator */

class ConstantDataArray final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ConstantDataArray();

  ConstantDataArray(const jlm::rvsdg::valuetype & type, size_t size)
      : simple_op(
          { size, type.copy() },
          { arraytype::Create(
              std::static_pointer_cast<const rvsdg::valuetype>(type.copy()),
              size) })
  {
    if (size == 0)
      throw jlm::util::error("size equals zero.");
  }

  virtual bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return static_cast<const arraytype *>(&result(0).type())->nelements();
  }

  const jlm::rvsdg::valuetype &
  type() const noexcept
  {
    return static_cast<const arraytype *>(&result(0).type())->element_type();
  }

  static std::unique_ptr<llvm::tac>
  create(const std::vector<const variable *> & elements)
  {
    if (elements.size() == 0)
      throw jlm::util::error("expected at least one element.");

    auto vt = dynamic_cast<const jlm::rvsdg::valuetype *>(&elements[0]->type());
    if (!vt)
      throw jlm::util::error("expected value type.");

    ConstantDataArray op(*vt, elements.size());
    return tac::create(op, elements);
  }

  static jlm::rvsdg::output *
  Create(const std::vector<jlm::rvsdg::output *> & elements)
  {
    if (elements.empty())
      throw jlm::util::error("Expected at least one element.");

    auto valueType = dynamic_cast<const jlm::rvsdg::valuetype *>(&elements[0]->type());
    if (!valueType)
    {
      throw jlm::util::error("Expected value type.");
    }

    ConstantDataArray operation(*valueType, elements.size());
    return jlm::rvsdg::simple_node::create_normalized(
        elements[0]->region(),
        operation,
        elements)[0];
  }
};

/* pointer compare operator */

enum class cmp
{
  eq,
  ne,
  gt,
  ge,
  lt,
  le
};

class ptrcmp_op final : public jlm::rvsdg::binary_op
{
public:
  virtual ~ptrcmp_op();

  inline ptrcmp_op(const PointerType & ptype, const llvm::cmp & cmp)
      : binary_op({ ptype.copy(), ptype.copy() }, jlm::rvsdg::bittype::Create(1)),
        cmp_(cmp)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  virtual jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * op1, const jlm::rvsdg::output * op2)
      const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::output * op1,
      jlm::rvsdg::output * op2) const override;

  inline llvm::cmp
  cmp() const noexcept
  {
    return cmp_;
  }

  static std::unique_ptr<llvm::tac>
  create(const llvm::cmp & cmp, const variable * op1, const variable * op2)
  {
    auto pt = dynamic_cast<const PointerType *>(&op1->type());
    if (!pt)
      throw jlm::util::error("expected pointer type.");

    ptrcmp_op op(*pt, cmp);
    return tac::create(op, { op1, op2 });
  }

private:
  llvm::cmp cmp_;
};

/* zext operator */

class zext_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~zext_op();

  inline zext_op(size_t nsrcbits, size_t ndstbits)
      : unary_op(jlm::rvsdg::bittype::Create(nsrcbits), jlm::rvsdg::bittype::Create(ndstbits))
  {
    if (ndstbits < nsrcbits)
      throw jlm::util::error("# destination bits must be greater than # source bits.");
  }

  inline zext_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected bitstring type.");

    auto dt = dynamic_cast<const jlm::rvsdg::bittype *>(dsttype.get());
    if (!dt)
      throw jlm::util::error("expected bitstring type.");

    if (dt->nbits() < st->nbits())
      throw jlm::util::error("# destination bits must be greater than # source bits.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  virtual jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * operand) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * operand)
      const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&argument(0).type())->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&result(0).type())->nbits();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto operandBitType = CheckAndExtractBitType(operand->type());
    auto resultBitType = CheckAndExtractBitType(type);

    zext_op operation(operandBitType.nbits(), resultBitType.nbits());
    return tac::create(operation, { operand });
  }

  static rvsdg::output &
  Create(rvsdg::output & operand, const rvsdg::type & resultType)
  {
    auto operandBitType = CheckAndExtractBitType(operand.type());
    auto resultBitType = CheckAndExtractBitType(resultType);

    zext_op operation(operandBitType.nbits(), resultBitType.nbits());
    return *rvsdg::simple_node::create_normalized(operand.region(), operation, { &operand })[0];
  }

private:
  static const rvsdg::bittype &
  CheckAndExtractBitType(const rvsdg::type & type)
  {
    if (auto bitType = dynamic_cast<const rvsdg::bittype *>(&type))
    {
      return *bitType;
    }

    throw util::type_error("bittype", type.debug_string());
  }
};

/* floating point constant operator */

class ConstantFP final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ConstantFP();

  inline ConstantFP(const fpsize & size, const ::llvm::APFloat & constant)
      : simple_op({}, { fptype::Create(size) }),
        constant_(constant)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  inline const ::llvm::APFloat &
  constant() const noexcept
  {
    return constant_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return static_cast<const fptype *>(&result(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const ::llvm::APFloat & constant, const jlm::rvsdg::type & type)
  {
    auto ft = dynamic_cast<const fptype *>(&type);
    if (!ft)
      throw jlm::util::error("expected floating point type.");

    ConstantFP op(ft->size(), constant);
    return tac::create(op, {});
  }

private:
  /* FIXME: I would not like to use the APFloat here,
     but I don't have a replacement right now. */
  ::llvm::APFloat constant_;
};

/* floating point comparison operator */

enum class fpcmp
{
  TRUE,
  FALSE,
  oeq,
  ogt,
  oge,
  olt,
  ole,
  one,
  ord,
  ueq,
  ugt,
  uge,
  ult,
  ule,
  une,
  uno
};

class fpcmp_op final : public jlm::rvsdg::binary_op
{
public:
  virtual ~fpcmp_op();

  inline fpcmp_op(const fpcmp & cmp, const fpsize & size)
      : binary_op({ fptype::Create(size), fptype::Create(size) }, jlm::rvsdg::bittype::Create(1)),
        cmp_(cmp)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * op1, const jlm::rvsdg::output * op2)
      const noexcept override;

  jlm::rvsdg::output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::output * op1,
      jlm::rvsdg::output * op2) const override;

  inline const fpcmp &
  cmp() const noexcept
  {
    return cmp_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return static_cast<const llvm::fptype *>(&argument(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const fpcmp & cmp, const variable * op1, const variable * op2)
  {
    auto ft = dynamic_cast<const fptype *>(&op1->type());
    if (!ft)
      throw jlm::util::error("expected floating point type.");

    fpcmp_op op(cmp, ft->size());
    return tac::create(op, { op1, op2 });
  }

private:
  fpcmp cmp_;
};

/** \brief UndefValueOperation class
 *
 * This operator is the Jlm equivalent of LLVM's UndefValue constant.
 */
class UndefValueOperation final : public jlm::rvsdg::simple_op
{
public:
  ~UndefValueOperation() noexcept override;

  explicit UndefValueOperation(const jlm::rvsdg::type & type)
      : simple_op({}, { type.copy() })
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

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  [[nodiscard]] const jlm::rvsdg::type &
  GetType() const noexcept
  {
    return result(0).type();
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region & region, const jlm::rvsdg::type & type)
  {
    UndefValueOperation operation(type);
    return jlm::rvsdg::simple_node::create_normalized(&region, operation, {})[0];
  }

  static std::unique_ptr<llvm::tac>
  Create(const jlm::rvsdg::type & type)
  {
    UndefValueOperation operation(type);
    return tac::create(operation, {});
  }

  static std::unique_ptr<llvm::tac>
  Create(const jlm::rvsdg::type & type, const std::string & name)
  {
    UndefValueOperation operation(type);
    return tac::create(operation, {}, { name });
  }

  static std::unique_ptr<llvm::tac>
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
class PoisonValueOperation final : public jlm::rvsdg::simple_op
{
public:
  ~PoisonValueOperation() noexcept override;

  explicit PoisonValueOperation(const jlm::rvsdg::valuetype & type)
      : jlm::rvsdg::simple_op({}, { type.copy() })
  {}

  PoisonValueOperation(const PoisonValueOperation &) = default;

  PoisonValueOperation(PoisonValueOperation &&) = delete;

  PoisonValueOperation &
  operator=(const PoisonValueOperation &) = delete;

  PoisonValueOperation &
  operator=(PoisonValueOperation &&) = delete;

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const jlm::rvsdg::valuetype &
  GetType() const noexcept
  {
    auto & type = result(0).type();
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::valuetype *>(&type));
    return *static_cast<const jlm::rvsdg::valuetype *>(&type);
  }

  static std::unique_ptr<llvm::tac>
  Create(const jlm::rvsdg::type & type)
  {
    auto & valueType = CheckAndConvertType(type);

    PoisonValueOperation operation(valueType);
    return tac::create(operation, {});
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region * region, const jlm::rvsdg::type & type)
  {
    auto & valueType = CheckAndConvertType(type);

    PoisonValueOperation operation(valueType);
    return jlm::rvsdg::simple_node::create_normalized(region, operation, {})[0];
  }

private:
  static const jlm::rvsdg::valuetype &
  CheckAndConvertType(const jlm::rvsdg::type & type)
  {
    if (auto valueType = dynamic_cast<const jlm::rvsdg::valuetype *>(&type))
      return *valueType;

    throw jlm::util::error("Expected value type.");
  }
};

/* floating point arithmetic operator */

enum class fpop
{
  add,
  sub,
  mul,
  div,
  mod
};

class fpbin_op final : public jlm::rvsdg::binary_op
{
public:
  virtual ~fpbin_op();

  inline fpbin_op(const llvm::fpop & op, const fpsize & size)
      : binary_op({ fptype::Create(size), fptype::Create(size) }, fptype::Create(size)),
        op_(op)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * op1, const jlm::rvsdg::output * op2)
      const noexcept override;

  jlm::rvsdg::output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::output * op1,
      jlm::rvsdg::output * op2) const override;

  inline const llvm::fpop &
  fpop() const noexcept
  {
    return op_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return static_cast<const fptype *>(&result(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const llvm::fpop & fpop, const variable * op1, const variable * op2)
  {
    auto ft = dynamic_cast<const fptype *>(&op1->type());
    if (!ft)
      throw jlm::util::error("expected floating point type.");

    fpbin_op op(fpop, ft->size());
    return tac::create(op, { op1, op2 });
  }

private:
  llvm::fpop op_;
};

/* fpext operator */

class fpext_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~fpext_op();

  inline fpext_op(const fpsize & srcsize, const fpsize & dstsize)
      : unary_op(fptype::Create(srcsize), fptype::Create(dstsize))
  {
    if (srcsize == fpsize::flt && dstsize == fpsize::half)
      throw jlm::util::error("destination type size must be bigger than source type size.");
  }

  inline fpext_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const fptype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const fptype *>(dsttype.get());
    if (!dt)
      throw jlm::util::error("expected floating point type.");

    if (st->size() == fpsize::flt && dt->size() == fpsize::half)
      throw jlm::util::error("destination type size must be bigger than source type size.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  inline const fpsize &
  srcsize() const noexcept
  {
    return static_cast<const fptype *>(&argument(0).type())->size();
  }

  inline const fpsize &
  dstsize() const noexcept
  {
    return static_cast<const fptype *>(&result(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const fptype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const fptype *>(&type);
    if (!dt)
      throw jlm::util::error("expected floating point type.");

    fpext_op op(st->size(), dt->size());
    return tac::create(op, { operand });
  }
};

/* fpneg operator */

class fpneg_op final : public jlm::rvsdg::unary_op
{
public:
  ~fpneg_op() override;

  fpneg_op(const fpsize & size)
      : unary_op(fptype::Create(size), fptype::Create(size))
  {}

  fpneg_op(const jlm::rvsdg::type & type)
      : unary_op(type.copy(), type.copy())
  {
    auto st = dynamic_cast<const fptype *>(&type);
    if (!st)
      throw jlm::util::error("expected floating point type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  const fpsize &
  size() const noexcept
  {
    return static_cast<const fptype *>(&argument(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand)
  {
    auto type = dynamic_cast<const fptype *>(&operand->type());
    if (!type)
      throw jlm::util::error("expected floating point type.");

    fpneg_op op(type->size());
    return tac::create(op, { operand });
  }
};

/* fptrunc operator */

class fptrunc_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~fptrunc_op();

  inline fptrunc_op(const fpsize & srcsize, const fpsize & dstsize)
      : unary_op(fptype::Create(srcsize), fptype::Create(dstsize))
  {
    if (srcsize == fpsize::half || (srcsize == fpsize::flt && dstsize != fpsize::half)
        || (srcsize == fpsize::dbl && dstsize == fpsize::dbl))
      throw jlm::util::error("destination tpye size must be smaller than source size type.");
  }

  inline fptrunc_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const fptype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const fptype *>(dsttype.get());
    if (!dt)
      throw jlm::util::error("expected floating point type.");

    if (st->size() == fpsize::half || (st->size() == fpsize::flt && dt->size() != fpsize::half)
        || (st->size() == fpsize::dbl && dt->size() == fpsize::dbl))
      throw jlm::util::error("destination tpye size must be smaller than source size type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  inline const fpsize &
  srcsize() const noexcept
  {
    return static_cast<const fptype *>(&argument(0).type())->size();
  }

  inline const fpsize &
  dstsize() const noexcept
  {
    return static_cast<const fptype *>(&result(0).type())->size();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const fptype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected floating point type.");

    auto dt = dynamic_cast<const fptype *>(&type);
    if (!dt)
      throw jlm::util::error("expected floating point type.");

    fptrunc_op op(st->size(), dt->size());
    return tac::create(op, { operand });
  }
};

/* valist operator */

class valist_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~valist_op();

  explicit valist_op(std::vector<std::shared_ptr<const jlm::rvsdg::type>> types)
      : simple_op(std::move(types), { varargtype::Create() })
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

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  create(const std::vector<const variable *> & arguments)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::type>> operands;
    for (const auto & argument : arguments)
      operands.push_back(argument->Type());

    valist_op op(std::move(operands));
    return tac::create(op, arguments);
  }

  static rvsdg::output *
  Create(rvsdg::region & region, const std::vector<rvsdg::output *> & operands)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> operandTypes;
    operandTypes.reserve(operands.size());
    for (auto & operand : operands)
      operandTypes.emplace_back(operand->Type());

    valist_op operation(std::move(operandTypes));
    return jlm::rvsdg::simple_node::create_normalized(&region, operation, operands)[0];
  }
};

/* bitcast operator */

class bitcast_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~bitcast_op();

  inline bitcast_op(const jlm::rvsdg::valuetype & srctype, const jlm::rvsdg::valuetype & dsttype)
      : unary_op(srctype.copy(), dsttype.copy())
  {}

  inline bitcast_op(
      std::shared_ptr<const jlm::rvsdg::valuetype> srctype,
      std::shared_ptr<const jlm::rvsdg::valuetype> dsttype)
      : unary_op(std::move(srctype), std::move(dsttype))
  {}

  inline bitcast_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    check_types(srctype, dsttype);
  }

  bitcast_op(const bitcast_op &) = default;

  bitcast_op(jlm::rvsdg::operation &&) = delete;

  bitcast_op &
  operator=(const jlm::rvsdg::operation &) = delete;

  bitcast_op &
  operator=(jlm::rvsdg::operation &&) = delete;

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto pair = check_types(operand->Type(), type.copy());

    bitcast_op op(pair.first, pair.second);
    return tac::create(op, { operand });
  }

  static jlm::rvsdg::output *
  create(jlm::rvsdg::output * operand, const jlm::rvsdg::type & rtype)
  {
    auto pair = check_types(operand->Type(), rtype.copy());

    bitcast_op op(pair.first, pair.second);
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, std::shared_ptr<const jlm::rvsdg::type> type)
  {
    auto pair = check_types(operand->Type(), type);

    bitcast_op op(pair.first, pair.second);
    return tac::create(op, { operand });
  }

  static jlm::rvsdg::output *
  create(jlm::rvsdg::output * operand, std::shared_ptr<const jlm::rvsdg::type> rtype)
  {
    auto pair = check_types(operand->Type(), rtype);

    bitcast_op op(pair.first, pair.second);
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }

private:
  static std::pair<
      std::shared_ptr<const jlm::rvsdg::valuetype>,
      std::shared_ptr<const jlm::rvsdg::valuetype>>
  check_types(
      const std::shared_ptr<const jlm::rvsdg::type> & otype,
      const std::shared_ptr<const jlm::rvsdg::type> & rtype)
  {
    auto ot = std::dynamic_pointer_cast<const jlm::rvsdg::valuetype>(otype);
    if (!ot)
      throw jlm::util::error("expected value type.");

    auto rt = std::dynamic_pointer_cast<const jlm::rvsdg::valuetype>(rtype);
    if (!rt)
      throw jlm::util::error("expected value type.");

    return std::make_pair(ot, rt);
  }
};

/* ConstantStruct operator */

class ConstantStruct final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ConstantStruct();

  inline ConstantStruct(std::shared_ptr<const StructType> type)
      : simple_op(create_srctypes(*type), { type })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const StructType &
  type() const noexcept
  {
    return *static_cast<const StructType *>(&result(0).type());
  }

  static std::unique_ptr<llvm::tac>
  create(const std::vector<const variable *> & elements, const jlm::rvsdg::type & type)
  {
    auto structType = CheckAndExtractStructType(type.copy());

    ConstantStruct op(std::move(structType));
    return tac::create(op, elements);
  }

  static rvsdg::output &
  Create(
      rvsdg::region & region,
      const std::vector<rvsdg::output *> & operands,
      const rvsdg::type & resultType)
  {
    auto structType = CheckAndExtractStructType(resultType.copy());

    ConstantStruct operation(std::move(structType));
    return *rvsdg::simple_node::create_normalized(&region, operation, operands)[0];
  }

  static rvsdg::output &
  Create(
      rvsdg::region & region,
      const std::vector<rvsdg::output *> & operands,
      std::shared_ptr<const rvsdg::type> resultType)
  {
    auto structType = CheckAndExtractStructType(std::move(resultType));

    ConstantStruct operation(std::move(structType));
    return *rvsdg::simple_node::create_normalized(&region, operation, operands)[0];
  }

private:
  static inline std::vector<std::shared_ptr<const rvsdg::type>>
  create_srctypes(const StructType & type)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types;
    for (size_t n = 0; n < type.GetDeclaration().NumElements(); n++)
      types.push_back(type.GetDeclaration().GetElement(n).copy());

    return types;
  }

  static std::shared_ptr<const StructType>
  CheckAndExtractStructType(std::shared_ptr<const rvsdg::type> type)
  {
    if (auto structType = std::dynamic_pointer_cast<const StructType>(type))
    {
      return structType;
    }

    throw util::type_error("StructType", type->debug_string());
  }
};

/* trunc operator */

class trunc_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~trunc_op();

  inline trunc_op(const jlm::rvsdg::bittype & otype, const jlm::rvsdg::bittype & rtype)
      : unary_op(otype.copy(), rtype.copy())
  {
    if (otype.nbits() < rtype.nbits())
      throw jlm::util::error("expected operand's #bits to be larger than results' #bits.");
  }

  inline trunc_op(
      std::shared_ptr<const jlm::rvsdg::type> optype,
      std::shared_ptr<const jlm::rvsdg::type> restype)
      : unary_op(optype, restype)
  {
    auto ot = dynamic_cast<const jlm::rvsdg::bittype *>(optype.get());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const jlm::rvsdg::bittype *>(restype.get());
    if (!rt)
      throw jlm::util::error("expected bits type.");

    if (ot->nbits() < rt->nbits())
      throw jlm::util::error("expected operand's #bits to be larger than results' #bits.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  virtual jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * operand) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * operand)
      const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&argument(0).type())->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return static_cast<const jlm::rvsdg::bittype *>(&result(0).type())->nbits();
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto ot = dynamic_cast<const jlm::rvsdg::bittype *>(&operand->type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const jlm::rvsdg::bittype *>(&type);
    if (!rt)
      throw jlm::util::error("expected bits type.");

    trunc_op op(*ot, *rt);
    return tac::create(op, { operand });
  }

  static jlm::rvsdg::output *
  create(size_t ndstbits, jlm::rvsdg::output * operand)
  {
    auto ot = dynamic_cast<const jlm::rvsdg::bittype *>(&operand->type());
    if (!ot)
      throw jlm::util::error("expected bits type.");

    trunc_op op(*ot, jlm::rvsdg::bittype(ndstbits));
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }
};

/* uitofp operator */

class uitofp_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~uitofp_op();

  inline uitofp_op(const jlm::rvsdg::bittype & srctype, const fptype & dsttype)
      : unary_op(srctype.copy(), dsttype.copy())
  {}

  inline uitofp_op(
      std::shared_ptr<const jlm::rvsdg::type> optype,
      std::shared_ptr<const jlm::rvsdg::type> restype)
      : unary_op(optype, restype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(optype.get());
    if (!st)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const fptype *>(restype.get());
    if (!rt)
      throw jlm::util::error("expected floating point type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  virtual jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * operand) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * operand)
      const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const fptype *>(&type);
    if (!rt)
      throw jlm::util::error("expected floating point type.");

    uitofp_op op(*st, *rt);
    return tac::create(op, { operand });
  }
};

/* sitofp operator */

class sitofp_op final : public jlm::rvsdg::unary_op
{
public:
  virtual ~sitofp_op();

  inline sitofp_op(const jlm::rvsdg::bittype & srctype, const fptype & dsttype)
      : unary_op(srctype.copy(), dsttype.copy())
  {}

  inline sitofp_op(
      std::shared_ptr<const jlm::rvsdg::type> srctype,
      std::shared_ptr<const jlm::rvsdg::type> dsttype)
      : unary_op(srctype, dsttype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(srctype.get());
    if (!st)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const fptype *>(dsttype.get());
    if (!rt)
      throw jlm::util::error("expected floating point type.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * output) const noexcept override;

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * output)
      const override;

  static std::unique_ptr<llvm::tac>
  create(const variable * operand, const jlm::rvsdg::type & type)
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(&operand->type());
    if (!st)
      throw jlm::util::error("expected bits type.");

    auto rt = dynamic_cast<const fptype *>(&type);
    if (!rt)
      throw jlm::util::error("expected floating point type.");

    sitofp_op op(*st, *rt);
    return tac::create(op, { operand });
  }
};

/* ConstantArray */

class ConstantArray final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ConstantArray();

  ConstantArray(const jlm::rvsdg::valuetype & type, size_t size)
      : jlm::rvsdg::simple_op(
          { size, type.copy() },
          { arraytype::Create(
              std::static_pointer_cast<const rvsdg::valuetype>(type.copy()),
              size) })
  {
    if (size == 0)
      throw jlm::util::error("size equals zero.\n");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return static_cast<const arraytype *>(&result(0).type())->nelements();
  }

  const jlm::rvsdg::valuetype &
  type() const noexcept
  {
    return static_cast<const arraytype *>(&result(0).type())->element_type();
  }

  static std::unique_ptr<llvm::tac>
  create(const std::vector<const variable *> & elements)
  {
    if (elements.size() == 0)
      throw jlm::util::error("expected at least one element.\n");

    auto vt = dynamic_cast<const jlm::rvsdg::valuetype *>(&elements[0]->type());
    if (!vt)
      throw jlm::util::error("expected value type.\n");

    ConstantArray op(*vt, elements.size());
    return tac::create(op, elements);
  }

  static rvsdg::output *
  Create(const std::vector<rvsdg::output *> & operands)
  {
    if (operands.empty())
      throw util::error("Expected at least one element.\n");

    auto valueType = dynamic_cast<const rvsdg::valuetype *>(&operands[0]->type());
    if (!valueType)
    {
      throw util::error("Expected value type.\n");
    }

    ConstantArray operation(*valueType, operands.size());
    return rvsdg::simple_node::create_normalized(operands[0]->region(), operation, operands)[0];
  }
};

/* ConstantAggregateZero operator */

class ConstantAggregateZero final : public jlm::rvsdg::simple_op
{
public:
  virtual ~ConstantAggregateZero();

  ConstantAggregateZero(std::shared_ptr<const jlm::rvsdg::type> type)
      : simple_op({}, { type })
  {
    auto st = dynamic_cast<const StructType *>(type.get());
    auto at = dynamic_cast<const arraytype *>(type.get());
    auto vt = dynamic_cast<const vectortype *>(type.get());
    if (!st && !at && !vt)
      throw jlm::util::error("expected array, struct, or vector type.\n");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  create(const jlm::rvsdg::type & type)
  {
    ConstantAggregateZero op(type.copy());
    return tac::create(op, {});
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region & region, const jlm::rvsdg::type & type)
  {
    ConstantAggregateZero operation(type.copy());
    return jlm::rvsdg::simple_node::create_normalized(&region, operation, {})[0];
  }

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region & region, std::shared_ptr<const jlm::rvsdg::type> type)
  {
    ConstantAggregateZero operation(std::move(type));
    return jlm::rvsdg::simple_node::create_normalized(&region, operation, {})[0];
  }
};

/* extractelement operator */

class extractelement_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~extractelement_op();

  inline extractelement_op(const vectortype & vtype, const jlm::rvsdg::bittype & btype)
      : simple_op({ vtype.copy(), btype.copy() }, { vtype.Type() })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static inline std::unique_ptr<llvm::tac>
  create(const llvm::variable * vector, const llvm::variable * index)
  {
    auto vt = dynamic_cast<const vectortype *>(&vector->type());
    if (!vt)
      throw jlm::util::error("expected vector type.");

    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&index->type());
    if (!bt)
      throw jlm::util::error("expected bit type.");

    extractelement_op op(*vt, *bt);
    return tac::create(op, { vector, index });
  }
};

/* shufflevector operator */

class shufflevector_op final : public jlm::rvsdg::simple_op
{
public:
  ~shufflevector_op() override;

  shufflevector_op(const fixedvectortype & v, const std::vector<int> & mask)
      : simple_op({ v.copy(), v.copy() }, { v.copy() }),
        Mask_(mask)
  {}

  shufflevector_op(const scalablevectortype & v, const std::vector<int> & mask)
      : simple_op({ v.copy(), v.copy() }, { v.copy() }),
        Mask_(mask)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const ::llvm::ArrayRef<int>
  Mask() const
  {
    return Mask_;
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * v1, const variable * v2, const std::vector<int> & mask)
  {
    if (is<fixedvectortype>(v1->type()) && is<fixedvectortype>(v2->type()))
      return CreateShuffleVectorTac<fixedvectortype>(v1, v2, mask);

    if (is<scalablevectortype>(v1->type()) && is<scalablevectortype>(v2->type()))
      return CreateShuffleVectorTac<scalablevectortype>(v1, v2, mask);

    throw jlm::util::error("Expected vector types as operands.");
  }

private:
  template<typename T>
  static std::unique_ptr<tac>
  CreateShuffleVectorTac(const variable * v1, const variable * v2, const std::vector<int> & mask)
  {
    auto vt = static_cast<const T *>(&v1->type());
    shufflevector_op op(*vt, mask);
    return tac::create(op, { v1, v2 });
  }

  std::vector<int> Mask_;
};

/* constantvector operator */

class constantvector_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~constantvector_op();

  inline constantvector_op(const vectortype & vt)
      : simple_op({ vt.size(), vt.Type() }, { vt.copy() })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static inline std::unique_ptr<llvm::tac>
  create(const std::vector<const variable *> & operands, const jlm::rvsdg::type & type)
  {
    auto vt = dynamic_cast<const vectortype *>(&type);
    if (!vt)
      throw jlm::util::error("expected vector type.");

    constantvector_op op(*vt);
    return tac::create(op, operands);
  }
};

/* insertelement operator */

class insertelement_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~insertelement_op();

  inline insertelement_op(
      const vectortype & vectype,
      const jlm::rvsdg::valuetype & vtype,
      const jlm::rvsdg::bittype & btype)
      : simple_op({ vectype.copy(), vtype.copy(), btype.copy() }, { vectype.copy() })
  {
    if (vectype.type() != vtype)
    {
      auto received = vtype.debug_string();
      auto expected = vectype.type().debug_string();
      throw jlm::util::error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static inline std::unique_ptr<llvm::tac>
  create(const llvm::variable * vector, const llvm::variable * value, const llvm::variable * index)
  {
    auto vct = dynamic_cast<const vectortype *>(&vector->type());
    if (!vct)
      throw jlm::util::error("expected vector type.");

    auto vt = dynamic_cast<const jlm::rvsdg::valuetype *>(&value->type());
    if (!vt)
      throw jlm::util::error("expected value type.");

    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&index->type());
    if (!bt)
      throw jlm::util::error("expected bit type.");

    insertelement_op op(*vct, *vt, *bt);
    return tac::create(op, { vector, value, index });
  }
};

/* vectorunary operator */

class vectorunary_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~vectorunary_op();

  inline vectorunary_op(
      const jlm::rvsdg::unary_op & op,
      const vectortype & operand,
      const vectortype & result)
      : simple_op({ operand.copy() }, { result.copy() }),
        op_(op.copy())
  {
    if (operand.type() != op.argument(0).type())
    {
      auto received = operand.type().debug_string();
      auto expected = op.argument(0).type().debug_string();
      throw jlm::util::error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }

    if (result.type() != op.result(0).type())
    {
      auto received = result.type().debug_string();
      auto expected = op.result(0).type().debug_string();
      throw jlm::util::error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  inline vectorunary_op(const vectorunary_op & other)
      : simple_op(other),
        op_(other.op_->copy())
  {}

  inline vectorunary_op(vectorunary_op && other)
      : simple_op(other),
        op_(std::move(other.op_))
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

  inline const jlm::rvsdg::unary_op &
  operation() const noexcept
  {
    return *static_cast<const jlm::rvsdg::unary_op *>(op_.get());
  }

  virtual bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static inline std::unique_ptr<llvm::tac>
  create(
      const jlm::rvsdg::unary_op & unop,
      const llvm::variable * operand,
      const jlm::rvsdg::type & type)
  {
    auto vct1 = dynamic_cast<const vectortype *>(&operand->type());
    auto vct2 = dynamic_cast<const vectortype *>(&type);
    if (!vct1 || !vct2)
      throw jlm::util::error("expected vector type.");

    vectorunary_op op(unop, *vct1, *vct2);
    return tac::create(op, { operand });
  }

private:
  std::unique_ptr<jlm::rvsdg::operation> op_;
};

/* vectorbinary operator */

class vectorbinary_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~vectorbinary_op();

  inline vectorbinary_op(
      const jlm::rvsdg::binary_op & binop,
      const vectortype & op1,
      const vectortype & op2,
      const vectortype & result)
      : simple_op({ op1.copy(), op2.copy() }, { result.copy() }),
        op_(binop.copy())
  {
    if (op1 != op2)
      throw jlm::util::error("expected the same vector types.");

    if (op1.type() != binop.argument(0).type())
    {
      auto received = op1.type().debug_string();
      auto expected = binop.argument(0).type().debug_string();
      throw jlm::util::error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }

    if (result.type() != binop.result(0).type())
    {
      auto received = result.type().debug_string();
      auto expected = binop.result(0).type().debug_string();
      throw jlm::util::error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  inline vectorbinary_op(const vectorbinary_op & other)
      : simple_op(other),
        op_(other.op_->copy())
  {}

  inline vectorbinary_op(vectorbinary_op && other)
      : simple_op(other),
        op_(std::move(other.op_))
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

  inline const jlm::rvsdg::binary_op &
  operation() const noexcept
  {
    return *static_cast<const jlm::rvsdg::binary_op *>(op_.get());
  }

  virtual bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static inline std::unique_ptr<llvm::tac>
  create(
      const jlm::rvsdg::binary_op & binop,
      const llvm::variable * op1,
      const llvm::variable * op2,
      const jlm::rvsdg::type & type)
  {
    auto vct1 = dynamic_cast<const vectortype *>(&op1->type());
    auto vct2 = dynamic_cast<const vectortype *>(&op2->type());
    auto vct3 = dynamic_cast<const vectortype *>(&type);
    if (!vct1 || !vct2 || !vct3)
      throw jlm::util::error("expected vector type.");

    vectorbinary_op op(binop, *vct1, *vct2, *vct3);
    return tac::create(op, { op1, op2 });
  }

private:
  std::unique_ptr<jlm::rvsdg::operation> op_;
};

/* constant data vector operator */

class constant_data_vector_op final : public jlm::rvsdg::simple_op
{
public:
  ~constant_data_vector_op() override;

private:
  constant_data_vector_op(const vectortype & vt)
      : simple_op({ vt.size(), vt.Type() }, { vt.copy() })
  {}

public:
  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return static_cast<const vectortype *>(&result(0).type())->size();
  }

  const jlm::rvsdg::valuetype &
  type() const noexcept
  {
    return static_cast<const vectortype *>(&result(0).type())->type();
  }

  static std::unique_ptr<tac>
  Create(const std::vector<const variable *> & elements)
  {
    if (elements.empty())
      throw jlm::util::error("Expected at least one element.");

    auto vt = std::dynamic_pointer_cast<const jlm::rvsdg::valuetype>(elements[0]->Type());
    if (!vt)
      throw jlm::util::error("Expected value type.");

    constant_data_vector_op op(fixedvectortype(vt, elements.size()));
    return tac::create(op, elements);
  }
};

/* ExtractValue operator */

class ExtractValue final : public jlm::rvsdg::simple_op
{
  typedef std::vector<unsigned>::const_iterator const_iterator;

public:
  virtual ~ExtractValue();

  inline ExtractValue(const jlm::rvsdg::type & aggtype, const std::vector<unsigned> & indices)
      : simple_op({ aggtype.copy() }, { dsttype(aggtype, indices) }),
        indices_(indices)
  {
    if (indices.empty())
      throw jlm::util::error("expected at least one index.");
  }

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
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

  const jlm::rvsdg::valuetype &
  type() const noexcept
  {
    return *static_cast<const jlm::rvsdg::valuetype *>(&argument(0).type());
  }

  static inline std::unique_ptr<llvm::tac>
  create(const llvm::variable * aggregate, const std::vector<unsigned> & indices)
  {
    ExtractValue op(aggregate->type(), indices);
    return tac::create(op, { aggregate });
  }

private:
  static inline std::vector<std::shared_ptr<const rvsdg::type>>
  dsttype(const jlm::rvsdg::type & aggtype, const std::vector<unsigned> & indices)
  {
    const jlm::rvsdg::type * type = &aggtype;
    for (const auto & index : indices)
    {
      if (auto st = dynamic_cast<const StructType *>(type))
      {
        if (index >= st->GetDeclaration().NumElements())
          throw jlm::util::error("extractvalue index out of bound.");

        type = &st->GetDeclaration().GetElement(index);
      }
      else if (auto at = dynamic_cast<const arraytype *>(type))
      {
        if (index >= at->nelements())
          throw jlm::util::error("extractvalue index out of bound.");

        type = &at->element_type();
      }
      else
        throw jlm::util::error("expected struct or array type.");
    }

    return { type->copy() };
  }

  std::vector<unsigned> indices_;
};

/* malloc operator */

class malloc_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~malloc_op();

  malloc_op(const jlm::rvsdg::bittype & btype)
      : simple_op({ btype.copy() }, { PointerType::Create(), MemoryStateType::Create() })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  const jlm::rvsdg::bittype &
  size_type() const noexcept
  {
    return *static_cast<const jlm::rvsdg::bittype *>(&argument(0).type());
  }

  FunctionType
  fcttype() const
  {
    JLM_ASSERT(narguments() == 1 && nresults() == 2);
    return FunctionType({ argument(0).Type() }, { result(0).Type(), result(1).Type() });
  }

  static std::unique_ptr<llvm::tac>
  create(const variable * size)
  {
    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&size->type());
    if (!bt)
      throw jlm::util::error("expected bits type.");

    malloc_op op(*bt);
    return tac::create(op, { size });
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output * size)
  {
    auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&size->type());
    if (!bt)
      throw jlm::util::error("expected bits type.");

    malloc_op op(*bt);
    return jlm::rvsdg::simple_node::create_normalized(size->region(), op, { size });
  }
};

/**
 * Represents the standard C library call free() used for freeing dynamically allocated memory.
 *
 * This operation has no equivalent LLVM instruction.
 */
class FreeOperation final : public jlm::rvsdg::simple_op
{
public:
  ~FreeOperation() noexcept override;

  explicit FreeOperation(size_t numMemoryStates)
      : simple_op(CreateOperandTypes(numMemoryStates), CreateResultTypes(numMemoryStates))
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * pointer,
      const std::vector<const variable *> & memoryStates,
      const variable * iOState)
  {
    std::vector<const variable *> operands;
    operands.push_back(pointer);
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());
    operands.push_back(iOState);

    FreeOperation operation(memoryStates.size());
    return tac::create(operation, operands);
  }

  static std::vector<jlm::rvsdg::output *>
  Create(
      jlm::rvsdg::output * pointer,
      const std::vector<jlm::rvsdg::output *> & memoryStates,
      jlm::rvsdg::output * iOState)
  {
    std::vector<jlm::rvsdg::output *> operands;
    operands.push_back(pointer);
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());
    operands.push_back(iOState);

    FreeOperation operation(memoryStates.size());
    return jlm::rvsdg::simple_node::create_normalized(pointer->region(), operation, operands);
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> memoryStates(
        numMemoryStates,
        MemoryStateType::Create());

    std::vector<std::shared_ptr<const rvsdg::type>> types({ PointerType::Create() });
    types.insert(types.end(), memoryStates.begin(), memoryStates.end());
    types.emplace_back(iostatetype::Create());

    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::type>> types(
        numMemoryStates,
        MemoryStateType::Create());
    types.emplace_back(iostatetype::Create());

    return types;
  }
};

}

#endif
