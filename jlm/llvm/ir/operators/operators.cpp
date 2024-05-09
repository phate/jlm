/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <llvm/ADT/SmallVector.h>

namespace jlm::llvm
{

/* phi operator */

phi_op::~phi_op() noexcept
{}

bool
phi_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const phi_op *>(&other);
  return op && op->nodes_ == nodes_ && op->result(0) == result(0);
}

std::string
phi_op::debug_string() const
{
  std::string str("[");
  for (size_t n = 0; n < narguments(); n++)
  {
    str += util::strfmt(node(n));
    if (n != narguments() - 1)
      str += ", ";
  }
  str += "]";

  return "PHI" + str;
}

std::unique_ptr<rvsdg::operation>
phi_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new phi_op(*this));
}

/* assignment operator */

assignment_op::~assignment_op() noexcept
{}

bool
assignment_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const assignment_op *>(&other);
  return op && op->argument(0) == argument(0);
}

std::string
assignment_op::debug_string() const
{
  return "ASSIGN";
}

std::unique_ptr<rvsdg::operation>
assignment_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new assignment_op(*this));
}

/* select operator */

select_op::~select_op() noexcept
{}

bool
select_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const select_op *>(&other);
  return op && op->result(0) == result(0);
}

std::string
select_op::debug_string() const
{
  return "SELECT";
}

std::unique_ptr<rvsdg::operation>
select_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new select_op(*this));
}

/* vectorselect operator */

vectorselect_op::~vectorselect_op() noexcept
{}

bool
vectorselect_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const vectorselect_op *>(&other);
  return op && op->type() == type();
}

std::string
vectorselect_op::debug_string() const
{
  return "VECTORSELECT";
}

std::unique_ptr<rvsdg::operation>
vectorselect_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new vectorselect_op(*this));
}

/* fp2ui operator */

fp2ui_op::~fp2ui_op() noexcept
{}

bool
fp2ui_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fp2ui_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
fp2ui_op::debug_string() const
{
  return "FP2UI";
}

std::unique_ptr<rvsdg::operation>
fp2ui_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fp2ui_op(*this));
}

rvsdg::unop_reduction_path_t
fp2ui_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
fp2ui_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented");
}

/* fp2si operator */

fp2si_op::~fp2si_op() noexcept
{}

bool
fp2si_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fp2si_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
fp2si_op::debug_string() const
{
  return "FP2UI";
}

std::unique_ptr<rvsdg::operation>
fp2si_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fp2si_op(*this));
}

rvsdg::unop_reduction_path_t
fp2si_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
fp2si_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* ctl2bits operator */

ctl2bits_op::~ctl2bits_op() noexcept
{}

bool
ctl2bits_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ctl2bits_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
ctl2bits_op::debug_string() const
{
  return "CTL2BITS";
}

std::unique_ptr<rvsdg::operation>
ctl2bits_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ctl2bits_op(*this));
}

/* branch operator */

branch_op::~branch_op() noexcept
{}

bool
branch_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const branch_op *>(&other);
  return op && op->argument(0) == argument(0);
}

std::string
branch_op::debug_string() const
{
  return "BRANCH";
}

std::unique_ptr<rvsdg::operation>
branch_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new branch_op(*this));
}

ConstantPointerNullOperation::~ConstantPointerNullOperation() noexcept = default;

bool
ConstantPointerNullOperation::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantPointerNullOperation *>(&other);
  return op && op->GetPointerType() == GetPointerType();
}

std::string
ConstantPointerNullOperation::debug_string() const
{
  return "ConstantPointerNull";
}

std::unique_ptr<rvsdg::operation>
ConstantPointerNullOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantPointerNullOperation(*this));
}

/* bits2ptr operator */

bits2ptr_op::~bits2ptr_op()
{}

bool
bits2ptr_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const bits2ptr_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
bits2ptr_op::debug_string() const
{
  return "BITS2PTR";
}

std::unique_ptr<rvsdg::operation>
bits2ptr_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new bits2ptr_op(*this));
}

rvsdg::unop_reduction_path_t
bits2ptr_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
bits2ptr_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* ptr2bits operator */

ptr2bits_op::~ptr2bits_op()
{}

bool
ptr2bits_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ptr2bits_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
ptr2bits_op::debug_string() const
{
  return "PTR2BITS";
}

std::unique_ptr<rvsdg::operation>
ptr2bits_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ptr2bits_op(*this));
}

rvsdg::unop_reduction_path_t
ptr2bits_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
ptr2bits_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

ConstantDataArray::~ConstantDataArray()
{}

bool
ConstantDataArray::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantDataArray *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantDataArray::debug_string() const
{
  return "ConstantDataArray";
}

std::unique_ptr<rvsdg::operation>
ConstantDataArray::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantDataArray(*this));
}

/* pointer compare operator */

ptrcmp_op::~ptrcmp_op()
{}

bool
ptrcmp_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ptrcmp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->cmp_ == cmp_;
}

std::string
ptrcmp_op::debug_string() const
{
  static std::unordered_map<llvm::cmp, std::string> map({ { cmp::eq, "eq" },
                                                          { cmp::ne, "ne" },
                                                          { cmp::gt, "gt" },
                                                          { cmp::ge, "ge" },
                                                          { cmp::lt, "lt" },
                                                          { cmp::le, "le" } });

  JLM_ASSERT(map.find(cmp()) != map.end());
  return "PTRCMP " + map[cmp()];
}

std::unique_ptr<rvsdg::operation>
ptrcmp_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ptrcmp_op(*this));
}

rvsdg::binop_reduction_path_t
ptrcmp_op::can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
ptrcmp_op::reduce_operand_pair(
    rvsdg::binop_reduction_path_t path,
    rvsdg::output * op1,
    rvsdg::output * op2) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* zext operator */

zext_op::~zext_op()
{}

bool
zext_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const zext_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
zext_op::debug_string() const
{
  return util::strfmt("ZEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::operation>
zext_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new zext_op(*this));
}

rvsdg::unop_reduction_path_t
zext_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  if (rvsdg::is<rvsdg::bitconstant_op>(producer(operand)))
    return rvsdg::unop_reduction_constant;

  return rvsdg::unop_reduction_none;
}

rvsdg::output *
zext_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  if (path == rvsdg::unop_reduction_constant)
  {
    auto c = static_cast<const rvsdg::bitconstant_op *>(&producer(operand)->operation());
    return create_bitconstant(
        rvsdg::node_output::node(operand)->region(),
        c->value().zext(ndstbits() - nsrcbits()));
  }

  return nullptr;
}

/* floating point constant operator */

ConstantFP::~ConstantFP()
{}

bool
ConstantFP::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantFP *>(&other);
  return op && size() == op->size() && constant().bitwiseIsEqual(op->constant());
}

std::string
ConstantFP::debug_string() const
{
  ::llvm::SmallVector<char, 32> v;
  constant().toString(v, 32, 0);

  std::string s("FP(");
  for (const auto & c : v)
    s += c;
  s += ")";

  return s;
}

std::unique_ptr<rvsdg::operation>
ConstantFP::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantFP(*this));
}

/* floating point comparison operator */

fpcmp_op::~fpcmp_op()
{}

bool
fpcmp_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fpcmp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->cmp_ == cmp_;
}

std::string
fpcmp_op::debug_string() const
{
  static std::unordered_map<fpcmp, std::string> map({ { fpcmp::oeq, "oeq" },
                                                      { fpcmp::ogt, "ogt" },
                                                      { fpcmp::oge, "oge" },
                                                      { fpcmp::olt, "olt" },
                                                      { fpcmp::ole, "ole" },
                                                      { fpcmp::one, "one" },
                                                      { fpcmp::ord, "ord" },
                                                      { fpcmp::ueq, "ueq" },
                                                      { fpcmp::ugt, "ugt" },
                                                      { fpcmp::uge, "uge" },
                                                      { fpcmp::ult, "ult" },
                                                      { fpcmp::ule, "ule" },
                                                      { fpcmp::une, "une" },
                                                      { fpcmp::uno, "uno" } });

  JLM_ASSERT(map.find(cmp()) != map.end());
  return "FPCMP " + map[cmp()];
}

std::unique_ptr<rvsdg::operation>
fpcmp_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fpcmp_op(*this));
}

rvsdg::binop_reduction_path_t
fpcmp_op::can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
fpcmp_op::reduce_operand_pair(
    rvsdg::binop_reduction_path_t path,
    rvsdg::output * op1,
    rvsdg::output * op2) const
{
  JLM_UNREACHABLE("Not implemented!");
}

UndefValueOperation::~UndefValueOperation() noexcept = default;

bool
UndefValueOperation::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const UndefValueOperation *>(&other);
  return op && op->GetType() == GetType();
}

std::string
UndefValueOperation::debug_string() const
{
  return "undef";
}

std::unique_ptr<rvsdg::operation>
UndefValueOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new UndefValueOperation(*this));
}

PoisonValueOperation::~PoisonValueOperation() noexcept = default;

bool
PoisonValueOperation::operator==(const operation & other) const noexcept
{
  auto operation = dynamic_cast<const PoisonValueOperation *>(&other);
  return operation && operation->GetType() == GetType();
}

std::string
PoisonValueOperation::debug_string() const
{
  return "poison";
}

std::unique_ptr<rvsdg::operation>
PoisonValueOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new PoisonValueOperation(*this));
}

/* floating point arithmetic operator */

fpbin_op::~fpbin_op()
{}

bool
fpbin_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fpbin_op *>(&other);
  return op && op->fpop() == fpop() && op->size() == size();
}

std::string
fpbin_op::debug_string() const
{
  static std::unordered_map<llvm::fpop, std::string> map({ { fpop::add, "add" },
                                                           { fpop::sub, "sub" },
                                                           { fpop::mul, "mul" },
                                                           { fpop::div, "div" },
                                                           { fpop::mod, "mod" } });

  JLM_ASSERT(map.find(fpop()) != map.end());
  return "FPOP " + map[fpop()];
}

std::unique_ptr<rvsdg::operation>
fpbin_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fpbin_op(*this));
}

rvsdg::binop_reduction_path_t
fpbin_op::can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
fpbin_op::reduce_operand_pair(
    rvsdg::binop_reduction_path_t path,
    rvsdg::output * op1,
    rvsdg::output * op2) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* fpext operator */

fpext_op::~fpext_op()
{}

bool
fpext_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fpext_op *>(&other);
  return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

std::string
fpext_op::debug_string() const
{
  return "fpext";
}

std::unique_ptr<rvsdg::operation>
fpext_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fpext_op(*this));
}

rvsdg::unop_reduction_path_t
fpext_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
fpext_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* fpneg operator */

fpneg_op::~fpneg_op()
{}

bool
fpneg_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fpneg_op *>(&other);
  return op && op->size() == size();
}

std::string
fpneg_op::debug_string() const
{
  return "fpneg";
}

std::unique_ptr<rvsdg::operation>
fpneg_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fpneg_op(*this));
}

rvsdg::unop_reduction_path_t
fpneg_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
fpneg_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* fptrunc operator */

fptrunc_op::~fptrunc_op()
{}

bool
fptrunc_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const fptrunc_op *>(&other);
  return op && op->srcsize() == srcsize() && op->dstsize() == dstsize();
}

std::string
fptrunc_op::debug_string() const
{
  return "fptrunc";
}

std::unique_ptr<rvsdg::operation>
fptrunc_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new fptrunc_op(*this));
}

rvsdg::unop_reduction_path_t
fptrunc_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
fptrunc_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* valist operator */

valist_op::~valist_op()
{}

bool
valist_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const valist_op *>(&other);
  if (!op || op->narguments() != narguments())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (op->argument(n) != argument(n))
      return false;
  }

  return true;
}

std::string
valist_op::debug_string() const
{
  return "VALIST";
}

std::unique_ptr<rvsdg::operation>
valist_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new valist_op(*this));
}

/* bitcast operator */

bitcast_op::~bitcast_op()
{}

bool
bitcast_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const bitcast_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
bitcast_op::debug_string() const
{
  return util::strfmt(
      "BITCAST[",
      argument(0).type().debug_string(),
      " -> ",
      result(0).type().debug_string(),
      "]");
}

std::unique_ptr<rvsdg::operation>
bitcast_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new bitcast_op(*this));
}

rvsdg::unop_reduction_path_t
bitcast_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
bitcast_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* ConstantStruct operator */

ConstantStruct::~ConstantStruct()
{}

bool
ConstantStruct::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantStruct *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantStruct::debug_string() const
{
  return "ConstantStruct";
}

std::unique_ptr<rvsdg::operation>
ConstantStruct::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantStruct(*this));
}

/* trunc operator */

trunc_op::~trunc_op()
{}

bool
trunc_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const trunc_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
trunc_op::debug_string() const
{
  return util::strfmt("TRUNC[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<rvsdg::operation>
trunc_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new trunc_op(*this));
}

rvsdg::unop_reduction_path_t
trunc_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
trunc_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* uitofp operator */

uitofp_op::~uitofp_op()
{}

bool
uitofp_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const uitofp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
uitofp_op::debug_string() const
{
  return "UITOFP";
}

std::unique_ptr<rvsdg::operation>
uitofp_op::copy() const
{
  return std::make_unique<uitofp_op>(*this);
}

rvsdg::unop_reduction_path_t
uitofp_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
uitofp_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* sitofp operator */

sitofp_op::~sitofp_op()
{}

bool
sitofp_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const sitofp_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

std::string
sitofp_op::debug_string() const
{
  return "SITOFP";
}

std::unique_ptr<rvsdg::operation>
sitofp_op::copy() const
{
  return std::make_unique<sitofp_op>(*this);
}

rvsdg::unop_reduction_path_t
sitofp_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
sitofp_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  JLM_UNREACHABLE("Not implemented!");
}

/* ConstantArray operator */

ConstantArray::~ConstantArray()
{}

bool
ConstantArray::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantArray *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantArray::debug_string() const
{
  return "ConstantArray";
}

std::unique_ptr<rvsdg::operation>
ConstantArray::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantArray(*this));
}

/* ConstantAggregateZero operator */

ConstantAggregateZero::~ConstantAggregateZero()
{}

bool
ConstantAggregateZero::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const ConstantAggregateZero *>(&other);
  return op && op->result(0) == result(0);
}

std::string
ConstantAggregateZero::debug_string() const
{
  return "ConstantAggregateZero";
}

std::unique_ptr<rvsdg::operation>
ConstantAggregateZero::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ConstantAggregateZero(*this));
}

/* extractelement operator */

extractelement_op::~extractelement_op()
{}

bool
extractelement_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const extractelement_op *>(&other);
  return op && op->argument(0) == argument(0) && op->argument(1) == argument(1);
}

std::string
extractelement_op::debug_string() const
{
  return "EXTRACTELEMENT";
}

std::unique_ptr<rvsdg::operation>
extractelement_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new extractelement_op(*this));
}

/* shufflevector operator */

shufflevector_op::~shufflevector_op()
{}

bool
shufflevector_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const shufflevector_op *>(&other);
  return op && op->argument(0) == argument(0) && op->Mask() == Mask();
}

std::string
shufflevector_op::debug_string() const
{
  return "SHUFFLEVECTOR";
}

std::unique_ptr<rvsdg::operation>
shufflevector_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new shufflevector_op(*this));
}

/* constantvector operator */

constantvector_op::~constantvector_op()
{}

bool
constantvector_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const constantvector_op *>(&other);
  return op && op->result(0) == result(0);
}

std::string
constantvector_op::debug_string() const
{
  return "CONSTANTVECTOR";
}

std::unique_ptr<rvsdg::operation>
constantvector_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new constantvector_op(*this));
}

/* insertelement operator */

insertelement_op::~insertelement_op()
{}

bool
insertelement_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const insertelement_op *>(&other);
  return op && op->argument(0) == argument(0) && op->argument(1) == argument(1)
      && op->argument(2) == argument(2);
}

std::string
insertelement_op::debug_string() const
{
  return "INSERTELEMENT";
}

std::unique_ptr<rvsdg::operation>
insertelement_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new insertelement_op(*this));
}

/* vectorunary operator */

vectorunary_op::~vectorunary_op()
{}

bool
vectorunary_op::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const vectorunary_op *>(&other);
  return op && op->operation() == operation();
}

std::string
vectorunary_op::debug_string() const
{
  return util::strfmt("VEC", operation().debug_string());
}

std::unique_ptr<rvsdg::operation>
vectorunary_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new vectorunary_op(*this));
}

/* vectorbinary operator */

vectorbinary_op::~vectorbinary_op()
{}

bool
vectorbinary_op::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const vectorbinary_op *>(&other);
  return op && op->operation() == operation();
}

std::string
vectorbinary_op::debug_string() const
{
  return util::strfmt("VEC", operation().debug_string());
}

std::unique_ptr<rvsdg::operation>
vectorbinary_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new vectorbinary_op(*this));
}

/* const data vector operator */

constant_data_vector_op::~constant_data_vector_op()
{}

bool
constant_data_vector_op::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const constant_data_vector_op *>(&other);
  return op && op->result(0) == result(0);
}

std::string
constant_data_vector_op::debug_string() const
{
  return "CONSTANTDATAVECTOR";
}

std::unique_ptr<rvsdg::operation>
constant_data_vector_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new constant_data_vector_op(*this));
}

/* extractvalue operator */

ExtractValue::~ExtractValue()
{}

bool
ExtractValue::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const ExtractValue *>(&other);
  return op && op->indices_ == indices_ && op->type() == type();
}

std::string
ExtractValue::debug_string() const
{
  return "ExtractValue";
}

std::unique_ptr<rvsdg::operation>
ExtractValue::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new ExtractValue(*this));
}

/* MemStateMerge operator */

MemStateMergeOperator::~MemStateMergeOperator()
{}

bool
MemStateMergeOperator::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemStateMergeOperator *>(&other);
  return operation && operation->narguments() == narguments();
}

std::string
MemStateMergeOperator::debug_string() const
{
  return "MemStateMerge";
}

std::unique_ptr<rvsdg::operation>
MemStateMergeOperator::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemStateMergeOperator(*this));
}

/* MemStateSplit operator */

MemStateSplitOperator::~MemStateSplitOperator()
{}

bool
MemStateSplitOperator::operator==(const rvsdg::operation & other) const noexcept
{
  auto operation = dynamic_cast<const MemStateSplitOperator *>(&other);
  return operation && operation->nresults() == nresults();
}

std::string
MemStateSplitOperator::debug_string() const
{
  return "MemStateSplit";
}

std::unique_ptr<rvsdg::operation>
MemStateSplitOperator::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new MemStateSplitOperator(*this));
}

/* malloc operator */

malloc_op::~malloc_op()
{}

bool
malloc_op::operator==(const operation & other) const noexcept
{
  /*
    Avoid CNE for malloc operator
  */
  return this == &other;
}

std::string
malloc_op::debug_string() const
{
  return "MALLOC";
}

std::unique_ptr<rvsdg::operation>
malloc_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new malloc_op(*this));
}

/* free operator */

FreeOperation::~FreeOperation() noexcept = default;

bool
FreeOperation::operator==(const operation & other) const noexcept
{
  // Avoid CNE for free operator
  return this == &other;
}

std::string
FreeOperation::debug_string() const
{
  return "FREE";
}

std::unique_ptr<rvsdg::operation>
FreeOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new FreeOperation(*this));
}

}
