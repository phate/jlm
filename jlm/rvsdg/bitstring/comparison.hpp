/*
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2014 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_COMPARISON_HPP
#define JLM_RVSDG_BITSTRING_COMPARISON_HPP

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::rvsdg
{

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
class MakeBitComparisonOperation final : public bitcompare_op
{
public:
  ~MakeBitComparisonOperation() noexcept override;

  explicit MakeBitComparisonOperation(std::size_t nbits) noexcept
      : bitcompare_op(bittype::Create(nbits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  compare_result
  reduce_constants(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  std::unique_ptr<bitcompare_op>
  create(size_t nbits) const override;

  static Output *
  create(size_t nbits, Output * op1, Output * op2)
  {
    return CreateOpNode<MakeBitComparisonOperation>({ op1, op2 }, nbits).output(0);
  }
};

struct reduce_eq;
extern const char BitEqLabel[];
using biteq_op =
    MakeBitComparisonOperation<reduce_eq, BitEqLabel, BinaryOperation::flags::commutative>;
extern template class MakeBitComparisonOperation<
    reduce_eq,
    BitEqLabel,
    BinaryOperation::flags::commutative>;

struct reduce_ne;
extern const char BitNeLabel[];
using bitne_op =
    MakeBitComparisonOperation<reduce_ne, BitNeLabel, BinaryOperation::flags::commutative>;
extern template class MakeBitComparisonOperation<
    reduce_ne,
    BitNeLabel,
    BinaryOperation::flags::commutative>;

struct reduce_sge;
extern const char BitSgeLabel[];
using bitsge_op = MakeBitComparisonOperation<reduce_sge, BitSgeLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_sge,
    BitSgeLabel,
    BinaryOperation::flags::none>;

struct reduce_sgt;
extern const char BitSgtLabel[];
using bitsgt_op = MakeBitComparisonOperation<reduce_sgt, BitSgtLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_sgt,
    BitSgtLabel,
    BinaryOperation::flags::none>;

struct reduce_sle;
extern const char BitSleLabel[];
using bitsle_op = MakeBitComparisonOperation<reduce_sle, BitSleLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_sle,
    BitSleLabel,
    BinaryOperation::flags::none>;

struct reduce_slt;
extern const char BitSltLabel[];
using bitslt_op = MakeBitComparisonOperation<reduce_slt, BitSltLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_slt,
    BitSltLabel,
    BinaryOperation::flags::none>;

struct reduce_uge;
extern const char BitUgeLabel[];
using bituge_op = MakeBitComparisonOperation<reduce_uge, BitUgeLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_uge,
    BitUgeLabel,
    BinaryOperation::flags::none>;

struct reduce_ugt;
extern const char BitUgtLabel[];
using bitugt_op = MakeBitComparisonOperation<reduce_ugt, BitUgtLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_ugt,
    BitUgtLabel,
    BinaryOperation::flags::none>;

struct reduce_ule;
extern const char BitUleLabel[];
using bitule_op = MakeBitComparisonOperation<reduce_ule, BitUleLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_ule,
    BitUleLabel,
    BinaryOperation::flags::none>;

struct reduce_ult;
extern const char BitUltLabel[];
using bitult_op = MakeBitComparisonOperation<reduce_ult, BitUltLabel, BinaryOperation::flags::none>;
extern template class MakeBitComparisonOperation<
    reduce_ult,
    BitUltLabel,
    BinaryOperation::flags::none>;

}

#endif
