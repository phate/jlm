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

template<typename reduction, const char * name, enum binary_op::flags opflags>
class MakeBitComparisonOperation final : public bitcompare_op
{
public:
  ~MakeBitComparisonOperation() noexcept override;

  explicit MakeBitComparisonOperation(const bittype & type) noexcept
      : bitcompare_op(type)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  enum binary_op::flags
  flags() const noexcept override;

  compare_result
  reduce_constants(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const override;

  std::string
  debug_string() const override;

  std::unique_ptr<operation>
  copy() const override;

  std::unique_ptr<bitcompare_op>
  create(size_t nbits) const override;

  static output *
  create(size_t nbits, output * op1, output * op2)
  {
    return simple_node::create_normalized(
        op1->region(),
        MakeBitComparisonOperation(nbits),
        { op1, op2 })[0];
  }
};

struct reduce_eq;
extern const char BitEqLabel[];
using biteq_op = MakeBitComparisonOperation<reduce_eq, BitEqLabel, binary_op::flags::commutative>;
extern template class MakeBitComparisonOperation<
    reduce_eq,
    BitEqLabel,
    binary_op::flags::commutative>;

struct reduce_ne;
extern const char BitNeLabel[];
using bitne_op = MakeBitComparisonOperation<reduce_ne, BitNeLabel, binary_op::flags::commutative>;
extern template class MakeBitComparisonOperation<
    reduce_ne,
    BitNeLabel,
    binary_op::flags::commutative>;

struct reduce_sge;
extern const char BitSgeLabel[];
using bitsge_op = MakeBitComparisonOperation<reduce_sge, BitSgeLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_sge, BitSgeLabel, binary_op::flags::none>;

struct reduce_sgt;
extern const char BitSgtLabel[];
using bitsgt_op = MakeBitComparisonOperation<reduce_sgt, BitSgtLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_sgt, BitSgtLabel, binary_op::flags::none>;

struct reduce_sle;
extern const char BitSleLabel[];
using bitsle_op = MakeBitComparisonOperation<reduce_sle, BitSleLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_sle, BitSleLabel, binary_op::flags::none>;

struct reduce_slt;
extern const char BitSltLabel[];
using bitslt_op = MakeBitComparisonOperation<reduce_slt, BitSltLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_slt, BitSltLabel, binary_op::flags::none>;

struct reduce_uge;
extern const char BitUgeLabel[];
using bituge_op = MakeBitComparisonOperation<reduce_uge, BitUgeLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_uge, BitUgeLabel, binary_op::flags::none>;

struct reduce_ugt;
extern const char BitUgtLabel[];
using bitugt_op = MakeBitComparisonOperation<reduce_ugt, BitUgtLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_ugt, BitUgtLabel, binary_op::flags::none>;

struct reduce_ule;
extern const char BitUleLabel[];
using bitule_op = MakeBitComparisonOperation<reduce_ule, BitUleLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_ule, BitUleLabel, binary_op::flags::none>;

struct reduce_ult;
extern const char BitUltLabel[];
using bitult_op = MakeBitComparisonOperation<reduce_ult, BitUltLabel, binary_op::flags::none>;
extern template class MakeBitComparisonOperation<reduce_ult, BitUltLabel, binary_op::flags::none>;

}

#endif
