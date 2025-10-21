/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_UNROLL_HPP
#define JLM_LLVM_OPT_UNROLL_HPP

#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Optimization that attempts to unroll loops (thetas).
 */
class LoopUnrolling final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~LoopUnrolling() noexcept override;

  explicit LoopUnrolling(const size_t factor)
      : Transformation("LoopUnrolling"),
        factor_(factor)
  {}

  /**
   * Given a module all inner most loops (thetas) are found and unrolled if possible.
   * All nodes in the module are traversed and if a theta is found and is the inner most theta
   * then an attempt is made to unroll it.
   *
   * \param module Module where the innermost loops are unrolled
   * \param statisticsCollector Statistics collector for collecting loop unrolling statistics.
   */
  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  size_t factor_;
};

class LoopUnrollInfo final
{
public:
  ~LoopUnrollInfo() noexcept = default;

private:
  LoopUnrollInfo(
      rvsdg::SimpleNode * cmpnode,
      rvsdg::SimpleNode * armnode,
      rvsdg::Output * idv,
      rvsdg::Output * step,
      rvsdg::Output * end)
      : end_(end),
        step_(step),
        cmpnode_(cmpnode),
        armnode_(armnode),
        idv_(idv)
  {}

public:
  LoopUnrollInfo(const LoopUnrollInfo &) = delete;

  LoopUnrollInfo(LoopUnrollInfo &&) = delete;

  LoopUnrollInfo &
  operator=(const LoopUnrollInfo &) = delete;

  LoopUnrollInfo &
  operator=(LoopUnrollInfo &&) = delete;

  inline rvsdg::ThetaNode *
  theta() const noexcept
  {
    auto node = idv()->region()->node();
    return util::assertedCast<rvsdg::ThetaNode>(node);
  }

  inline bool
  has_known_init() const noexcept
  {
    return is_known(init());
  }

  inline bool
  has_known_step() const noexcept
  {
    return is_known(step());
  }

  inline bool
  has_known_end() const noexcept
  {
    return is_known(end());
  }

  inline bool
  is_known() const noexcept
  {
    return has_known_init() && has_known_step() && has_known_end();
  }

  std::unique_ptr<jlm::rvsdg::BitValueRepresentation>
  niterations() const noexcept;

  rvsdg::SimpleNode *
  cmpnode() const noexcept
  {
    return cmpnode_;
  }

  [[nodiscard]] const rvsdg::SimpleOperation &
  cmpoperation() const noexcept
  {
    return cmpnode()->GetOperation();
  }

  inline rvsdg::SimpleNode *
  armnode() const noexcept
  {
    return armnode_;
  }

  [[nodiscard]] const rvsdg::SimpleOperation &
  armoperation() const noexcept
  {
    return armnode()->GetOperation();
  }

  inline rvsdg::Output *
  idv() const noexcept
  {
    return idv_;
  }

  inline jlm::rvsdg::Output *
  init() const noexcept
  {
    return theta()->MapPreLoopVar(*idv()).input->origin();
  }

  inline const jlm::rvsdg::BitValueRepresentation *
  init_value() const noexcept
  {
    return value(init());
  }

  inline rvsdg::Output *
  step() const noexcept
  {
    return step_;
  }

  inline const jlm::rvsdg::BitValueRepresentation *
  step_value() const noexcept
  {
    return value(step());
  }

  inline rvsdg::Output *
  end() const noexcept
  {
    return end_;
  }

  inline const jlm::rvsdg::BitValueRepresentation *
  end_value() const noexcept
  {
    return value(end());
  }

  inline bool
  is_additive() const noexcept
  {
    return jlm::rvsdg::is<jlm::rvsdg::bitadd_op>(armnode());
  }

  inline bool
  is_subtractive() const noexcept
  {
    return jlm::rvsdg::is<jlm::rvsdg::bitsub_op>(armnode());
  }

  inline size_t
  nbits() const noexcept
  {
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::BitCompareOperation *>(&cmpnode()->GetOperation()));
    return static_cast<const rvsdg::BitCompareOperation *>(&cmpnode()->GetOperation())
        ->type()
        .nbits();
  }

  inline jlm::rvsdg::BitValueRepresentation
  remainder(size_t factor) const noexcept
  {
    return niterations()->umod({ nbits(), (int64_t)factor });
  }

  static std::unique_ptr<LoopUnrollInfo>
  create(rvsdg::ThetaNode * theta);

private:
  bool
  is_known(const rvsdg::Output * output) const noexcept
  {
    auto & tracedOutput = rvsdg::traceOutputIntraProcedurally(*output);
    auto [_, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::bitconstant_op>(tracedOutput);
    return constantOperation && constantOperation->value().is_known();
  }

  const rvsdg::BitValueRepresentation *
  value(const rvsdg::Output * output) const noexcept
  {
    if (!is_known(output))
      return nullptr;

    auto & tracedOutput = rvsdg::traceOutputIntraProcedurally(*output);
    auto [_, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::bitconstant_op>(tracedOutput);
    return constantOperation == nullptr ? nullptr : &constantOperation->value();
  }

  rvsdg::Output * end_;
  rvsdg::Output * step_;
  rvsdg::SimpleNode * cmpnode_;
  rvsdg::SimpleNode * armnode_;
  rvsdg::Output * idv_;
};

/**
 * Try to unroll the given theta.
 *
 * \param node The theta to attempt the unrolling on.
 * \param factor The number of times to unroll the loop, e.g., if the factor is two then the loop
 * body is duplicated in the unrolled loop.
 */
void
unroll(rvsdg::ThetaNode * node, size_t factor);

}

#endif
