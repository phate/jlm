/*
 * Copyright 2017 Nico Reißmann <nico.reißmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_UNROLL_HPP
#define JLM_LLVM_OPT_UNROLL_HPP

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

class LlvmRvsdgModule;

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
      rvsdg::SimpleNode * cmpNode,
      rvsdg::SimpleNode * armNode,
      rvsdg::Output * idv,
      rvsdg::Output * step,
      rvsdg::Output * end)
      : end_(end),
        step_(step),
        cmpNode_(cmpNode),
        armNode_(armNode),
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
  hasKnownInit() const noexcept
  {
    return isKnown(init());
  }

  inline bool
  hasKnownStep() const noexcept
  {
    return isKnown(step());
  }

  inline bool
  hasKnownEnd() const noexcept
  {
    return isKnown(end());
  }

  inline bool
  isKnown() const noexcept
  {
    return hasKnownInit() && hasKnownStep() && hasKnownEnd();
  }

  std::unique_ptr<jlm::llvm::IntegerValueRepresentation>
  niterations() const noexcept;

  rvsdg::SimpleNode *
  cmpNode() const noexcept
  {
    return cmpNode_;
  }

  [[nodiscard]] const rvsdg::SimpleOperation &
  cmpOperation() const noexcept
  {
    return cmpNode()->GetOperation();
  }

  inline rvsdg::SimpleNode *
  armNode() const noexcept
  {
    return armNode_;
  }

  [[nodiscard]] const rvsdg::SimpleOperation &
  armOperation() const noexcept
  {
    return armNode()->GetOperation();
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

  inline const jlm::llvm::IntegerValueRepresentation *
  initValue() const noexcept
  {
    return value(init());
  }

  inline rvsdg::Output *
  step() const noexcept
  {
    return step_;
  }

  inline const jlm::llvm::IntegerValueRepresentation *
  stepValue() const noexcept
  {
    return value(step());
  }

  inline rvsdg::Output *
  end() const noexcept
  {
    return end_;
  }

  inline const jlm::llvm::IntegerValueRepresentation *
  endValue() const noexcept
  {
    return value(end());
  }

  inline bool
  isAdditive() const noexcept
  {
    return jlm::rvsdg::is<IntegerAddOperation>(armNode());
  }

  inline bool
  isSubtractive() const noexcept
  {
    return jlm::rvsdg::is<IntegerSubOperation>(armNode());
  }

  inline size_t
  nbits() const noexcept
  {
    JLM_ASSERT(dynamic_cast<const IntegerBinaryOperation *>(&cmpNode()->GetOperation()));
    return static_cast<const IntegerBinaryOperation *>(&cmpNode()->GetOperation())->Type().nbits();
  }

  inline jlm::llvm::IntegerValueRepresentation
  remainder(size_t factor) const noexcept
  {
    return niterations()->umod({ nbits(), (int64_t)factor });
  }

  static std::unique_ptr<LoopUnrollInfo>
  create(rvsdg::ThetaNode * theta);

private:
  /**
   * Returns the constant value behind the given output, if it is known.
   */
  const IntegerValueRepresentation *
  value(const rvsdg::Output * output) const noexcept
  {
    auto & tracedOutput = llvm::traceOutput(*output, false);

    const auto [_, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(tracedOutput);
    if (!constantOperation)
      return nullptr;

    return constantOperation->Representation().is_known() ? &constantOperation->Representation()
                                                          : nullptr;
  }

  bool
  isKnown(const rvsdg::Output * output) const noexcept
  {
    return value(output) != nullptr;
  }

  rvsdg::Output * end_;
  rvsdg::Output * step_;
  rvsdg::SimpleNode * cmpNode_;
  rvsdg::SimpleNode * armNode_;
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
