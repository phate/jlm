/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_UNROLL_HPP
#define JLM_LLVM_OPT_UNROLL_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Optimization that attempts to unroll loops (thetas).
 */
class loopunroll final : public optimization
{
public:
  virtual ~loopunroll();

  constexpr loopunroll(size_t factor)
      : factor_(factor)
  {}

  /**
   * Given a module all inner most loops (thetas) are found and unrolled if possible.
   * All nodes in the module are traversed and if a theta is found and is the inner most theta
   * then an attempt is made to unroll it.
   *
   * \param module Module where the innermost loops are unrolled
   * \param statisticsCollector Statistics collector for collecting loop unrolling statistics.
   */
  virtual void
  run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  size_t factor_;
};

class unrollinfo final
{
public:
  inline ~unrollinfo()
  {}

private:
  inline unrollinfo(
      jlm::rvsdg::node * cmpnode,
      jlm::rvsdg::node * armnode,
      rvsdg::RegionArgument * idv,
      rvsdg::RegionArgument * step,
      rvsdg::RegionArgument * end)
      : end_(end),
        step_(step),
        cmpnode_(cmpnode),
        armnode_(armnode),
        idv_(idv)
  {}

public:
  unrollinfo(const unrollinfo &) = delete;

  unrollinfo(unrollinfo &&) = delete;

  unrollinfo &
  operator=(const unrollinfo &) = delete;

  unrollinfo &
  operator=(unrollinfo &&) = delete;

  inline jlm::rvsdg::theta_node *
  theta() const noexcept
  {
    auto node = idv()->region()->node();
    JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(node));
    return static_cast<jlm::rvsdg::theta_node *>(node);
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

  std::unique_ptr<jlm::rvsdg::bitvalue_repr>
  niterations() const noexcept;

  inline jlm::rvsdg::node *
  cmpnode() const noexcept
  {
    return cmpnode_;
  }

  inline const jlm::rvsdg::simple_op &
  cmpoperation() const noexcept
  {
    return *static_cast<const jlm::rvsdg::simple_op *>(&cmpnode()->operation());
  }

  inline jlm::rvsdg::node *
  armnode() const noexcept
  {
    return armnode_;
  }

  inline const jlm::rvsdg::simple_op &
  armoperation() const noexcept
  {
    return *static_cast<const jlm::rvsdg::simple_op *>(&armnode()->operation());
  }

  inline rvsdg::RegionArgument *
  idv() const noexcept
  {
    return idv_;
  }

  inline jlm::rvsdg::output *
  init() const noexcept
  {
    return idv()->input()->origin();
  }

  inline const jlm::rvsdg::bitvalue_repr *
  init_value() const noexcept
  {
    return value(init());
  }

  inline rvsdg::RegionArgument *
  step() const noexcept
  {
    return step_;
  }

  inline const jlm::rvsdg::bitvalue_repr *
  step_value() const noexcept
  {
    return value(step());
  }

  inline rvsdg::RegionArgument *
  end() const noexcept
  {
    return end_;
  }

  inline const jlm::rvsdg::bitvalue_repr *
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
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::bitcompare_op *>(&cmpnode()->operation()));
    return static_cast<const jlm::rvsdg::bitcompare_op *>(&cmpnode()->operation())->type().nbits();
  }

  inline jlm::rvsdg::bitvalue_repr
  remainder(size_t factor) const noexcept
  {
    return niterations()->umod({ nbits(), (int64_t)factor });
  }

  static std::unique_ptr<unrollinfo>
  create(jlm::rvsdg::theta_node * theta);

private:
  inline bool
  is_known(jlm::rvsdg::output * output) const noexcept
  {
    auto p = producer(output);
    if (!p)
      return false;

    auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&p->operation());
    return op && op->value().is_known();
  }

  inline const jlm::rvsdg::bitvalue_repr *
  value(jlm::rvsdg::output * output) const noexcept
  {
    if (!is_known(output))
      return nullptr;

    auto p = producer(output);
    return &static_cast<const jlm::rvsdg::bitconstant_op *>(&p->operation())->value();
  }

  rvsdg::RegionArgument * end_;
  rvsdg::RegionArgument * step_;
  jlm::rvsdg::node * cmpnode_;
  jlm::rvsdg::node * armnode_;
  rvsdg::RegionArgument * idv_;
};

/**
 * Try to unroll the given theta.
 *
 * \param node The theta to attempt the unrolling on.
 * \param factor The number of times to unroll the loop, e.g., if the factor is two then the loop
 * body is duplicated in the unrolled loop.
 */
void
unroll(jlm::rvsdg::theta_node * node, size_t factor);

}

#endif
