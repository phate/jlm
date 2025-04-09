/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_IFCONVERSION_HPP
#define JLM_LLVM_OPT_IFCONVERSION_HPP

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Region;
}

namespace jlm::llvm
{

// FIXME: add documentation
class IfConversion final : public rvsdg::Transformation
{
public:
  ~IfConversion() noexcept override;

  IfConversion();

  IfConversion(const IfConversion &) = delete;

  IfConversion(IfConversion &&) = delete;

  IfConversion &
  operator=(const IfConversion &) = delete;

  IfConversion &
  operator=(IfConversion &&) = delete;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  HandleRegion(rvsdg::Region & region);

  static void
  HandleGammaNode(const rvsdg::GammaNode & gammaNode);
};

}

#endif // JLM_LLVM_OPT_IFCONVERSION_HPP
