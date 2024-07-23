/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_THETA_HPP
#define JLM_LLVM_IR_OPERATORS_THETA_HPP

#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

/*
  FIXME: This should be defined in librvsdg.
*/
static inline const jlm::rvsdg::result *
is_theta_result(const jlm::rvsdg::input * input)
{
  using namespace jlm::rvsdg;

  auto r = dynamic_cast<const jlm::rvsdg::result *>(input);
  if (r && is<theta_op>(r->region()->node()))
    return r;

  return nullptr;
}

}

#endif
