/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_GAMMA_HPP
#define JLM_LLVM_IR_OPERATORS_GAMMA_HPP

#include <jlm/rvsdg/gamma.hpp>

namespace jlm::llvm
{

/*
  FIXME: This should be defined in librvsdg.
*/
static inline const rvsdg::argument *
is_gamma_argument(const rvsdg::output * output)
{
  using namespace rvsdg;

  auto a = dynamic_cast<const rvsdg::argument *>(output);
  if (a && is<gamma_op>(a->region()->node()))
    return a;

  return nullptr;
}

/*
  FIXME: This should be defined in librvsdg.
*/
static inline const rvsdg::result *
is_gamma_result(const rvsdg::input * input)
{
  using namespace rvsdg;

  auto r = dynamic_cast<const result *>(input);
  if (r && is<gamma_op>(r->region()->node()))
    return r;

  return nullptr;
}

}

#endif
