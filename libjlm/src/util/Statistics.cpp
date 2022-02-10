/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

namespace jlm {

void
StatisticsDescriptor::print_stat(const Statistics & s) const noexcept
{
  fprintf(file_.fd(), "%s\n", s.ToString().c_str());
}

Statistics::~Statistics()
= default;

}
