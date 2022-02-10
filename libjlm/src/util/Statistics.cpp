/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

namespace jlm {

void
StatisticsDescriptor::PrintStatistics(const Statistics & s) const noexcept
{
  if (IsPrintable(s.GetStatisticsId()))
    fprintf(file_.fd(), "%s\n", s.ToString().c_str());
}

Statistics::~Statistics()
= default;

}
