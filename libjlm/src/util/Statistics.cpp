/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

namespace jlm {

Statistics::~Statistics()
= default;

void
StatisticsDescriptor::PrintStatistics(const Statistics & statistics) const noexcept
{
  if (IsDemanded(statistics.GetId()))
    fprintf(File_.fd(), "%s\n", statistics.ToString().c_str());
}

}
