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

void
StatisticsCollector::PrintStatistics() const
{
  if (NumCollectedStatistics() == 0)
    return;

  jlm::file file(GetSettings().GetFilePath());
  file.open("a");

  for (auto & statistics : CollectedStatistics())
  {
    fprintf(file.fd(), "%s\n", statistics.ToString().c_str());
  }

  file.close();
}

}
