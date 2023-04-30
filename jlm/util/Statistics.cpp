/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

namespace jlm {

Statistics::~Statistics()
= default;

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
