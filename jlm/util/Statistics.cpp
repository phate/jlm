/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

namespace jlm::util
{

Statistics::~Statistics() = default;

void
StatisticsCollector::PrintStatistics() const
{
  if (NumCollectedStatistics() == 0)
    return;

  auto & filePath = GetSettings().GetFilePath();
  if (filePath.Exists() && !filePath.IsFile())
  {
    return;
  }

  util::file file(filePath);
  file.open("w");

  for (auto & statistics : CollectedStatistics())
  {
    fprintf(file.fd(), "%s\n", statistics.ToString().c_str());
  }

  file.close();
}

}
