/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

#include <jlm/util/strfmt.hpp>

namespace jlm::util
{

Statistics::~Statistics() = default;

std::string
Statistics::Serialize() const
{
  std::ostringstream ss;
  for (const auto & [mName, measurement] : Measurements_)
  {
    if (ss.tellp() != 0)
      ss << " ";

    ss << mName << ":";
    std::visit(
        [&](const auto & value)
        {
          ss << value;
        },
        measurement);
  }
  for (const auto & [mName, timer] : Timers_)
  {
    if (ss.tellp() != 0)
      ss << " ";

    ss << mName << "[ns]:" << timer.ns();
  }

  return ss.str();
}

bool
Statistics::HasMeasurement(const std::string & name) const noexcept
{
  for (const auto & [mName, _] : Measurements_)
    if (mName == name)
      return true;
  return false;
}

Statistics::Measurement &
Statistics::GetMeasurement(const std::string & name)
{
  for (auto & [mName, measurement] : Measurements_)
    if (mName == name)
      return measurement;
  throw util::error(util::strfmt("Unknown measurement: ", name));
}

const Statistics::Measurement &
Statistics::GetMeasurement(const std::string & name) const
{
  return const_cast<Statistics *>(this)->GetMeasurement(name);
}

util::iterator_range<Statistics::MeasurementList::const_iterator>
Statistics::GetMeasurements() const
{
  return {Measurements_.begin(), Measurements_.end()};
}

bool
Statistics::HasTimer(const std::string & name) const noexcept
{
  for (const auto & [mName, _] : Timers_)
    if (mName == name)
      return true;
  return false;
}

util::timer &
Statistics::GetTimer(const std::string & name)
{
  for (auto & [mName, timer] : Timers_)
    if (mName == name)
      return timer;
  throw util::error(util::strfmt("Unknown timer: ", name));
}

const util::timer &
Statistics::GetTimer(const std::string & name) const
{
  return const_cast<Statistics *>(this)->GetTimer(name);
}

util::iterator_range<Statistics::TimerList::const_iterator>
Statistics::GetTimers() const
{
  return Timers_;
}

util::timer &
Statistics::AddTimer(std::string name)
{
  JLM_ASSERT(!HasTimer(name));
  Timers_.emplace_back(std::make_pair(std::move(name), util::timer()));
  auto & timer = Timers_.back().second;
  return timer;
}

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
