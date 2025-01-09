/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/Statistics.hpp>

#include <jlm/util/BijectiveMap.hpp>
#include <jlm/util/strfmt.hpp>

#include <string_view>

namespace jlm::util
{

// Mapping between each statistics id and identifier used when serializing the statistic
static const util::BijectiveMap<Statistics::Id, std::string_view> &
GetStatisticsIdNames()
{
  static util::BijectiveMap<Statistics::Id, std::string_view> mapping = {
    { Statistics::Id::Aggregation, "Aggregation" },
    { Statistics::Id::AgnosticMemoryNodeProvisioning, "AgnosticMemoryNodeProvider" },
    { Statistics::Id::AndersenAnalysis, "AndersenAnalysis" },
    { Statistics::Id::Annotation, "Annotation" },
    { Statistics::Id::CommonNodeElimination, "CNE" },
    { Statistics::Id::ControlFlowRecovery, "ControlFlowRestructuring" },
    { Statistics::Id::DataNodeToDelta, "DataNodeToDeltaStatistics" },
    { Statistics::Id::DeadNodeElimination, "DeadNodeElimination" },
    { Statistics::Id::FunctionInlining, "ILN" },
    { Statistics::Id::JlmToRvsdgConversion, "ControlFlowGraphToLambda" },
    { Statistics::Id::LoopUnrolling, "UNROLL" },
    { Statistics::Id::InvariantValueRedirection, "InvariantValueRedirection" },
    { Statistics::Id::MemoryStateEncoder, "MemoryStateEncoder" },
    { Statistics::Id::PullNodes, "PULL" },
    { Statistics::Id::PushNodes, "PUSH" },
    { Statistics::Id::ReduceNodes, "RED" },
    { Statistics::Id::RegionAwareMemoryNodeProvisioning, "RegionAwareMemoryNodeProvision" },
    { Statistics::Id::RvsdgConstruction, "InterProceduralGraphToRvsdg" },
    { Statistics::Id::RvsdgDestruction, "RVSDGDESTRUCTION" },
    { Statistics::Id::RvsdgOptimization, "RVSDGOPTIMIZATION" },
    { Statistics::Id::RvsdgTreePrinter, "RvsdgTreePrinter" },
    { Statistics::Id::SteensgaardAnalysis, "SteensgaardAnalysis" },
    { Statistics::Id::ThetaGammaInversion, "IVT" },
    { Statistics::Id::TopDownMemoryNodeEliminator, "TopDownMemoryNodeEliminator" }
  };
  // Make sure every Statistic is mentioned in the mapping
  auto lastIdx = static_cast<size_t>(Statistics::Id::LastEnumValue);
  auto firstIdx = static_cast<size_t>(Statistics::Id::FirstEnumValue);
  JLM_ASSERT(mapping.Size() == lastIdx - firstIdx - 1);
  return mapping;
}

Statistics::~Statistics() = default;

std::string_view
Statistics::GetName() const
{
  return GetStatisticsIdNames().LookupKey(StatisticsId_);
}

std::optional<filepath>
Statistics::GetSourceFile() const
{
  return SourceFile_;
}

std::string
Statistics::Serialize(char fieldSeparator, char nameValueSeparator) const
{
  std::ostringstream ss;

  ss << GetName() << fieldSeparator;
  ss << (GetSourceFile().has_value() ? GetSourceFile().value().to_str() : "None");

  for (const auto & [mName, measurement] : Measurements_)
  {
    if (ss.tellp() != 0)
      ss << fieldSeparator;

    ss << mName << nameValueSeparator;
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
      ss << fieldSeparator;

    ss << mName << "[ns]" << nameValueSeparator << timer.ns();
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

const Statistics::Measurement &
Statistics::GetMeasurement(const std::string & name) const
{
  for (const auto & [mName, measurement] : Measurements_)
    if (mName == name)
      return measurement;
  JLM_UNREACHABLE("Unknown measurement");
}

util::iterator_range<Statistics::MeasurementList::const_iterator>
Statistics::GetMeasurements() const
{
  return { Measurements_.begin(), Measurements_.end() };
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
  JLM_UNREACHABLE("Unknown timer");
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
    fprintf(file.fd(), "%s\n", statistics.Serialize(' ', ':').c_str());
  }

  file.close();
}

}
