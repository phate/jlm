/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Statistics.hpp>

#include <atomic>
#include <fstream>
#include <memory>
#include <sstream>

class MyTestStatistics final : public jlm::util::Statistics
{
public:
  MyTestStatistics(jlm::util::Statistics::Id id, const jlm::util::filepath & sourceFile)
      : jlm::util::Statistics(id, sourceFile)
  {}

  void
  Start(uint64_t count, double weight)
  {
    AddMeasurement("count", count);
    AddMeasurement("weight", weight);
    AddTimer("Timer").start();
  }

  void
  Stop(int64_t bankAccount, std::string state)
  {
    AddMeasurement("bankAccount", bankAccount);
    AddMeasurement("state", std::move(state));
    GetTimer("Timer").stop();
  }
};

void
TestStatisticsMeasurements()
{
  using namespace jlm::util;

  // Arrange
  filepath path("file.ll");
  MyTestStatistics statistics(Statistics::Id::Aggregation, path);

  // Act
  statistics.Start(10, 6.0);
  // Pretend to do real work
  std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
  statistics.Stop(-400, "poor");

  // Assert
  assert(statistics.GetId() == Statistics::Id::Aggregation);
  assert(statistics.GetSourceFile() == path);

  assert(statistics.HasMeasurement("count"));
  assert(!statistics.HasMeasurement("height"));
  assert(statistics.HasTimer("Timer"));
  assert(!statistics.HasTimer("SpinLockTimer"));

  assert(statistics.GetMeasurementValue<uint64_t>("count") == 10);
  assert(statistics.GetMeasurementValue<double>("weight") == 6.0);
  assert(statistics.GetMeasurementValue<int64_t>("bankAccount") == -400);
  assert(statistics.GetMeasurementValue<std::string>("state") == "poor");
  assert(statistics.GetTimerElapsedNanoseconds("Timer") > 0);

  // Ensure order is preserved
  auto measurements = statistics.GetMeasurements();
  auto it = measurements.begin();
  assert(it->first == "count");
  it++;
  assert(it->first == "weight");
  it++;
  assert(it->first == "bankAccount");
  it++;
  assert(it->first == "state");
  it++;
  assert(it == measurements.end());

  auto timers = statistics.GetTimers();
  assert(timers.begin()->first == "Timer");
}

void
TestStatisticsCollection()
{
  using namespace jlm::util;
  /*
   * Arrange
   */
  StatisticsCollectorSettings settings(filepath("stats.txt"), { Statistics::Id::Aggregation });
  StatisticsCollector collector(std::move(settings));

  filepath path("file.ll");
  std::unique_ptr<Statistics> testStatistics1(
      new MyTestStatistics(Statistics::Id::Aggregation, path));
  std::unique_ptr<Statistics> testStatistics2(
      new MyTestStatistics(Statistics::Id::LoopUnrolling, path));

  /*
   * Act
   */
  collector.CollectDemandedStatistics(std::move(testStatistics1));
  collector.CollectDemandedStatistics(std::move(testStatistics2));

  /*
   * Assert
   */
  auto numCollectedStatistics =
      std::distance(collector.CollectedStatistics().begin(), collector.CollectedStatistics().end());

  assert(numCollectedStatistics == 1);
  for (auto & statistic : collector.CollectedStatistics())
  {
    assert(statistic.GetId() == Statistics::Id::Aggregation);
  }
}

void
TestStatisticsPrinting()
{
  using namespace jlm::util;

  // Arrange
  filepath filePath(std::string(std::filesystem::temp_directory_path()) + "/TestStatistics");

  // Ensure file is not around from last test run.
  std::remove(filePath.to_str().c_str());

  StatisticsCollectorSettings settings(filePath, { Statistics::Id::Aggregation });
  StatisticsCollector collector(std::move(settings));

  filepath path("file.ll");
  std::unique_ptr<MyTestStatistics> testStatistics(
      new MyTestStatistics(Statistics::Id::Aggregation, path));

  collector.CollectDemandedStatistics(std::move(testStatistics));

  // Act
  collector.PrintStatistics();

  // Assert
  std::stringstream stringStream;
  std::ifstream file(filePath.to_str());
  stringStream << file.rdbuf();

  assert(stringStream.str() == "Aggregation file.ll");
}

static int
TestStatistics()
{
  TestStatisticsMeasurements();
  TestStatisticsCollection();
  TestStatisticsPrinting();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestStatistics", TestStatistics)
