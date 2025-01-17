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

int
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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestStatistics-TestStatisticsMeasurements",
    TestStatisticsMeasurements)

int
TestStatisticsCollection()
{
  using namespace jlm::util;

  // Arrange
  StatisticsCollectorSettings settings({ Statistics::Id::Aggregation });
  StatisticsCollector collector(std::move(settings));

  filepath path("file.ll");
  std::unique_ptr<Statistics> testStatistics1(
      new MyTestStatistics(Statistics::Id::Aggregation, path));
  std::unique_ptr<Statistics> testStatistics2(
      new MyTestStatistics(Statistics::Id::LoopUnrolling, path));

  // Act
  collector.CollectDemandedStatistics(std::move(testStatistics1));
  collector.CollectDemandedStatistics(std::move(testStatistics2));

  // Assert
  auto numCollectedStatistics =
      std::distance(collector.CollectedStatistics().begin(), collector.CollectedStatistics().end());

  assert(numCollectedStatistics == 1);
  for (auto & statistic : collector.CollectedStatistics())
  {
    assert(statistic.GetId() == Statistics::Id::Aggregation);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestStatistics-TestStatisticsCollection", TestStatisticsCollection)

int
TestStatisticsPrinting()
{
  using namespace jlm::util;

  // Arrange
  auto testOutputDir = filepath::TempDirectoryPath().Join("jlm-test-statistics");

  // Remove the output dir if it was not properly cleaned up last time
  std::filesystem::remove(testOutputDir.to_str());
  assert(!testOutputDir.Exists());

  StatisticsCollectorSettings settings(
      { Statistics::Id::Aggregation },
      testOutputDir,
      "test-module");

  StatisticsCollector collector(settings);

  filepath path("file.ll");
  std::unique_ptr<MyTestStatistics> statistics(
      new MyTestStatistics(Statistics::Id::Aggregation, path));
  statistics->Start(10, 6.0);
  statistics->Stop(-400, "poor");

  collector.CollectDemandedStatistics(std::move(statistics));

  // Act
  collector.PrintStatistics();

  // Assert
  assert(testOutputDir.IsDirectory());

  const auto outputFileName = "test-module-" + settings.GetUniqueString() + "-statistics.log";
  std::ifstream file(testOutputDir.Join(outputFileName).to_str());
  std::string name, fileName, measurement;
  file >> name >> fileName >> measurement;

  assert(name == "Aggregation");
  assert(fileName == path.to_str());
  assert(measurement == "count:10");

  // Cleanup
  std::filesystem::remove_all(testOutputDir.to_str());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestStatistics-TestStatisticsPrinting", TestStatisticsPrinting)

int
TestCreateOutputFile()
{
  using namespace jlm::util;

  // Arrange
  StatisticsCollectorSettings settings(
      { Statistics::Id::Aggregation },
      filepath("."),
      "test-module");
  settings.SetUniqueString("ABC");
  StatisticsCollector collector(std::move(settings));

  // Act
  const auto statsFile = collector.CreateOutputFile("stats.log");

  const auto cool0 = collector.CreateOutputFile("cool", true);
  const auto cool1 = collector.CreateOutputFile("cool", true);

  const auto nice0 = collector.CreateOutputFile("nice.txt", true);
  const auto nice1 = collector.CreateOutputFile("nice.txt", true);

  // Assert
  assert(statsFile.path() == "./test-module-ABC-stats.log");

  assert(cool0.path() == "./test-module-ABC-cool-0");
  assert(cool1.path() == "./test-module-ABC-cool-1");

  assert(nice0.path() == "./test-module-ABC-nice-0.txt");
  assert(nice1.path() == "./test-module-ABC-nice-1.txt");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestStatistics-TestCreateOutputFile", TestCreateOutputFile)
