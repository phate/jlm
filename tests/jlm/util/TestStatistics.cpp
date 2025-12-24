/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/Statistics.hpp>

#include <atomic>
#include <fstream>
#include <memory>

class MyTestStatistics final : public jlm::util::Statistics
{
public:
  MyTestStatistics(jlm::util::Statistics::Id id, const jlm::util::FilePath & sourceFile)
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

TEST(StatisticsTests, TestStatisticsMeasurements)
{
  using namespace jlm::util;

  // Arrange
  FilePath path("file.ll");
  MyTestStatistics statistics(Statistics::Id::Aggregation, path);

  // Act
  statistics.Start(10, 6.0);
  // Pretend to do real work
  std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
  statistics.Stop(-400, "poor");

  // Assert
  EXPECT_EQ(statistics.GetId(), Statistics::Id::Aggregation);
  EXPECT_EQ(statistics.GetSourceFile(), path);

  EXPECT_TRUE(statistics.HasMeasurement("count"));
  EXPECT_FALSE(statistics.HasMeasurement("height"));
  EXPECT_TRUE(statistics.HasTimer("Timer"));
  EXPECT_FALSE(statistics.HasTimer("SpinLockTimer"));

  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("count"), 10);
  EXPECT_EQ(statistics.GetMeasurementValue<double>("weight"), 6.0);
  EXPECT_EQ(statistics.GetMeasurementValue<int64_t>("bankAccount"), -400);
  EXPECT_EQ(statistics.GetMeasurementValue<std::string>("state"), "poor");
  EXPECT_GT(statistics.GetTimerElapsedNanoseconds("Timer"), 0);

  // Ensure order is preserved
  auto measurements = statistics.GetMeasurements();
  auto it = measurements.begin();
  EXPECT_EQ(it->first, "count");
  it++;
  EXPECT_EQ(it->first, "weight");
  it++;
  EXPECT_EQ(it->first, "bankAccount");
  it++;
  EXPECT_EQ(it->first, "state");
  it++;
  EXPECT_EQ(it, measurements.end());

  auto timers = statistics.GetTimers();
  EXPECT_EQ(timers.begin()->first, "Timer");
}

TEST(StatisticsTests, TestStatisticsCollection)
{
  using namespace jlm::util;

  // Arrange
  StatisticsCollectorSettings settings({ Statistics::Id::Aggregation });
  StatisticsCollector collector(std::move(settings));

  FilePath path("file.ll");
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

  EXPECT_EQ(numCollectedStatistics, 1);
  for (auto & statistic : collector.CollectedStatistics())
  {
    EXPECT_EQ(statistic.GetId(), Statistics::Id::Aggregation);
  }
}

TEST(StatisticsTests, TestStatisticsPrinting)
{
  using namespace jlm::util;

  // Arrange
  auto testOutputDir = FilePath::TempDirectoryPath().Join("jlm-test-statistics");

  // Remove the output dir if it was not properly cleaned up last time
  std::filesystem::remove(testOutputDir.to_str());
  EXPECT_FALSE(testOutputDir.Exists());

  StatisticsCollectorSettings settings(
      { Statistics::Id::Aggregation },
      testOutputDir,
      "test-module");

  StatisticsCollector collector(settings);

  FilePath path("file.ll");
  std::unique_ptr<MyTestStatistics> statistics(
      new MyTestStatistics(Statistics::Id::Aggregation, path));
  statistics->Start(10, 6.0);
  statistics->Stop(-400, "poor");

  collector.CollectDemandedStatistics(std::move(statistics));

  // Act
  collector.PrintStatistics();

  // Assert
  EXPECT_TRUE(testOutputDir.IsDirectory());

  const auto outputFileName = "test-module-" + settings.GetUniqueString() + "-statistics.log";
  std::ifstream file(testOutputDir.Join(outputFileName).to_str());
  std::string name, fileName, measurement;
  file >> name >> fileName >> measurement;

  EXPECT_EQ(name, "Aggregation");
  EXPECT_EQ(fileName, path.to_str());
  EXPECT_EQ(measurement, "count:10");

  // Cleanup
  std::filesystem::remove_all(testOutputDir.to_str());
}

TEST(StatisticsTests, TestCreateOutputFile)
{
  using namespace jlm::util;

  // Arrange
  StatisticsCollectorSettings settings(
      { Statistics::Id::Aggregation },
      FilePath("/tmp"),
      "test-module");
  settings.SetUniqueString("ABC");
  StatisticsCollector collector(std::move(settings));

  // Act
  const auto statsFile = collector.createOutputFile("stats.log");

  const auto cool0 = collector.createOutputFile("cool", true);
  const auto cool1 = collector.createOutputFile("cool", true);

  const auto nice0 = collector.createOutputFile("nice.txt", true);
  const auto nice1 = collector.createOutputFile("nice.txt", true);

  // Assert
  EXPECT_EQ(statsFile.path(), "/tmp/test-module-ABC-stats.log");

  EXPECT_EQ(cool0.path(), "/tmp/test-module-ABC-cool-0");
  EXPECT_EQ(cool1.path(), "/tmp/test-module-ABC-cool-1");

  EXPECT_EQ(nice0.path(), "/tmp/test-module-ABC-nice-0.txt");
  EXPECT_EQ(nice1.path(), "/tmp/test-module-ABC-nice-1.txt");
}
