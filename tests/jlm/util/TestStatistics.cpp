/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Statistics.hpp>

#include <fstream>
#include <memory>
#include <sstream>

class MyTestStatistics final : public jlm::util::Statistics
{
public:
  MyTestStatistics(jlm::util::Statistics::Id id, jlm::util::filepath sourceFile)
      : jlm::util::Statistics(id, std::move(sourceFile))
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
  filepath path("file.ll");
  MyTestStatistics statistics(Statistics::Id::Aggregation, path);

  statistics.Start(10, 6.0);
  statistics.Stop(-400, "poor");

  assert(statistics.GetMeasurementValue<uint64_t>("count") == 10);
  assert(statistics.GetMeasurementValue<double>("weight") == 6.0);
  assert(statistics.GetMeasurementValue<int64_t>("bankAccount") == -400);
  assert(statistics.GetMeasurementValue<std::string>("count") == "poor");
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
