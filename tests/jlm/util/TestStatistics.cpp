/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Statistics.hpp>

#include <fstream>
#include <memory>
#include <sstream>

class MyTestStatistics final : public jlm::util::Statistics {
public:
  explicit
  MyTestStatistics(
    jlm::util::Statistics::Id id,
    std::string text)
    : jlm::util::Statistics(id)
    , Text_(std::move(text))
  {}

  [[nodiscard]] std::string
  ToString() const noexcept override
  {
    return Text_;
  }

private:
  std::string Text_;
};

void
TestStatisticsCollection()
{
  using namespace jlm::util;
  /*
   * Arrange
   */
  std::unique_ptr<Statistics> testStatistics1(new MyTestStatistics(Statistics::Id::Aggregation, ""));
  std::unique_ptr<Statistics> testStatistics2(new MyTestStatistics(Statistics::Id::LoopUnrolling, ""));

  StatisticsCollectorSettings settings(
    filepath(""),
    {Statistics::Id::Aggregation});

  StatisticsCollector collector(std::move(settings));

  /*
   * Act
   */
  collector.CollectDemandedStatistics(std::move(testStatistics1));
  collector.CollectDemandedStatistics(std::move(testStatistics2));

  /*
   * Assert
   */
  auto numCollectedStatistics = std::distance(
    collector.CollectedStatistics().begin(),
    collector.CollectedStatistics().end());

  assert(numCollectedStatistics == 1);
}

void
TestStatisticsPrinting()
{
  using namespace jlm::util;

  // Arrange
  filepath filePath(std::string(std::filesystem::temp_directory_path()) + "/TestStatistics");

  // Ensure file is not around from last test run.
  std::remove(filePath.to_str().c_str());

  std::string myText("MyTestStatistics");
  std::unique_ptr<Statistics> testStatistics(new MyTestStatistics(
    Statistics::Id::Aggregation,
    myText));

  StatisticsCollectorSettings settings(
    filePath,
    {Statistics::Id::Aggregation});

  StatisticsCollector collector(std::move(settings));
  collector.CollectDemandedStatistics(std::move(testStatistics));

  // Act
  collector.PrintStatistics();

  // Assert
  std::stringstream stringStream;
  std::ifstream file(filePath.to_str());
  stringStream << file.rdbuf();

  assert(stringStream.str() == (myText + "\n"));
}

static int
TestStatistics()
{
  TestStatisticsCollection();
  TestStatisticsPrinting();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestStatistics", TestStatistics)