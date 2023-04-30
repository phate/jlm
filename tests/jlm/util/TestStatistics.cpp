/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Statistics.hpp>

#include <fstream>
#include <memory>
#include <sstream>

class MyTestStatistics final : public jlm::Statistics {
public:
  explicit
  MyTestStatistics(
    jlm::Statistics::Id id,
    std::string text)
    : jlm::Statistics(id)
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
  /*
   * Arrange
   */
  std::unique_ptr<jlm::Statistics> testStatistics1(new MyTestStatistics(jlm::Statistics::Id::Aggregation, ""));
  std::unique_ptr<jlm::Statistics> testStatistics2(new MyTestStatistics(jlm::Statistics::Id::LoopUnrolling, ""));

  jlm::StatisticsCollectorSettings settings(
    jlm::filepath(""),
    {jlm::Statistics::Id::Aggregation});

  jlm::StatisticsCollector collector(std::move(settings));

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
  /*
   * Arrange
   */
  jlm::filepath filePath("/tmp/TestStatistics");

  /*
   * Ensure file is not around from last test run.
   */
  std::remove(filePath.to_str().c_str());

  std::string myText("MyTestStatistics");
  std::unique_ptr<jlm::Statistics> testStatistics(new MyTestStatistics(
    jlm::Statistics::Id::Aggregation,
    myText));

  jlm::StatisticsCollectorSettings settings(
    filePath,
    {jlm::Statistics::Id::Aggregation});

  jlm::StatisticsCollector collector(std::move(settings));
  collector.CollectDemandedStatistics(std::move(testStatistics));

  /*
   * Act
   */
  collector.PrintStatistics();

  /*
   * Assert
   */
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