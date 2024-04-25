/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/tooling/CommandLine.hpp>

static void
TestOptimizationCommandLineArgumentConversion()
{
  using namespace jlm::tooling;

  for (size_t n =
           static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::FirstEnumValue) + 1;
       n != static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::LastEnumValue);
       n++)
  {
    auto expectedOptimizationId = static_cast<JlmOptCommandLineOptions::OptimizationId>(n);
    auto commandLineArgument =
        JlmOptCommandLineOptions::ToCommandLineArgument(expectedOptimizationId);
    auto receivedOptimizationId =
        JlmOptCommandLineOptions::FromCommandLineArgumentToOptimizationId(commandLineArgument);

    assert(receivedOptimizationId == expectedOptimizationId);
  }
}

static void
TestStatisticsCommandLineArgumentConversion()
{
  using namespace jlm::tooling;
  for (size_t n = static_cast<std::size_t>(jlm::util::Statistics::Id::FirstEnumValue) + 1;
       n != static_cast<std::size_t>(jlm::util::Statistics::Id::LastEnumValue);
       n++)
  {
    auto expectedStatisticsId = static_cast<jlm::util::Statistics::Id>(n);
    auto commandLineArgument =
        JlmOptCommandLineOptions::ToCommandLineArgument(expectedStatisticsId);
    auto receivedStatisticsId =
        JlmOptCommandLineOptions::FromCommandLineArgumentToStatisticsId(commandLineArgument);

    assert(receivedStatisticsId == expectedStatisticsId);
  }
}

static void
TestOptimizationIdToOptimizationTranslation()
{
  using namespace jlm::tooling;

  for (size_t n =
           static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::FirstEnumValue) + 1;
       n != static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::LastEnumValue);
       n++)
  {
    auto optimizationId = static_cast<JlmOptCommandLineOptions::OptimizationId>(n);

    // throws exception on failure
    JlmOptCommandLineOptions::GetOptimization(optimizationId);
  }
}

static int
TestOutputFormatToCommandLineArgument()
{
  using namespace jlm::tooling;

  // Arrange
  auto start = static_cast<std::size_t>(JlmOptCommandLineOptions::OutputFormat::FirstEnumValue) + 1;
  auto end = static_cast<std::size_t>(JlmOptCommandLineOptions::OutputFormat::LastEnumValue);

  // Act & Assert
  for (size_t n = start; n != end; n++)
  {
    auto outputFormat = static_cast<JlmOptCommandLineOptions::OutputFormat>(n);

    // throws exception / asserts on failure
    JlmOptCommandLineOptions::ToCommandLineArgument(outputFormat);
  }

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/tooling/TestJlmOptCommandLineParser-TestOutputFormatToCommandLineArgument",
    TestOutputFormatToCommandLineArgument)

static int
Test()
{
  TestOptimizationCommandLineArgumentConversion();
  TestStatisticsCommandLineArgumentConversion();
  TestOptimizationIdToOptimizationTranslation();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlmOptCommandLineParser", Test)
