/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/tooling/CommandLine.hpp>

#include <cstring>

static const jlm::tooling::JlmOptCommandLineOptions &
ParseCommandLineArguments(const std::vector<std::string> & commandLineArguments)
{
  std::vector<const char *> cStrings;
  for (const auto & commandLineArgument : commandLineArguments)
  {
    cStrings.push_back(commandLineArgument.c_str());
  }

  static jlm::tooling::JlmOptCommandLineParser commandLineParser;
  return commandLineParser.ParseCommandLineArguments(
      static_cast<int>(cStrings.size()),
      cStrings.data());
}

TEST(JlmOptCommandLinerParserTests, TestOptimizationCommandLineArgumentConversion)
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

    EXPECT_EQ(receivedOptimizationId, expectedOptimizationId);
  }
}

TEST(JlmOptCommandLinerParserTests, TestStatisticsCommandLineArgumentConversion)
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

    EXPECT_EQ(receivedStatisticsId, expectedStatisticsId);
  }
}

TEST(JlmOptCommandLinerParserTests, TestOutputFormatToCommandLineArgument)
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
}

TEST(JlmOptCommandLinerParserTests, OutputFormatParsing)
{
  using namespace jlm::tooling;

  auto testOutputFormatParsing =
      [](std::string_view outputFormatString, JlmOptCommandLineOptions::OutputFormat outputFormat)
  {
    // Arrange
    std::vector<std::string> commandLineArguments(
        { "jlm-opt", "--output-format", std::string(outputFormatString), "foo.c" });

    // Act
    auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

    // Assert
    EXPECT_EQ(commandLineOptions.GetOutputFormat(), outputFormat);
  };

  auto start = static_cast<std::size_t>(JlmOptCommandLineOptions::OutputFormat::FirstEnumValue) + 1;
  auto end = static_cast<std::size_t>(JlmOptCommandLineOptions::OutputFormat::LastEnumValue);

  for (size_t n = start; n != end; n++)
  {
    auto outputFormat = static_cast<JlmOptCommandLineOptions::OutputFormat>(n);
#ifndef ENABLE_MLIR
    if (outputFormat == JlmOptCommandLineOptions::OutputFormat::Mlir)
      continue;
#endif

    auto outputFormatString = JlmOptCommandLineOptions::ToCommandLineArgument(outputFormat);
    testOutputFormatParsing(outputFormatString, outputFormat);
  }
}
