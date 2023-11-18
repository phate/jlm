/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/tooling/Command.hpp>
#include <jlm/util/strfmt.hpp>

static void
TestStatistics()
{
  using namespace jlm::tooling;

  // Arrange
  std::string expectedStatisticsDir = "/myStatisticsDir/";

  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      jlm::util::filepath(expectedStatisticsDir + "myStatisticsFile"),
      { jlm::util::Statistics::Id::SteensgaardAnalysis });

  JlmOptCommandLineOptions commandLineOptions(
      jlm::util::filepath("inputFile.ll"),
      jlm::util::filepath("outputFile.ll"),
      JlmOptCommandLineOptions::OutputFormat::Llvm,
      statisticsCollectorSettings,
      { JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
        JlmOptCommandLineOptions::OptimizationId::LoopUnrolling });

  JlmOptCommand command("jlm-opt", commandLineOptions);

  // Act
  auto receivedCommandLine = command.ToString();

  // Assert
  std::string expectedCommandLine = jlm::util::strfmt(
      "jlm-opt ",
      "--llvm ",
      "--DeadNodeElimination --LoopUnrolling ",
      "-s " + expectedStatisticsDir + " ",
      "--print-steensgaard-analysis ",
      "-o outputFile.ll ",
      "inputFile.ll");

  assert(receivedCommandLine == expectedCommandLine);
}

static int
TestJlmOptCommand()
{
  TestStatistics();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlmOptCommand", TestJlmOptCommand)
