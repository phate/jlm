/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <cassert>

static void
TestJlcCompiling()
{
  using namespace jlm::tooling;
  using namespace jlm::util;

  // Arrange
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back({ FilePath("foo.c"),
                                               FilePath("foo.d"),
                                               FilePath("foo.o"),
                                               "foo.o",
                                               true,
                                               true,
                                               true,
                                               false });

  // Act
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  // Assert
  assert(commandGraph->NumNodes() == 5);
  auto & commandNode = commandGraph->GetExitNode().IncomingEdges().begin()->GetSource();
  auto command = dynamic_cast<const LlcCommand *>(&commandNode.GetCommand());
  assert(command && command->OutputFile() == "foo.o");
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tooling/TestJlcCommandGraphGenerator-TestJlcCompiling",
    TestJlcCompiling);

static void
TestJlcLinking()
{
  using namespace jlm::tooling;
  using namespace jlm::util;

  // Arrange
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back(
      { FilePath("foo.o"), FilePath(""), FilePath("foo.o"), "foo.o", false, false, false, true });
  commandLineOptions.OutputFile_ = FilePath("foobar");

  // Act
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  // Assert
  assert(commandGraph->NumNodes() == 3);
  auto & commandNode = commandGraph->GetExitNode().IncomingEdges().begin()->GetSource();
  auto command = dynamic_cast<const ClangCommand *>(&commandNode.GetCommand());
  assert(command->InputFiles()[0] == "foo.o" && command->OutputFile() == "foobar");
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlcCommandGraphGenerator-TestJlcLinking", TestJlcLinking);

static void
TestJlmOptOptimizations()
{
  using namespace jlm::tooling;
  using namespace jlm::util;

  // Arrange
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back(
      { FilePath("foo.o"), FilePath(""), FilePath("foo.o"), "foo.o", true, true, true, true });
  commandLineOptions.OutputFile_ = FilePath("foobar");
  commandLineOptions.JlmOptOptimizations_.push_back(
      JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  commandLineOptions.JlmOptOptimizations_.push_back(
      JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);

  // Act
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  // Assert
  auto & clangCommandNode = commandGraph->GetEntryNode().OutgoingEdges().begin()->GetSink();
  auto & jlmOptCommandNode = clangCommandNode.OutgoingEdges().begin()->GetSink();
  auto & jlmOptCommand = *dynamic_cast<const JlmOptCommand *>(&jlmOptCommandNode.GetCommand());
  auto & optimizations = jlmOptCommand.GetCommandLineOptions().GetOptimizationIds();

  assert(optimizations.size() == 2);
  assert(optimizations[0] == JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  assert(optimizations[1] == JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tooling/TestJlcCommandGraphGenerator-TestJlmOptOptimizations",
    TestJlmOptOptimizations);

static void
TestJlmOptStatistics()
{
  using namespace jlm::util;

  // Arrange
  HashSet<Statistics::Id> expectedStatistics(
      { Statistics::Id::Aggregation, Statistics::Id::AndersenAnalysis });

  jlm::tooling::JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back(
      { FilePath("foo.o"), FilePath(""), FilePath("foo.o"), "foo.o", true, true, true, true });
  commandLineOptions.OutputFile_ = FilePath("foobar");
  commandLineOptions.JlmOptPassStatistics_ = expectedStatistics;

  // Act
  auto commandGraph = jlm::tooling::JlcCommandGraphGenerator::Generate(commandLineOptions);

  // Assert
  auto & clangCommandNode = commandGraph->GetEntryNode().OutgoingEdges().begin()->GetSink();
  auto & jlmOptCommandNode = clangCommandNode.OutgoingEdges().begin()->GetSink();
  auto & jlmOptCommand =
      *dynamic_cast<const jlm::tooling::JlmOptCommand *>(&jlmOptCommandNode.GetCommand());
  auto & statisticsCollectorSettings =
      jlmOptCommand.GetCommandLineOptions().GetStatisticsCollectorSettings();

  assert(statisticsCollectorSettings.GetDemandedStatistics() == expectedStatistics);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tooling/TestJlcCommandGraphGenerator-TestJlmOptStatistics",
    TestJlmOptStatistics);
