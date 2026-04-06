/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <cassert>

TEST(JlcCommandGraphGeneratorTests, TestJlcCompiling)
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
  EXPECT_EQ(commandGraph->NumNodes(), 5u);
  auto & commandNode = commandGraph->GetExitNode().IncomingEdges().begin()->GetSource();
  auto command = dynamic_cast<const LlcCommand *>(&commandNode.GetCommand());
  EXPECT_NE(command, nullptr);
  EXPECT_EQ(command->OutputFile(), "foo.o");
}

TEST(JlcCommandGraphGeneratorTests, TestJlcLinking)
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
  EXPECT_EQ(commandGraph->NumNodes(), 3u);
  auto & commandNode = commandGraph->GetExitNode().IncomingEdges().begin()->GetSource();
  auto command = dynamic_cast<const ClangCommand *>(&commandNode.GetCommand());
  EXPECT_EQ(command->InputFiles()[0], "foo.o");
  EXPECT_EQ(command->OutputFile(), "foobar");
}

TEST(JlcCommandGraphGeneratorTests, TestJlmOptOptimizations)
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

  EXPECT_EQ(optimizations.size(), 2u);
  EXPECT_EQ(optimizations[0], JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  EXPECT_EQ(optimizations[1], JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);
}

TEST(JlcCommandGraphGeneratorTests, TestJlmOptStatistics)
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

  EXPECT_EQ(statisticsCollectorSettings.GetDemandedStatistics(), expectedStatistics);
}
