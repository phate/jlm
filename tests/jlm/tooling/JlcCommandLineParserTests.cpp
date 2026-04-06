/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/tooling/CommandLine.hpp>

#include <cstring>

static const jlm::tooling::JlcCommandLineOptions &
ParseCommandLineArguments(const std::vector<std::string> & commandLineArguments)
{
  std::vector<const char *> cStrings;
  for (const auto & commandLineArgument : commandLineArguments)
  {
    cStrings.push_back(commandLineArgument.c_str());
  }

  static jlm::tooling::JlcCommandLineParser commandLineParser;
  return commandLineParser.ParseCommandLineArguments(
      static_cast<int>(cStrings.size()),
      cStrings.data());
}

TEST(JlcCommandLineParserTests, Test1)
{
  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "-c", "-o", "foo.o", "foo.c" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.Compilations_.size(), 1u);
  auto & compilation = commandLineOptions.Compilations_[0];

  EXPECT_EQ(compilation.RequiresLinking(), false);
  EXPECT_EQ(compilation.OutputFile(), "foo.o");
}

TEST(JlcCommandLineParserTests, Test2)
{
  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "-o", "foobar", "/tmp/f1.o" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.Compilations_.size(), 1u);
  EXPECT_EQ(commandLineOptions.OutputFile_, "foobar");

  auto & compilation = commandLineOptions.Compilations_[0];
  EXPECT_FALSE(compilation.RequiresParsing());
  EXPECT_FALSE(compilation.RequiresOptimization());
  EXPECT_FALSE(compilation.RequiresAssembly());
  EXPECT_TRUE(compilation.RequiresLinking());
}

TEST(JlcCommandLineParserTests, Test3)
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "-O", "foobar.c" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.OptimizationLevel_, JlcCommandLineOptions::OptimizationLevel::O0);
}

TEST(JlcCommandLineParserTests, Test4)
{
  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "foobar.c", "-c" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.Compilations_.size(), 1u);

  auto & compilation = commandLineOptions.Compilations_[0];
  EXPECT_FALSE(compilation.RequiresLinking());
  EXPECT_EQ(compilation.OutputFile(), "foobar.o");
}

TEST(JlcCommandLineParserTests, TestJlmOptOptimizations)
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments(
      { "jlc", "foobar.c", "-JCommonNodeElimination", "-JDeadNodeElimination" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.JlmOptOptimizations_.size(), 2u);
  EXPECT_EQ(
      commandLineOptions.JlmOptOptimizations_[0],
      JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  EXPECT_EQ(
      commandLineOptions.JlmOptOptimizations_[1],
      JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);
}

TEST(JlcCommandLineParserTests, TestFalseJlmOptOptimization)
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "-JFoobar", "foobar.c" });

  // Act & Assert
  EXPECT_THROW(ParseCommandLineArguments(commandLineArguments), CommandLineParser::Exception);
}

TEST(JlcCommandLineParserTests, TestJlmOptPassStatistics)
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc",
                                                  "--JlmOptPassStatistics=print-aggregation-time",
                                                  "--JlmOptPassStatistics=print-andersen-analysis",
                                                  "foobar.c" });

  jlm::util::HashSet expectedStatistics(
      { jlm::util::Statistics::Id::Aggregation, jlm::util::Statistics::Id::AndersenAnalysis });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  EXPECT_EQ(commandLineOptions.JlmOptPassStatistics_, expectedStatistics);
}
