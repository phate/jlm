/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/tooling/CommandLine.hpp>

#include <cassert>
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

static void
Test1()
{
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({ "jlc", "-c", "-o", "foo.o", "foo.c" });

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.Compilations_.size() == 1);
  auto & compilation = commandLineOptions.Compilations_[0];

  assert(compilation.RequiresLinking() == false);
  assert(compilation.OutputFile() == "foo.o");
}

static void
Test2()
{
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({ "jlc", "-o", "foobar", "/tmp/f1.o" });

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.Compilations_.size() == 1);
  assert(commandLineOptions.OutputFile_ == "foobar");

  auto & compilation = commandLineOptions.Compilations_[0];
  assert(compilation.RequiresParsing() == false);
  assert(compilation.RequiresOptimization() == false);
  assert(compilation.RequiresAssembly() == false);
  assert(compilation.RequiresLinking() == true);
}

static void
Test3()
{
  using namespace jlm::tooling;

  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({ "jlc", "-O", "foobar.c" });

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.OptimizationLevel_ == JlcCommandLineOptions::OptimizationLevel::O0);
}

static void
Test4()
{
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({ "jlc", "foobar.c", "-c" });

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.Compilations_.size() == 1);

  auto & compilation = commandLineOptions.Compilations_[0];
  assert(compilation.RequiresLinking() == false);
  assert(compilation.OutputFile() == "foobar.o");
}

static void
TestJlmOptOptimizations()
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments(
      { "jlc", "foobar.c", "-JCommonNodeElimination", "-JDeadNodeElimination" });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  assert(commandLineOptions.JlmOptOptimizations_.size() == 2);
  assert(
      commandLineOptions.JlmOptOptimizations_[0]
      == JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  assert(
      commandLineOptions.JlmOptOptimizations_[1]
      == JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);
}

static void
TestFalseJlmOptOptimization()
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc", "-JFoobar", "foobar.c" });

  // Act & Assert
  bool exceptionThrown = false;
  try
  {
    ParseCommandLineArguments(commandLineArguments);
  }
  catch (CommandLineParser::Exception &)
  {
    exceptionThrown = true;
  }

  assert(exceptionThrown);
}

static void
TestJlmOptPassStatistics()
{
  using namespace jlm::tooling;

  // Arrange
  std::vector<std::string> commandLineArguments({ "jlc",
                                                  "--JlmOptPassStatistics=print-aggregation-time",
                                                  "--JlmOptPassStatistics=print-andersen-analysis",
                                                  "foobar.c" });

  jlm::util::HashSet<jlm::util::Statistics::Id> expectedStatistics(
      { jlm::util::Statistics::Id::Aggregation, jlm::util::Statistics::Id::AndersenAnalysis });

  // Act
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  // Assert
  assert(commandLineOptions.JlmOptPassStatistics_ == expectedStatistics);
}

static void
Test()
{
  Test1();
  Test2();
  Test3();
  Test4();
  TestJlmOptOptimizations();
  TestFalseJlmOptOptimization();
  TestJlmOptPassStatistics();
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlcCommandLineParser", Test)
