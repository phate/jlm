/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/tooling/CommandLine.hpp>

#include <cassert>
#include <cstring>

static const jlm::JlcCommandLineOptions &
ParseCommandLineArguments(const std::vector<std::string> & commandLineArguments)
{
  std::vector<char*> array;
  for (const auto & commandLineArgument : commandLineArguments) {
    array.push_back(new char[commandLineArgument.size() + 1]);
    strncpy(array.back(), commandLineArgument.data(), commandLineArgument.size());
    array.back()[commandLineArgument.size()] = '\0';
  }

  static jlm::JlcCommandLineParser commandLineParser;
  auto & commandLineOptions = commandLineParser.ParseCommandLineArguments(
    static_cast<int>(array.size()),
    &array[0]);

  for (const auto & ptr : array)
    delete[] ptr;

  return commandLineOptions;
}

static void
Test1()
{
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({"jlc", "-c", "-o", "foo.o", "foo.c"});

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
  std::vector<std::string> commandLineArguments({"jlc", "-o", "foobar", "/tmp/f1.o"});

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
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({"jlc", "-O", "foobar.c"});

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.OptimizationLevel_ == jlm::JlcCommandLineOptions::OptimizationLevel::O0);
}

static void
Test4()
{
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({"jlc", "foobar.c", "-c"});

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
  /*
   * Arrange
   */
  std::vector<std::string> commandLineArguments({"jlc", "foobar.c", "-Jcne", "-Jdne"});

  /*
   * Act
   */
  auto & commandLineOptions = ParseCommandLineArguments(commandLineArguments);

  /*
   * Assert
   */
  assert(commandLineOptions.JlmOptOptimizations_[0].compare("cne") == 0 && \
         commandLineOptions.JlmOptOptimizations_[1].compare("dne") == 0);
}

static int
Test()
{
  Test1();
  Test2();
  Test3();
  Test4();
  TestJlmOptOptimizations();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/tooling/TestJlcCommandLineParser", Test)
