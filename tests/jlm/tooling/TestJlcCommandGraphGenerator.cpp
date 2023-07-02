/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandLine.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>

#include <cassert>

static void
Test1()
{
  using namespace jlm::tooling;

  /*
   * Arrange
   */
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back({
                                               {"foo.c"},
                                               {"foo.d"},
                                               {"foo.o"},
                                               "foo.o",
                                               true,
                                               true,
                                               true,
                                               false});

  /*
   * Act
   */
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  /*
   * Assert
   */
  auto & commandNode = (*commandGraph->GetExitNode().IncomingEdges().begin()).GetSource();
  auto command = dynamic_cast<const LlcCommand*>(&commandNode.GetCommand());
  assert(command && command->OutputFile() == "foo.o");
}

static void
Test2()
{
  using namespace jlm::tooling;

  /*
   * Arrange
   */
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back({
                                               {"foo.o"},
                                               {""},
                                               {"foo.o"},
                                               "foo.o",
                                               false,
                                               false,
                                               false,
                                               true});
  commandLineOptions.OutputFile_ = {"foobar"};

  /*
   * Act
   */
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  /*
   * Assert
   */
  assert(commandGraph->NumNodes() == 3);

  auto & commandNode = (*commandGraph->GetExitNode().IncomingEdges().begin()).GetSource();
  auto command = dynamic_cast<const ClangCommand*>(&commandNode.GetCommand());
  assert(command->InputFiles()[0] == "foo.o" && command->OutputFile() == "foobar");
}

static void
TestJlmOptOptimizations()
{
  using namespace jlm::tooling;

  /*
   * Arrange
   */
  JlcCommandLineOptions commandLineOptions;
  commandLineOptions.Compilations_.push_back({
                                               {"foo.o"},
                                               {""},
                                               {"foo.o"},
                                               "foo.o",
                                               true,
                                               true,
                                               true,
                                               true});
  commandLineOptions.OutputFile_ = {"foobar"};
  commandLineOptions.JlmOptOptimizations_.push_back(JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  commandLineOptions.JlmOptOptimizations_.push_back(JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);

  /*
   * Act
   */
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  /*
   * Assert
   */
  auto & clangCommandNode = (*commandGraph->GetEntryNode().OutgoingEdges().begin()).GetSink();
  auto & jlmOptCommandNode = (clangCommandNode.OutgoingEdges().begin())->GetSink();
  auto & jlmOptCommand = *dynamic_cast<const JlmOptCommand*>(&jlmOptCommandNode.GetCommand());
  auto & optimizations = jlmOptCommand.GetCommandLineOptions().GetOptimizationIds();

  assert(optimizations.size() == 2);
  assert(optimizations[0] == JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination);
  assert(optimizations[1] == JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination);
}

static int
Test()
{
  Test1();
  Test2();
  TestJlmOptOptimizations();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlcCommandGraphGenerator", Test)
