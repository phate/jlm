/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/tooling/Command.hpp>
#include <jlm/llvm/tooling/CommandLine.hpp>
#include <jlm/llvm/tooling/CommandGraphGenerator.hpp>

#include <cassert>

static void
Test1()
{
  using namespace jlm;

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
  using namespace jlm;

  /*
   * Arrange
   */
  jlm::JlcCommandLineOptions commandLineOptions;
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
  using namespace jlm;

  /*
   * Arrange
   */
  jlm::JlcCommandLineOptions commandLineOptions;
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
  commandLineOptions.JlmOptOptimizations_.push_back("cne");
  commandLineOptions.JlmOptOptimizations_.push_back("dne");

  /*
   * Act
   */
  auto commandGraph = JlcCommandGraphGenerator::Generate(commandLineOptions);

  /*
   * Assert
   */
  auto & clangCommandNode = (*commandGraph->GetEntryNode().OutgoingEdges().begin()).GetSink();
  auto & jlmOptCommandNode = (clangCommandNode.OutgoingEdges().begin())->GetSink();
  auto command = dynamic_cast<const jlm::JlmOptCommand*>(&jlmOptCommandNode.GetCommand());
  auto optimizations = command->Optimizations();
  assert(optimizations.size() == 2);
  assert(optimizations[0] == jlm::JlmOptCommand::Optimization::CommonNodeElimination && \
         optimizations[1] == jlm::JlmOptCommand::Optimization::DeadNodeElimination);
}

static int
Test()
{
  Test1();
  Test2();
  TestJlmOptOptimizations();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/tooling/TestJlcCommandGraphGenerator", Test)
