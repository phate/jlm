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

static int
Test()
{
  Test1();
  Test2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/tooling/TestJlcCommandGraphGenerator", Test)