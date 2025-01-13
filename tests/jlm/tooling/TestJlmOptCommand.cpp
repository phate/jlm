/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/util/strfmt.hpp>

#include <fstream>

static void
TestStatistics()
{
  using namespace jlm::llvm;
  using namespace jlm::tooling;
  using namespace jlm::util;

  // Arrange
  std::string expectedStatisticsDir = "/myStatisticsDir/";

  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::SteensgaardAnalysis },
      expectedStatisticsDir,
      "inputFile");

  JlmOptCommandLineOptions commandLineOptions(
      jlm::util::filepath("inputFile.ll"),
      JlmOptCommandLineOptions::InputFormat::Llvm,
      jlm::util::filepath("outputFile.ll"),
      JlmOptCommandLineOptions::OutputFormat::Llvm,
      statisticsCollectorSettings,
      RvsdgTreePrinter::Configuration({}),
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

  std::cout << "rec: " << receivedCommandLine << std::endl;
  std::cout << "exp: " << expectedCommandLine << std::endl;

  assert(receivedCommandLine == expectedCommandLine);
}

static int
TestJlmOptCommand()
{
  TestStatistics();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlmOptCommand", TestJlmOptCommand)

static int
OptimizationIdToOptimizationTranslation()
{
  using namespace jlm::llvm;
  using namespace jlm::tooling;
  using namespace jlm::util;

  // Arrange
  std::vector<JlmOptCommandLineOptions::OptimizationId> optimizationIds;
  for (size_t n =
           static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::FirstEnumValue) + 1;
       n != static_cast<std::size_t>(JlmOptCommandLineOptions::OptimizationId::LastEnumValue);
       n++)
  {
    auto optimizationId = static_cast<JlmOptCommandLineOptions::OptimizationId>(n);
    optimizationIds.emplace_back(optimizationId);
  }

  JlmOptCommandLineOptions options(
      filepath(""),
      JlmOptCommandLineOptions::InputFormat::Llvm,
      filepath(""),
      JlmOptCommandLineOptions::OutputFormat::Llvm,
      StatisticsCollectorSettings(),
      RvsdgTreePrinter::Configuration({}),
      optimizationIds);

  // Act & Assert
  // terminates on unhandled optimization id
  JlmOptCommand command("jlm-opt", options);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tooling/TestJlmOptCommand-OptimizationIdToOptimizationTranslation",
    OptimizationIdToOptimizationTranslation)

static int
PrintRvsdgTreeToFile()
{
  using namespace jlm;

  // Arrange
  util::filepath outputFile("/tmp/RvsdgTree");

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");
  util::StatisticsCollector statisticsCollector;

  // Act
  tooling::JlmOptCommand::PrintRvsdgModule(
      rvsdgModule,
      outputFile,
      tooling::JlmOptCommandLineOptions::OutputFormat::Tree,
      statisticsCollector);

  // Assert
  std::stringstream buffer;
  std::ifstream istream(outputFile.to_str());
  buffer << istream.rdbuf();

  assert(buffer.str() == "RootRegion\n");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/tooling/TestJlmOptCommand-PrintRvsdgTreeToFile", PrintRvsdgTreeToFile)
