/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>

#include <fstream>

static int
PrintRvsdgTree()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  std::string fileName = "PrintTreeTest";
  auto rvsdgModule = RvsdgModule::Create({ fileName }, "", "");

  auto functionType =
      FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });
  auto lambda = lambda::node::create(
      rvsdgModule->Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);
  auto lambdaOutput = lambda->finalize({ lambda->fctargument(0) });
  GraphExport::Create(*lambdaOutput, "f");

  auto tempDirectory = std::filesystem::temp_directory_path();
  RvsdgTreePrinter::Configuration configuration({ tempDirectory });
  RvsdgTreePrinter printer(configuration);

  // Act
  printer.run(*rvsdgModule);

  // Assert
  auto outputFilePath = tempDirectory.string() + "/" + fileName + "-rvsdgTree-0";

  std::ifstream file(outputFilePath);
  std::stringstream buffer;
  buffer << file.rdbuf();

  assert(buffer.str() == "RootRegion\n-LAMBDA[f]\n--Region[0]\n\n");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/RvsdgTreePrinterTests-PrintRvsdgTree", PrintRvsdgTree)
