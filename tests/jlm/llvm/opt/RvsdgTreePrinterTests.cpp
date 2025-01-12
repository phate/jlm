/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/util/Statistics.hpp>

#include <fstream>

static std::string
ReadFile(const jlm::util::filepath & outputFilePath)
{
  std::ifstream file(outputFilePath.to_str());
  std::stringstream buffer;
  buffer << file.rdbuf();

  return buffer.str();
}

/**
 * Runs the given RvsdgTreePrinter on the given module, reads the file back in, and deletes the file
 */
static std::string
RunAndExtractFile(jlm::llvm::RvsdgModule & module, jlm::llvm::RvsdgTreePrinter & printer)
{
  using namespace jlm::util;

  const auto tmpDir = filepath::TempDirectoryPath();
  StatisticsCollectorSettings settings({}, tmpDir, "TestTreePrinter");
  StatisticsCollector collector(settings);

  printer.Run(module, collector);

  auto fileName = tmpDir.Join("TestTreePrinter-" + settings.GetUniqueString() + "-rvsdgTree-0.txt");
  auto result = ReadFile(fileName);

  // Cleanup
  std::filesystem::remove(fileName.to_str());

  return result;
}

static int
PrintRvsdgTree()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");

  auto functionType = jlm::rvsdg::FunctionType::Create(
      { MemoryStateType::Create() },
      { MemoryStateType::Create() });
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  auto lambdaOutput = lambda->finalize({ lambda->GetFunctionArguments()[0] });
  jlm::tests::GraphExport::Create(*lambdaOutput, "f");

  RvsdgTreePrinter::Configuration configuration({});
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree = "RootRegion\n"
                      "-LAMBDA[f]\n"
                      "--Region[0]\n\n";

  assert(tree == expectedTree);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/RvsdgTreePrinterTests-PrintRvsdgTree", PrintRvsdgTree)

static int
PrintNumRvsdgNodesAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");
  auto rootRegion = &rvsdgModule->Rvsdg().GetRootRegion();

  auto structuralNode = jlm::tests::structural_node::create(rootRegion, 2);
  jlm::tests::test_op::create(structuralNode->subregion(0), {}, {});
  jlm::tests::test_op::create(structuralNode->subregion(1), {}, {});

  jlm::tests::test_op::create(rootRegion, {}, {});

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumRvsdgNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree = "RootRegion NumRvsdgNodes:2\n"
                      "-STRUCTURAL_TEST_NODE NumRvsdgNodes:2\n"
                      "--Region[0] NumRvsdgNodes:1\n"
                      "--Region[1] NumRvsdgNodes:1\n\n";

  assert(tree == expectedTree);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/RvsdgTreePrinterTests-PrintNumRvsdgNodesAnnotation",
    PrintNumRvsdgNodesAnnotation)

static int
PrintNumMemoryStateInputsOutputsAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::tests::valuetype::Create();

  auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto & x = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & y = jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto structuralNode = jlm::tests::structural_node::create(&rvsdg.GetRootRegion(), 2);
  auto & ix = structuralNode->AddInputWithArguments(x);
  auto & iy = structuralNode->AddInputWithArguments(y);

  auto & ox = structuralNode->AddOutputWithResults({ &ix.Argument(0), &ix.Argument(1) });
  auto & oy = structuralNode->AddOutputWithResults({ &iy.Argument(0), &iy.Argument(1) });

  jlm::tests::GraphExport::Create(ox, "x");
  jlm::tests::GraphExport::Create(oy, "y");

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumMemoryStateInputsOutputs });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree =
      "RootRegion NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n"
      "-STRUCTURAL_TEST_NODE NumMemoryStateTypeInputs:1 NumMemoryStateTypeOutputs:1\n"
      "--Region[0] NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n"
      "--Region[1] NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n\n";

  assert(tree == expectedTree);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/RvsdgTreePrinterTests-PrintNumMemoryStateInputsOutputsAnnotation",
    PrintNumMemoryStateInputsOutputsAnnotation)
