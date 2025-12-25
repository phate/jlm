/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/util/Statistics.hpp>

#include <fstream>
#include <jlm/rvsdg/TestType.hpp>

static std::string
ReadFile(const jlm::util::FilePath & outputFilePath)
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

  const auto tmpDir = FilePath::TempDirectoryPath();
  StatisticsCollectorSettings settings({}, tmpDir, "TestTreePrinter");
  StatisticsCollector collector(settings);

  printer.Run(module, collector);

  auto fileName = tmpDir.Join("TestTreePrinter-" + settings.GetUniqueString() + "-rvsdgTree-0.txt");
  auto result = ReadFile(fileName);

  // Cleanup
  std::filesystem::remove(fileName.to_str());

  return result;
}

static void
PrintRvsdgTree()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(FilePath(""), "", "");

  auto functionType = jlm::rvsdg::FunctionType::Create(
      { MemoryStateType::Create() },
      { MemoryStateType::Create() });
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto lambdaOutput = lambda->finalize({ lambda->GetFunctionArguments()[0] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

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
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/RvsdgTreePrinterTests-PrintRvsdgTree", PrintRvsdgTree)

static void
PrintNumRvsdgNodesAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = RvsdgModule::Create(FilePath(""), "", "");
  auto rootRegion = &rvsdgModule->Rvsdg().GetRootRegion();

  auto structuralNode = jlm::tests::TestStructuralNode::create(rootRegion, 2);
  jlm::tests::TestOperation::create(structuralNode->subregion(0), {}, {});
  jlm::tests::TestOperation::create(structuralNode->subregion(1), {}, {});

  jlm::tests::TestOperation::create(rootRegion, {}, {});

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumRvsdgNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree = "RootRegion NumRvsdgNodes:2\n"
                      "-TestStructuralOperation NumRvsdgNodes:2\n"
                      "--Region[0] NumRvsdgNodes:1\n"
                      "--Region[1] NumRvsdgNodes:1\n\n";

  assert(tree == expectedTree);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/RvsdgTreePrinterTests-PrintNumRvsdgNodesAnnotation",
    PrintNumRvsdgNodesAnnotation)

static void
PrintNumLoadNodesAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = jlm::rvsdg::TestType::Create(jlm::rvsdg::TypeKind::Value);

  auto rvsdgModule = RvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  auto rootRegion = &rvsdg.GetRootRegion();

  auto & address = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "a");
  auto & memoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "m");

  auto structuralNode = jlm::tests::TestStructuralNode::create(rootRegion, 3);
  const auto addressInput = structuralNode->addInputWithArguments(address);
  const auto memoryStateInput = structuralNode->addInputWithArguments(memoryState);
  LoadNonVolatileOperation::Create(
      addressInput.argument[0],
      { memoryStateInput.argument[0] },
      valueType,
      4);
  jlm::tests::TestOperation::create(structuralNode->subregion(1), {}, {});
  LoadNonVolatileOperation::Create(
      addressInput.argument[2],
      { memoryStateInput.argument[2] },
      valueType,
      4);

  jlm::tests::TestOperation::create(rootRegion, {}, {});

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumLoadNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree = "RootRegion NumLoadNodes:0\n"
                      "-TestStructuralOperation NumLoadNodes:2\n"
                      "--Region[0] NumLoadNodes:1\n"
                      "--Region[1] NumLoadNodes:0\n"
                      "--Region[2] NumLoadNodes:1\n\n";

  assert(tree == expectedTree);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/RvsdgTreePrinterTests-PrintNumLoadNodesAnnotation",
    PrintNumLoadNodesAnnotation)

static void
PrintNumMemoryStateInputsOutputsAnnotation()
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::Create(jlm::rvsdg::TypeKind::Value);

  auto rvsdgModule = RvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto & x = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & y = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");

  auto structuralNode = jlm::tests::TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);
  const auto inputVarX = structuralNode->addInputWithArguments(x);
  const auto inputVarY = structuralNode->addInputWithArguments(y);

  auto outputVarX =
      structuralNode->addOutputWithResults({ inputVarX.argument[0], inputVarX.argument[1] });
  auto outputVarY =
      structuralNode->addOutputWithResults({ inputVarY.argument[0], inputVarY.argument[1] });

  jlm::rvsdg::GraphExport::Create(*outputVarX.output, "x");
  jlm::rvsdg::GraphExport::Create(*outputVarY.output, "y");

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumMemoryStateInputsOutputs });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree =
      "RootRegion NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n"
      "-TestStructuralOperation NumMemoryStateTypeInputs:1 NumMemoryStateTypeOutputs:1\n"
      "--Region[0] NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n"
      "--Region[1] NumMemoryStateTypeArguments:1 NumMemoryStateTypeResults:1\n\n";

  assert(tree == expectedTree);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/RvsdgTreePrinterTests-PrintNumMemoryStateInputsOutputsAnnotation",
    PrintNumMemoryStateInputsOutputsAnnotation)
