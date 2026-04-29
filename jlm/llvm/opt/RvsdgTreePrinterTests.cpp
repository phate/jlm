/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/Statistics.hpp>

#include <fstream>

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
RunAndExtractFile(jlm::llvm::LlvmRvsdgModule & module, jlm::llvm::RvsdgTreePrinter & printer)
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

TEST(RvsdgTreePrinterTests, PrintRvsdgTree)
{
  using namespace jlm::llvm;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = LlvmRvsdgModule::Create(FilePath(""), "", "");

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
  auto expectedTree =
      "{\"StructuralNodes\":[{\"DebugString\":\"LAMBDA[f]\",\"Subregions\":[{}]}]}\n";

  EXPECT_EQ(tree, expectedTree);
}

TEST(RvsdgTreePrinterTests, PrintNumRvsdgNodesAnnotation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(FilePath(""), "", "");
  auto rootRegion = &rvsdgModule->Rvsdg().GetRootRegion();

  auto structuralNode = TestStructuralNode::create(rootRegion, 2);
  TestOperation::createNode(structuralNode->subregion(0), {}, {});
  TestOperation::createNode(structuralNode->subregion(1), {}, {});

  TestOperation::createNode(rootRegion, {}, {});

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumRvsdgNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree =
      "{\"NumRvsdgNodes\":2,\"StructuralNodes\":[{\"DebugString\":\"TestStructuralOperation\","
      "\"NumRvsdgNodes\":2,\"Subregions\":[{\"NumRvsdgNodes\":1},{\"NumRvsdgNodes\":1}]}]}\n";

  EXPECT_EQ(tree, expectedTree);
}

TEST(RvsdgTreePrinterTests, PrintNumLoadNodesAnnotation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto valueType = jlm::rvsdg::TestType::createValueType();

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  auto rootRegion = &rvsdg.GetRootRegion();

  auto & address = jlm::rvsdg::GraphImport::Create(rvsdg, pointerType, "a");
  auto & memoryState = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "m");

  auto structuralNode = TestStructuralNode::create(rootRegion, 3);
  const auto addressInput = structuralNode->addInputWithArguments(address);
  const auto memoryStateInput = structuralNode->addInputWithArguments(memoryState);
  LoadNonVolatileOperation::Create(
      addressInput.argument[0],
      { memoryStateInput.argument[0] },
      valueType,
      4);
  TestOperation::createNode(structuralNode->subregion(1), {}, {});
  LoadNonVolatileOperation::Create(
      addressInput.argument[2],
      { memoryStateInput.argument[2] },
      valueType,
      4);

  TestOperation::createNode(rootRegion, {}, {});

  RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumLoadNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  auto expectedTree = "{\"NumLoadNodes\":0,\"StructuralNodes\":[{\"DebugString\":"
                      "\"TestStructuralOperation\",\"NumLoadNodes\":2,\"Subregions\":"
                      "[{\"NumLoadNodes\":1},{\"NumLoadNodes\":0},{\"NumLoadNodes\":1}]}]}\n";

  EXPECT_EQ(tree, expectedTree);
}

TEST(RvsdgTreePrinterTests, PrintNumMemoryStateInputsOutputsAnnotation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = jlm::rvsdg::TestType::createValueType();

  auto rvsdgModule = jlm::llvm::LlvmRvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto & x = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & y = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");

  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);
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
      "{\"NumMemoryStateTypeArguments\":1,\"NumMemoryStateTypeResults\":1,\"StructuralNodes\":"
      "[{\"DebugString\":\"TestStructuralOperation\",\"NumMemoryStateTypeInputs\":1,"
      "\"NumMemoryStateTypeOutputs\":1,\"Subregions\":[{\"NumMemoryStateTypeArguments\":"
      "1,\"NumMemoryStateTypeResults\":1},{\"NumMemoryStateTypeArguments\":"
      "1,\"NumMemoryStateTypeResults\":1}]}]}\n";

  EXPECT_EQ(tree, expectedTree);
}

TEST(RvsdgTreePrinterTests, printDebugIds)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();

  auto rvsdgModule = LlvmRvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  TestStructuralNode::create(&rvsdg.GetRootRegion(), 2);

  const RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::DebugIds });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  const auto expectedTree =
      "{\"RegionId\":0,\"StructuralNodes\":[{\"DebugString\":\"TestStructuralOperation\","
      "\"NodeId\":0,\"Subregions\":[{\"RegionId\":1},{\"RegionId\":2}]}]}\n";

  EXPECT_EQ(tree, expectedTree);
}

TEST(RvsdgTreePrinterTests, printAllocaNodes)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto structType = StructType::CreateIdentified({ valueType, valueType }, false);

  auto rvsdgModule = LlvmRvsdgModule::Create(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto & one = IntegerConstantOperation::Create(rvsdg.GetRootRegion(), 32, 1);

  auto allocaResults = AllocaOperation::create(valueType, one.output(0), 4);
  auto aggregatedAllocaResults = AllocaOperation::create(structType, one.output(0), 4);

  const RvsdgTreePrinter::Configuration configuration(
      { RvsdgTreePrinter::Configuration::Annotation::NumAggregateAllocaNodes,
        RvsdgTreePrinter::Configuration::Annotation::NumAllocaNodes });
  RvsdgTreePrinter printer(configuration);

  // Act
  auto tree = RunAndExtractFile(*rvsdgModule, printer);
  std::cout << tree;

  // Assert
  const auto expectedTree = "{\"NumAllocaNodes\":2,\"NumAggregateAllocaNodes\":1}\n";

  EXPECT_EQ(tree, expectedTree);
}
