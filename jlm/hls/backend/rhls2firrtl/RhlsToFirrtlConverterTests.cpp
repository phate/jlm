/*
 * Copyright 2026 Nordic University of Technology
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/lambda.hpp>

#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Instructions.h>

#include <fstream>
#include <regex>
#include <string>

using namespace jlm::hls;

// Helper function to check if a string contains a substring
static bool
ContainsSubstring(const std::string & str, const std::string & substr)
{
  return str.find(substr) != std::string::npos;
}

// Helper function to count occurrences of a substring
static size_t
CountSubstring(const std::string & str, const std::string & substr)
{
  size_t count = 0;
  size_t pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos)
  {
    ++count;
    pos += substr.length();
  }
  return count;
}

// Helper to extract FIRRTL output from a simple lambda function
static std::string
GenerateFirrtlFromLambda(
    const std::string & name,
    const std::vector<std::shared_ptr<const jlm::rvsdg::Type>> & arguments,
    const std::vector<std::shared_ptr<const jlm::rvsdg::Type>> & results)
{
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(arguments, results),
          name,
          jlm::llvm::Linkage::externalLinkage));

  // Finalize the lambda
  std::vector<jlm::rvsdg::Output *> resultOutputs;
  for (size_t i = 0; i < results.size(); ++i)
  {
    resultOutputs.push_back(lambdaNode->GetFunctionArguments()[i]);
  }
  auto f = lambdaNode->finalize(resultOutputs);
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Convert to FIRRTL
  RhlsToFirrtlConverter converter;
  return converter.ToString(rm);
}

// Test that a simple module with no inputs/outputs generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestSimpleModuleNoPorts)
{
  // Arrange
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments;
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results;

  // Act
  auto firrtl = GenerateFirrtlFromLambda("simple_no_ports", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
}

// Test that a simple module with inputs generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestSimpleModuleWithInputs)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments = { bitType };
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results;

  // Act
  auto firrtl = GenerateFirrtlFromLambda("simple_inputs", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "input")); // Should have input ports
}

// Test that a simple module with outputs generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestSimpleModuleWithOutputs)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments;
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results = { bitType };

  // Act
  auto firrtl = GenerateFirrtlFromLambda("simple_outputs", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "output")); // Should have output ports
}

// Test that a module with both inputs and outputs generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestSimpleModuleWithInOut)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments = { bitType };
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results = { bitType };

  // Act
  auto firrtl = GenerateFirrtlFromLambda("simple_inout", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "input"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "output"));
}

// Test that FIRRTL output has valid structure
TEST(RhlsToFirrtlConverterTests, TestFirrtlStructure)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments = { bitType };
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results = { bitType };

  // Act
  auto firrtl = GenerateFirrtlFromLambda("struct_test", arguments, results);

  // Assert - basic FIRRTL structure
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module")) << "Missing 'module' keyword";

  // Check for circuit name
  std::regex circuitRegex(R"(circuit\s+(\w+)\s*:)");
  EXPECT_TRUE(std::regex_search(firrtl, circuitRegex)) << "Invalid circuit declaration";

  // Check for module name
  std::regex moduleRegex(R"(module\s+(\w+)\s*\()");
  EXPECT_TRUE(std::regex_search(firrtl, moduleRegex)) << "Invalid module declaration";
}

// Test that a module with multiple inputs generates correct FIRRTL
TEST(RhlsToFirrtlConverterTests, TestMultipleInputs)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments = { bitType, bitType, bitType };
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results;

  // Act
  auto firrtl = GenerateFirrtlFromLambda("multi_inputs", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  // Count occurrences of "input" - should have at least 3 inputs
  EXPECT_GE(CountSubstring(firrtl, "input"), 3) << "Expected at least 3 input ports";
}

// Test that a module with multiple outputs generates correct FIRRTL
TEST(RhlsToFirrtlConverterTests, TestMultipleOutputs)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments;
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results = { bitType, bitType, bitType };

  // Act
  auto firrtl = GenerateFirrtlFromLambda("multi_outputs", arguments, results);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  // Count occurrences of "output" - should have at least 3 outputs
  EXPECT_GE(CountSubstring(firrtl, "output"), 3) << "Expected at least 3 output ports";
}

// Test that a module with combinational logic (add operation) generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestCombinationalLogicAdd)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "add_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an add operation in the lambda body
  auto & addNode = jlm::llvm::IntegerAddOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ addNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
  // Add operation should generate FIRRTL add
  EXPECT_TRUE(ContainsSubstring(firrtl, "+")) << "Expected add operation in FIRRTL";
}

// Test that a module with integer subtract generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestCombinationalLogicSub)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "sub_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add a subtract operation in the lambda body
  auto & subNode = jlm::llvm::IntegerSubOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ subNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
}

// Test that a module with integer AND generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestCombinationalLogicAnd)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "and_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an AND operation in the lambda body
  auto & andNode = jlm::llvm::IntegerAndOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ andNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
}

// Test that a module with integer XOR generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestCombinationalLogicXor)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "xor_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an XOR operation in the lambda body
  auto & xorNode = jlm::llvm::IntegerXorOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ xorNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
}

// Test that a module with integer OR generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestCombinationalLogicOr)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "or_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an OR operation in the lambda body
  auto & orNode = jlm::llvm::IntegerOrOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ orNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
}

// Test that a module with comparison operations generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestComparisonOperations)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "cmp_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an equality comparison in the lambda body
  auto & cmpNode = jlm::llvm::IntegerEqOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ cmpNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "module"));
  EXPECT_TRUE(ContainsSubstring(firrtl, "==")) << "Expected equality comparison in FIRRTL";
}

// Test that FIRRTL output for a simple add function contains expected structure
TEST(RhlsToFirrtlConverterTests, TestAddFunctionStructure)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { bitType }),
          "add_func",
          jlm::llvm::Linkage::externalLinkage));

  // Add an add operation in the lambda body
  auto & addNode = jlm::llvm::IntegerAddOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda
  auto f = lambdaNode->finalize({ addNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert - check for FIRRTL specific syntax
  // FIRRTL modules have specific structure with ports
  EXPECT_TRUE(ContainsSubstring(firrtl, "input")) << "Expected 'input' in FIRRTL module";
  EXPECT_TRUE(ContainsSubstring(firrtl, "output")) << "Expected 'output' in FIRRTL module";
  EXPECT_TRUE(ContainsSubstring(firrtl, "wire")) << "Expected 'wire' in FIRRTL module";
}

// Test that FIRRTL output preserves module names
TEST(RhlsToFirrtlConverterTests, TestModuleNaming)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments = { bitType };
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results = { bitType };

  // Act
  auto firrtl = GenerateFirrtlFromLambda("my_module_name", arguments, results);

  // Assert - module name should be in FIRRTL
  EXPECT_TRUE(ContainsSubstring(firrtl, "my_module_name"))
      << "Expected module name 'my_module_name' in FIRRTL output";
}