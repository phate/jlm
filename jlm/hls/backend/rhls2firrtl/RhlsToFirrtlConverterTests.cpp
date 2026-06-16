/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
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

  // Create constant values for results if there are no function arguments
  std::vector<jlm::rvsdg::Output *> resultOutputs;
  if (arguments.empty() && !results.empty())
  {
    // For functions with no inputs but outputs, create constant values in the subregion
    auto & region = *lambdaNode->subregion();
    for (size_t i = 0; i < results.size(); ++i)
    {
      // Create a constant value (0) of the appropriate type
      auto resultType = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(results[i]);
      if (resultType)
      {
        auto & constantNode = jlm::llvm::IntegerConstantOperation::Create(
            region,
            resultType->nbits(),
            0);
        resultOutputs.push_back(constantNode.output(0));
      }
    }
  }
  else
  {
    // For functions with inputs, use function arguments as results
    for (size_t i = 0; i < results.size(); ++i)
    {
      resultOutputs.push_back(lambdaNode->GetFunctionArguments()[i]);
    }
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

  // Assert - module name in FIRRTL has _lambda_mod suffix
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module simple_no_ports_lambda_mod :"))
      << "Expected 'module simple_no_ports_lambda_mod :' in FIRRTL output";
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
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module simple_inputs_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for UInt<32> type declaration (FIRRTL uses bit-width annotations)
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
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
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module simple_outputs_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for UInt<32> type declaration
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
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
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module simple_inout_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for UInt<32> type declarations
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
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

  // Assert - basic FIRRTL structure with type information
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module struct_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for proper type annotations (bit-width)
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' bit-width annotation in FIRRTL output";
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
  EXPECT_TRUE(ContainsSubstring(firrtl, "module multi_inputs_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for correct type on each input (UInt<32> should appear at least 3 times for 3 inputs + reset)
  int uint32Count = CountSubstring(firrtl, "UInt<32>");
  EXPECT_GE(uint32Count, 3) << "Expected at least 3 occurrences of UInt<32>";
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
  EXPECT_TRUE(ContainsSubstring(firrtl, "module multi_outputs_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for correct type on each output (UInt<32> should appear at least 3 times)
  int uint32Count = CountSubstring(firrtl, "UInt<32>");
  EXPECT_GE(uint32Count, 3) << "Expected at least 3 occurrences of UInt<32>";
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

  // Assert - ADD operations should be present as 'add' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module add_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  // Check that the ADD operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "add("))
      << "Expected 'add(' operation in FIRRTL output for IntegerAddOperation";
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

  // Assert - SUB operations should be present as 'sub' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module sub_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  // Check that the SUB operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "sub("))
      << "Expected 'sub(' operation in FIRRTL output for IntegerSubOperation";
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

  // Assert - AND operations should be present as 'and' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module and_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  // Check that the AND operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "and("))
      << "Expected 'and(' operation in FIRRTL output for IntegerAndOperation";
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

  // Assert - XOR operations should be present as 'xor' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module xor_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  // Check that the XOR operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "xor("))
      << "Expected 'xor(' operation in FIRRTL output for IntegerXorOperation";
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

  // Assert - OR operations should be present as 'or' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module or_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  // Check that the OR operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "or("))
      << "Expected 'or(' operation in FIRRTL output for IntegerOrOperation";
}

// Test that a module with comparison operations generates valid FIRRTL
TEST(RhlsToFirrtlConverterTests, TestComparisonOperations)
{
  // Arrange - use 1-bit type for equality comparison output
  auto bitType = jlm::rvsdg::BitType::Create(32);
  auto boolType = jlm::rvsdg::BitType::Create(1);  // Equality returns 1-bit result
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType, bitType }, { boolType }),
          "cmp_test",
          jlm::llvm::Linkage::externalLinkage));

  // Add an equality comparison in the lambda body
  auto & cmpNode = jlm::llvm::IntegerEqOperation::createNode(
      bitType->nbits(),
      *lambdaNode->GetFunctionArguments()[0],
      *lambdaNode->GetFunctionArguments()[1]);

  // Finalize the lambda with 1-bit result
  auto f = lambdaNode->finalize({ cmpNode.output(0) });
  (void)f;

  jlm::rvsdg::GraphExport::Create(*f, "output");

  // Act
  RhlsToFirrtlConverter converter;
  auto firrtl = converter.ToString(rm);

  // Assert - equality comparison should generate 'eq' in FIRRTL
  EXPECT_FALSE(firrtl.empty());
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Missing 'circuit' keyword";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module cmp_test_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for both UInt<32> (inputs) and UInt<1> (output)
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' type annotation in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<1>"))
      << "Expected 'UInt<1>' type annotation for comparison result";
  // Check that the equality operation is generated
  EXPECT_TRUE(ContainsSubstring(firrtl, "eq("))
      << "Expected 'eq(' (equality) operation in FIRRTL output for IntegerEqOperation";
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

  // Assert - check for FIRRTL specific syntax with type annotations
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Expected 'circuit' in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module add_func_lambda_mod :"))
      << "Expected module with correct name in FIRRTL output";
  // Check for proper type annotations
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' bit-width annotation in FIRRTL output";
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

  // Assert - module name should be in FIRRTL with _lambda_mod suffix
  EXPECT_TRUE(ContainsSubstring(firrtl, "circuit")) << "Expected 'circuit' in FIRRTL output";
  EXPECT_TRUE(ContainsSubstring(firrtl, "module my_module_name_lambda_mod :"))
      << "Expected 'module my_module_name_lambda_mod :' FIRRTL syntax";
  // Verify type annotation is present
  EXPECT_TRUE(ContainsSubstring(firrtl, "UInt<32>"))
      << "Expected 'UInt<32>' bit-width annotation in FIRRTL output";
}
