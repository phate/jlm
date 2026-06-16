/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

using namespace jlm::hls;

// Test helper class that exposes protected methods for testing
class TestableBaseHLS : public jlm::hls::BaseHLS
{
public:
  // Expose protected methods as public for testing
  using jlm::hls::BaseHLS::get_mem_reqs;
  using jlm::hls::BaseHLS::get_mem_resps;
  using jlm::hls::BaseHLS::get_node_name;
  using jlm::hls::BaseHLS::get_port_name;
  using jlm::hls::BaseHLS::get_reg_args;
  using jlm::hls::BaseHLS::get_reg_results;
  using jlm::hls::BaseHLS::JlmSize;

  // Expose protected members as public for testing
  using jlm::hls::BaseHLS::node_map;
  using jlm::hls::BaseHLS::output_map;

  // Override pure virtual GetText to make the class instantiable
  std::string
  GetText([[maybe_unused]] jlm::llvm::LlvmRvsdgModule & rm) override
  {
    return "";
  }

  std::string
  extension() override
  {
    return ".txt";
  }
};

// Test isForbiddenChar
TEST(BaseHlsTests, TestIsForbiddenChar)
{
  // Valid characters - should return false
  EXPECT_FALSE(isForbiddenChar('A'));
  EXPECT_FALSE(isForbiddenChar('Z'));
  EXPECT_FALSE(isForbiddenChar('a'));
  EXPECT_FALSE(isForbiddenChar('z'));
  EXPECT_FALSE(isForbiddenChar('0'));
  EXPECT_FALSE(isForbiddenChar('9'));
  EXPECT_FALSE(isForbiddenChar('_'));

  // Invalid characters - should return true
  EXPECT_TRUE(isForbiddenChar('-'));
  EXPECT_TRUE(isForbiddenChar('.'));
  EXPECT_TRUE(isForbiddenChar('!'));
  EXPECT_TRUE(isForbiddenChar('@'));
  EXPECT_TRUE(isForbiddenChar(' '));
  EXPECT_TRUE(isForbiddenChar('\n'));
  EXPECT_TRUE(isForbiddenChar('\t'));
}

// Test get_port_name with Input
TEST(BaseHlsTests, TestGetPortNameInput)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & import = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "input0");

  // Act
  auto portName = TestableBaseHLS().get_port_name(&import);

  // Assert
  EXPECT_EQ(portName, "a0");
}

// Test get_port_name with Output
TEST(BaseHlsTests, TestGetPortNameOutput)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto node = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { valueType });

  // Act
  auto portName = TestableBaseHLS().get_port_name(node->output(0));

  // Assert
  EXPECT_EQ(portName, "o0");
}

// Test get_port_name with RegionArgument
TEST(BaseHlsTests, TestGetPortNameRegionArgument)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto portName = TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[0]);

  // Assert
  EXPECT_EQ(portName, "a0");
}

// Test get_port_name with different indices
TEST(BaseHlsTests, TestGetPortNameWithDifferentIndices)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { valueType, valueType, valueType },
              { valueType, valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Finalize the lambda to create results
  lambda->finalize({ lambda->GetFunctionArguments()[0], lambda->GetFunctionArguments()[1] });

  // Act & Assert
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[0]), "a0");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[1]), "a1");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[2]), "a2");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionResults()[0]), "r0");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionResults()[1]), "r1");
}

// Test get_node_name
TEST(BaseHlsTests, TestGetNodeName)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto node =
      jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType, bitType });

  TestableBaseHLS baseHls;

  // Create a mock node map to avoid the pure virtual function requirement
  baseHls.node_map[node] = "test_node";

  // Act
  auto nodeName = baseHls.get_node_name(node);

  // Assert - node name should not be empty
  EXPECT_FALSE(nodeName.empty());
}

// Test get_node_name fallback when node is not in map
TEST(BaseHlsTests, TestGetNodeNameFallback)
{
  // Arrange - use BitType to ensure JlmSize works
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create a node that will NOT have an entry in node_map
  auto node = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });

  TestableBaseHLS baseHls;

  // Act - get_node_name should generate name using fallback logic
  auto nodeName = baseHls.get_node_name(node);

  // Assert - node name should be generated with expected format
  EXPECT_FALSE(nodeName.empty());
  // Name should start with "op_"
  EXPECT_EQ(nodeName.substr(0, 3), "op_");
  // Should contain the DebugString of the node (TestOperation)
  EXPECT_NE(std::string::npos, nodeName.find("TestOperation"));
  // Should end with _N where N is the node_map size at time of creation
}

// Test get_node_name with multiple nodes
TEST(BaseHlsTests, TestGetNodeNameMultipleNodes)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto node1 = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });

  auto node2 = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });

  TestableBaseHLS baseHls;

  // Create mock node maps to avoid the pure virtual function requirement
  baseHls.node_map[node1] = "test_node1";
  baseHls.node_map[node2] = "test_node2";

  // Act
  auto nodeName1 = baseHls.get_node_name(node1);
  auto nodeName2 = baseHls.get_node_name(node2);

  // Assert
  EXPECT_FALSE(nodeName1.empty());
  EXPECT_FALSE(nodeName2.empty());
  EXPECT_NE(nodeName1, nodeName2);
}

// Test get_reg_args with only register arguments
TEST(BaseHlsTests, TestGetRegArgsOnlyRegisters)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);

  // Assert
  EXPECT_EQ(regArgs.size(), 2);
  EXPECT_EQ(regArgs[0], lambda->GetFunctionArguments()[0]);
  EXPECT_EQ(regArgs[1], lambda->GetFunctionArguments()[1]);
}

// Test get_reg_args with memory responses
TEST(BaseHlsTests, TestGetRegArgsWithMemoryResponses)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  elements.emplace_back("data", valueType);
  auto memResType = std::make_shared<jlm::hls::BundleType>(std::move(elements));

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, memResType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);

  // Assert - only the first argument (valueType) should be in reg_args
  EXPECT_EQ(regArgs.size(), 1);
  EXPECT_EQ(regArgs[0], lambda->GetFunctionArguments()[0]);
}

// Test get_reg_args with empty lambda
TEST(BaseHlsTests, TestGetRegArgsEmpty)
{
  // Arrange
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({}, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);

  // Assert
  EXPECT_TRUE(regArgs.empty());
}

// Test get_reg_results with only register results
TEST(BaseHlsTests, TestGetRegResultsOnlyRegisters)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType, valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Finalize the lambda to create results
  lambda->finalize({ lambda->GetFunctionArguments()[0], lambda->GetFunctionArguments()[0] });

  // Act
  auto regResults = TestableBaseHLS().get_reg_results(*lambda);

  // Assert - use GetFunctionResults() for consistency
  EXPECT_EQ(regResults.size(), 2);
  EXPECT_EQ(regResults[0], lambda->GetFunctionResults()[0]);
  EXPECT_EQ(regResults[1], lambda->GetFunctionResults()[1]);
}

// Test get_reg_results with empty lambda
TEST(BaseHlsTests, TestGetRegResultsEmpty)
{
  // Arrange
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({}, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regResults = TestableBaseHLS().get_reg_results(*lambda);

  // Assert
  EXPECT_TRUE(regResults.empty());
}

// Test get_mem_reqs with no memory requests
TEST(BaseHlsTests, TestGetMemReqsNone)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto memReqs = TestableBaseHLS().get_mem_reqs(*lambda);

  // Assert
  EXPECT_TRUE(memReqs.empty());
}

// Test get_mem_resps with no memory responses
TEST(BaseHlsTests, TestGetMemRespsNone)
{
  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Finalize the lambda to create results
  lambda->finalize({ lambda->GetFunctionArguments()[0] });

  // Act
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert
  EXPECT_TRUE(memResps.empty());
}

// Test JlmSize with different types
TEST(BaseHlsTests, TestJlmSize)
{
  // Test with bit type
  auto bitType = jlm::rvsdg::BitType::Create(32);
  EXPECT_EQ(TestableBaseHLS().JlmSize(bitType.get()), 32);

  // Test with pointer type
  auto ptrType = jlm::llvm::PointerType::Create();
  size_t expectedPtrSize = sizeof(void *) * 8;
  EXPECT_EQ(TestableBaseHLS().JlmSize(ptrType.get()), expectedPtrSize);

  // Test with 64-bit bit type for coverage
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  EXPECT_EQ(TestableBaseHLS().JlmSize(bit64Type.get()), 64);

  // Test with 8-bit bit type for coverage
  auto bit8Type = jlm::rvsdg::BitType::Create(8);
  EXPECT_EQ(TestableBaseHLS().JlmSize(bit8Type.get()), 8);

  // Test with control type (returns ceil(log2(nalternatives())))
  auto controlType = jlm::rvsdg::ControlType::Create(4);
  EXPECT_EQ(TestableBaseHLS().JlmSize(controlType.get()), 2); // ceil(log2(4)) = 2

  // Test with float type (32-bit)
  auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::flt);
  EXPECT_EQ(TestableBaseHLS().JlmSize(floatType.get()), 32);

  // Test with double type (64-bit)
  auto doubleType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
  EXPECT_EQ(TestableBaseHLS().JlmSize(doubleType.get()), 64);

  // Test with array type
  auto elementBitType = jlm::rvsdg::BitType::Create(32);
  auto arrayType = jlm::llvm::ArrayType::Create(elementBitType, 4);
  EXPECT_EQ(TestableBaseHLS().JlmSize(arrayType.get()), 128); // 32 * 4

  // Test with vector type (FixedVectorType)
  auto vectorElementType = jlm::rvsdg::BitType::Create(32);
  auto vectorType = jlm::llvm::FixedVectorType::Create(vectorElementType, 2);
  EXPECT_EQ(TestableBaseHLS().JlmSize(vectorType.get()), 64); // 32 * 2
}

// Test node name generation with forbidden characters
TEST(BaseHlsTests, TestNodeNameWithForbiddenChars)
{
  // Arrange - use BitType instead of TestType which doesn't have JlmSize support
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create a node that will generate a name
  auto node = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });

  TestableBaseHLS baseHls;

  // Act
  auto nodeName = baseHls.get_node_name(node);

  // Assert - no forbidden characters should be in the name
  for (size_t i = 0; i < nodeName.size(); ++i)
  {
    EXPECT_FALSE(isForbiddenChar(nodeName[i]));
  }
}

// Test port naming with complex types
TEST(BaseHlsTests, TestPortNamingWithComplexType)
{
  // Arrange
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ bitType }, { bitType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Finalize the lambda to create results
  lambda->finalize({ lambda->GetFunctionArguments()[0] });

  // Act & Assert
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[0]), "a0");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionResults()[0]), "r0");
}

// Test edge cases - empty lambda with only memory responses
TEST(BaseHlsTests, TestEdgeCaseEmptyLambdaWithMemResp)
{
  // Arrange
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memResType = std::make_shared<jlm::hls::BundleType>(std::move(elements));

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memResType }, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert
  EXPECT_TRUE(regArgs.empty());  // No register args
  EXPECT_EQ(memResps.size(), 1); // But has memory response
}

// Test memory request extraction with BundleType results
TEST(BaseHlsTests, TestMemReqsWithBundleType)
{
  // Note: This test demonstrates that get_mem_reqs() checks for BundleType results.
  // However, creating actual BundleType results requires finalize() to be called,
  // which needs inputs. Since we can't create a lambda with only BundleType outputs
  // without inputs (no values to finalize), this test verifies the behavior when
  // no memory requests exist in a typical lambda structure.

  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create lambda with type (value) -> value (no memory requests)
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Finalize to create the result
  lambda->finalize({ lambda->GetFunctionArguments()[0] });

  // Act
  auto memReqs = TestableBaseHLS().get_mem_reqs(*lambda);

  // Assert - no memory requests since there's no BundleType results
  EXPECT_TRUE(memReqs.empty());
}

// Test memory response extraction with BundleType arguments
TEST(BaseHlsTests, TestMemRespsWithBundleType)
{
  // Arrange: Create a BundleType for memory responses {addr}
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements;
  respElements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memRespType = std::make_shared<jlm::hls::BundleType>(std::move(respElements));

  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create lambda with type (mem_resp) -> value
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert - should extract the BundleType argument as memory response
  EXPECT_EQ(memResps.size(), 1);
  EXPECT_EQ(memResps[0], lambda->GetFunctionArguments()[0]);
}

// Test port naming with many inputs/outputs to verify index counter behavior
TEST(BaseHlsTests, TestPortNamingMaxIndices)
{
  // Arrange: Create lambda with multiple inputs (a0, a1, ..., aN)
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> argTypes;
  for (int i = 0; i < 5; ++i)
  {
    argTypes.push_back(jlm::rvsdg::BitType::Create(32));
  }

  auto valueType = jlm::rvsdg::TestType::createValueType();
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes;
  for (int i = 0; i < 3; ++i)
  {
    resultTypes.push_back(valueType);
  }

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(argTypes, resultTypes),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act & Assert: Verify all argument indices are unique and sequential
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[0]), "a0");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[1]), "a1");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[2]), "a2");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[3]), "a3");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[4]), "a4");
}

// Test lambda with only memory responses (no register args)
TEST(BaseHlsTests, TestEmptyLambdaAllMemory)
{
  // Arrange: Create a BundleType for memory responses {addr}
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements;
  respElements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memRespType = std::make_shared<jlm::hls::BundleType>(std::move(respElements));

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create lambda with type (mem_resp) -> {}
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType }, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert: No register args, but has memory response
  EXPECT_TRUE(regArgs.empty());  // No register args
  EXPECT_EQ(memResps.size(), 1); // But has memory response
}

// Test lambda with both register and memory arguments
TEST(BaseHlsTests, TestMixedRegAndMemArgs)
{
  // Arrange: Create a BundleType for memory responses {addr}
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements;
  respElements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memRespType = std::make_shared<jlm::hls::BundleType>(std::move(respElements));

  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create lambda with type (value, mem_resp) -> value
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, memRespType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert: One register arg, one memory response
  EXPECT_EQ(regArgs.size(), 1); // Only 'value' is register arg
  EXPECT_EQ(regArgs[0], lambda->GetFunctionArguments()[0]);
  EXPECT_EQ(memResps.size(), 1); // mem_resp is memory response
  EXPECT_EQ(memResps[0], lambda->GetFunctionArguments()[1]);
}

// Test multiple BundleType arguments (multiple memory responses)
TEST(BaseHlsTests, TestMultipleMemResponses)
{
  // Arrange: Create two different BundleType responses
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements1;
  respElements1.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memRespType1 = std::make_shared<jlm::hls::BundleType>(std::move(respElements1));

  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements2;
  respElements2.emplace_back("data", jlm::rvsdg::BitType::Create(32));
  auto memRespType2 = std::make_shared<jlm::hls::BundleType>(std::move(respElements2));

  auto valueType = jlm::rvsdg::TestType::createValueType();
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Create lambda with type (mem_resp1, mem_resp2) -> value
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType1, memRespType2 }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda);
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda);

  // Assert: No register args, two memory responses
  EXPECT_TRUE(regArgs.empty());  // No register args
  EXPECT_EQ(memResps.size(), 2); // Two memory responses
}
