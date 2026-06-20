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

  // Test with state type (returns 1 for StateKind types)
  auto stateType = jlm::rvsdg::TestType::createStateType();
  EXPECT_EQ(TestableBaseHLS().JlmSize(stateType.get()), 1);

  // Test with bundle type (returns 0 - this is a known hack in the implementation)
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  elements.emplace_back("data", bitType);
  auto bundleType = std::make_shared<jlm::hls::BundleType>(std::move(elements));
  EXPECT_EQ(TestableBaseHLS().JlmSize(bundleType.get()), 0);

  // Test that exception is thrown for unknown types
  auto testType = jlm::rvsdg::TestType::createValueType();
  EXPECT_THROW(TestableBaseHLS().JlmSize(testType.get()), std::logic_error);
}

// Test get_port_name with various port types and indices
TEST(BaseHlsTests, TestGetPortName)
{
  // Arrange: Create a lambda with multiple arguments and results
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

  auto baseHls = TestableBaseHLS();

  // Act & Assert: Test GraphImport (input) - should be "a0"
  auto & import = jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "input0");
  EXPECT_EQ(baseHls.get_port_name(&import), "a0");

  // Test NodeOutput
  auto node = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { valueType });
  EXPECT_EQ(baseHls.get_port_name(node->output(0)), "o0");

  // Test RegionArgument with various indices
  EXPECT_EQ(baseHls.get_port_name(lambda->GetFunctionArguments()[0]), "a0");
  EXPECT_EQ(baseHls.get_port_name(lambda->GetFunctionArguments()[1]), "a1");
  EXPECT_EQ(baseHls.get_port_name(lambda->GetFunctionArguments()[2]), "a2");

  // Test RegionResult with various indices
  EXPECT_EQ(baseHls.get_port_name(lambda->GetFunctionResults()[0]), "r0");
  EXPECT_EQ(baseHls.get_port_name(lambda->GetFunctionResults()[1]), "r1");
}

// Test get_node_name and node name generation
TEST(BaseHlsTests, TestGetNodeName)
{
  auto bitType = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();
  TestableBaseHLS baseHls;

  // Test node name generation with mock entry
  auto node1 = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });
  baseHls.node_map[node1] = "test_node";
  EXPECT_FALSE(baseHls.get_node_name(node1).empty());

  // Test fallback when node is not in map
  auto node2 = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });
  auto nodeName2 = baseHls.get_node_name(node2);
  EXPECT_FALSE(nodeName2.empty());
  EXPECT_EQ(nodeName2.substr(0, 3), "op_");
  EXPECT_NE(std::string::npos, nodeName2.find("TestOperation"));

  // Test that multiple nodes get different names
  auto node3 = jlm::rvsdg::TestOperation::createNode(&rvsdg.GetRootRegion(), {}, { bitType });
  baseHls.node_map[node3] = "test_node";
  EXPECT_NE(baseHls.get_node_name(node2), baseHls.get_node_name(node3));

  // Test that no forbidden characters are in generated names
  for (size_t i = 0; i < nodeName2.size(); ++i)
  {
    EXPECT_FALSE(isForbiddenChar(nodeName2[i]));
  }
}

// Test get_reg_args with various argument types
TEST(BaseHlsTests, TestGetRegArgs)
{
  auto valueType = jlm::rvsdg::TestType::createValueType();
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  elements.emplace_back("data", valueType);
  auto memResType = std::make_shared<jlm::hls::BundleType>(std::move(elements));
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Test with only register arguments
  auto lambda1 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda1);
  EXPECT_EQ(regArgs.size(), 2);

  // Test with memory responses (should filter them out)
  auto lambda2 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, memResType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda2);
  EXPECT_EQ(regArgs.size(), 1); // Only the first argument is in reg_args
  EXPECT_EQ(regArgs[0], lambda2->GetFunctionArguments()[0]);

  // Test with empty lambda
  auto lambda3 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({}, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda3);
  EXPECT_TRUE(regArgs.empty());

  // Test with only memory responses (no register args)
  auto lambda4 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memResType }, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda4);
  EXPECT_TRUE(regArgs.empty());

  // Test with mixed register and memory arguments
  auto lambda5 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, memResType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda5);
  EXPECT_EQ(regArgs.size(), 1);
  EXPECT_EQ(regArgs[0], lambda5->GetFunctionArguments()[0]);
}

// Test get_reg_results with various result types
TEST(BaseHlsTests, TestGetRegResults)
{
  auto valueType = jlm::rvsdg::TestType::createValueType();
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memReqType = std::make_shared<jlm::hls::BundleType>(std::move(elements));
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Test with only register results
  auto lambda1 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType, valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  lambda1->finalize({ lambda1->GetFunctionArguments()[0], lambda1->GetFunctionArguments()[0] });
  auto regResults = TestableBaseHLS().get_reg_results(*lambda1);
  EXPECT_EQ(regResults.size(), 2);
  EXPECT_EQ(regResults[0], lambda1->GetFunctionResults()[0]);
  EXPECT_EQ(regResults[1], lambda1->GetFunctionResults()[1]);

  // Test with empty lambda
  auto lambda2 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({}, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regResults = TestableBaseHLS().get_reg_results(*lambda2);
  EXPECT_TRUE(regResults.empty());
}

// Test get_mem_reqs and get_mem_resps
TEST(BaseHlsTests, TestMemoryHandling)
{
  auto valueType = jlm::rvsdg::TestType::createValueType();
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", jlm::llvm::PointerType::Create());
  elements.emplace_back("data", valueType);
  auto memReqType = std::make_shared<jlm::hls::BundleType>(std::move(elements));
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  // Test no memory requests
  auto lambda1 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  auto memReqs = TestableBaseHLS().get_mem_reqs(*lambda1);
  EXPECT_TRUE(memReqs.empty());

  // Test no memory responses
  auto lambda2 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  auto memResps = TestableBaseHLS().get_mem_resps(*lambda2);
  EXPECT_TRUE(memResps.empty());

  // Test memory response extraction with BundleType arguments
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements;
  respElements.emplace_back("addr", jlm::llvm::PointerType::Create());
  auto memRespType = std::make_shared<jlm::hls::BundleType>(std::move(respElements));

  auto lambda3 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  memResps = TestableBaseHLS().get_mem_resps(*lambda3);
  EXPECT_EQ(memResps.size(), 1);
  EXPECT_EQ(memResps[0], lambda3->GetFunctionArguments()[0]);

  // Test empty lambda with only memory responses
  auto lambda4 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType }, {}),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  auto regArgs = TestableBaseHLS().get_reg_args(*lambda4);
  memResps = TestableBaseHLS().get_mem_resps(*lambda4);
  EXPECT_TRUE(regArgs.empty());
  EXPECT_EQ(memResps.size(), 1);

  // Test mixed register and memory arguments
  auto lambda5 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ valueType, memRespType }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda5);
  memResps = TestableBaseHLS().get_mem_resps(*lambda5);
  EXPECT_EQ(regArgs.size(), 1);
  EXPECT_EQ(memResps.size(), 1);

  // Test multiple BundleType arguments (multiple memory responses)
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> respElements2;
  respElements2.emplace_back("data", jlm::rvsdg::BitType::Create(32));
  auto memRespType2 = std::make_shared<jlm::hls::BundleType>(std::move(respElements2));

  auto lambda6 = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ memRespType, memRespType2 }, { valueType }),
          "f",
          jlm::llvm::Linkage::externalLinkage));
  regArgs = TestableBaseHLS().get_reg_args(*lambda6);
  memResps = TestableBaseHLS().get_mem_resps(*lambda6);
  EXPECT_TRUE(regArgs.empty());
  EXPECT_EQ(memResps.size(), 2);
}

// Test port naming with many inputs/outputs to verify index counter behavior
TEST(BaseHlsTests, TestPortNamingMaxIndices)
{
  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { jlm::rvsdg::BitType::Create(32),
                jlm::rvsdg::BitType::Create(32),
                jlm::rvsdg::BitType::Create(32),
                jlm::rvsdg::BitType::Create(32),
                jlm::rvsdg::BitType::Create(32) },
              { jlm::rvsdg::TestType::createValueType(),
                jlm::rvsdg::TestType::createValueType(),
                jlm::rvsdg::TestType::createValueType() }),
          "f",
          jlm::llvm::Linkage::externalLinkage));

  // Act & Assert: Verify all argument indices are unique and sequential
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[0]), "a0");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[1]), "a1");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[2]), "a2");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[3]), "a3");
  EXPECT_EQ(TestableBaseHLS().get_port_name(lambda->GetFunctionArguments()[4]), "a4");
}
