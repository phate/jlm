/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/hls/opt/IOBarrierRemoval.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/simple-node.hpp>

static int
IOBarrierRemoval()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto i32Type = bittype::Create(32);
  auto ioStateType = IOStateType::Create();
  const auto functionType =
      FunctionType::Create({ i32Type, i32Type, ioStateType }, { i32Type, ioStateType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & rvsdg = rvsdgModule.Rvsdg();

  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  auto x = lambdaNode->GetFunctionArguments()[0];
  auto y = lambdaNode->GetFunctionArguments()[1];
  auto ioState = lambdaNode->GetFunctionArguments()[2];

  auto & ioBarrierNode = jlm::rvsdg::CreateOpNode<IOBarrierOperation>({ x, ioState }, i32Type);

  auto & sdivNode =
      jlm::rvsdg::CreateOpNode<IntegerSDivOperation>({ ioBarrierNode.output(0), y }, 32);

  const auto lambdaOutput = lambdaNode->finalize({ sdivNode.output(0), ioState });

  jlm::llvm::GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::hls::IOBarrierRemoval ioBarrierRemoval;
  jlm::util::StatisticsCollector statisticsCollector;
  ioBarrierRemoval.Run(rvsdgModule, statisticsCollector);

  // Assert
  assert(!Region::ContainsOperation<IOBarrierOperation>(rvsdg.GetRootRegion(), true));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/opt/IOBarrierRemoval", IOBarrierRemoval)
