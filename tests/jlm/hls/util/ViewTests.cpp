/*
 * Copyright 2024 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestDumpDot()
{
  std::cout << std::endl << "### Test dump_dot ###" << std::endl << std::endl;

  using namespace jlm;
  using namespace jlm::hls;
  using namespace jlm::llvm;

  // Arrange
  auto b32 = rvsdg::BitType::Create(32);
  auto ft = rvsdg::FunctionType::Create({}, { b32 });

  rvsdg::Graph graph;

  auto lambda = rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto bitConstant = rvsdg::BitConstantOperation::create(lambda->subregion(), { 32, 0 });

  auto f = lambda->finalize({ bitConstant });
  rvsdg::GraphExport::Create(*f, "");

  rvsdg::view(graph, stdout);

  std::unordered_map<rvsdg::Output *, ViewColors> outputColor;
  std::unordered_map<rvsdg::Input *, ViewColors> inputColor;
  std::unordered_map<rvsdg::Output *, ViewColors> tailLabel;

  // Act
  auto dotOutput = ToDot(lambda->region(), outputColor, inputColor, tailLabel);

  // Assert
  assert(dotOutput.size() > 0);
  assert(dotOutput.find("digraph G {") != std::string::npos);
  assert(dotOutput.find("subgraph cluster_") != std::string::npos);
  assert(dotOutput.find("tooltip=\"bit32\"") != std::string::npos);
  assert(dotOutput.find("BITS32_0_") != std::string::npos);
}

JLM_UNIT_TEST_REGISTER("jlm/hls/util/TestView", TestDumpDot)

static void
TestDumpDotTheta()
{
  using namespace jlm;
  using namespace jlm::hls;
  using namespace jlm::llvm;

  // Arrange
  auto b32 = rvsdg::BitType::Create(32);
  auto ft = rvsdg::FunctionType::Create({ b32, b32, b32 }, { b32, b32, b32 });

  rvsdg::Graph graph;

  auto lambda = rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto theta = rvsdg::ThetaNode::create(lambda->subregion());
  auto idv = theta->AddLoopVar(lambda->GetFunctionArguments()[0]);
  auto lvs = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  auto lve = theta->AddLoopVar(lambda->GetFunctionArguments()[2]);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idv.pre, lvs.pre }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, lve.pre }, 32).output(0);
  auto match = rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv.post->divert_to(arm);
  theta->set_predicate(match);

  auto f = lambda->finalize({ theta->output(0), theta->output(1), theta->output(2) });
  rvsdg::GraphExport::Create(*f, "");

  rvsdg::view(graph, stdout);

  std::unordered_map<rvsdg::Output *, ViewColors> outputColor;
  std::unordered_map<rvsdg::Input *, ViewColors> inputColor;
  std::unordered_map<rvsdg::Output *, ViewColors> tailLabel;
  // Act
  auto dotOutput = ToDot(lambda->region(), outputColor, inputColor, tailLabel);

  // Assert
  assert(dotOutput.size() > 0);
  assert(dotOutput.find("digraph G {") != std::string::npos);
  assert(dotOutput.find("subgraph cluster_") != std::string::npos);
  assert(dotOutput.find("tooltip=\"bit32\"") != std::string::npos);
  assert(dotOutput.find("tooltip=\"ctl(2)\"") != std::string::npos);
}

JLM_UNIT_TEST_REGISTER("jlm/hls/util/TestViewTheta", TestDumpDotTheta)
