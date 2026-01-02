/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 *                Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/GammaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(GammaConversionTests, TestWithMatch)
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ft = jlm::rvsdg::FunctionType::Create({ jlm::rvsdg::BitType::Create(1), vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambda->GetFunctionArguments()[0]);
  auto gamma = jlm::rvsdg::GammaNode::create(match, 2);
  auto ev1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto ev2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto ex = gamma->AddExitVar({ ev1.branchArgument[0], ev2.branchArgument[1] });

  auto f = lambda->finalize({ ex.output });
  jlm::rvsdg::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::hls::GammaNodeConversion::CreateAndRun(rm, statisticsCollector);

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<jlm::hls::MuxOperation>(*lambda->subregion(), true));
}

TEST(GammaConversionTests, TestWithoutMatch)
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ft =
      jlm::rvsdg::FunctionType::Create({ jlm::rvsdg::ControlType::Create(2), vt, vt }, { vt });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");

  /* Setup graph */

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(ft, "f", Linkage::externalLinkage));

  auto gamma = jlm::rvsdg::GammaNode::create(lambda->GetFunctionArguments()[0], 2);
  auto ev1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto ev2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto ex = gamma->AddExitVar({ ev1.branchArgument[0], ev2.branchArgument[1] });

  auto f = lambda->finalize({ ex.output });
  jlm::rvsdg::GraphExport::Create(*f, "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::hls::GammaNodeConversion::CreateAndRun(rm, statisticsCollector);

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Assert
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<jlm::hls::MuxOperation>(*lambda->subregion(), true));
}
