/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rhls2firrtl/dot-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
stringToFile(std::string output, std::string fileName)
{
  std::ofstream outputFile;
  outputFile.open(fileName);
  outputFile << output;
  outputFile.close();
}

TEST(LoopPassThroughTests, test)
{
  using namespace jlm;

  auto ft = jlm::rvsdg::FunctionType::Create(
      { rvsdg::BitType::Create(1), rvsdg::BitType::Create(8), rvsdg::BitType::Create(8) },
      { rvsdg::BitType::Create(8) });

  jlm::llvm::RvsdgModule rm(util::FilePath(""), "", "");

  /* setup graph */

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rm.Rvsdg().GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(ft, "f", jlm::llvm::Linkage::externalLinkage));

  auto loop = hls::LoopNode::create(lambda->subregion());

  auto loop_out = loop->AddLoopVar(lambda->GetFunctionArguments()[1]);

  auto f = lambda->finalize({ loop_out });
  rvsdg::GraphExport::Create(*f, "");

  rvsdg::view(rm.Rvsdg(), stdout);
  hls::DotHLS dhls;
  stringToFile(dhls.run(rm), "/tmp/jlm_hls_test_before.dot");

  util::StatisticsCollector statisticsCollector;
  hls::RhlsDeadNodeElimination::CreateAndRun(rm, statisticsCollector);

  hls::DotHLS dhls2;
  stringToFile(dhls2.run(rm), "/tmp/jlm_hls_test_after.dot");

  // The whole loop gets eliminated, leading to a direct connection
  EXPECT_EQ(lambda->GetFunctionResults()[0]->origin(), lambda->GetFunctionArguments()[1]);
}
