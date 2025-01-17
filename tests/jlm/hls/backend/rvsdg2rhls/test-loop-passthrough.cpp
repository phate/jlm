/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/backend/rhls2firrtl/dot-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/control.hpp>
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

static int
test()
{
  using namespace jlm;

  auto ft = jlm::rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(1), rvsdg::bittype::Create(8), rvsdg::bittype::Create(8) },
      { rvsdg::bittype::Create(8) });

  jlm::llvm::RvsdgModule rm(util::filepath(""), "", "");

  /* setup graph */

  auto lambda = jlm::llvm::lambda::node::create(
      &rm.Rvsdg().GetRootRegion(),
      ft,
      "f",
      jlm::llvm::linkage::external_linkage);

  auto loop = hls::loop_node::create(lambda->subregion());

  auto loop_out = loop->AddLoopVar(lambda->GetFunctionArguments()[1]);

  auto f = lambda->finalize({ loop_out });
  jlm::llvm::GraphExport::Create(*f, "");

  rvsdg::view(rm.Rvsdg(), stdout);
  hls::DotHLS dhls;
  stringToFile(dhls.run(rm), "/tmp/jlm_hls_test_before.dot");

  hls::dne(rm);
  hls::DotHLS dhls2;
  stringToFile(dhls2.run(rm), "/tmp/jlm_hls_test_after.dot");

  // The whole loop gets eliminated, leading to a direct connection
  assert(lambda->GetFunctionResults()[0]->origin() == lambda->GetFunctionArguments()[1]);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/test-loop-passthrough", test)
