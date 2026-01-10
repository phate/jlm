/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/Statistics.hpp>

TEST(ExportTests, test)
{
  using namespace jlm::llvm;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt }, { vt });

  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  auto d = DataNode::Create(im.ipgraph(), "d", vt, Linkage::externalLinkage, "", false);
  auto f = FunctionNode::create(im.ipgraph(), "f", ft, Linkage::externalLinkage);

  im.create_global_value(d);
  im.create_variable(f);

  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(im, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // We should have no exports in the RVSDG. The data and function
  // node should be converted to RVSDG imports as they do not have
  // a body, i.e., either a CFG or a initialization.
  EXPECT_EQ(rvsdgModule->Rvsdg().GetRootRegion().nresults(), 0u);
}
