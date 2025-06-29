/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

static void
test()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ft = jlm::rvsdg::FunctionType::Create({ vt }, { vt });

  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  auto d = data_node::Create(im.ipgraph(), "d", vt, linkage::external_linkage, "", false);
  auto f = function_node::create(im.ipgraph(), "f", ft, linkage::external_linkage);

  im.create_global_value(d);
  im.create_variable(f);

  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(im, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  /*
    We should have no exports in the RVSDG. The data and function
    node should be converted to RVSDG imports as they do not have
    a body, i.e., either a CFG or a initialization.
  */
  assert(rvsdgModule->Rvsdg().GetRootRegion().nresults() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-export", test)
