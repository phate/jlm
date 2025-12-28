/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
test()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto pointerType = PointerType::Create();
  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  auto d0 = DataNode::Create(im.ipgraph(), "d0", vt, Linkage::externalLinkage, "", false);

  auto d1 = DataNode::Create(im.ipgraph(), "d1", vt, Linkage::externalLinkage, "", false);
  auto d2 = DataNode::Create(im.ipgraph(), "d2", vt, Linkage::externalLinkage, "", false);

  auto v0 = im.create_global_value(d0);
  auto v1 = im.create_global_value(d1);
  auto v2 = im.create_global_value(d2);

  d1->add_dependency(d0);
  d1->add_dependency(d2);
  d2->add_dependency(d0);
  d2->add_dependency(d1);

  tacsvector_t tvec1, tvec2;
  tvec1.push_back(ThreeAddressCode::create(
      TestOperation::create({ pointerType, pointerType }, { vt }),
      { v0, v2 }));
  tvec2.push_back(ThreeAddressCode::create(
      TestOperation::create({ pointerType, pointerType }, { vt }),
      { v0, v1 }));

  d1->set_initialization(std::make_unique<DataNodeInit>(std::move(tvec1)));
  d2->set_initialization(std::make_unique<DataNodeInit>(std::move(tvec2)));

  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(im, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-recursive-data", test)
