/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunInvariantValueRedirection(jlm::RvsdgModule & rvsdgModule)
{
  jlm::StatisticsCollector statisticsCollector;
  jlm::InvariantValueRedirection invariantValueRedirection;
  invariantValueRedirection.run(rvsdgModule, statisticsCollector);
}

static void
TestGamma()
{
  auto SetupRvsdg = []()
  {
    using namespace jlm;

    jlm::valuetype valueType;
    jive::ctltype controlType(2);

    auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");
    auto & graph = rvsdgModule->Rvsdg();
    auto c = graph.add_import({controlType, "c"});
    auto x = graph.add_import({valueType, "x"});
    auto y = graph.add_import({valueType, "y"});

    auto gammaNode1 = jive::gamma_node::create(c, 2);
    auto gammaInput1 = gammaNode1->add_entryvar(c);
    auto gammaInput2 = gammaNode1->add_entryvar(x);
    auto gammaInput3 = gammaNode1->add_entryvar(y);

    auto gammaNode2 = jive::gamma_node::create(gammaInput1->argument(0), 2);
    auto gammaInput4 = gammaNode2->add_entryvar(gammaInput2->argument(0));
    auto gammaInput5 = gammaNode2->add_entryvar(gammaInput3->argument(0));
    gammaNode2->add_exitvar({gammaInput4->argument(0), gammaInput4->argument(1)});
    gammaNode2->add_exitvar({gammaInput5->argument(0), gammaInput5->argument(1)});

    gammaNode1->add_exitvar({gammaNode2->output(0), gammaInput2->argument(1)});
    gammaNode1->add_exitvar({gammaNode2->output(1), gammaInput3->argument(1)});

    graph.add_export(gammaNode1->output(0), {gammaNode1->output(0)->type(), "x"});
    graph.add_export(gammaNode1->output(1), {gammaNode1->output(1)->type(), "y"});

    return rvsdgModule;
  };

  /*
   * Arrange
   */
  auto rvsdgModule = SetupRvsdg();
  auto rootRegion = rvsdgModule->Rvsdg().root();

  /*
   * Act
   */
  jive::view(rootRegion, stdout);
  RunInvariantValueRedirection(*rvsdgModule);
  jive::view(rootRegion, stdout);

  /*
   * Assert
   */
  assert(rootRegion->result(0)->origin() == rootRegion->argument(1));
  assert(rootRegion->result(1)->origin() == rootRegion->argument(2));
}

static void
TestTheta()
{
  auto SetupRvsdg = []()
  {
    using namespace jlm;

    loopstatetype loopStateType;
    jlm::valuetype valueType;
    jive::ctltype controlType(2);

    auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");
    auto & graph = rvsdgModule->Rvsdg();
    auto c = graph.add_import({controlType, "c"});
    auto x = graph.add_import({valueType, "x"});
    auto l = graph.add_import({loopStateType, "l"});

    auto thetaNode1 = jive::theta_node::create(graph.root());
    auto thetaOutput1 = thetaNode1->add_loopvar(c);
    auto thetaOutput2 = thetaNode1->add_loopvar(x);
    auto thetaOutput3 = thetaNode1->add_loopvar(l);

    auto thetaNode2 = jive::theta_node::create(thetaNode1->subregion());
    auto thetaOutput4 = thetaNode2->add_loopvar(thetaOutput1->argument());
    thetaNode2->add_loopvar(thetaOutput2->argument());
    auto thetaOutput5 = thetaNode2->add_loopvar(thetaOutput3->argument());
    thetaNode2->set_predicate(thetaOutput4->argument());

    thetaOutput3->result()->divert_to(thetaOutput5);
    thetaNode1->set_predicate(thetaOutput1->argument());

    graph.add_export(thetaOutput1, {thetaOutput1->type(), "c"});
    graph.add_export(thetaOutput2, {thetaOutput2->type(), "x"});
    graph.add_export(thetaOutput3, {thetaOutput3->type(), "l"});

    return std::make_tuple(
      std::move(rvsdgModule),
      thetaOutput3);
  };

  /*
   * Arrange
   */
  auto [rvsdgModule, thetaOutput3] = SetupRvsdg();
  auto rootRegion = rvsdgModule->Rvsdg().root();

  /*
   * Act
   */
  jive::view(rootRegion, stdout);
  RunInvariantValueRedirection(*rvsdgModule);
  jive::view(rootRegion, stdout);

  /*
   * Assert
   */
  assert(rootRegion->result(0)->origin() == rootRegion->argument(0));
  assert(rootRegion->result(1)->origin() == rootRegion->argument(1));
  assert(rootRegion->result(2)->origin() == thetaOutput3);
}

static int
TestInvariantValueRedirection()
{
  TestGamma();
  TestTheta();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestInvariantValueRedirection", TestInvariantValueRedirection)
