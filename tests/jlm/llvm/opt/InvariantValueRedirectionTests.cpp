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

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunInvariantValueRedirection(jlm::llvm::RvsdgModule & rvsdgModule)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::InvariantValueRedirection invariantValueRedirection;
  invariantValueRedirection.run(rvsdgModule, statisticsCollector);
}

static int
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::tests::valuetype valueType;
  jlm::rvsdg::ctltype controlType(2);
  FunctionType functionType({ &controlType, &valueType, &valueType }, { &valueType, &valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode =
      lambda::node::create(rvsdg.root(), functionType, "test", linkage::external_linkage);

  auto c = lambdaNode->fctargument(0);
  auto x = lambdaNode->fctargument(1);
  auto y = lambdaNode->fctargument(2);

  auto gammaNode1 = jlm::rvsdg::gamma_node::create(c, 2);
  auto gammaInput1 = gammaNode1->add_entryvar(c);
  auto gammaInput2 = gammaNode1->add_entryvar(x);
  auto gammaInput3 = gammaNode1->add_entryvar(y);

  auto gammaNode2 = jlm::rvsdg::gamma_node::create(gammaInput1->argument(0), 2);
  auto gammaInput4 = gammaNode2->add_entryvar(gammaInput2->argument(0));
  auto gammaInput5 = gammaNode2->add_entryvar(gammaInput3->argument(0));
  gammaNode2->add_exitvar({ gammaInput4->argument(0), gammaInput4->argument(1) });
  gammaNode2->add_exitvar({ gammaInput5->argument(0), gammaInput5->argument(1) });

  gammaNode1->add_exitvar({ gammaNode2->output(0), gammaInput2->argument(1) });
  gammaNode1->add_exitvar({ gammaNode2->output(1), gammaInput3->argument(1) });

  auto lambdaOutput = lambdaNode->finalize({ gammaNode1->output(0), gammaNode1->output(1) });

  rvsdg.add_export(lambdaOutput, { lambdaOutput->type(), "test" });

  // Act
  jlm::rvsdg::view(rvsdg, stdout);
  RunInvariantValueRedirection(*rvsdgModule);
  jlm::rvsdg::view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->fctresult(0)->origin() == x);
  assert(lambdaNode->fctresult(1)->origin() == y);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestInvariantValueRedirection-Gamma", TestGamma)

static int
TestTheta()
{
  // Arrange
  using namespace jlm::llvm;

  iostatetype ioStateType;
  jlm::tests::valuetype valueType;
  jlm::rvsdg::ctltype controlType(2);
  FunctionType functionType(
      { &controlType, &valueType, &ioStateType },
      { &controlType, &valueType, &ioStateType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode =
      lambda::node::create(rvsdg.root(), functionType, "test", linkage::external_linkage);

  auto c = lambdaNode->fctargument(0);
  auto x = lambdaNode->fctargument(1);
  auto l = lambdaNode->fctargument(2);

  auto thetaNode1 = jlm::rvsdg::theta_node::create(lambdaNode->subregion());
  auto thetaOutput1 = thetaNode1->add_loopvar(c);
  auto thetaOutput2 = thetaNode1->add_loopvar(x);
  auto thetaOutput3 = thetaNode1->add_loopvar(l);

  auto thetaNode2 = jlm::rvsdg::theta_node::create(thetaNode1->subregion());
  auto thetaOutput4 = thetaNode2->add_loopvar(thetaOutput1->argument());
  thetaNode2->add_loopvar(thetaOutput2->argument());
  auto thetaOutput5 = thetaNode2->add_loopvar(thetaOutput3->argument());
  thetaNode2->set_predicate(thetaOutput4->argument());

  thetaOutput3->result()->divert_to(thetaOutput5);
  thetaNode1->set_predicate(thetaOutput1->argument());

  auto lambdaOutput = lambdaNode->finalize({ thetaOutput1, thetaOutput2, thetaOutput3 });

  rvsdg.add_export(lambdaOutput, { lambdaOutput->type(), "test" });

  // Act
  jlm::rvsdg::view(rvsdg, stdout);
  RunInvariantValueRedirection(*rvsdgModule);
  jlm::rvsdg::view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->fctresult(0)->origin() == c);
  assert(lambdaNode->fctresult(1)->origin() == x);
  assert(lambdaNode->fctresult(2)->origin() == thetaOutput3);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/TestInvariantValueRedirection-Theta", TestTheta)
