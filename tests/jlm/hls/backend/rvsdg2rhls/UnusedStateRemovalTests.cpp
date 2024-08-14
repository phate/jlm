/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

static void
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto p = &jlm::tests::GraphImport::Create(rvsdg, jlm::rvsdg::ctltype::Create(2), "p");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto gammaNode = jlm::rvsdg::gamma_node::create(p, 2);

  auto gammaInput1 = gammaNode->add_entryvar(x);
  auto gammaInput2 = gammaNode->add_entryvar(y);
  auto gammaInput3 = gammaNode->add_entryvar(z);
  auto gammaInput4 = gammaNode->add_entryvar(x);
  auto gammaInput5 = gammaNode->add_entryvar(x);
  auto gammaInput6 = gammaNode->add_entryvar(x);
  auto gammaInput7 = gammaNode->add_entryvar(x);

  auto gammaOutput1 =
      gammaNode->add_exitvar({ gammaInput1->argument(0), gammaInput1->argument(1) });
  auto gammaOutput2 =
      gammaNode->add_exitvar({ gammaInput2->argument(0), gammaInput3->argument(1) });
  auto gammaOutput3 =
      gammaNode->add_exitvar({ gammaInput4->argument(0), gammaInput5->argument(1) });
  auto gammaOutput4 =
      gammaNode->add_exitvar({ gammaInput6->argument(0), gammaInput6->argument(1) });
  auto gammaOutput5 =
      gammaNode->add_exitvar({ gammaInput6->argument(0), gammaInput7->argument(1) });

  rvsdg.add_export(gammaOutput1, { valueType, "" });
  rvsdg.add_export(gammaOutput2, { valueType, "" });
  rvsdg.add_export(gammaOutput3, { valueType, "" });
  rvsdg.add_export(gammaOutput4, { valueType, "" });
  rvsdg.add_export(gammaOutput5, { valueType, "" });

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(gammaNode->ninputs() == 7);  // gammaInput1 was removed
  assert(gammaNode->noutputs() == 4); // gammaOutput1 was removed
  assert(gammaInput2->index() == 1);
  assert(gammaOutput2->index() == 0);
  // FIXME: The transformation is way too conservative here. The only input and output it removes
  // are gammaInput1 and gammaOutput1, respectively. However, it could also remove gammaOutput3,
  // gammaOutput4, and gammaOutput5 as they are all invariant. This in turn would also render some
  // more inputs dead.
}

static void
TestTheta()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create(
      { jlm::rvsdg::ctltype::Create(2), valueType, valueType, valueType },
      { valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  auto p = rvsdg.add_import({ jlm::rvsdg::ctltype::Create(2), "p" });
  auto x = rvsdg.add_import({ valueType, "x" });
  auto y = rvsdg.add_import({ valueType, "y" });
  auto z = rvsdg.add_import({ valueType, "z" });

  auto thetaNode = jlm::rvsdg::theta_node::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(p);
  auto thetaOutput1 = thetaNode->add_loopvar(x);
  auto thetaOutput2 = thetaNode->add_loopvar(y);
  auto thetaOutput3 = thetaNode->add_loopvar(z);

  thetaOutput2->result()->divert_to(thetaOutput3->argument());
  thetaOutput3->result()->divert_to(thetaOutput2->argument());
  thetaNode->set_predicate(thetaOutput0->argument());

  auto result = jlm::tests::SimpleNode::Create(
                    *rvsdg.root(),
                    { thetaOutput0, thetaOutput1, thetaOutput2, thetaOutput3 },
                    { valueType })
                    .output(0);

  rvsdg.add_export(result, { valueType, "f" });

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  // This assert is only here so that we do not forget this test when we refactor the code
  assert(thetaNode->ninputs() == 1);

  // FIXME: This transformation is broken for theta nodes. For the setup above, it
  // removes all inputs/outputs, except the predicate. However, the only
  // input and output it should remove are input 1 and output 0, respectively.
}

static void
TestLambda()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = FunctionType::Create(
      { valueType, valueType },
      { valueType, valueType, valueType, valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = rvsdg.add_import({ valueType, "x" });

  auto lambdaNode =
      lambda::node::create(rvsdg.root(), functionType, "f", linkage::external_linkage);
  auto argument0 = lambdaNode->fctargument(0);
  auto argument1 = lambdaNode->fctargument(1);
  auto argument2 = lambdaNode->add_ctxvar(x);
  auto argument3 = lambdaNode->add_ctxvar(x);

  auto result1 =
      jlm::tests::SimpleNode::Create(*lambdaNode->subregion(), { argument1 }, { valueType })
          .output(0);
  auto result3 =
      jlm::tests::SimpleNode::Create(*lambdaNode->subregion(), { argument3 }, { valueType })
          .output(0);

  auto lambdaOutput = lambdaNode->finalize({ argument0, result1, argument2, result3 });

  rvsdg.add_export(lambdaOutput, { PointerType::Create(), "f" });

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(rvsdg.root()->nnodes() == 1);
  auto & newLambdaNode = dynamic_cast<const lambda::node &>(*rvsdg.root()->nodes.begin());
  assert(newLambdaNode.ninputs() == 2);
  assert(newLambdaNode.subregion()->narguments() == 3);
  assert(newLambdaNode.subregion()->nresults() == 2);
  // FIXME For lambdas, the transformation has the following issues:
  // 1. It works only for lambda nodes in the root region. It throws an assert for all other lambdas
  // 2. It does not check whether the lambda is only exported. Removing passthrough values works
  // only for lambda nodes that do not have any calls.
  // 3. It removes all pass through values, regardless of whether they are value or state types.
  // Removing value types does change the signature of a lambda node.
  // 4. It does not remove the arguments and inputs of context variables that are just passed
  // through. It only renders them dead.
  //
  // There might be more issues.
}

static int
TestUnusedStateRemoval()
{
  TestGamma();
  TestTheta();
  TestLambda();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/UnusedStateRemovalTests", TestUnusedStateRemoval)
