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

  auto p = &jlm::tests::GraphImport::Create(rvsdg, jlm::rvsdg::ControlType::Create(2), "p");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto gammaNode = jlm::rvsdg::GammaNode::create(p, 2);

  auto gammaInput1 = gammaNode->AddEntryVar(x);
  auto gammaInput2 = gammaNode->AddEntryVar(y);
  auto gammaInput3 = gammaNode->AddEntryVar(z);
  auto gammaInput4 = gammaNode->AddEntryVar(x);
  auto gammaInput5 = gammaNode->AddEntryVar(x);
  auto gammaInput6 = gammaNode->AddEntryVar(x);
  auto gammaInput7 = gammaNode->AddEntryVar(x);

  auto gammaOutput1 = gammaNode->AddExitVar(gammaInput1.branchArgument);
  auto gammaOutput2 =
      gammaNode->AddExitVar({ gammaInput2.branchArgument[0], gammaInput3.branchArgument[1] });
  auto gammaOutput3 =
      gammaNode->AddExitVar({ gammaInput4.branchArgument[0], gammaInput5.branchArgument[1] });
  auto gammaOutput4 =
      gammaNode->AddExitVar({ gammaInput6.branchArgument[0], gammaInput6.branchArgument[1] });
  auto gammaOutput5 =
      gammaNode->AddExitVar({ gammaInput6.branchArgument[0], gammaInput7.branchArgument[1] });

  GraphExport::Create(*gammaOutput1.output, "");
  GraphExport::Create(*gammaOutput2.output, "");
  GraphExport::Create(*gammaOutput3.output, "");
  GraphExport::Create(*gammaOutput4.output, "");
  GraphExport::Create(*gammaOutput5.output, "");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(gammaNode->ninputs() == 7);  // gammaInput1 was removed
  assert(gammaNode->noutputs() == 4); // gammaOutput1 was removed
  assert(gammaInput2.input->index() == 1);
  assert(gammaOutput2.output->index() == 0);
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
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType, valueType },
      { valueType });

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  auto p = &jlm::tests::GraphImport::Create(rvsdg, jlm::rvsdg::ControlType::Create(2), "p");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(p);
  auto thetaOutput1 = thetaNode->AddLoopVar(x);
  auto thetaOutput2 = thetaNode->AddLoopVar(y);
  auto thetaOutput3 = thetaNode->AddLoopVar(z);

  thetaOutput2.post->divert_to(thetaOutput3.pre);
  thetaOutput3.post->divert_to(thetaOutput2.pre);
  thetaNode->set_predicate(thetaOutput0.pre);

  auto result =
      jlm::tests::SimpleNode::Create(
          rvsdg.GetRootRegion(),
          { thetaOutput0.output, thetaOutput1.output, thetaOutput2.output, thetaOutput3.output },
          { valueType })
          .output(0);

  GraphExport::Create(*result, "f");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(thetaNode->ninputs() == 3);
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

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode =
      lambda::node::create(&rvsdg.GetRootRegion(), functionType, "f", linkage::external_linkage);
  auto argument0 = lambdaNode->GetFunctionArguments()[0];
  auto argument1 = lambdaNode->GetFunctionArguments()[1];
  auto argument2 = lambdaNode->AddContextVar(*x).inner;
  auto argument3 = lambdaNode->AddContextVar(*x).inner;

  auto result1 =
      jlm::tests::SimpleNode::Create(*lambdaNode->subregion(), { argument1 }, { valueType })
          .output(0);
  auto result3 =
      jlm::tests::SimpleNode::Create(*lambdaNode->subregion(), { argument3 }, { valueType })
          .output(0);

  auto lambdaOutput = lambdaNode->finalize({ argument0, result1, argument2, result3 });

  GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto & newLambdaNode = dynamic_cast<const lambda::node &>(*rvsdg.GetRootRegion().Nodes().begin());
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
