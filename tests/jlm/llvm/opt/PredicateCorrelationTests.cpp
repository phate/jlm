/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/opt/PredicateCorrelation.hpp>

static void
testThetaGamma()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto valueType = ValueType::Create();
  auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create({ valueType, valueType }, { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto functionArgument0 = lambdaNode->GetFunctionArguments()[0];
  auto functionArgument1 = lambdaNode->GetFunctionArguments()[1];

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());
  auto loopVar0 = thetaNode->AddLoopVar(functionArgument0);
  auto loopVar1 = thetaNode->AddLoopVar(functionArgument1);

  auto dummyNodeTheta = TestOperation::create(thetaNode->subregion(), {}, { valueType });

  auto predicate = TestOperation::create(thetaNode->subregion(), {}, { controlType })->output(0);
  auto gammaNode = GammaNode::create(predicate, 2);
  auto entryVar0 = gammaNode->AddEntryVar(loopVar0.pre);
  auto entryVar1 = gammaNode->AddEntryVar(dummyNodeTheta->output(0));

  auto ctlConstant0 = ctlconstant_ auto dummyNodeGamma1 =
      TestOperation::create(gammaNode->subregion(1), {}, { valueType });

  auto controlConstant0 =
      ctlconstant_op::create(gammaNode->subregion(0), ControlValueRepresentation(0, 2));
  auto controlConstant1 =
      ctlconstant_op::create(gammaNode->subregion(1), ControlValueRepresentation(1, 2));

  auto controlExitVar = gammaNode->AddExitVar({ controlConstant0, controlConstant1 });
  auto exitVar0 =
      gammaNode->AddExitVar({ dummyNodeGamma0->output(0), entryVar0.branchArgument[1] });
  auto exitVar1 =
      gammaNode->AddExitVar({ entryVar1.branchArgument[0], dummyNodeGamma1->output(0) });

  thetaNode->predicate()->divert_to(controlExitVar.output);
  loopVar0.post->divert_to(exitVar0.output);
  loopVar1.post->divert_to(exitVar1.output);

  auto lambdaOutput = lambdaNode->finalize({ loopVar1.output });

  GraphExport::Create(*lambdaOutput, "test");

  // Act
  RunInvariantValueRedirection(*rvsdgModule);

  // Assert
  // We expect that the post value of both loop variables does not originate from the gamma any
  // longer.
  auto loopVars = thetaNode->GetLoopVars();
  assert(loopVars.size() == 2);

  // Loop variable 0 was dead after the loop, which means it is irrelevant what happens to it in
  // the last iteration of the loop. As the loop predicate originates from a control constant in
  // one of the gamma nodes' subregions, the loop variables' value is always the same as the one
  // from the gamma subregion with control constant 1 (i.e. loop repetition). This means we could
  // redirect the loop variable from the gamma to the respective entry variables' origin.
  assert(loopVars[0].post->origin() == loopVars[0].pre);

  // Loop variable 1 was dead at the beginning of each loop iteration, which means it is irrelevant
  // what happens to it except in the last iteration of the loop. As the loop predicate originates
  // from a control constant in a one of the gamma nodes' subregions, the loop variables' value is
  // always the same as the one from the gamma subregion with control constant 0 (i.e. loop exit).
  // This means we could redirect the loop variable from the gamma to the respective entry
  // variables' origin.
  assert(loopVars[1].post->origin() == dummyNodeTheta->output(0));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/PredicateCorrelationTests-testThetaGamma", testThetaGamma)
