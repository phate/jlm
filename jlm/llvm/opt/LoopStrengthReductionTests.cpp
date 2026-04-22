/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/view.hpp>

#include <gtest/gtest.h>

void
RunLoopStrengthReduction(jlm::rvsdg::RvsdgModule & rvsdgModule)
{
  jlm::llvm::LoopStrengthReduction loopStrengthReduction;
  jlm::util::StatisticsCollector statisticsCollector;
  loopStrengthReduction.Run(rvsdgModule, statisticsCollector);
}

TEST(LoopStrengthReductionTests, SimpleArithmeticCandidateOperation)
{
  // Tests strength reduction of a simple candidate operation j = 3 * i
  // i has the recurrence {0,+,2}. Applying strength reduction should result in us creating a new
  // induction variable for j with the recurrence {0,+,6}
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // i

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { mulNode.output(0), memoryState.pre },
      { memoryStateType });

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(addNode.output(0));
  memoryState.post->divert_to(testOperation->output(0));

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 1);

  auto newIV = theta->GetLoopVars()[numLoopVarsAfter - 1];

  // Check the start value
  const auto & IVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(IVInputNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&IVInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & IVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(IVPostOrigin->GetOperation()));

  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(IVPostOrigin->input(0)->origin(), newIV.pre);

  // Check that RHS of the ADD is an integer constant with the step value (6)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_NE(rhsConstantOperation, nullptr);
  EXPECT_EQ(rhsConstantOperation->Representation().to_uint(), 6u);

  // Check that the test operation now uses the new induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), newIV.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationDependentOnInvalidInductionVariable)
{
  // Tests that applying strength reduction on a loop with candidate operation j = 3 * i, where i is
  // an invalid (geometric) induction variable i, results in no change
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // i

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & mulNode1 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode2 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { mulNode2.output(0), memoryState.pre },
      { memoryStateType });

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ mulNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(mulNode1.output(0));
  memoryState.post->divert_to(testOperation->output(0));

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that no loop variables were added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore);

  // Check that the test operation still uses the MUL node (no change)
  EXPECT_EQ(testOperation->input(0)->origin(), mulNode2.output(0));
}

TEST(LoopStrengthReductionTests, NestedArithmeticCandidateOperation)
{
  // Tests that we don't create unnecessary induction variables for nested operations.
  // It applies strength reduction on a loop with candidate operations j = 3 * i and k = j + 1.
  // i has the recurrence {0,+,2}. Even though k is defined with an addition, it's definition (scev
  // tree) includes the multiplication from j, and it is therefore a candidate to be reduced.
  // Since j has no other users than k, applying strength reduction should result in us only
  // creating a new induction variable for k with the recurrence {1,+,6}
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // i

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ mulNode.output(0), c1.output(0) }, 32);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { addNode2.output(0), memoryState.pre },
      { memoryStateType });

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(testOperation->output(0));

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that only one loop variable was added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 1);

  auto newIV = theta->GetLoopVars()[numLoopVarsAfter - 1];

  // Check the start value
  const auto & IVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(IVInputNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&IVInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 1u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & IVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(IVPostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(IVPostOrigin->input(0)->origin(), newIV.pre);
  // Check that RHS of the ADD is an integer constant with the step value (6)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_NE(rhsConstantOperation, nullptr);
  EXPECT_EQ(rhsConstantOperation->Representation().to_uint(), 6u);

  // Check that the test operation now uses the new induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), newIV.pre);
}

TEST(LoopStrengthReductionTests, NestedArithmeticCandidateOperationWithUsersForBoth)
{
  // Tests strength reduction of the two candidate operations j = 3 * i and k = j + 1 where both j
  // and k have users. i has the recurrence {0,+,2}. Since both j and k are being used, applying
  // strength reduction should result in us creating two new induction variables, one for j with the
  // recurrence {0,+,6} and one for k with the recurrence {1,+,6}
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // i

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ mulNode.output(0), c1.output(0) }, 32);

  auto testOperation1 = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { mulNode.output(0), memoryState.pre },
      { memoryStateType });

  auto testOperation2 = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { addNode2.output(0), testOperation1->output(0) },
      { memoryStateType });

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(testOperation2->output(0));

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that two new loop variables were added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 2);

  auto newIV1 = theta->GetLoopVars()[numLoopVarsAfter - 1];
  auto newIV2 = theta->GetLoopVars()[numLoopVarsAfter - 2];

  // Check their start values
  const auto & IV1InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV1.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(IV1InputNode->GetOperation()));
  auto constantOperation1 =
      dynamic_cast<const IntegerConstantOperation *>(&IV1InputNode->GetOperation());
  EXPECT_NE(constantOperation1, nullptr);
  EXPECT_EQ(constantOperation1->Representation().to_uint(), 1u);

  const auto & IV2InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV2.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(IV2InputNode->GetOperation()));
  auto constantOperation2 =
      dynamic_cast<const IntegerConstantOperation *>(&IV2InputNode->GetOperation());
  EXPECT_NE(constantOperation2, nullptr);
  EXPECT_EQ(constantOperation2->Representation().to_uint(), 0u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & IV1PostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV1.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(IV1PostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV1
  EXPECT_EQ(IV1PostOrigin->input(0)->origin(), newIV1.pre);
  // Check that RHS of the ADD is an integer constant with the step value (6)
  const auto & addRhsInputNode1 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IV1PostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode1));
  const auto & rhsConstantOperation1 =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode1->GetOperation());
  EXPECT_NE(rhsConstantOperation1, nullptr);
  EXPECT_EQ(rhsConstantOperation1->Representation().to_uint(), 6u);

  const auto & IV2PostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV2.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(IV2PostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV2
  EXPECT_EQ(IV2PostOrigin->input(0)->origin(), newIV2.pre);
  // Check that RHS of the ADD is an integer constant with the step value (6)
  const auto & addRhsInputNode2 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IV2PostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode2));
  const auto & rhsConstantOperation2 =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode2->GetOperation());
  EXPECT_NE(rhsConstantOperation2, nullptr);
  EXPECT_EQ(rhsConstantOperation2->Representation().to_uint(), 6u);

  // Check that the test operations now use their respective new induction variables
  EXPECT_EQ(testOperation1->input(0)->origin(), newIV2.pre);
  EXPECT_EQ(testOperation2->input(0)->origin(), newIV1.pre);
}

TEST(LoopStrengthReductionTests, SimpleGEPCandidateOperation)
{
  // Tests strength reduction of a GEP operation which takes j = 3 * i as index. i has the
  // recurrence {0,+,2} and j has recurrence {0,+,6}. Since the GEP has int type, the recurrence for
  // the GEP is {Init(a1),+,24}. We expect to strength reduce the original GEP to a new loop
  // variable which is incremented by the value of a new GEP with byte type and offset 24
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(arrPtr);         // arr ptr

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & addNode1 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & sExtNode = SExtOperation::create(64, mulNode.output(0));

  const auto & c0_2 = IntegerConstantOperation::Create(*theta->subregion(), 64, 0);
  const auto gep = GetElementPtrOperation::Create(
      lv2.pre,
      { c0_2.output(0), sExtNode },
      intArrayType,
      pointerType);

  auto loadOutputs = LoadNonVolatileOperation::Create(gep, { memoryState.pre }, intType, 32);
  auto & subNode =
      jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ loadOutputs[0], c2.output(0) }, 32);

  auto storeOutputs =
      StoreNonVolatileOperation::Create(gep, subNode.output(0), { loadOutputs[1] }, 4);
  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(storeOutputs[0]);
  theta->set_predicate(matchResult);

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  std::vector<jlm::rvsdg::Input *> oldGepNodeUsers;

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 1);
  auto newIV = theta->GetLoopVars()[numLoopVarsAfter - 1];

  // Check that the post value of the new IV comes from a new GEP node
  const auto & IVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<GetElementPtrOperation>(IVPostOrigin->GetOperation()));
  const auto & gepOperation =
      dynamic_cast<const GetElementPtrOperation *>(&IVPostOrigin->GetOperation());
  EXPECT_NE(gepOperation, nullptr);
  // Check that it has the right type
  EXPECT_EQ(gepOperation->GetPointeeType(), *jlm::rvsdg::BitType::Create(8));
  // Check that base address of the GEP is the pre value of the new IV
  EXPECT_EQ(IVPostOrigin->input(0)->origin(), newIV.pre);
  // Check that index of the GEP is an integer constant with the step value
  const auto & gepIndexInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(gepIndexInputNode));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&gepIndexInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().nbits(), 64u);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 24u);

  // Check that the both the load and store nodes use the new induction variable as address
  auto loadNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*loadOutputs[0]);
  EXPECT_NE(loadNode, nullptr);
  EXPECT_TRUE(jlm::rvsdg::is<LoadNonVolatileOperation>(loadNode->GetOperation()));
  EXPECT_EQ(LoadOperation::AddressInput(*loadNode).origin(), newIV.pre);

  auto storeNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*storeOutputs[0]);
  EXPECT_NE(storeNode, nullptr);
  EXPECT_TRUE(jlm::rvsdg::is<StoreNonVolatileOperation>(storeNode->GetOperation()));
  EXPECT_EQ(StoreOperation::AddressInput(*storeNode).origin(), newIV.pre);
}

TEST(LoopStrengthReductionTests, GEPCandidateOperationWithNAryStart)
{
  // Tests strength reduction of a GEP operation which takes j = (3 * i) + 4 as index. i has the
  // recurrence {0,+,2} and j has recurrence {4,+,6}. Since the GEP has int type, the recurrence for
  // the GEP is {(Init(a1) + 16),+,24}. We expect to strength reduce the original GEP to a new loop
  // variable which has as input value a GEP operation with the ptr Init(a1) and 16 as offset,
  // and is incremented by the value of a new GEP with byte type and offset 24
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState = theta->AddLoopVar(mem);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(arrPtr);         // arr ptr

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);
  const auto & c4 = IntegerConstantOperation::Create(*theta->subregion(), 32, 4);
  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ mulNode.output(0), c4.output(0) }, 32);

  const auto & sExtNode = SExtOperation::create(64, addNode2.output(0));

  const auto & c0_2 = IntegerConstantOperation::Create(*theta->subregion(), 64, 0);
  const auto gep = GetElementPtrOperation::Create(
      lv2.pre,
      { c0_2.output(0), sExtNode },
      intArrayType,
      pointerType);

  const auto & c10 = IntegerConstantOperation::Create(*theta->subregion(), 32, 10);

  auto storeOutputs = StoreNonVolatileOperation::Create(gep, c10.output(0), { memoryState.pre }, 4);
  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(storeOutputs[0]);
  theta->set_predicate(matchResult);

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  std::vector<jlm::rvsdg::Input *> oldGepNodeUsers;

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 1);
  auto newIV = theta->GetLoopVars()[numLoopVarsAfter - 1];

  // Check that the base address of the new induction variable is a GEP operation with the original
  // array ptr as lhs and the right offset as rhs
  const auto & newIVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<GetElementPtrOperation>(newIVInputNode->GetOperation()));
  auto lhs = newIVInputNode->input(0)->origin();
  auto rhs = newIVInputNode->input(1)->origin();
  EXPECT_EQ(lhs, arrPtr);
  const auto rhsNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*rhs);
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(rhsNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&rhsNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 16u);

  // Check that the post value of the new IV comes from a new GEP node
  const auto & IVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<GetElementPtrOperation>(IVPostOrigin->GetOperation()));
  const auto & stepGepOperation =
      dynamic_cast<const GetElementPtrOperation *>(&IVPostOrigin->GetOperation());
  EXPECT_NE(stepGepOperation, nullptr);
  // Check that it has the right type
  EXPECT_EQ(stepGepOperation->GetPointeeType(), *jlm::rvsdg::BitType::Create(8));

  // Check that index of the GEP is an integer constant with the step value
  const auto & gepIndexInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*IVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(gepIndexInputNode));
  const auto & indexConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&gepIndexInputNode->GetOperation());
  EXPECT_NE(indexConstantOperation, nullptr);
  EXPECT_EQ(indexConstantOperation->Representation().nbits(), 64u);
  EXPECT_EQ(indexConstantOperation->Representation().to_uint(), 24u);

  // Check that the both the load and store nodes use the new induction variable as address
  auto storeNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*storeOutputs[0]);
  EXPECT_NE(storeNode, nullptr);
  EXPECT_TRUE(jlm::rvsdg::is<StoreNonVolatileOperation>(storeNode->GetOperation()));
  EXPECT_EQ(StoreOperation::AddressInput(*storeNode).origin(), newIV.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationInNestedLoopTest)
{
  // Tests strength reduction of a variable with the recurrence {{0,+,3}<1>}<2>. Since {0,+,3}<1> is
  // invariant in the loop with ID 2, this is treated as a constant, and is hoisted into the loop
  // "preheader", and used as the input value of a new loop variable.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState1 = theta1->AddLoopVar(mem);
  const auto lv1_1 = theta1->AddLoopVar(c0.output(0)); // i

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_1.pre, c1.output(0) }, 32);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1_1.post->divert_to(addNode1.output(0));

  const auto & theta2 = jlm::rvsdg::ThetaNode::create(theta1->subregion());

  const auto lv1_2 = theta2->AddLoopVar(lv1_1.pre);  // i (but in inner loop)
  const auto lv2 = theta2->AddLoopVar(c1.output(0)); // x
  const auto memoryState2 = theta2->AddLoopVar(memoryState1.pre);

  const auto & c2 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 2);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1_2.pre, c3.output(0) }, 32);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode2.output(0), c10.output(0) }, 32);
  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta2->subregion(),
      { mulNode.output(0), memoryState2.pre },
      { memoryStateType });

  theta2->set_predicate(matchResult2);
  lv2.post->divert_to(addNode2.output(0));
  memoryState2.post->divert_to(testOperation->output(0));

  memoryState1.post->divert_to(memoryState2.output);
  jlm::rvsdg::GraphExport::Create(*memoryState1.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsBefore = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsBefore = theta2->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsAfter = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsAfter = theta2->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added in both the inner and outer loop
  EXPECT_EQ(outerNumLoopVarsAfter, outerNumLoopVarsBefore + 1);
  EXPECT_EQ(innerNumLoopVarsAfter, innerNumLoopVarsBefore + 1);

  auto outerNewIV = theta1->GetLoopVars()[outerNumLoopVarsAfter - 1];
  auto innerNewIV = theta2->GetLoopVars()[innerNumLoopVarsAfter - 1];

  // Check the start value
  const auto & outerIVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(outerIVInputNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&outerIVInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & outerIVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(outerIVPostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(outerIVPostOrigin->input(0)->origin(), outerNewIV.pre);
  // Check that RHS of the ADD is an integer constant with the step value (3)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerIVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_NE(rhsConstantOperation, nullptr);
  EXPECT_EQ(rhsConstantOperation->Representation().to_uint(), 3u);

  // Check that the new inner induction variable uses the pre value of the outer IV as start value
  EXPECT_EQ(innerNewIV.input->origin(), outerNewIV.pre);
  // And that it is not modified in the inner loop
  EXPECT_EQ(innerNewIV.post->origin(), innerNewIV.pre);

  // Check that the test operation now uses the new inner induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), innerNewIV.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationInNestedLoopWithGammaTest)
{
  // Does essentially the same as CandidateOperationInNestedLoopTest, but simulating the structure
  // of a head-controlled loop. Meaning we have interchanging theta and gamma nodes. This means that
  // the hoisted value must be routed down through from the outer loop to the inner loop.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();
  const auto controlType = jlm::rvsdg::ControlType::Create(2);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, controlType, "");
  auto m = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto gamma1 = jlm::rvsdg::GammaNode::create(c, 2);
  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  auto ev1 = gamma1->AddEntryVar(c0.output(0));
  auto mem1 = gamma1->AddEntryVar(m);

  const auto & theta1 = jlm::rvsdg::ThetaNode::create(gamma1->subregion(1));

  const auto memoryState1 = theta1->AddLoopVar(mem1.branchArgument[1]);
  const auto lv1_1 = theta1->AddLoopVar(ev1.branchArgument[1]); // i

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_1.pre, c1.output(0) }, 32);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1_1.post->divert_to(addNode1.output(0));

  const auto & c10 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 10);
  auto & sltNode2 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ c1.output(0), c10.output(0) }, 32);
  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  const auto & gamma2 = jlm::rvsdg::GammaNode::create(matchResult2, 2);
  auto ev2 = gamma2->AddEntryVar(c1.output(0));
  auto mem2 = gamma2->AddEntryVar(memoryState1.pre);
  auto ev3 = gamma2->AddEntryVar(lv1_1.pre);

  const auto & theta2 = jlm::rvsdg::ThetaNode::create(gamma2->subregion(1));

  const auto lv2 = theta2->AddLoopVar(ev2.branchArgument[1]);   // x
  const auto lv1_2 = theta2->AddLoopVar(ev3.branchArgument[1]); // i (but in inner loop)
  const auto memoryState2 = theta2->AddLoopVar(mem2.branchArgument[1]);

  const auto & c2 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 2);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1_2.pre, c3.output(0) }, 32);

  const auto & c10_2 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode3 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode2.output(0), c10_2.output(0) }, 32);
  const auto matchResult3 =
      jlm::rvsdg::MatchOperation::Create(*sltNode3.output(0), { { 1, 1 } }, 0, 2);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta2->subregion(),
      { mulNode.output(0), memoryState2.pre },
      { memoryStateType });

  theta2->set_predicate(matchResult3);
  lv2.post->divert_to(addNode2.output(0));
  memoryState2.post->divert_to(testOperation->output(0));

  auto exitVar1 = gamma2->AddExitVar({ mem2.branchArgument[0], memoryState2.output });
  memoryState1.post->divert_to(exitVar1.output);
  auto exitVar2 = gamma1->AddExitVar({ mem1.branchArgument[0], memoryState1.output });

  jlm::rvsdg::GraphExport::Create(*exitVar2.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsBefore = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsBefore = theta2->GetLoopVars().size();

  const auto outerNumEntryVarsBefore = gamma1->GetEntryVars().size();
  const auto innerNumEntryVarsBefore = gamma2->GetEntryVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsAfter = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsAfter = theta2->GetLoopVars().size();

  const auto outerNumEntryVarsAfter = gamma1->GetEntryVars().size();
  const auto innerNumEntryVarsAfter = gamma2->GetEntryVars().size();

  // Assert

  // Check that a new loop variable was added in both the inner and outer loop
  EXPECT_EQ(outerNumLoopVarsAfter, outerNumLoopVarsBefore + 1);
  EXPECT_EQ(innerNumLoopVarsAfter, innerNumLoopVarsBefore + 1);

  // Check that a new entry variable was added in the inner gamma, but not in the outer gamma
  EXPECT_EQ(innerNumEntryVarsAfter, innerNumEntryVarsBefore + 1);
  EXPECT_EQ(outerNumEntryVarsAfter, outerNumEntryVarsBefore);

  auto outerNewIV = theta1->GetLoopVars()[outerNumLoopVarsAfter - 1];
  auto innerNewIV = theta2->GetLoopVars()[innerNumLoopVarsAfter - 1];

  // Check the start value
  const auto & outerIVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(outerIVInputNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&outerIVInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & outerIVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(outerIVPostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(outerIVPostOrigin->input(0)->origin(), outerNewIV.pre);
  // Check that RHS of the ADD is an integer constant with the step value (3)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerIVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_NE(rhsConstantOperation, nullptr);
  EXPECT_EQ(rhsConstantOperation->Representation().to_uint(), 3u);

  // Check that the new outer IV has been routed correctly through the gamma nodes
  auto innerNewEV = gamma2->GetEntryVars()[innerNumEntryVarsAfter - 1];
  // Check that the new entry variable has the pre value of the new IV as input
  EXPECT_EQ(innerNewEV.input->origin(), outerNewIV.pre);

  // Check that the new inner IV has the branch argument of the new inner entry variable as input
  EXPECT_EQ(innerNewIV.input->origin(), innerNewEV.branchArgument[1]);

  // Check that the new inner IV is not modified in the inner loop
  EXPECT_EQ(innerNewIV.post->origin(), innerNewIV.pre);

  // Check that the test operation now uses the new inner induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), innerNewIV.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationInThreeLevelNestedLoopTest)
{
  // Similar to the previous two tests, but with one layer deeper of nesting. This test exists to
  // ensure that the routing of hoisted variables functions correctly.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  const auto memoryState1 = theta1->AddLoopVar(mem);
  const auto lv1_1 = theta1->AddLoopVar(c0.output(0)); // i

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_1.pre, c1.output(0) }, 32);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1_1.post->divert_to(addNode1.output(0));

  const auto & theta2 = jlm::rvsdg::ThetaNode::create(theta1->subregion());

  const auto lv1_2 = theta2->AddLoopVar(lv1_1.pre);  // i (but in middle loop)
  const auto lv2 = theta2->AddLoopVar(c1.output(0)); // x
  const auto memoryState2 = theta2->AddLoopVar(memoryState1.pre);

  const auto & c2 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 2);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c2.output(0) }, 32);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode2.output(0), c10.output(0) }, 32);
  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  theta2->set_predicate(matchResult2);
  lv2.post->divert_to(addNode2.output(0));

  const auto & theta3 = jlm::rvsdg::ThetaNode::create(theta2->subregion());

  const auto lv1_3 = theta3->AddLoopVar(lv1_2.pre);  // i (but in inner loop)
  const auto lv3 = theta3->AddLoopVar(c2.output(0)); // y
  const auto memoryState3 = theta3->AddLoopVar(memoryState2.pre);

  const auto & c4 = IntegerConstantOperation::Create(*theta3->subregion(), 32, 4);
  auto & addNode3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, c4.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta3->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1_3.pre, c3.output(0) }, 32);

  const auto & c20 = IntegerConstantOperation::Create(*theta3->subregion(), 32, 20);
  auto & sltNode3 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode3.output(0), c20.output(0) }, 32);
  const auto matchResult3 =
      jlm::rvsdg::MatchOperation::Create(*sltNode3.output(0), { { 1, 1 } }, 0, 2);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta3->subregion(),
      { mulNode.output(0), memoryState3.pre },
      { memoryStateType });

  theta3->set_predicate(matchResult3);
  lv3.post->divert_to(addNode3.output(0));
  memoryState3.post->divert_to(testOperation->output(0));

  lv1_2.post->divert_to(lv1_3.output);
  memoryState2.post->divert_to(memoryState3.output);

  memoryState1.post->divert_to(memoryState2.output);
  jlm::rvsdg::GraphExport::Create(*memoryState1.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsBefore = theta1->GetLoopVars().size();
  const auto middleNumLoopVarsBefore = theta2->GetLoopVars().size();
  const auto innerNumLoopVarsBefore = theta3->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsAfter = theta1->GetLoopVars().size();
  const auto middleNumLoopVarsAfter = theta2->GetLoopVars().size();
  const auto innerNumLoopVarsAfter = theta3->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added in all three loops
  EXPECT_EQ(outerNumLoopVarsAfter, outerNumLoopVarsBefore + 1);
  EXPECT_EQ(middleNumLoopVarsAfter, middleNumLoopVarsBefore + 1);
  EXPECT_EQ(innerNumLoopVarsAfter, innerNumLoopVarsBefore + 1);

  auto outerNewIV = theta1->GetLoopVars()[outerNumLoopVarsAfter - 1];
  auto middleNewIV = theta2->GetLoopVars()[middleNumLoopVarsAfter - 1];
  auto innerNewIV = theta3->GetLoopVars()[innerNumLoopVarsAfter - 1];

  // Check the start value
  const auto & outerIVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(outerIVInputNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&outerIVInputNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 0u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & outerIVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(outerIVPostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(outerIVPostOrigin->input(0)->origin(), outerNewIV.pre);
  // Check that RHS of the ADD is an integer constant with the step value (3)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerIVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_NE(rhsConstantOperation, nullptr);
  EXPECT_EQ(rhsConstantOperation->Representation().to_uint(), 3u);

  // Check that the new middle induction variable uses the pre value of the outer IV as start value
  EXPECT_EQ(middleNewIV.input->origin(), outerNewIV.pre);
  // And that it is not modified in the middle loop
  EXPECT_EQ(middleNewIV.post->origin(), middleNewIV.pre);

  // Check that the new inner induction variable uses the pre value of the middle IV as start value
  EXPECT_EQ(innerNewIV.input->origin(), middleNewIV.pre);
  // And that it is not modified in the inner loop
  EXPECT_EQ(innerNewIV.post->origin(), innerNewIV.pre);

  // Check that the test operation now uses the new inner induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), innerNewIV.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationWithInitTest)
{
  // Tests strength reduction of an operation with the following chrec:
  // {(Init(a1) * 3),+,(Init(a2) * 3)}<2>. In this example, we need to create new loop variables for
  // both the start value and the chrec, and add them together in the loop's subregion. This test
  // verifies that the hoisting of SCEV expressions functions correctly.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");
  auto i = &jlm::rvsdg::GraphImport::Create(graph, intType, "i");
  auto k = &jlm::rvsdg::GraphImport::Create(graph, intType, "k");

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ intType, intType }, { memoryStateType }),
          "f",
          Linkage::externalLinkage));
  auto cv1 = lambda->AddContextVar(*mem).inner;
  auto cv2 = lambda->AddContextVar(*i).inner;
  auto cv3 = lambda->AddContextVar(*k).inner;

  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(cv1); // memory state
  const auto lv1 = theta->AddLoopVar(cv2);         // i
  const auto lv2 = theta->AddLoopVar(cv3);         // k

  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv2.pre }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta->subregion(),
      { mulNode.output(0), memoryState.pre },
      { memoryStateType });

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(addNode.output(0));
  memoryState.post->divert_to(testOperation->output(0));

  auto res = lambda->finalize({ memoryState.output });

  jlm::rvsdg::GraphExport::Create(*res, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert
  // Check that two new loop variables were added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore + 2);

  auto newIV1 = theta->GetLoopVars()[numLoopVarsAfter - 1];
  auto newIV2 = theta->GetLoopVars()[numLoopVarsAfter - 2];

  // We are checking that the input value of the first IV is the value of i times 3.
  // This corresponds with the start value of the chrec being (Init(a1) * 3)
  const auto & IV1InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV1.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerMulOperation>(IV1InputNode->GetOperation()));
  auto lhs1 = IV1InputNode->input(0)->origin();
  auto rhs1 = IV1InputNode->input(1)->origin();
  EXPECT_EQ(lhs1, cv2);
  const auto rhsNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*rhs1);
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(rhsNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&rhsNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 3u);

  // We are checking that the input value of the second IV is the value of k times 3.
  // This corresponds with the start value of the chrec being (Init(a2) * 3)
  const auto & IV2InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV2.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerMulOperation>(IV2InputNode->GetOperation()));
  auto lhs2 = IV2InputNode->input(0)->origin();
  auto rhs2 = IV2InputNode->input(1)->origin();
  EXPECT_EQ(lhs2, cv3);
  const auto rhsNode2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*rhs2);
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(rhsNode2->GetOperation()));
  const auto & constantOperation2 =
      dynamic_cast<const IntegerConstantOperation *>(&rhsNode2->GetOperation());
  EXPECT_NE(constantOperation2, nullptr);
  EXPECT_EQ(constantOperation2->Representation().to_uint(), 3u);

  // Check that the post value of the new IV2 comes from a new ADD node
  const auto & IV1PostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV1.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(IV1PostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV1
  EXPECT_EQ(IV1PostOrigin->input(0)->origin(), newIV1.pre);
  // Check that RHS of the ADD is the pre value of the new IV2
  EXPECT_EQ(IV1PostOrigin->input(1)->origin(), newIV2.pre);

  // Check that the test operation now uses the new induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), newIV1.pre);
}

TEST(LoopStrengthReductionTests, CandidateOperationWithInitAndTracingTest)
{
  // In some cases, a recurrence from a higher level loop can be add-/mul-folded with a SCEV in
  // a loop that is at a lower level. In those cases, when hoisting the value, we need to trace the
  // output of the SCEV up to the corresponding output in the outermost loop's region.

  // Tests strength reduction of an operation with the following chrec:
  // {{(Init(a1) * 3),+,-1}<2>}<3>, where Init(a1) is in the loop with ID 3, which is at a lower
  // level than the recurrence it is in (loop with ID 2). In this example, we need to hoist the
  // recurrence {(Init(a1) * 3),+,-1}<2> which is invariant in the loop with ID 3. This is done by
  // creating a new induction variable in the outermost loop with (Init(a1) * 3) as the start value,
  // and which is decremented by 1 each iteration.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto mem = &jlm::rvsdg::GraphImport::Create(graph, memoryStateType, "");
  auto a = &jlm::rvsdg::GraphImport::Create(graph, intType, "a");
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ intType }, { memoryStateType }),
          "f",
          Linkage::externalLinkage));
  auto cv1 = lambda->AddContextVar(*mem).inner;
  auto cv2 = lambda->AddContextVar(*a).inner;

  const auto & c0 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & theta1 = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState1 = theta1->AddLoopVar(cv1);
  const auto lv1_1 = theta1->AddLoopVar(c0.output(0)); // i
  const auto lv2_1 = theta1->AddLoopVar(cv2);          // a

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_1.pre, c1.output(0) }, 32);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1_1.post->divert_to(addNode1.output(0));

  const auto & theta2 = jlm::rvsdg::ThetaNode::create(theta1->subregion());

  const auto lv1_2 = theta2->AddLoopVar(lv1_1.pre);  // i (but in inner loop)
  const auto lv2_2 = theta2->AddLoopVar(lv2_1.pre);  // a (but in inner loop)
  const auto lv3 = theta2->AddLoopVar(c1.output(0)); // k
  const auto memoryState2 = theta2->AddLoopVar(memoryState1.pre);

  const auto & c2 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 2);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv2_2.pre, c3.output(0) }, 32);

  auto & subNode =
      jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ mulNode.output(0), lv1_2.pre }, 32);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode2.output(0), c10.output(0) }, 32);
  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  auto testOperation = jlm::rvsdg::TestOperation::createNode(
      theta2->subregion(),
      { subNode.output(0), memoryState2.pre },
      { memoryStateType });

  theta2->set_predicate(matchResult2);
  lv3.post->divert_to(addNode2.output(0));
  memoryState2.post->divert_to(testOperation->output(0));

  memoryState1.post->divert_to(memoryState2.output);
  auto res = lambda->finalize({ memoryState1.output });
  jlm::rvsdg::GraphExport::Create(*res, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsBefore = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsBefore = theta2->GetLoopVars().size();

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto outerNumLoopVarsAfter = theta1->GetLoopVars().size();
  const auto innerNumLoopVarsAfter = theta2->GetLoopVars().size();

  // Assert

  // Check that a new loop variable was added in both the inner and outer loop
  EXPECT_EQ(outerNumLoopVarsAfter, outerNumLoopVarsBefore + 1);
  EXPECT_EQ(innerNumLoopVarsAfter, innerNumLoopVarsBefore + 1);

  auto outerNewIV = theta1->GetLoopVars()[outerNumLoopVarsAfter - 1];
  auto innerNewIV = theta2->GetLoopVars()[innerNumLoopVarsAfter - 1];

  // We are checking that the input value of the outer new IV is the value of a times 3.
  // This corresponds with the start value of the chrec being (Init(a1) * 3)
  const auto & outerIVInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerMulOperation>(outerIVInputNode->GetOperation()));
  auto lhs = outerIVInputNode->input(0)->origin();
  auto rhs = outerIVInputNode->input(1)->origin();
  EXPECT_EQ(lhs, cv2);
  const auto rhsNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*rhs);
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(rhsNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&rhsNode->GetOperation());
  EXPECT_EQ(constantOperation->Representation().to_uint(), 3u);

  // Check that the post value of the new IV comes from a new ADD node
  const auto & outerIVPostOrigin =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerNewIV.post->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerAddOperation>(outerIVPostOrigin->GetOperation()));
  // Check that LHS of the ADD is the pre value of the new IV
  EXPECT_EQ(outerIVPostOrigin->input(0)->origin(), outerNewIV.pre);
  // Check that RHS of the ADD is an integer constant with the step value (-1)
  const auto & addRhsInputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*outerIVPostOrigin->input(1)->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(addRhsInputNode));
  const auto & rhsConstantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&addRhsInputNode->GetOperation());
  EXPECT_EQ(rhsConstantOperation->Representation().to_int(), -1);

  // Check that the new inner induction variable uses the pre value of the outer IV as start value
  EXPECT_EQ(innerNewIV.input->origin(), outerNewIV.pre);
  // And that it is not modified in the inner loop
  EXPECT_EQ(innerNewIV.post->origin(), innerNewIV.pre);

  // Check that the test operation now uses the new inner induction variable
  EXPECT_EQ(testOperation->input(0)->origin(), innerNewIV.pre);
}
