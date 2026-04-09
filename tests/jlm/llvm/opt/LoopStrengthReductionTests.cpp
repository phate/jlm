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
  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&IVPostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);

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

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, ArithmeticCandidateOperationDependentOnInvalidInductionVariable)
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
  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode2.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

  // Act
  RunLoopStrengthReduction(rvsdgModule);

  // std::cout << "After: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsAfter = theta->GetLoopVars().size();

  // Assert

  // Check that no loop variables were added
  EXPECT_EQ(numLoopVarsAfter, numLoopVarsBefore);

  // Check that all users of the MUL node still use the MUL node
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), mulNode2.output(0));
    }
  }
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
  std::vector<jlm::rvsdg::Input *> oldAddNodeUsers;
  for (auto & user : addNode2.output(0)->Users())
  {
    oldAddNodeUsers.push_back(&user);
  }

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
  // TODO: This can probably be changed to TryGetSimpleNodeAndOptionalOp
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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&IVPostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);

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

  // Check that all users of the old ADD node now use the new induction variable
  for (auto & user : oldAddNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV.pre);
    }
  }
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

  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }
  std::vector<jlm::rvsdg::Input *> oldAddNodeUsers;
  for (auto & user : addNode2.output(0)->Users())
  {
    oldAddNodeUsers.push_back(&user);
  }

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
  const auto & addOperation1 =
      dynamic_cast<const IntegerAddOperation *>(&IV1PostOrigin->GetOperation());
  EXPECT_NE(addOperation1, nullptr);

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
  const auto & addOperation2 =
      dynamic_cast<const IntegerAddOperation *>(&IV2PostOrigin->GetOperation());
  EXPECT_NE(addOperation2, nullptr);

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

  // Check that all users of the old ADD node now use the new induction variable
  for (auto & user : oldAddNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV1.pre);
    }
  }

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV2.pre);
    }
  }
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

  auto loadNode = LoadNonVolatileOperation::Create(gep, { memoryState.pre }, intType, 32);
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ loadNode[0], c2.output(0) }, 32);

  auto storeNode = StoreNonVolatileOperation::Create(gep, subNode.output(0), { loadNode[1] }, 4);
  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(storeNode[0]);
  theta->set_predicate(matchResult);

  jlm::rvsdg::GraphExport::Create(*memoryState.output, "");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();

  std::vector<jlm::rvsdg::Input *> oldGepNodeUsers;
  for (auto & user : gep->Users())
  {
    oldGepNodeUsers.push_back(&user);
  }

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

  // Check that all users of the old GEP node now use the new induction variable
  for (auto & user : oldGepNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV.pre);
    }
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

TEST(LoopStrengthReductionTests, PlaceholderNestedLoopTest)
{
  // TODO
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
  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&outerIVPostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);

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

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), innerNewIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, PlaceholderNestedLoopWithGammaTest)
{
  // TODO
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

  // TODO: Probably a way to just use some graph import or something here
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

  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&outerIVPostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);

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

  // Check that all users of the old MUL node now use the new inner induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), innerNewIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, NestedThreeLoopTest)
{
  // TODO
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
  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&outerIVPostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);

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

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), innerNewIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, PlaceholderInitTest)
{
  // TODO
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
  std::vector<jlm::rvsdg::Input *> oldMulNodeUsers;
  for (auto & user : mulNode.output(0)->Users())
  {
    oldMulNodeUsers.push_back(&user);
  }

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

  // TODO
  const auto & IV1InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV1.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerMulOperation>(IV1InputNode->GetOperation()));
  const auto & mulOperation =
      dynamic_cast<const IntegerMulOperation *>(&IV1InputNode->GetOperation());
  EXPECT_NE(mulOperation, nullptr);

  auto lhs1 = IV1InputNode->input(0)->origin();
  auto rhs1 = IV1InputNode->input(1)->origin();
  EXPECT_EQ(lhs1, cv2);
  const auto rhsNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*rhs1);
  EXPECT_TRUE(jlm::rvsdg::is<IntegerConstantOperation>(rhsNode->GetOperation()));
  const auto & constantOperation =
      dynamic_cast<const IntegerConstantOperation *>(&rhsNode->GetOperation());
  EXPECT_NE(constantOperation, nullptr);
  EXPECT_EQ(constantOperation->Representation().to_uint(), 3u);

  // TODO
  const auto & IV2InputNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*newIV2.input->origin());
  EXPECT_TRUE(jlm::rvsdg::is<IntegerMulOperation>(IV2InputNode->GetOperation()));
  const auto & mulOperation2 =
      dynamic_cast<const IntegerMulOperation *>(&IV2InputNode->GetOperation());
  EXPECT_NE(mulOperation2, nullptr);

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
  const auto & addOperation =
      dynamic_cast<const IntegerAddOperation *>(&IV1PostOrigin->GetOperation());
  EXPECT_NE(addOperation, nullptr);
  // Check that LHS of the ADD is the pre value of the new IV1
  EXPECT_EQ(IV1PostOrigin->input(0)->origin(), newIV1.pre);
  // Check that LHS of the ADD is the pre value of the new IV2
  EXPECT_EQ(IV1PostOrigin->input(1)->origin(), newIV2.pre);

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user->origin())
    {
      EXPECT_EQ(user->origin(), newIV1.pre);
    }
  }
}

#pragma GCC diagnostic pop
