/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

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
#include <jlm/rvsdg/view.hpp>

#include <gtest/gtest.h>

void
RunLoopStrengthReduction(jlm::rvsdg::RvsdgModule & rvsdgModule)
{
  jlm::llvm::LoopStrengthReduction loopStrengthReduction;
  jlm::util::StatisticsCollector statisticsCollector;
  loopStrengthReduction.Run(rvsdgModule, statisticsCollector);
}

TEST(LoopStrengthReductionTests, SimpleCandidateOperation)
{
  // Tests strength reduction of a simple candidate operation j = 3 * i
  // i has the recurrence {0,+,2}. Applying strength reduction should result in us creating a new
  // induction variable for j with the recurrence {0,+,6}
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { pointerType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  const auto cv1 = lambda->AddContextVar(*arrPtr).inner;

  const auto & c0_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(cv1);            // arr ptr

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

  auto lambdaOutput = lambda->finalize({ lv2.output, memoryState.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "arrPtr");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();
  const auto oldMulNodeUsers = mulNode.output(0)->Users();

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
    if (user.origin())
    {
      EXPECT_EQ(user.origin(), newIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, CandidateOperationDependentOnInvalidInductionVariable)
{
  // Tests that applying strength reduction on a loop with candidate operation j = 3 * i, where i is
  // an invalid (geometric) induction variable i, results in no change
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { pointerType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  const auto cv1 = lambda->AddContextVar(*arrPtr).inner;

  const auto & c1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(cv1);          // arr ptr

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & mulNode1 =
      jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode2 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & sExtNode = SExtOperation::create(64, mulNode2.output(0));

  const auto & c0 = IntegerConstantOperation::Create(*theta->subregion(), 64, 0);
  const auto gep = GetElementPtrOperation::Create(
      lv2.pre,
      { c0.output(0), sExtNode },
      intArrayType,
      pointerType);

  auto loadNode = LoadNonVolatileOperation::Create(gep, { memoryState.pre }, intType, 32);
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ loadNode[0], c2.output(0) }, 32);

  auto storeNode = StoreNonVolatileOperation::Create(gep, subNode.output(0), { loadNode[1] }, 4);
  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);

  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ mulNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  lv1.post->divert_to(mulNode1.output(0));
  memoryState.post->divert_to(storeNode[0]);
  theta->set_predicate(matchResult);

  auto lambdaOutput = lambda->finalize({ lv2.output, memoryState.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "arrPtr");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();
  const auto oldMulNodeUsers = mulNode2.output(0)->Users();

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
    if (user.origin())
    {
      EXPECT_EQ(user.origin(), mulNode2.output(0));
    }
  }
}

TEST(LoopStrengthReductionTests, NestedCandidateOperation)
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
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { pointerType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  const auto cv1 = lambda->AddContextVar(*arrPtr).inner;

  const auto & c0_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(cv1);            // arr ptr

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & addNode1 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  const auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ mulNode.output(0), c1.output(0) }, 32);

  const auto & sExtNode = SExtOperation::create(64, addNode2.output(0));

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

  auto lambdaOutput = lambda->finalize({ lv2.output, memoryState.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "arrPtr");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();
  const auto oldAddNodeUsers = addNode2.output(0)->Users();

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
    if (user.origin())
    {
      EXPECT_EQ(user.origin(), newIV.pre);
    }
  }
}

TEST(LoopStrengthReductionTests, NestedCandidateOperationWithUsersForBoth)
{
  // Tests strength reduction of the two candidate operations j = 3 * i and k = j + 1 where both j
  // and k have users. i has the recurrence {0,+,2}. Since both j and k are being used, applying
  // strength reduction should result in us creating two new induction variables, one for j with the
  // recurrence {0,+,6} and one for k with the recurrence {1,+,6}
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { pointerType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  auto arrPtr = &jlm::rvsdg::GraphImport::Create(graph, pointerType, "arrPtr");
  const auto cv1 = lambda->AddContextVar(*arrPtr).inner;

  const auto & c0_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(cv1);            // arr ptr

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & addNode1 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c3.output(0) }, 32);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  const auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ mulNode.output(0), c1.output(0) }, 32);

  const auto & c0_2 = IntegerConstantOperation::Create(*theta->subregion(), 64, 0);
  const auto & sExtNode1 = SExtOperation::create(64, mulNode.output(0));
  const auto gep1 = GetElementPtrOperation::Create(
      lv2.pre,
      { c0_2.output(0), sExtNode1 },
      intArrayType,
      pointerType);
  auto loadNode = LoadNonVolatileOperation::Create(gep1, { memoryState.pre }, intType, 32);

  const auto & sExtNode2 = SExtOperation::create(64, addNode2.output(0));
  const auto gep2 = GetElementPtrOperation::Create(
      lv2.pre,
      { c0_2.output(0), sExtNode2 },
      intArrayType,
      pointerType);
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ loadNode[0], c2.output(0) }, 32);
  auto storeNode = StoreNonVolatileOperation::Create(gep2, subNode.output(0), { loadNode[1] }, 4);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ addNode1.output(0), c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  lv1.post->divert_to(addNode1.output(0));
  memoryState.post->divert_to(storeNode[0]);
  theta->set_predicate(matchResult);

  auto lambdaOutput = lambda->finalize({ lv2.output, memoryState.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "arrPtr");

  // std::cout << "Before: \n";
  // jlm::rvsdg::view(graph, stdout);

  const auto numLoopVarsBefore = theta->GetLoopVars().size();
  const auto oldMulNodeUsers = mulNode.output(0)->Users();
  const auto oldAddNodeUsers = addNode2.output(0)->Users();

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
    if (user.origin())
    {
      EXPECT_EQ(user.origin(), newIV1.pre);
    }
  }

  // Check that all users of the old MUL node now use the new induction variable
  for (auto & user : oldMulNodeUsers)
  {
    if (user.origin())
    {
      EXPECT_EQ(user.origin(), newIV2.pre);
    }
  }
}
