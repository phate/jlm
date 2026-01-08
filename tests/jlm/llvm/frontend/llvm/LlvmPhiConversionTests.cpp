/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IRReader/IRReader.h>

#include <string>

/**
 * Tests converting instances of ::llvm::PHINode.
 * Some of the operands have constant values, and some are results from the predecessors.
 * One of the phi node has its own result as one of its operands.
 *
 * The function corresponds to the C code
 * uint64_t popcount(uint64_t x) {
 *     uint64_t popcnt = 0;
 *
 *     while(true) {
 *         // 2 phi nodes with 3 possible predecessors here
 *         // one for x, and one for popcnt
 *
 *         if (x == 0)
 *             break;
 *         uint64_t rem = x % 2 == 1;
 *         x >>= 1;
 *         if (rem) {
 *             popcnt++;
 *             continue;
 *         }
 *         else {
 *             continue;
 *         }
 *     }
 *     return popcnt;
 * }
 */
TEST(LlvmPhiConversionTests, TestPhiConversion)
{
  // Arrange
  llvm::LLVMContext ctx;
  llvm::Module module("popcount.c", ctx);

  // Build LLVM module
  {
    llvm::IRBuilder builder(ctx);

    auto i64 = builder.getInt64Ty();
    auto prototype = llvm::FunctionType::get(i64, { i64 }, false);
    llvm::Function * function =
        llvm::Function::Create(prototype, llvm::Function::ExternalLinkage, "popcount", module);

    auto bb1 = llvm::BasicBlock::Create(ctx, "bb1", function);
    auto bb2 = llvm::BasicBlock::Create(ctx, "bb2", function);
    auto bb3 = llvm::BasicBlock::Create(ctx, "bb3", function);
    auto bb4 = llvm::BasicBlock::Create(ctx, "bb4", function);
    auto bb5 = llvm::BasicBlock::Create(ctx, "bb5", function);
    auto bb6 = llvm::BasicBlock::Create(ctx, "bb6", function);

    builder.SetInsertPoint(bb1); // Entry block
    builder.CreateBr(bb2);

    builder.SetInsertPoint(bb2); // Predecessors: bb1, bb4, bb5
    auto phiX = builder.CreatePHI(i64, 3);
    auto phiPopcount = builder.CreatePHI(i64, 3);

    auto xIs0 = builder.CreateICmpEQ(phiX, llvm::ConstantInt::get(i64, 0, false));
    builder.CreateCondBr(xIs0, bb6, bb3);

    builder.SetInsertPoint(bb3); // Predecessors: bb2
    auto rem = builder.CreateURem(phiX, llvm::ConstantInt::get(i64, 2, false));
    auto remEq1 = builder.CreateICmpEQ(rem, llvm::ConstantInt::get(i64, 1, false));
    auto halfX = builder.CreateLShr(phiX, llvm::ConstantInt::get(i64, 1, false));
    builder.CreateCondBr(remEq1, bb4, bb5);

    builder.SetInsertPoint(bb4); // Predecessor: bb3
    auto popcountPlus1 = builder.CreateAdd(phiPopcount, llvm::ConstantInt::get(i64, 1, false));
    builder.CreateBr(bb2);

    builder.SetInsertPoint(bb5); // Predecessor: bb3
    builder.CreateBr(bb2);

    builder.SetInsertPoint(bb6); // Predecessor: bb2
    builder.CreateRet(phiPopcount);

    // Finally give the phi nodes their operands
    phiX->addIncoming(function->getArg(0), bb1);
    phiX->addIncoming(halfX, bb4);
    phiX->addIncoming(halfX, bb5);

    phiPopcount->addIncoming(llvm::ConstantInt::get(i64, 0, false), bb1);
    phiPopcount->addIncoming(popcountPlus1, bb4);
    phiPopcount->addIncoming(phiPopcount, bb5);
  }

  // jlm::tests::print(module);

  // Act
  auto ipgmod = jlm::llvm::ConvertLlvmModule(module);

  // print(*ipgmod, stdout);

  // Assert
  // First traverse from the function's entry node to bb2
  auto popcount =
      jlm::util::assertedCast<const jlm::llvm::FunctionNode>(ipgmod->ipgraph().find("popcount"));
  auto entry_node = popcount->cfg()->entry();
  EXPECT_TRUE(entry_node->single_successor());
  auto bb1_node = entry_node->OutEdge(0)->sink();
  EXPECT_TRUE(bb1_node->single_successor());
  auto bb2_node = bb1_node->OutEdge(0)->sink();
  auto bb2 = jlm::util::assertedCast<jlm::llvm::BasicBlock>(bb2_node);

  // The first two three address codes should be the phi representing x and popcnt respectively
  auto tacs = bb2->begin();
  auto & phiX = *tacs;
  auto & phiPopcnt = *std::next(tacs);

  // Check that they are both phi operations
  auto phiXOp = *jlm::util::assertedCast<const jlm::llvm::SsaPhiOperation>(&phiX->operation());
  auto phiPopcntOp =
      *jlm::util::assertedCast<const jlm::llvm::SsaPhiOperation>(&phiPopcnt->operation());

  // Both phi nodes should have 3 operands, representing the loop entry, and the two "continue"s
  EXPECT_EQ(phiX->noperands(), 3);
  // The phi node for x takes its value from the function arg in the first operand
  EXPECT_EQ(phiX->operand(0), popcount->cfg()->entry()->argument(0));
  // The last two predecessor basic blocks both use the same value for x
  EXPECT_EQ(phiX->operand(1), phiX->operand(2));

  EXPECT_EQ(phiPopcnt->noperands(), 3);
  // The first operand of the phi node is the constant integer 0
  auto constant0variable =
      jlm::util::assertedCast<const jlm::llvm::ThreeAddressCodeVariable>(phiPopcnt->operand(0));
  auto constant0op = jlm::util::assertedCast<const jlm::llvm::IntegerConstantOperation>(
      &constant0variable->tac()->operation());
  EXPECT_EQ(constant0op->Representation(), 0);
  // The last operand of the popcnt phi is the result of the phi itself
  EXPECT_EQ(phiPopcnt->operand(2), phiPopcnt->result(0));
}

/**
 * Tests converting instances of ::llvm::PHINode where some of the predecessors are "dead".
 * A dead predecessor is a basic block that is not reachable from the function's entry.
 * This test has one phi node with 4 operands, where two of them are dead.
 */
TEST(LlvmPhiConversionTests, TestPhiOperandElision)
{
  // Arrange
  llvm::LLVMContext ctx;
  llvm::Module module("phi-elide.c", ctx);

  // Build LLVM module
  {
    llvm::IRBuilder builder(ctx);

    auto i64 = builder.getInt64Ty();
    auto prototype = llvm::FunctionType::get(i64, { i64 }, false);
    llvm::Function * function =
        llvm::Function::Create(prototype, llvm::Function::ExternalLinkage, "phi_elide", module);

    auto bb1 = llvm::BasicBlock::Create(ctx, "bb1", function);
    auto bb2 = llvm::BasicBlock::Create(ctx, "bb2", function);
    auto bb3 = llvm::BasicBlock::Create(ctx, "bb3", function);
    auto bb4 = llvm::BasicBlock::Create(ctx, "bb4", function);
    auto bb5 = llvm::BasicBlock::Create(ctx, "bb5", function);

    builder.SetInsertPoint(bb1); // entry block
    auto xIs0 = builder.CreateICmpEQ(function->getArg(0), llvm::ConstantInt::get(i64, 0));
    builder.CreateCondBr(xIs0, bb4, bb5);

    builder.SetInsertPoint(bb2); // No predecessors (dead)
    auto xPlus1 = builder.CreateAdd(function->getArg(0), llvm::ConstantInt::get(i64, 1));
    auto xIs1 = builder.CreateICmpEQ(function->getArg(0), llvm::ConstantInt::get(i64, 1));
    builder.CreateCondBr(xIs1, bb3, bb5);

    builder.SetInsertPoint(bb3); // Predecessors: bb2 (dead)
    builder.CreateBr(bb5);

    builder.SetInsertPoint(bb4); // Predecessors: bb1
    auto xPlus2 = builder.CreateAdd(function->getArg(0), llvm::ConstantInt::get(i64, 2));
    builder.CreateBr(bb5);

    builder.SetInsertPoint(bb5); // Predecessors: bb1, bb2 (dead), bb3 (dead), bb4
    auto bb5phi = builder.CreatePHI(i64, 4);
    builder.CreateRet(bb5phi);

    bb5phi->addIncoming(llvm::ConstantInt::get(i64, 0), bb1);
    bb5phi->addIncoming(xPlus1, bb2);              // Dead
    bb5phi->addIncoming(function->getArg(0), bb3); // Dead
    bb5phi->addIncoming(xPlus2, bb4);
  }

  module.print(llvm::errs(), nullptr);

  // Act
  auto ipgmod = jlm::llvm::ConvertLlvmModule(module);

  print(*ipgmod, stdout);

  // Assert
  // Get the CFG of the function
  auto phi_elide =
      jlm::util::assertedCast<const jlm::llvm::FunctionNode>(ipgmod->ipgraph().find("phi_elide"));

  // Traverse the cfg and save every phi node
  size_t numBasicBlocks = 0;
  std::vector<jlm::llvm::ThreeAddressCode *> phiTacs;
  for (auto & bb : *phi_elide->cfg())
  {
    numBasicBlocks++;
    for (auto tac : bb)
    {
      if (jlm::rvsdg::is<jlm::llvm::SsaPhiOperation>(tac->operation()))
        phiTacs.push_back(tac);
    }
  }

  // There should be 3 basic blocks left (bb1, bb4, bb5)
  EXPECT_EQ(numBasicBlocks, 3);
  // There should be exactly one phi three address code
  EXPECT_EQ(phiTacs.size(), 1);
  auto phiTac = phiTacs[0];
  // The phi should have two operands
  EXPECT_EQ(phiTac->noperands(), 2);
  // The first phi operand should be a constant 0
  auto constant0variable =
      jlm::util::assertedCast<const jlm::llvm::ThreeAddressCodeVariable>(phiTac->operand(0));
  auto constant0op = jlm::util::assertedCast<const jlm::llvm::IntegerConstantOperation>(
      &constant0variable->tac()->operation());
  EXPECT_EQ(constant0op->Representation(), 0);
}
