/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <string>

/**
 * Tests converting instances of ::llvm::PHINode.
 * Some of the operands have constant values, and some are results from the predecessors.
 * One of the phi node has its own result as one its operands.
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
static int
TestPhiConversion()
{
  static const std::string POPCOUNT_PROGRAM =
      "source_filename = \"popcount.c\"\n"
      "target datalayout = "
      "\"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n"
      "target triple = \"x86_64-pc-linux-gnu\"\n"
      "\n"
      "define dso_local i64 @popcount(i64 noundef %0) #0 {\n"
      "  br label %bb2\n"
      "\n"
      "bb2:\n"
      "  ; The current value of x\n"
      "  %3 = phi i64 [%0, %1], [%8, %bb4], [%8, %bb5]\n"
      "  ; The current value of popcnt (1 bits seen)\n"
      "  ; Note that this phi is self-referential!\n"
      "  %4 = phi i64 [0, %1], [%9, %bb4], [%4, %bb5]\n"
      "\n"
      "  ; First check if x is 0, and if so jump to the exit\n"
      "  %5 = icmp eq i64 %3, 0\n"
      "  br i1 %5, label %bb6, label %bb3\n"
      "\n"
      "bb3:\n"
      "  ; Check if x % 2 is 1\n"
      "  %6 = urem i64 %3, 2\n"
      "  %7 = icmp eq i64 %6, 1\n"
      "\n"
      "  ; Also calculate x>>1 now\n"
      "  %8 = lshr i64 %3, 1\n"
      "\n"
      "  br i1 %7, label %bb4, label %bb5\n"
      "\n"
      "bb4:\n"
      "  ; Here x was odd, calculate popcnt+1\n"
      "  %9 = add i64 %4, 1\n"
      "  br label %bb2\n"
      "\n"
      "bb5:\n"
      "  ; Here x was even, no need to calculate anything\n"
      "  br label %bb2\n"
      "\n"
      "bb6:\n"
      "  ret i64 %4\n"
      "}\n"
      "\n"
      "attributes #0 = { noinline nounwind sspstrong uwtable \"frame-pointer\"=\"all\" "
      "\"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" "
      "\"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" "
      "\"target-features\"=\"+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"tune-cpu\"=\"generic\" }\n"
      "\n"
      "!llvm.module.flags = !{!0, !1, !2, !3, !4}\n"
      "!llvm.ident = !{!5}\n"
      "\n"
      "!0 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!1 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!2 = !{i32 7, !\"PIE Level\", i32 2}\n"
      "!3 = !{i32 7, !\"uwtable\", i32 2}\n"
      "!4 = !{i32 7, !\"frame-pointer\", i32 2}\n"
      "!5 = !{!\"clang version 18.1.8\"}\n"
      "!6 = distinct !{!6, !7}\n"
      "!7 = !{!\"llvm.loop.mustprogress\"}";

  // Arrange
  llvm::LLVMContext ctx;
  llvm::MemoryBufferRef program(POPCOUNT_PROGRAM, "popcount.ll");
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIR(program, diagnostic, ctx);

  // diagnostic.print("", llvm::outs());
  // llvm::outs().flush();
  assert(module);

  jlm::tests::print(*module);

  // Act
  auto ipgmod = jlm::llvm::ConvertLlvmModule(*module);

  // print(*ipgmod, stdout);

  // Assert
  // First traverse from the function's entry node to bb2
  auto popcount =
      jlm::util::AssertedCast<const jlm::llvm::function_node>(ipgmod->ipgraph().find("popcount"));
  auto entry_node = popcount->cfg()->entry();
  assert(entry_node->single_successor());
  auto bb1_node = entry_node->outedge(0)->sink();
  assert(bb1_node->single_successor());
  auto bb2_node = bb1_node->outedge(0)->sink();
  auto bb2 = jlm::util::AssertedCast<jlm::llvm::basic_block>(bb2_node);

  // The first two tac instructions should be the phi representing x and popcnt respectively
  auto tacs = bb2->begin();
  auto & phiX = *tacs;
  auto & phiPopcnt = *std::next(tacs);

  // Check that they are both phi operations
  auto phiXOp = *jlm::util::AssertedCast<const jlm::llvm::phi_op>(&phiX->operation());
  auto phiPopcntOp = *jlm::util::AssertedCast<const jlm::llvm::phi_op>(&phiPopcnt->operation());

  // Both phi nodes should have 3 operands, representing the loop entry, and the two "continue"s
  assert(phiX->noperands() == 3);
  // The phi node for x takes its value from the function arg in the first operand
  assert(phiX->operand(0) == popcount->cfg()->entry()->argument(0));
  // The last two predecessor basic blocks both use the same value for x
  assert(phiX->operand(1) == phiX->operand(2));

  assert(phiPopcnt->noperands() == 3);
  // The first operand of the phi node is the constant integer 0
  auto constant0variable =
      jlm::util::AssertedCast<const jlm::llvm::tacvariable>(phiPopcnt->operand(0));
  auto constant0op = jlm::util::AssertedCast<const jlm::rvsdg::bitconstant_op>(
      &constant0variable->tac()->operation());
  assert(constant0op->value() == 0);
  // The last operand of the popcnt phi is the result of the phi itself
  assert(phiPopcnt->operand(2) == phiPopcnt->result(0));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/TestLlvmPhiConversion-TestPhiConversion",
    TestPhiConversion)

/**
 * Tests converting instances of ::llvm::PHINode where some of the predecessors are "dead".
 * A dead predecessor is a basic block that is not reachable from the function's entry.
 * This test has one phi node with 4 operands, where two of them are dead,
 * and one with 2 operands, where one of them is dead.
 * The first should be converted to a jlm::llvm::phi_op with two operands,
 * while the second should become a direct reference to the value from the only alive predecessor.
 * Due to straightening, this last basic block is also merged into its predecessor.
 */
static int
TestPhiOperandElision()
{
  static const std::string PHI_OPERAND_ELISION =
      "source_filename = \"phi_elide.c\"\n"
      "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n"
      "target triple = \"x86_64-pc-linux-gnu\"\n"
      "\n"
      "; Function Attrs: noinline nounwind sspstrong uwtable\n"
      "define dso_local i64 @phi_elide(i64 noundef %0) #0 {\n"
      "  %2 = icmp eq i64 %0, 0\n"
      "  br i1 %2, label %bb4, label %bb5\n"
      "\n"
      "bb2: ; No predecessors (dead)\n"
      "  %3 = add i64 %0, 1\n"
      "  %4 = icmp eq i64 %0, 1\n"
      "  br i1 %4, label %bb3, label %bb5\n"
      "\n"
      "bb3: ; predecessor = bb2 (dead)\n"
      "  br label %bb5\n"
      "\n"
      "bb4: ; predecessor = entry-bb (%1)\n"
      "  %5 = add i64 %0, 2\n"
      "  br label %bb5\n"
      "\n"
      "bb5: ; predecessors = entry-bb, bb2 (dead), bb3 (dead), bb4\n"
      "  %6 = phi i64 [0, %1], [%3, %bb2], [%0, %bb3], [%5, %bb4]\n"
      "  br label %bb7\n"
      "\n"
      "bb6: ; No predecessors\n"
      "  br label %bb7\n"
      "\n"
      "bb7:\n"
      "  %7 = phi i64 [%6, %bb5], [poison, %bb6]\n"
      "  %8 = mul i64 %7, 10\n"
      "  ret i64 %8\n"
      "}\n"
      "\n"
      "attributes #0 = { noinline nounwind sspstrong uwtable \"frame-pointer\"=\"all\" \"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"tune-cpu\"=\"generic\" }\n"
      "\n"
      "!llvm.module.flags = !{!0, !1, !2, !3, !4}\n"
      "!llvm.ident = !{!5}\n"
      "\n"
      "!0 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!1 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!2 = !{i32 7, !\"PIE Level\", i32 2}\n"
      "!3 = !{i32 7, !\"uwtable\", i32 2}\n"
      "!4 = !{i32 7, !\"frame-pointer\", i32 2}\n"
      "!5 = !{!\"clang version 18.1.8\"}\n"
      "!6 = distinct !{!6, !7}\n"
      "!7 = !{!\"llvm.loop.mustprogress\"}";

  // Arrange
  llvm::LLVMContext ctx;
  llvm::MemoryBufferRef program(PHI_OPERAND_ELISION, "phi-elide.ll");
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIR(program, diagnostic, ctx);

  // diagnostic.print("", llvm::outs());
  // llvm::outs().flush();
  assert(module);

  // jlm::tests::print(*module);

  // Act
  auto ipgmod = jlm::llvm::ConvertLlvmModule(*module);

  // print(*ipgmod, stdout);

  // Assert
  // Get the CFG of the function
  auto phi_elide =
      jlm::util::AssertedCast<const jlm::llvm::function_node>(ipgmod->ipgraph().find("phi_elide"));

  // Traverse the cfg and save every phi node
  size_t numBasicBlocks = 0;
  std::vector<jlm::llvm::tac *> phiTacs;
  for (auto & bb : *phi_elide->cfg()) {
    numBasicBlocks++;
    for (auto tac : bb)
    {
      if (jlm::rvsdg::is<jlm::llvm::phi_op>(tac->operation()))
        phiTacs.push_back(tac);
    }
  }

  // There should be 3 basic blocks left (bb1, bb5, bb7)
  assert(numBasicBlocks == 3);
  // There should be exactly one phi tac
  assert(phiTacs.size() == 1);
  auto phiTac = phiTacs[0];
  // The phi should have two operands
  assert(phiTac->noperands() == 2);
  // The first phi operand should be a constant 0
  auto constant0variable =
      jlm::util::AssertedCast<const jlm::llvm::tacvariable>(phiTac->operand(0));
  auto constant0op = jlm::util::AssertedCast<const jlm::rvsdg::bitconstant_op>(
      &constant0variable->tac()->operation());
  assert(constant0op->value() == 0);

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/TestLlvmPhiConversion-TestPhiOperandElision",
    TestPhiOperandElision)