/*
 * Copyright 2024 Halvor Linder Henriksen <halvorlinder@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace
{

// Structure to hold all the info needed to test an integer binary operation
template<typename JlmOperation, typename MlirOperation>
struct IntegerBinaryOpTest
{
  using JlmOpType = JlmOperation;
  using MlirOpType = MlirOperation;
  const char * name;
};

// Template function to test an integer binary operation
template<typename JlmOperation, typename MlirOperation>
static void
TestIntegerBinaryOperation()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  const size_t nbits = 64;
  const uint64_t val1 = 2;
  const uint64_t val2 = 3;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto constOp1 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { nbits, val1 });
    auto constOp2 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { nbits, val2 });
    auto binaryOp = JlmOperation(nbits);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), binaryOp.copy(), { constOp1, constOp2 });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool opFound = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirBinaryOp = ::mlir::dyn_cast<MlirOperation>(&op);
      if (mlirBinaryOp)
      {
        auto inputBitType1 =
            mlirBinaryOp.getOperand(0).getType().template dyn_cast<::mlir::IntegerType>();
        auto inputBitType2 =
            mlirBinaryOp.getOperand(1).getType().template dyn_cast<::mlir::IntegerType>();
        EXPECT_NE(inputBitType1, nullptr);
        EXPECT_EQ(inputBitType1.getWidth(), nbits);
        EXPECT_NE(inputBitType2, nullptr);
        EXPECT_EQ(inputBitType2.getWidth(), nbits);
        auto outputBitType =
            mlirBinaryOp.getResult().getType().template dyn_cast<::mlir::IntegerType>();
        EXPECT_NE(outputBitType, nullptr);
        EXPECT_EQ(outputBitType.getWidth(), nbits);
        opFound = true;
      }
    }
    EXPECT_TRUE(opFound);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto convertedRvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &convertedRvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 3);
      bool foundBinaryOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedBinaryOp = dynamic_cast<const JlmOperation *>(&node.GetOperation());
        if (convertedBinaryOp)
        {
          EXPECT_EQ(convertedBinaryOp->nresults(), 1);
          EXPECT_EQ(convertedBinaryOp->narguments(), 2);
          auto inputBitType1 = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedBinaryOp->argument(0).get());
          EXPECT_EQ(inputBitType1->nbits(), nbits);
          auto inputBitType2 = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedBinaryOp->argument(1).get());
          EXPECT_EQ(inputBitType2->nbits(), nbits);
          auto outputBitType = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedBinaryOp->result(0).get());
          EXPECT_EQ(outputBitType->nbits(), nbits);
          foundBinaryOp = true;
        }
      }
      EXPECT_TRUE(foundBinaryOp);
    }
  }
}

// Macro to define and register a test for an integer binary operation
#define REGISTER_INT_BINARY_OP_TEST(JLM_OP, MLIR_NS, MLIR_OP, TEST_NAME) \
  TEST(IntegerOperationConversionTests, TEST_NAME)                       \
  {                                                                      \
    return TestIntegerBinaryOperation<                                   \
        jlm::llvm::Integer##JLM_OP##Operation,                           \
        ::mlir::MLIR_NS::MLIR_OP>();                                     \
  }

// Register tests for all the integer binary operations
REGISTER_INT_BINARY_OP_TEST(Add, arith, AddIOp, Add)
REGISTER_INT_BINARY_OP_TEST(Sub, arith, SubIOp, Sub)
REGISTER_INT_BINARY_OP_TEST(Mul, arith, MulIOp, Mul)
REGISTER_INT_BINARY_OP_TEST(SDiv, arith, DivSIOp, DivSI)
REGISTER_INT_BINARY_OP_TEST(UDiv, arith, DivUIOp, DivUI)
REGISTER_INT_BINARY_OP_TEST(SRem, arith, RemSIOp, RemSI)
REGISTER_INT_BINARY_OP_TEST(URem, arith, RemUIOp, RemUI)
REGISTER_INT_BINARY_OP_TEST(Shl, LLVM, ShlOp, ShLI)
REGISTER_INT_BINARY_OP_TEST(AShr, LLVM, AShrOp, ShRSI)
REGISTER_INT_BINARY_OP_TEST(LShr, LLVM, LShrOp, ShRUI)
REGISTER_INT_BINARY_OP_TEST(And, arith, AndIOp, AndI)
REGISTER_INT_BINARY_OP_TEST(Or, arith, OrIOp, OrI)
REGISTER_INT_BINARY_OP_TEST(Xor, arith, XOrIOp, XOrI)

// Structure to hold all the info needed to test an integer comparison operation
template<typename JlmOperation>
struct IntegerComparisonOpTest
{
  using JlmOpType = JlmOperation;
  ::mlir::arith::CmpIPredicate predicate;
  const char * name;
};

// Template function to test an integer comparison operation
template<typename JlmOperation>
static void
TestIntegerComparisonOperation(const IntegerComparisonOpTest<JlmOperation> & test)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  const size_t nbits = 64;
  const uint64_t val1 = 2;
  const uint64_t val2 = 3;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto constOp1 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { nbits, val1 });
    auto constOp2 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { nbits, val2 });
    auto compOp = JlmOperation(nbits);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), compOp.copy(), { constOp1, constOp2 });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool opFound = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirCompOp = ::mlir::dyn_cast<::mlir::arith::CmpIOp>(&op);
      if (mlirCompOp)
      {
        auto inputBitType1 = mlirCompOp.getOperand(0).getType().dyn_cast<::mlir::IntegerType>();
        auto inputBitType2 = mlirCompOp.getOperand(1).getType().dyn_cast<::mlir::IntegerType>();
        EXPECT_NE(inputBitType1, nullptr);
        EXPECT_EQ(inputBitType1.getWidth(), nbits);
        EXPECT_NE(inputBitType2, nullptr);
        EXPECT_EQ(inputBitType2.getWidth(), nbits);

        // Check the output type is i1 (boolean)
        auto outputType = mlirCompOp.getResult().getType().dyn_cast<::mlir::IntegerType>();
        EXPECT_NE(outputType, nullptr);
        EXPECT_EQ(outputType.getWidth(), 1);

        // Verify the predicate is correct
        EXPECT_EQ(mlirCompOp.getPredicate(), test.predicate);
        opFound = true;
      }
    }
    EXPECT_TRUE(opFound);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto convertedRvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &convertedRvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 3);
      bool foundCompOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedCompOp = dynamic_cast<const JlmOperation *>(&node.GetOperation());
        if (convertedCompOp)
        {
          EXPECT_EQ(convertedCompOp->nresults(), 1);
          EXPECT_EQ(convertedCompOp->narguments(), 2);
          auto inputBitType1 = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedCompOp->argument(0).get());
          EXPECT_EQ(inputBitType1->nbits(), nbits);
          auto inputBitType2 = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedCompOp->argument(1).get());
          EXPECT_EQ(inputBitType2->nbits(), nbits);

          // Check the output type is bit1 (boolean)
          auto outputBitType =
              jlm::util::assertedCast<const jlm::rvsdg::BitType>(convertedCompOp->result(0).get());
          EXPECT_EQ(outputBitType->nbits(), 1);

          foundCompOp = true;
        }
      }
      EXPECT_TRUE(foundCompOp);
    }
  }
}

// Macro to define and register a test for an integer comparison operation
#define REGISTER_INT_COMP_OP_TEST(JLM_OP, PREDICATE, TEST_NAME)             \
  TEST(IntegerOperationConversionTests, TEST_NAME)                          \
  {                                                                         \
    IntegerComparisonOpTest<jlm::llvm::Integer##JLM_OP##Operation> test = { \
      ::mlir::arith::CmpIPredicate::PREDICATE,                              \
      #TEST_NAME                                                            \
    };                                                                      \
    return TestIntegerComparisonOperation(test);                            \
  }

// Register tests for all the integer comparison operations
REGISTER_INT_COMP_OP_TEST(Eq, eq, Eq)
REGISTER_INT_COMP_OP_TEST(Ne, ne, Ne)
REGISTER_INT_COMP_OP_TEST(Slt, slt, Slt)
REGISTER_INT_COMP_OP_TEST(Sle, sle, Sle)
REGISTER_INT_COMP_OP_TEST(Sgt, sgt, Sgt)
REGISTER_INT_COMP_OP_TEST(Sge, sge, Sge)
REGISTER_INT_COMP_OP_TEST(Ult, ult, Ult)
REGISTER_INT_COMP_OP_TEST(Ule, ule, Ule)
REGISTER_INT_COMP_OP_TEST(Ugt, ugt, Ugt)
REGISTER_INT_COMP_OP_TEST(Uge, uge, Uge)

}
