/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestOperations.hpp>

#include <mlir/IR/OwningOpRef.h>

using namespace jlm::hls;
using namespace jlm::rvsdg;
using namespace jlm::llvm;

namespace jlm::hls
{
class TestableRhlsToFirrtlConverter : public RhlsToFirrtlConverter
{
public:
  bool
  TestIsIdentityMapping(const rvsdg::MatchOperation & op)
  {
    return IsIdentityMapping(op);
  }

  circt::firrtl::CircuitOp
  TestMlirGen(const rvsdg::LambdaNode * lambdaNode)
  {
    return MlirGen(lambdaNode);
  }
};
}

/* ================================================================== */
/*  Base fixture: module + assert helpers                             */
/* ================================================================== */

class FirrtlTestBase : public ::testing::Test
{
protected:
  std::unique_ptr<LlvmRvsdgModule> Module_{};
  LambdaNode * Lambda_ = nullptr;

  template<typename OpT>
  bool
  AssertFirrtlOpExists(mlir::Operation * circuit)
  {
    bool found = false;
    circuit->walk(
        [&found](mlir::Operation * op)
        {
          if (::mlir::isa<OpT>(op))
            found = true;
        });
    return found;
  }

  template<typename OpT>
  bool
  AssertFirrtlOpExists(mlir::OwningOpRef<circt::firrtl::CircuitOp> & circuit)
  {
    return AssertFirrtlOpExists<OpT>(circuit->getOperation());
  }

  void
  SetUp() override
  {
    Module_ = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
    Lambda_ = nullptr;
  }
};

/* ================================================================== */
/*  Fixture: default lambda with 2x32-bit inputs, 32-bit output       */
/* ================================================================== */

class FirrtlConversionTest : public FirrtlTestBase
{
protected:
  LambdaNode *
  CreateLambda(
      const std::vector<std::shared_ptr<const Type>> & inputs,
      const std::vector<std::shared_ptr<const Type>> & outputs)
  {
    auto functionType = FunctionType::Create(inputs, outputs);
    Lambda_ = LambdaNode::Create(
        Module_->Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    return Lambda_;
  }

  template<typename OpT>
  void
  ExpectFirrtlOp()
  {
    TestableRhlsToFirrtlConverter converter;
    mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
    EXPECT_TRUE(AssertFirrtlOpExists<OpT>(circuit->getOperation()));
  }

  void
  SetUp() override
  {
    FirrtlTestBase::SetUp();
    CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(32) });
  }
};

/* ================================================================== */
/*  IsIdentityMapping tests                                           */
/* ================================================================== */

class IdentityMappingTest : public ::testing::Test
{
protected:
  std::unique_ptr<LlvmRvsdgModule> Module_{};
  TestableRhlsToFirrtlConverter Converter_;

  const MatchOperation *
  CreateMatchOp(const std::unordered_map<uint64_t, uint64_t> & mapping, uint64_t nalternatives)
  {
    auto * predicate =
        IntegerConstantOperation::Create(Module_->Rvsdg().GetRootRegion(), 2, 0).output(0);
    auto & node = MatchOperation::CreateNode(*predicate, mapping, 0, nalternatives);
    return dynamic_cast<const MatchOperation *>(&node.GetOperation());
  }

  void
  SetUp() override
  {
    Module_ = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  }
};

TEST_F(IdentityMappingTest, IdentityMapping)
{
  auto * matchOp = CreateMatchOp({ { 0, 0 }, { 1, 1 }, { 2, 2 } }, 3);
  EXPECT_TRUE(Converter_.TestIsIdentityMapping(*matchOp));
}

TEST_F(IdentityMappingTest, NonIdentityMapping)
{
  auto * matchOp = CreateMatchOp({ { 0, 1 }, { 1, 0 } }, 2);
  EXPECT_FALSE(Converter_.TestIsIdentityMapping(*matchOp));
}

TEST_F(IdentityMappingTest, EmptyMapping)
{
  auto * matchOp = CreateMatchOp({}, 2);
  EXPECT_TRUE(Converter_.TestIsIdentityMapping(*matchOp));
}

TEST_F(IdentityMappingTest, MixedMappingOneViolates)
{
  auto * matchOp = CreateMatchOp({ { 0, 0 }, { 1, 2 } }, 3);
  EXPECT_FALSE(Converter_.TestIsIdentityMapping(*matchOp));
}

/* ================================================================== */
/*  MatchOperation semantic tests                                     */
/* ================================================================== */

TEST_F(IdentityMappingTest, MappedAlternative)
{
  auto * matchOp = CreateMatchOp({ { 0, 1 }, { 1, 3 }, { 2, 0 } }, 4);
  EXPECT_EQ(matchOp->alternative(0), 1u);
  EXPECT_EQ(matchOp->alternative(1), 3u);
  EXPECT_EQ(matchOp->alternative(2), 0u);
}

TEST_F(IdentityMappingTest, DefaultAlternative)
{
  auto * predicate =
      IntegerConstantOperation::Create(Module_->Rvsdg().GetRootRegion(), 4, 0).output(0);
  auto & node = MatchOperation::CreateNode(*predicate, { { 0, 1 }, { 2, 3 } }, 7, 8);
  auto * matchOp = dynamic_cast<const MatchOperation *>(&node.GetOperation());
  EXPECT_EQ(matchOp->alternative(1), 7u);
  EXPECT_EQ(matchOp->alternative(3), 7u);
}

TEST_F(IdentityMappingTest, DefaultAlternativeAccessor)
{
  auto * matchOp = CreateMatchOp({ { 0, 1 } }, 4);
  EXPECT_EQ(matchOp->default_alternative(), 0u);
}

TEST_F(IdentityMappingTest, Nalternatives)
{
  auto * matchOp = CreateMatchOp({ { 0, 0 }, { 1, 1 } }, 16);
  EXPECT_EQ(matchOp->nalternatives(), 16u);
}

TEST_F(IdentityMappingTest, Nbits)
{
  auto * predicate =
      IntegerConstantOperation::Create(Module_->Rvsdg().GetRootRegion(), 32, 0).output(0);
  auto & node = MatchOperation::CreateNode(*predicate, { { 0, 0 } }, 0, 2);
  auto * matchOp = dynamic_cast<const MatchOperation *>(&node.GetOperation());
  EXPECT_EQ(matchOp->nbits(), 32u);
}

TEST_F(IdentityMappingTest, IteratorTraversal)
{
  auto * matchOp = CreateMatchOp({ { 0, 10 }, { 1, 20 }, { 2, 30 } }, 3);
  std::unordered_map<uint64_t, uint64_t> collected;
  for (auto it = matchOp->begin(); it != matchOp->end(); ++it)
    collected[it->first] = it->second;
  EXPECT_EQ(collected.size(), 3u);
  EXPECT_EQ(collected.at(0), 10u);
  EXPECT_EQ(collected.at(1), 20u);
  EXPECT_EQ(collected.at(2), 30u);
}

TEST_F(IdentityMappingTest, Equality)
{
  auto * matchOp1 = CreateMatchOp({ { 0, 1 }, { 1, 0 } }, 3);
  auto * matchOp2 = CreateMatchOp({ { 0, 1 }, { 1, 0 } }, 3);
  EXPECT_EQ(*matchOp1, *matchOp2);
}

TEST_F(IdentityMappingTest, Inequality)
{
  auto * matchOp1 = CreateMatchOp({ { 0, 0 }, { 1, 1 } }, 2);
  auto * matchOp2 = CreateMatchOp({ { 0, 1 }, { 1, 0 } }, 2);
  EXPECT_NE(*matchOp1, *matchOp2);
}

/* ================================================================== */
/*  Binary operation FIRRTL conversion tests                          */
/* ================================================================== */

TEST_F(FirrtlConversionTest, AddOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & addNode = IntegerAddOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ addNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::AddPrimOp>();
}

TEST_F(FirrtlConversionTest, SubOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & subNode = IntegerSubOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ subNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::SubPrimOp>();
}

TEST_F(FirrtlConversionTest, MulOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & mulNode = IntegerMulOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ mulNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::MulPrimOp>();
}

TEST_F(FirrtlConversionTest, AndOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & andNode = IntegerAndOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ andNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::AndPrimOp>();
}

TEST_F(FirrtlConversionTest, OrOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & orNode = IntegerOrOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ orNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::OrPrimOp>();
}

TEST_F(FirrtlConversionTest, XorOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & xorNode = IntegerXorOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ xorNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::XorPrimOp>();
}

/* ================================================================== */
/*  Shift operation FIRRTL conversion tests                           */
/* ================================================================== */

TEST_F(FirrtlConversionTest, ShlOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & shlNode = IntegerShlOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ shlNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::DShlPrimOp>();
}

TEST_F(FirrtlConversionTest, LShrOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & lshrNode = IntegerLShrOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ lshrNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::DShrPrimOp>();
}

/* ================================================================== */
/*  Signed arithmetic operation FIRRTL conversion tests               */
/* ================================================================== */

TEST_F(FirrtlConversionTest, SDivOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sdivNode = IntegerSDivOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sdivNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::DivPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsUIntPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, AShrOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & ashrNode = IntegerAShrOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ ashrNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::DShrPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsUIntPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, SRemOperation)
{
  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sremNode = IntegerSRemOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sremNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RemPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsUIntPrimOp>(circuit));
}

/* ================================================================== */
/*  Comparison operation FIRRTL conversion tests                      */
/* ================================================================== */

TEST_F(FirrtlConversionTest, ComparisonEqualOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & eqNode = IntegerEqOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ eqNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::EQPrimOp>();
}

TEST_F(FirrtlConversionTest, ComparisonNotEqualOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & neqNode = IntegerNeOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ neqNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::NEQPrimOp>();
}

TEST_F(FirrtlConversionTest, ComparisonSgtOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sgtNode = IntegerSgtOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sgtNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::GTPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, ComparisonSltOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sltNode = IntegerSltOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sltNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::LTPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, ComparisonSleOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sleNode = IntegerSleOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sleNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::LEQPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, ComparisonSgeOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & sgeNode = IntegerSgeOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ sgeNode.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::GEQPrimOp>(circuit));
}

TEST_F(FirrtlConversionTest, ComparisonUltOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & ultNode = IntegerUltOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ ultNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::LTPrimOp>();
}

TEST_F(FirrtlConversionTest, ComparisonUleOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & uleNode = IntegerUleOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ uleNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::LEQPrimOp>();
}

TEST_F(FirrtlConversionTest, ComparisonUgtOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & ugtNode = IntegerUgtOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ ugtNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::GTPrimOp>();
}

TEST_F(FirrtlConversionTest, ComparisonUgeOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32), BitType::Create(32) }, { BitType::Create(1) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & ugeNode = IntegerUgeOperation::createNode(32, arg0, arg1);
  Lambda_->finalize({ ugeNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::GEQPrimOp>();
}

/* ================================================================== */
/*  Unary operation FIRRTL conversion tests                           */
/* ================================================================== */

TEST_F(FirrtlConversionTest, UnaryTruncOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32) }, { BitType::Create(16) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & truncOutput = TruncOperation::create(16, arg0);
  Lambda_->finalize({ &truncOutput });

  ExpectFirrtlOp<circt::firrtl::BitsPrimOp>();
}

TEST_F(FirrtlConversionTest, UnarySExtOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(16) }, { BitType::Create(32) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & sextOutput = SExtOperation::create(32, arg0);
  Lambda_->finalize({ &sextOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::PadPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsUIntPrimOp>(circuit));
}

/* ================================================================== */
/*  Constant operation FIRRTL conversion tests                         */
/* ================================================================== */

TEST_F(FirrtlConversionTest, ConstantIntegerOperation)
{
  Lambda_ = CreateLambda({}, { BitType::Create(32) });

  auto & constantNode = IntegerConstantOperation::Create(*Lambda_->subregion(), 32, 42);
  Lambda_->finalize({ constantNode.output(0) });

  ExpectFirrtlOp<circt::firrtl::ConstantOp>();
}

TEST_F(FirrtlConversionTest, ConstantUndefValueOperation)
{
  Lambda_ = CreateLambda({}, { BitType::Create(32) });

  auto * undefOutput = UndefValueOperation::Create(*Lambda_->subregion(), BitType::Create(32));
  Lambda_->finalize({ undefOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::InvalidValueOp>(circuit));
}

/* ================================================================== */
/*  Pass-through operation FIRRTL conversion tests                    */
/* ================================================================== */

TEST_F(FirrtlConversionTest, PassThroughBitCastOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(32) }, { BitType::Create(32) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto * bitcastOutput = BitCastOperation::create(&arg0, BitType::Create(32));
  Lambda_->finalize({ bitcastOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}

TEST_F(FirrtlConversionTest, PassThroughZExtOperation)
{
  Lambda_ = CreateLambda({ BitType::Create(16) }, { BitType::Create(32) });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & zextOutput = ZExtOperation::create(32, arg0);
  Lambda_->finalize({ &zextOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}

/* ================================================================== */
/*  IntegerToPointerOperation FIRRTL conversion test                  */
/* ================================================================== */

TEST_F(FirrtlConversionTest, IntegerToPointerOperation)
{
  auto ptrType = jlm::llvm::PointerType::Create();
  Lambda_ = CreateLambda({ BitType::Create(32) }, { ptrType });

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto * itopOutput = IntToPtrOperation::create(&arg0);
  Lambda_->finalize({ itopOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}

/* ================================================================== */
/*  MatchOperation non-identity FIRRTL conversion tests               */
/* ================================================================== */

class FirrtlMatchConversionTest : public FirrtlTestBase
{
protected:
  LambdaNode *
  CreateMatchLambda(int predicateBits, int outBits)
  {
    auto functionType =
        FunctionType::Create({ BitType::Create(predicateBits) }, { ControlType::Create(outBits) });
    Lambda_ = LambdaNode::Create(
        Module_->Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    return Lambda_;
  }
};

TEST_F(FirrtlMatchConversionTest, MatchOperationNonIdentityMapping)
{
  CreateMatchLambda(32, 4);

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & node = MatchOperation::CreateNode(
      predicate,
      std::unordered_map<uint64_t, uint64_t>{ { 0, 3 }, { 1, 2 }, { 2, 1 }, { 3, 0 } },
      0,
      4);
  Lambda_->finalize({ node.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::MuxPrimOp>(circuit));
}

TEST_F(FirrtlMatchConversionTest, MatchOperationPartialMapping)
{
  CreateMatchLambda(32, 4);

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & node = MatchOperation::CreateNode(
      predicate,
      std::unordered_map<uint64_t, uint64_t>{ { 0, 5 }, { 2, 3 } },
      7,
      4);
  Lambda_->finalize({ node.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::MuxPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::ConstantOp>(circuit));
}

TEST_F(FirrtlMatchConversionTest, MatchOperationIdentityWithTruncation)
{
  CreateMatchLambda(32, 4);

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & node =
      MatchOperation::CreateNode(predicate, std::unordered_map<uint64_t, uint64_t>{}, 0, 4);
  Lambda_->finalize({ node.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::BitsPrimOp>(circuit));
}

TEST_F(FirrtlMatchConversionTest, MatchOperationIdentityEqualSizes)
{
  auto functionType = FunctionType::Create({ BitType::Create(2) }, { ControlType::Create(4) });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & node =
      MatchOperation::CreateNode(predicate, std::unordered_map<uint64_t, uint64_t>{}, 0, 4);
  Lambda_->finalize({ node.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
  EXPECT_FALSE(AssertFirrtlOpExists<circt::firrtl::BitsPrimOp>(circuit));
  EXPECT_FALSE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
  EXPECT_FALSE(AssertFirrtlOpExists<circt::firrtl::MuxPrimOp>(circuit));
}

/* ================================================================== */
/*  ControlConstantOperation FIRRTL conversion test                   */
/* ================================================================== */

class FirrtlControlConstantTest : public FirrtlTestBase
{
protected:
  void
  SetUp() override
  {
    FirrtlTestBase::SetUp();
    auto functionType = FunctionType::Create({}, { ControlType::Create(4) });
    Lambda_ = LambdaNode::Create(
        Module_->Rvsdg().GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
  }
};

TEST_F(FirrtlControlConstantTest, ControlConstantOperation)
{
  auto & controlValue = ControlConstantOperation::create(*Lambda_->subregion(), 4, 2);
  Lambda_->finalize({ &controlValue });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::ConstantOp>(circuit));
}

/* ================================================================== */
/*  Memory state pass-through FIRRTL conversion tests                 */
/* ================================================================== */

TEST_F(FirrtlTestBase, MemoryStateMergeOperation)
{
  auto memoryStateType = jlm::llvm::MemoryStateType::Create();
  auto functionType =
      FunctionType::Create({ memoryStateType, memoryStateType }, { memoryStateType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto * mergeOutput =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::Output *>{ &arg0, &arg1 });
  Lambda_->finalize({ mergeOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}

TEST_F(FirrtlTestBase, LambdaExitMemoryStateMergeOperation)
{
  auto memoryStateType = jlm::llvm::MemoryStateType::Create();
  auto functionType =
      FunctionType::Create({ memoryStateType, memoryStateType }, { memoryStateType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & arg1 = *Lambda_->GetFunctionArguments()[1];
  auto & mergeOutput = jlm::llvm::LambdaExitMemoryStateMergeOperation::CreateNode(
      *Lambda_->subregion(),
      { &arg0, &arg1 },
      { 0, 1 });
  Lambda_->finalize({ mergeOutput.output(0) });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}

/* ================================================================== */
/*  Error handling tests                                              */
/* ================================================================== */

TEST_F(FirrtlTestBase, UnimplementedSimpleNodeThrows)
{
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({}, { bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto * testNode = jlm::rvsdg::TestOperation::createNode(Lambda_->subregion(), {}, { bitType });
  Lambda_->finalize({ testNode->output(0) });

  TestableRhlsToFirrtlConverter converter;
  bool exceptionThrown = false;

  try
  {
    converter.TestMlirGen(Lambda_);
  }
  catch (const std::logic_error &)
  {
    exceptionThrown = true;
  }

  EXPECT_TRUE(exceptionThrown);
}

/* ================================================================== */
/*  MuxOperation FIRRTL conversion tests                               */
/* ================================================================== */

TEST_F(FirrtlTestBase, MuxOperationTwoAlternatives)
{
  auto controlType = ControlType::Create(2);
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType, bitType, bitType }, { bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & value0 = *Lambda_->GetFunctionArguments()[1];
  auto & value1 = *Lambda_->GetFunctionArguments()[2];
  auto muxOutputs = MuxOperation::create(predicate, { &value0, &value1 }, true, false);
  Lambda_->finalize({ muxOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
}

TEST_F(FirrtlTestBase, MuxOperationThreeAlternatives)
{
  auto controlType = ControlType::Create(3);
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType, bitType, bitType, bitType }, { bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & value0 = *Lambda_->GetFunctionArguments()[1];
  auto & value1 = *Lambda_->GetFunctionArguments()[2];
  auto & value2 = *Lambda_->GetFunctionArguments()[3];
  auto muxOutputs = MuxOperation::create(predicate, { &value0, &value1, &value2 }, true, false);
  Lambda_->finalize({ muxOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
}

/* ================================================================== */
/*  GetElementPtrOperation FIRRTL conversion test                     */
/* ================================================================== */

TEST_F(FirrtlTestBase, GetElementPtrOperationArrayType)
{
  auto ptrType = PointerType::Create();
  auto bitType = BitType::Create(32);
  auto arrayType = ArrayType::Create(bitType, 10);
  auto functionType = FunctionType::Create({ ptrType, BitType::Create(32) }, { ptrType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & ptr = *Lambda_->GetFunctionArguments()[0];
  auto & index = *Lambda_->GetFunctionArguments()[1];
  auto * gepOutput = GetElementPtrOperation::create(&ptr, { &index }, arrayType);
  Lambda_->finalize({ gepOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::CvtPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AsSIntPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::MulPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AddPrimOp>(circuit));
}

/* ================================================================== */
/*  HLS SinkOperation FIRRTL conversion test                           */
/* ================================================================== */

TEST_F(FirrtlTestBase, SinkOperation)
{
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ bitType }, {});
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto sinkOutputs = SinkOperation::create(arg0);
  Lambda_->finalize({});

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::ConstantOp>(circuit));
}

/* ================================================================== */
/*  HLS TriggerOperation FIRRTL conversion test                       */
/* ================================================================== */

TEST_F(FirrtlTestBase, TriggerOperation)
{
  auto triggerType = TriggerType::Create();
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ triggerType, bitType }, { bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & tg = *Lambda_->GetFunctionArguments()[0];
  auto & value = *Lambda_->GetFunctionArguments()[1];
  auto triggerOutputs = TriggerOperation::create(tg, value);
  Lambda_->finalize({ triggerOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AndPrimOp>(circuit));
}

/* ================================================================== */
/*  HLS BranchOperation FIRRTL conversion test                        */
/* ================================================================== */

TEST_F(FirrtlTestBase, BranchOperation)
{
  auto bitType = BitType::Create(32);
  auto controlType = ControlType::Create(2);
  auto functionType = FunctionType::Create({ controlType, bitType }, { bitType, bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & predicate = *Lambda_->GetFunctionArguments()[0];
  auto & value = *Lambda_->GetFunctionArguments()[1];
  auto branchOutputs = ::jlm::hls::BranchOperation::create(predicate, value);
  Lambda_->finalize({ branchOutputs[0], branchOutputs[1] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS MuxOperation (NDMux) FIRRTL conversion test                   */
/* ================================================================== */

TEST_F(FirrtlTestBase, NDMuxOperation)
{
  auto controlType = ControlType::Create(3);
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType, bitType, bitType, bitType }, { bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & tag = *Lambda_->GetFunctionArguments()[0];
  auto & val0 = *Lambda_->GetFunctionArguments()[1];
  auto & val1 = *Lambda_->GetFunctionArguments()[2];
  auto & val2 = *Lambda_->GetFunctionArguments()[3];
  auto muxOutputs = MuxOperation::create(tag, { &val0, &val1, &val2 }, false, false);
  Lambda_->finalize({ muxOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::EQPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::AndPrimOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS ForkOperation FIRRTL conversion tests                         */
/* ================================================================== */

TEST_F(FirrtlTestBase, ForkOperation)
{
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ bitType }, { bitType, bitType, bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto forkOutputs = ForkOperation::create(3, arg0, false);
  Lambda_->finalize({ forkOutputs[0], forkOutputs[1], forkOutputs[2] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS StateGateOperation FIRRTL conversion test                     */
/* ================================================================== */

TEST_F(FirrtlTestBase, StateGateOperation)
{
  auto bitType = BitType::Create(32);
  auto memStateType = MemoryStateType::Create();
  auto functionType = FunctionType::Create({ bitType, memStateType }, { bitType, memStateType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto & state = *Lambda_->GetFunctionArguments()[1];
  auto stateGateOutputs = StateGateOperation::create(arg0, { &state });
  Lambda_->finalize({ stateGateOutputs[0], stateGateOutputs[1] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS PredicateBufferOperation FIRRTL conversion test               */
/* ================================================================== */

TEST_F(FirrtlTestBase, PredicateBufferOperation)
{
  auto controlType = ControlType::Create(2);
  auto functionType = FunctionType::Create({ controlType }, { controlType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & arg0 = *Lambda_->GetFunctionArguments()[0];
  auto predBufOutputs = PredicateBufferOperation::create(arg0);
  Lambda_->finalize({ predBufOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::MuxPrimOp>(circuit));
}

/* ================================================================== */
/*  HLS AddressQueueOperation FIRRTL conversion test                  */
/* ================================================================== */

TEST_F(FirrtlTestBase, AddressQueueOperation)
{
  auto ptrType = PointerType::Create();
  auto memStateType = MemoryStateType::Create();
  auto functionType = FunctionType::Create({ ptrType, ptrType, memStateType }, { ptrType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & check = *Lambda_->GetFunctionArguments()[0];
  auto & enq = *Lambda_->GetFunctionArguments()[1];
  auto & deq = *Lambda_->GetFunctionArguments()[2];
  auto addrQueueOutput = AddressQueueOperation::create(check, enq, deq, false, 4);
  Lambda_->finalize({ addrQueueOutput });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WireOp>(circuit));
}

/* ================================================================== */
/*  HLS LoadOperation FIRRTL conversion test                          */
/* ================================================================== */

TEST_F(FirrtlTestBase, JlmHlsLoadOperation)
{
  auto ptrType = PointerType::Create();
  auto bitType = BitType::Create(32);
  auto memStateType = MemoryStateType::Create();
  // HLS LoadOperation: inputs {Pointer addr, MemoryState*, ValueType load_result}
  //                     outputs {ValueType data, ...MemoryState*, Pointer addr}
  auto functionType =
      FunctionType::Create({ ptrType, memStateType, bitType }, { bitType, memStateType, ptrType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & addr = *Lambda_->GetFunctionArguments()[0];
  auto & state = *Lambda_->GetFunctionArguments()[1];
  auto & loadResult = *Lambda_->GetFunctionArguments()[2];
  auto loadOutputs = ::jlm::hls::LoadOperation::create(addr, { &state }, loadResult);
  Lambda_->finalize({ loadOutputs[0], loadOutputs[1], loadOutputs[2] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS StoreOperation FIRRTL conversion test                         */
/* ================================================================== */

TEST_F(FirrtlTestBase, HlsStoreOperation)
{
  auto ptrType = PointerType::Create();
  auto bitType = BitType::Create(32);
  auto memStateType = MemoryStateType::Create();
  // HLS StoreOperation: inputs {Pointer addr, ValueType value, ...MemoryState*, MemoryState* resp}
  //                    outputs {MemoryState*, PointerType, ValueType data}
  auto functionType =
      FunctionType::Create({ ptrType, bitType, memStateType }, { memStateType, ptrType, bitType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & addr = *Lambda_->GetFunctionArguments()[0];
  auto & value = *Lambda_->GetFunctionArguments()[1];
  auto & state = *Lambda_->GetFunctionArguments()[2];
  auto storeOutputs = ::jlm::hls::StoreOperation::create(addr, value, { &state }, state);
  Lambda_->finalize({ storeOutputs[0], storeOutputs[1], storeOutputs[2] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS DecoupledLoadOperation FIRRTL conversion test                 */
/* ================================================================== */

TEST_F(FirrtlTestBase, DecoupledLoadOperation)
{
  auto ptrType = PointerType::Create();
  auto bitType = BitType::Create(32);
  auto functionType = FunctionType::Create({ ptrType, bitType }, { bitType, ptrType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & addr = *Lambda_->GetFunctionArguments()[0];
  auto & loadResult = *Lambda_->GetFunctionArguments()[1];
  auto dloadOutputs = DecoupledLoadOperation::create(addr, loadResult, 4);
  Lambda_->finalize({ dloadOutputs[0], dloadOutputs[1] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::FExtModuleOp>(circuit));
}

/* ================================================================== */
/*  HLS LocalLoadOperation FIRRTL conversion test                     */
/* ================================================================== */

TEST_F(FirrtlTestBase, LocalLoadOperation)
{
  auto bitType = BitType::Create(32);
  auto memStateType = MemoryStateType::Create();
  // LocalLoadOperation: inputs {Bit(64) index, MemoryState*, ValueType load_result}
  //                    outputs {ValueType data, ...MemoryState*, Bit(64) addr}
  auto functionType = FunctionType::Create(
      { BitType::Create(64), memStateType, bitType },
      { bitType, memStateType, BitType::Create(64) });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & index = *Lambda_->GetFunctionArguments()[0];
  auto & state = *Lambda_->GetFunctionArguments()[1];
  auto & loadResult = *Lambda_->GetFunctionArguments()[2];
  auto loadOutputs = ::jlm::hls::LocalLoadOperation::create(index, { &state }, loadResult);
  Lambda_->finalize({ loadOutputs[0], loadOutputs[1], loadOutputs[2] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::RegResetOp>(circuit));
  EXPECT_TRUE(AssertFirrtlOpExists<circt::firrtl::WhenOp>(circuit));
}

/* ================================================================== */
/*  HLS LocalStoreOperation FIRRTL conversion test                    */
/* ================================================================== */

TEST_F(FirrtlTestBase, LocalStoreOperation)
{
  auto bitType = BitType::Create(32);
  auto memStateType = MemoryStateType::Create();
  // LocalStoreOperation: inputs {Bit(64) index, ValueType value, MemoryState*}
  //                    outputs {...MemoryState*, Bit(64) addr}
  auto functionType =
      FunctionType::Create({ BitType::Create(64), bitType, memStateType }, { memStateType });
  Lambda_ = LambdaNode::Create(
      Module_->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto & index = *Lambda_->GetFunctionArguments()[0];
  auto & value = *Lambda_->GetFunctionArguments()[1];
  auto & state = *Lambda_->GetFunctionArguments()[2];
  auto storeOutputs = ::jlm::hls::LocalStoreOperation::create(index, value, { &state });
  Lambda_->finalize({ storeOutputs[0] });

  TestableRhlsToFirrtlConverter converter;
  mlir::OwningOpRef<circt::firrtl::CircuitOp> circuit(converter.TestMlirGen(Lambda_));
  EXPECT_TRUE(circuit);
}
