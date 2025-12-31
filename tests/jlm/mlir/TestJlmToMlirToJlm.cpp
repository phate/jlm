/*
 * Copyright 2024 Halvor Linder Henriksen <halvorlinder@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

TEST(JlmToMlirToJlmTests, TestUndef)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create an undef operation
    std::cout << "Undef Operation" << std::endl;
    UndefValueOperation::Create(graph->GetRootRegion(), jlm::rvsdg::BitType::Create(32));

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    EXPECT_EQ(omegaRegion.getBlocks().size(), 1);
    auto & omegaBlock = omegaRegion.front();
    // 1 undef + omegaResult
    EXPECT_EQ(omegaBlock.getOperations().size(), 2);
    EXPECT_TRUE(mlir::isa<mlir::jlm::Undef>(omegaBlock.front()));
    auto mlirUndefOp = mlir::dyn_cast<::mlir::jlm::Undef>(&omegaBlock.front());
    mlirUndefOp.dump();

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);

      // Get the undef op
      auto convertedUndef =
          dynamic_cast<const UndefValueOperation *>(&region->Nodes().begin()->GetOperation());

      EXPECT_NE(convertedUndef, nullptr);

      auto outputType = convertedUndef->result(0);
      EXPECT_TRUE(jlm::rvsdg::is<const jlm::rvsdg::BitType>(outputType));
      EXPECT_EQ(std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(outputType)->nbits(), 32);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestAlloca)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create a bits node for alloc size
    std::cout << "Bit Constanr" << std::endl;
    auto bits = &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 32, 1 });

    // Create alloca node
    std::cout << "Alloca Operation" << std::endl;
    jlm::rvsdg::CreateOpNode<AllocaOperation>(
        { bits },
        jlm::rvsdg::BitType::Create(64),
        jlm::rvsdg::BitType::Create(32),
        4);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    std::cout << "Checking blocks and operations count" << std::endl;
    auto & omegaRegion = omega.getRegion();
    EXPECT_EQ(omegaRegion.getBlocks().size(), 1);
    auto & omegaBlock = omegaRegion.front();

    // Bit-contant + alloca + omegaResult
    EXPECT_EQ(omegaBlock.getOperations().size(), 3);

    bool foundAlloca = false;
    for (auto & op : omegaBlock)
    {
      if (mlir::isa<mlir::jlm::Alloca>(op))
      {
        auto mlirAllocaOp = mlir::cast<mlir::jlm::Alloca>(op);
        EXPECT_EQ(mlirAllocaOp.getAlignment(), 4);
        EXPECT_EQ(mlirAllocaOp.getNumResults(), 2);

        auto valueType = mlir::cast<mlir::IntegerType>(mlirAllocaOp.getValueType());
        EXPECT_NE(valueType, nullptr);
        EXPECT_EQ(valueType.getWidth(), 64);
        foundAlloca = true;
      }
    }
    EXPECT_TRUE(foundAlloca);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);

      bool foundAlloca = false;
      for (auto & node : region->Nodes())
      {
        if (auto allocaOp = dynamic_cast<const AllocaOperation *>(&node.GetOperation()))
        {
          EXPECT_EQ(allocaOp->alignment(), 4);

          EXPECT_TRUE(jlm::rvsdg::is<jlm::rvsdg::BitType>(allocaOp->ValueType()));
          auto valueBitType =
              dynamic_cast<const jlm::rvsdg::BitType *>(allocaOp->ValueType().get());
          EXPECT_EQ(valueBitType->nbits(), 64);

          EXPECT_EQ(allocaOp->narguments(), 1);

          EXPECT_TRUE(jlm::rvsdg::is<jlm::rvsdg::BitType>(allocaOp->argument(0)));
          auto inputBitType =
              dynamic_cast<const jlm::rvsdg::BitType *>(allocaOp->argument(0).get());
          EXPECT_EQ(inputBitType->nbits(), 32);

          EXPECT_EQ(allocaOp->nresults(), 2);

          EXPECT_TRUE(jlm::rvsdg::is<PointerType>(allocaOp->result(0)));
          EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::MemoryStateType>(allocaOp->result(1)));

          foundAlloca = true;
        }
      }
      EXPECT_TRUE(foundAlloca);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestLoad)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto iOStateArgument = lambda->GetFunctionArguments().at(0);
    auto memoryStateArgument = lambda->GetFunctionArguments().at(1);
    auto pointerArgument = lambda->GetFunctionArguments().at(2);

    // Create load operation
    auto loadType = jlm::rvsdg::BitType::Create(32);
    auto loadOp = jlm::llvm::LoadNonVolatileOperation(loadType, 1, 4);
    auto & subregion = *(lambda->subregion());
    LoadNonVolatileOperation::Create(
        subregion,
        std::make_unique<LoadNonVolatileOperation>(loadOp),
        { pointerArgument, memoryStateArgument });

    lambda->finalize({ iOStateArgument, memoryStateArgument });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();
    auto & mlirOp = mlirLambdaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::jlm::Load>(mlirOp));

    auto mlirLoad = mlir::cast<mlir::jlm::Load>(mlirOp);
    EXPECT_EQ(mlirLoad.getAlignment(), 4);
    EXPECT_EQ(mlirLoad.getInputMemStates().size(), 1);
    EXPECT_EQ(mlirLoad.getNumOperands(), 2);
    EXPECT_EQ(mlirLoad.getNumResults(), 2);

    auto outputType = mlirLoad.getOutput().getType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(outputType));
    auto integerType = mlir::cast<mlir::IntegerType>(outputType);
    EXPECT_EQ(integerType.getWidth(), 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_TRUE(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(is<LoadNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedLoad = convertedLambda->subregion()->Nodes().begin().ptr();
      auto loadOperation =
          dynamic_cast<const LoadNonVolatileOperation *>(&convertedLoad->GetOperation());

      EXPECT_EQ(loadOperation->GetAlignment(), 4);
      EXPECT_EQ(loadOperation->NumMemoryStates(), 1);

      EXPECT_TRUE(is<jlm::llvm::PointerType>(convertedLoad->input(0)->Type()));
      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedLoad->input(1)->Type()));

      EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedLoad->output(0)->Type()));
      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedLoad->output(1)->Type()));

      auto outputBitType =
          std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(convertedLoad->output(0)->Type());
      EXPECT_EQ(outputBitType->nbits(), 32);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestStore)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitsType = jlm::rvsdg::BitType::Create(32);
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create(), bitsType },
        { IOStateType::Create(), MemoryStateType::Create() });
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto iOStateArgument = lambda->GetFunctionArguments().at(0);
    auto memoryStateArgument = lambda->GetFunctionArguments().at(1);
    auto pointerArgument = lambda->GetFunctionArguments().at(2);
    auto bitsArgument = lambda->GetFunctionArguments().at(3);

    // Create store operation
    auto storeOp = jlm::llvm::StoreNonVolatileOperation(bitsType, 1, 4);
    jlm::llvm::StoreNonVolatileOperation::Create(
        *lambda->subregion(),
        std::make_unique<StoreNonVolatileOperation>(storeOp),
        { pointerArgument, bitsArgument, memoryStateArgument });

    lambda->finalize({ iOStateArgument, memoryStateArgument });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();
    auto & mlirOp = mlirLambdaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::jlm::Store>(mlirOp));

    auto mlirStore = mlir::cast<mlir::jlm::Store>(mlirOp);
    EXPECT_EQ(mlirStore.getAlignment(), 4);
    EXPECT_EQ(mlirStore.getInputMemStates().size(), 1);
    EXPECT_EQ(mlirStore.getNumOperands(), 3);

    auto inputType = mlirStore.getValue().getType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(inputType));
    auto integerType = mlir::cast<mlir::IntegerType>(inputType);
    EXPECT_EQ(integerType.getWidth(), 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_TRUE(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(is<StoreNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedStore = convertedLambda->subregion()->Nodes().begin().ptr();
      auto convertedStoreOperation =
          dynamic_cast<const StoreNonVolatileOperation *>(&convertedStore->GetOperation());

      EXPECT_EQ(convertedStoreOperation->GetAlignment(), 4);
      EXPECT_EQ(convertedStoreOperation->NumMemoryStates(), 1);

      EXPECT_TRUE(is<jlm::llvm::PointerType>(convertedStore->input(0)->Type()));
      EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedStore->input(1)->Type()));
      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedStore->input(2)->Type()));

      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedStore->output(0)->Type()));

      auto inputBitType =
          std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(convertedStore->input(1)->Type());
      EXPECT_EQ(inputBitType->nbits(), 32);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestSext)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {

    auto bitsType = jlm::rvsdg::BitType::Create(32);
    auto functionType = jlm::rvsdg::FunctionType::Create({ bitsType }, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto bitsArgument = lambda->GetFunctionArguments().at(0);

    // Create sext operation
    auto sextOp = jlm::llvm::SExtOperation::create((size_t)64, bitsArgument);
    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*sextOp);
    EXPECT_NE(node, nullptr);

    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();
    auto & mlirOp = mlirLambdaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::arith::ExtSIOp>(mlirOp));

    auto mlirSext = mlir::cast<mlir::arith::ExtSIOp>(mlirOp);
    auto inputType = mlirSext.getOperand().getType();
    auto outputType = mlirSext.getType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(inputType));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(outputType));
    EXPECT_TRUE(mlir::cast<mlir::IntegerType>(inputType).getWidth() == 32);
    EXPECT_TRUE(mlir::cast<mlir::IntegerType>(outputType).getWidth() == 64);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_TRUE(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(is<SExtOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSext = dynamic_cast<const SExtOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      EXPECT_EQ(convertedSext->ndstbits(), 64);
      EXPECT_EQ(convertedSext->nsrcbits(), 32);
      EXPECT_EQ(convertedSext->nresults(), 1);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestSitofp)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {

    auto bitsType = jlm::rvsdg::BitType::Create(32);
    auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
    auto functionType = jlm::rvsdg::FunctionType::Create({ bitsType }, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto bitsArgument = lambda->GetFunctionArguments().at(0);

    // Create sitofp operation
    jlm::rvsdg::CreateOpNode<SIToFPOperation>({ bitsArgument }, bitsType, floatType);

    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();
    auto & mlirOp = mlirLambdaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::arith::SIToFPOp>(mlirOp));

    auto mlirSitofp = mlir::cast<mlir::arith::SIToFPOp>(mlirOp);
    auto inputType = mlirSitofp.getOperand().getType();
    auto outputType = mlirSitofp.getType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(inputType));
    EXPECT_TRUE(mlir::cast<mlir::IntegerType>(inputType).getWidth() == 32);
    EXPECT_TRUE(mlir::isa<mlir::Float64Type>(outputType));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(
          is<SIToFPOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSitofp = dynamic_cast<const SIToFPOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      EXPECT_TRUE(jlm::rvsdg::is<jlm::rvsdg::BitType>(*convertedSitofp->argument(0).get()));
      EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedSitofp->result(0).get()));
    }
  }
}

TEST(JlmToMlirToJlmTests, TestConstantFP)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto functionType = jlm::rvsdg::FunctionType::Create({}, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

    // Create sitofp operation
    jlm::rvsdg::CreateOpNode<ConstantFP>(*lambda->subregion(), fpsize::dbl, ::llvm::APFloat(2.0));

    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & mlirOp = omega.getRegion().front().front().getRegion(0).front().front();

    EXPECT_TRUE(mlir::isa<mlir::arith::ConstantFloatOp>(mlirOp));

    auto mlirConst = mlir::cast<mlir::arith::ConstantFloatOp>(mlirOp);
    EXPECT_TRUE(mlirConst.value().isExactlyValue(2.0));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(is<ConstantFP>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedConst = dynamic_cast<const ConstantFP *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedConst->result(0).get()));
      EXPECT_TRUE(convertedConst->constant().isExactlyValue(2.0));
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFpBinary)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;
  auto binOps = std::vector<fpop>{ fpop::add, fpop::sub, fpop::mul, fpop::div, fpop::mod };
  for (auto binOp : binOps)
  {
    auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
    auto graph = &rvsdgModule->Rvsdg();
    {
      auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
      auto functionType = jlm::rvsdg::FunctionType::Create({ floatType, floatType }, {});
      auto lambda = jlm::rvsdg::LambdaNode::Create(
          graph->GetRootRegion(),
          LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

      auto floatArgument1 = lambda->GetFunctionArguments().at(0);
      auto floatArgument2 = lambda->GetFunctionArguments().at(1);

      jlm::rvsdg::CreateOpNode<FBinaryOperation>(
          { floatArgument1, floatArgument2 },
          binOp,
          floatType);

      lambda->finalize({});

      // Convert the RVSDG to MLIR
      std::cout << "Convert to MLIR" << std::endl;
      jlm::mlir::JlmToMlirConverter mlirgen;
      auto omega = mlirgen.ConvertModule(*rvsdgModule);

      // Validate the generated MLIR
      std::cout << "Validate MLIR" << std::endl;
      auto & mlirOp = omega.getRegion().front().front().getRegion(0).front().front();
      switch (binOp)
      {
      case fpop::add:
        EXPECT_TRUE(mlir::isa<mlir::arith::AddFOp>(mlirOp));
        break;
      case fpop::sub:
        EXPECT_TRUE(mlir::isa<mlir::arith::SubFOp>(mlirOp));
        break;
      case fpop::mul:
        EXPECT_TRUE(mlir::isa<mlir::arith::MulFOp>(mlirOp));
        break;
      case fpop::div:
        EXPECT_TRUE(mlir::isa<mlir::arith::DivFOp>(mlirOp));
        break;
      case fpop::mod:
        EXPECT_TRUE(mlir::isa<mlir::arith::RemFOp>(mlirOp));
        break;
      default:
        FAIL();
      }

      // Convert the MLIR to RVSDG and check the result
      std::cout << "Converting MLIR to RVSDG" << std::endl;
      std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
      rootBlock->push_back(omega);
      auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
      auto region = &rvsdgModule->Rvsdg().GetRootRegion();
      {
        using namespace jlm::llvm;

        EXPECT_EQ(region->numNodes(), 1);
        auto convertedLambda =
            jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
        EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);

        auto node = convertedLambda->subregion()->Nodes().begin().ptr();
        auto convertedFpbin =
            jlm::util::assertedCast<const FBinaryOperation>(&node->GetOperation());
        EXPECT_EQ(convertedFpbin->fpop(), binOp);
        EXPECT_EQ(convertedFpbin->nresults(), 1);
        EXPECT_EQ(convertedFpbin->narguments(), 2);
      }
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFMulAddOp)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
    auto functionType =
        jlm::rvsdg::FunctionType::Create({ floatType, floatType, floatType }, { floatType });
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

    auto floatArgument1 = lambda->GetFunctionArguments().at(0);
    auto floatArgument2 = lambda->GetFunctionArguments().at(1);
    auto floatArgument3 = lambda->GetFunctionArguments().at(2);

    auto & node = jlm::rvsdg::CreateOpNode<jlm::llvm::FMulAddIntrinsicOperation>(
        { floatArgument1, floatArgument2, floatArgument3 },
        floatType);

    lambda->finalize({ node.output(0) });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & mlirOp = omega.getRegion().front().front().getRegion(0).front().front();
    EXPECT_TRUE(mlir::isa<mlir::LLVM::FMulAddOp>(mlirOp));

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto roundTripModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);

    // Assert
    auto region = &roundTripModule->Rvsdg().GetRootRegion();
    EXPECT_EQ(region->numNodes(), 1);
    auto convertedLambda =
        jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
    EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
    const auto arguments = convertedLambda->GetFunctionArguments();
    const auto results = convertedLambda->GetFunctionResults();
    EXPECT_EQ(arguments.size(), 3);
    EXPECT_EQ(results.size(), 1);

    auto & convertedNode = *convertedLambda->subregion()->Nodes().begin();
    EXPECT_TRUE(is<jlm::llvm::FMulAddIntrinsicOperation>(&convertedNode));
    EXPECT_EQ(convertedNode.input(0)->origin(), arguments[0]);
    EXPECT_EQ(convertedNode.input(1)->origin(), arguments[1]);
    EXPECT_EQ(convertedNode.input(2)->origin(), arguments[2]);
    EXPECT_EQ(results[0]->origin(), convertedNode.output(0));
  }
}

TEST(JlmToMlirToJlmTests, TestGetElementPtr)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto pointerType = PointerType::Create();
    auto bitType = jlm::rvsdg::BitType::Create(32);

    auto functionType = jlm::rvsdg::FunctionType::Create({ pointerType, bitType }, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

    auto pointerArgument = lambda->GetFunctionArguments().at(0);
    auto bitArgument = lambda->GetFunctionArguments().at(1);

    auto arrayType = ArrayType::Create(bitType, 2);

    GetElementPtrOperation::Create(
        pointerArgument,
        { bitArgument, bitArgument },
        arrayType,
        pointerType);

    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & op = omega.getRegion().front().front().getRegion(0).front().front();

    EXPECT_TRUE(mlir::isa<mlir::LLVM::GEPOp>(op));

    auto mlirGep = mlir::cast<mlir::LLVM::GEPOp>(op);
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirGep.getBase().getType()));
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirGep.getType()));

    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMArrayType>(mlirGep.getElemType()));
    auto mlirArrayType = mlir::cast<mlir::LLVM::LLVMArrayType>(mlirGep.getElemType());

    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(mlirArrayType.getElementType()));
    EXPECT_EQ(mlirArrayType.getNumElements(), 2);

    auto indices = mlirGep.getIndices();
    EXPECT_EQ(indices.size(), 2);
    auto index0 = indices[0].dyn_cast<mlir::Value>();
    auto index1 = indices[1].dyn_cast<mlir::Value>();
    EXPECT_NE(index0, nullptr);
    EXPECT_NE(index1, nullptr);
    EXPECT_TRUE(index0.getType().isa<mlir::IntegerType>());
    EXPECT_TRUE(index1.getType().isa<mlir::IntegerType>());
    EXPECT_EQ(index0.getType().getIntOrFloatBitWidth(), 32);
    EXPECT_EQ(index1.getType().getIntOrFloatBitWidth(), 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);

      auto op = convertedLambda->subregion()->Nodes().begin();
      EXPECT_TRUE(is<GetElementPtrOperation>(op->GetOperation()));
      auto convertedGep = dynamic_cast<const GetElementPtrOperation *>(&op->GetOperation());

      EXPECT_TRUE(is<ArrayType>(convertedGep->GetPointeeType()));
      EXPECT_TRUE(is<PointerType>(convertedGep->result(0)));
      EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedGep->argument(1)));
      EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedGep->argument(2)));
    }
  }
}

TEST(JlmToMlirToJlmTests, TestDelta)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto bitType = jlm::rvsdg::BitType::Create(32);

    auto delta1 = jlm::rvsdg::DeltaNode::Create(
        &graph->GetRootRegion(),
        jlm::llvm::DeltaOperation::Create(
            bitType,
            "non-constant-delta",
            Linkage::externalLinkage,
            "section",
            false));

    auto bitConstant = &jlm::rvsdg::BitConstantOperation::create(*delta1->subregion(), { 32, 1 });
    delta1->finalize(bitConstant);

    auto delta2 = jlm::rvsdg::DeltaNode::Create(
        &graph->GetRootRegion(),
        jlm::llvm::DeltaOperation::Create(
            bitType,
            "constant-delta",
            Linkage::externalLinkage,
            "section",
            true));
    auto bitConstant2 = &jlm::rvsdg::BitConstantOperation::create(*delta2->subregion(), { 32, 1 });
    delta2->finalize(bitConstant2);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;

    auto & omegaBlock = omega.getRegion().front();
    EXPECT_EQ(omegaBlock.getOperations().size(), 3); // 2 delta nodes + 1 omegaresult
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirDeltaNode = ::mlir::dyn_cast<::mlir::rvsdg::DeltaNode>(&op);
      auto mlirOmegaResult = ::mlir::dyn_cast<::mlir::rvsdg::OmegaResult>(&op);

      EXPECT_TRUE(mlirDeltaNode || mlirOmegaResult);

      if (mlirOmegaResult)
      {
        break;
      }

      if (mlirDeltaNode.getConstant())
      {
        EXPECT_EQ(mlirDeltaNode.getName().str(), "constant-delta");
      }
      else
      {
        EXPECT_EQ(mlirDeltaNode.getName().str(), "non-constant-delta");
      }

      EXPECT_EQ(mlirDeltaNode.getSection(), "section");
      EXPECT_EQ(mlirDeltaNode.getLinkage(), "external_linkage");
      EXPECT_TRUE(mlirDeltaNode.getType().isa<mlir::LLVM::LLVMPointerType>());
      auto terminator = mlirDeltaNode.getRegion().front().getTerminator();
      EXPECT_NE(terminator, nullptr);
      EXPECT_EQ(terminator->getNumOperands(), 1);
      EXPECT_TRUE(terminator->getOperand(0).getType().isa<mlir::IntegerType>());
    }

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);
      for (auto & node : region->Nodes())
      {
        auto convertedDelta = jlm::util::assertedCast<jlm::rvsdg::DeltaNode>(&node);
        EXPECT_EQ(convertedDelta->subregion()->numNodes(), 1);
        auto dop = jlm::util::assertedCast<const jlm::llvm::DeltaOperation>(&node.GetOperation());

        if (convertedDelta->constant())
        {
          EXPECT_EQ(dop->name(), "constant-delta");
        }
        else
        {
          EXPECT_EQ(dop->name(), "non-constant-delta");
        }

        EXPECT_TRUE(is<jlm::rvsdg::BitType>(*dop->Type()));
        EXPECT_EQ(dop->linkage(), Linkage::externalLinkage);
        EXPECT_EQ(dop->Section(), "section");

        auto op = convertedDelta->subregion()->Nodes().begin();
        EXPECT_TRUE(is<jlm::llvm::IntegerConstantOperation>(op->GetOperation()));
      }
    }
  }
}

TEST(JlmToMlirToJlmTests, TestConstantDataArray)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitConstant1 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 32, 1 });
    auto bitConstant2 =
        &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 32, 2 });
    auto bitType = jlm::rvsdg::BitType::Create(32);
    jlm::llvm::ConstantDataArray::Create({ bitConstant1, bitConstant2 });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundConstantDataArray = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirConstantDataArray = ::mlir::dyn_cast<::mlir::jlm::ConstantDataArray>(&op);
      if (mlirConstantDataArray)
      {
        EXPECT_EQ(mlirConstantDataArray.getNumOperands(), 2);
        EXPECT_TRUE(mlirConstantDataArray.getOperand(0).getType().isa<mlir::IntegerType>());
        EXPECT_TRUE(mlirConstantDataArray.getOperand(1).getType().isa<mlir::IntegerType>());
        auto mlirConstantDataArrayResultType =
            mlirConstantDataArray.getResult().getType().dyn_cast<mlir::LLVM::LLVMArrayType>();
        EXPECT_NE(mlirConstantDataArrayResultType, nullptr);
        EXPECT_TRUE(mlirConstantDataArrayResultType.getElementType().isa<mlir::IntegerType>());
        EXPECT_EQ(mlirConstantDataArrayResultType.getNumElements(), 2);
        foundConstantDataArray = true;
      }
    }
    EXPECT_TRUE(foundConstantDataArray);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 3);
      bool foundConstantDataArray = false;
      for (auto & node : region->Nodes())
      {
        if (auto constantDataArray = dynamic_cast<const ConstantDataArray *>(&node.GetOperation()))
        {
          foundConstantDataArray = true;
          EXPECT_EQ(constantDataArray->nresults(), 1);
          EXPECT_EQ(constantDataArray->narguments(), 2);
          auto resultType = constantDataArray->result(0);
          auto arrayType = dynamic_cast<const jlm::llvm::ArrayType *>(resultType.get());
          EXPECT_NE(arrayType, nullptr);
          EXPECT_TRUE(is<jlm::rvsdg::BitType>(arrayType->element_type()));
          EXPECT_EQ(arrayType->nelements(), 2);
          EXPECT_TRUE(is<jlm::rvsdg::BitType>(constantDataArray->argument(0)));
          EXPECT_TRUE(is<jlm::rvsdg::BitType>(constantDataArray->argument(1)));
        }
      }
      EXPECT_TRUE(foundConstantDataArray);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestConstantAggregateZero)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType = jlm::rvsdg::BitType::Create(32);
    auto arrayType = jlm::llvm::ArrayType::Create(bitType, 2);
    ConstantAggregateZeroOperation::Create(graph->GetRootRegion(), arrayType);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto mlirConstantAggregateZero = ::mlir::dyn_cast<::mlir::LLVM::ZeroOp>(&omegaBlock.front());
    EXPECT_NE(mlirConstantAggregateZero, nullptr);
    auto mlirConstantAggregateZeroResultType =
        mlirConstantAggregateZero.getType().dyn_cast<mlir::LLVM::LLVMArrayType>();
    EXPECT_NE(mlirConstantAggregateZeroResultType, nullptr);
    EXPECT_TRUE(mlirConstantAggregateZeroResultType.getElementType().isa<mlir::IntegerType>());
    EXPECT_EQ(mlirConstantAggregateZeroResultType.getNumElements(), 2);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto const convertedConstantAggregateZero =
          jlm::util::assertedCast<const ConstantAggregateZeroOperation>(
              &region->Nodes().begin().ptr()->GetOperation());
      EXPECT_EQ(convertedConstantAggregateZero->nresults(), 1);
      EXPECT_EQ(convertedConstantAggregateZero->narguments(), 0);
      auto resultType = convertedConstantAggregateZero->result(0);
      auto arrayType = dynamic_cast<const jlm::llvm::ArrayType *>(resultType.get());
      EXPECT_NE(arrayType, nullptr);
      EXPECT_TRUE(is<jlm::rvsdg::BitType>(arrayType->element_type()));
      EXPECT_EQ(arrayType->nelements(), 2);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestVarArgList)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType = jlm::rvsdg::BitType::Create(32);
    auto bits1 = &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 32, 1 });
    auto bits2 = &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 32, 2 });
    jlm::llvm::VariadicArgumentListOperation::Create(graph->GetRootRegion(), { bits1, bits2 });

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundVarArgOp = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirVarArgOp = ::mlir::dyn_cast<::mlir::jlm::CreateVarArgList>(&op);
      if (mlirVarArgOp)
      {
        EXPECT_EQ(mlirVarArgOp.getOperands().size(), 2);
        EXPECT_TRUE(mlirVarArgOp.getOperands()[0].getType().isa<mlir::IntegerType>());
        EXPECT_TRUE(mlirVarArgOp.getOperands()[1].getType().isa<mlir::IntegerType>());
        EXPECT_TRUE(mlirVarArgOp.getResult().getType().isa<mlir::jlm::VarargListType>());
        foundVarArgOp = true;
      }
    }
    EXPECT_TRUE(foundVarArgOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 3);
      bool foundVarArgOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedVarArgOp =
            dynamic_cast<const VariadicArgumentListOperation *>(&node.GetOperation());
        if (convertedVarArgOp)
        {
          EXPECT_EQ(convertedVarArgOp->nresults(), 1);
          EXPECT_EQ(convertedVarArgOp->narguments(), 2);
          auto resultType = convertedVarArgOp->result(0);
          EXPECT_TRUE(is<jlm::llvm::VariableArgumentType>(resultType));
          EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedVarArgOp->argument(0)));
          EXPECT_TRUE(is<jlm::rvsdg::BitType>(convertedVarArgOp->argument(1)));
          foundVarArgOp = true;
        }
      }
      EXPECT_TRUE(foundVarArgOp);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFNeg)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::flt);
    auto & constNode = jlm::rvsdg::CreateOpNode<ConstantFP>(
        graph->GetRootRegion(),
        floatType,
        ::llvm::APFloat(2.0));
    jlm::rvsdg::CreateOpNode<FNegOperation>({ constNode.output(0) }, jlm::llvm::fpsize::flt);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundFNegOp = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirFNegOp = ::mlir::dyn_cast<::mlir::arith::NegFOp>(&op);
      if (mlirFNegOp)
      {
        auto inputFloatType = mlirFNegOp.getOperand().getType().dyn_cast<mlir::FloatType>();
        EXPECT_NE(inputFloatType, nullptr);
        EXPECT_EQ(inputFloatType.getWidth(), 32);
        auto outputFloatType = mlirFNegOp.getResult().getType().dyn_cast<mlir::FloatType>();
        EXPECT_NE(outputFloatType, nullptr);
        EXPECT_EQ(outputFloatType.getWidth(), 32);
        foundFNegOp = true;
      }
    }
    EXPECT_TRUE(foundFNegOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);
      bool foundFNegOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedFNegOp = dynamic_cast<const FNegOperation *>(&node.GetOperation());
        if (convertedFNegOp)
        {
          EXPECT_EQ(convertedFNegOp->nresults(), 1);
          EXPECT_EQ(convertedFNegOp->narguments(), 1);
          auto inputFloatType = jlm::util::assertedCast<const jlm::llvm::FloatingPointType>(
              convertedFNegOp->argument(0).get());
          EXPECT_EQ(inputFloatType->size(), jlm::llvm::fpsize::flt);
          auto outputFloatType = jlm::util::assertedCast<const jlm::llvm::FloatingPointType>(
              convertedFNegOp->result(0).get());
          EXPECT_EQ(outputFloatType->size(), jlm::llvm::fpsize::flt);
          foundFNegOp = true;
        }
      }
      EXPECT_TRUE(foundFNegOp);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFPExt)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto floatType1 = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::flt);
    auto floatType2 = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
    auto & constNode = jlm::rvsdg::CreateOpNode<ConstantFP>(
        graph->GetRootRegion(),
        floatType1,
        ::llvm::APFloat(2.0));
    jlm::rvsdg::CreateOpNode<FPExtOperation>({ constNode.output(0) }, floatType1, floatType2);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundFPExtOp = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirFPExtOp = ::mlir::dyn_cast<::mlir::arith::ExtFOp>(&op);
      if (mlirFPExtOp)
      {
        auto inputFloatType = mlirFPExtOp.getOperand().getType().dyn_cast<mlir::FloatType>();
        EXPECT_NE(inputFloatType, nullptr);
        EXPECT_EQ(inputFloatType.getWidth(), 32);
        auto outputFloatType = mlirFPExtOp.getResult().getType().dyn_cast<mlir::FloatType>();
        EXPECT_NE(outputFloatType, nullptr);
        EXPECT_EQ(outputFloatType.getWidth(), 64);
        foundFPExtOp = true;
      }
    }
    EXPECT_TRUE(foundFPExtOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);
      bool foundFPExtOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedFPExtOp = dynamic_cast<const FPExtOperation *>(&node.GetOperation());
        if (convertedFPExtOp)
        {
          EXPECT_EQ(convertedFPExtOp->nresults(), 1);
          EXPECT_EQ(convertedFPExtOp->narguments(), 1);
          auto inputFloatType = jlm::util::assertedCast<const jlm::llvm::FloatingPointType>(
              convertedFPExtOp->argument(0).get());
          EXPECT_EQ(inputFloatType->size(), jlm::llvm::fpsize::flt);
          auto outputFloatType = jlm::util::assertedCast<const jlm::llvm::FloatingPointType>(
              convertedFPExtOp->result(0).get());
          EXPECT_EQ(outputFloatType->size(), jlm::llvm::fpsize::dbl);
          foundFPExtOp = true;
        }
      }
      EXPECT_TRUE(foundFPExtOp);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestTrunc)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType1 = jlm::rvsdg::BitType::Create(64);
    auto bitType2 = jlm::rvsdg::BitType::Create(32);
    auto constOp = &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 64, 2 });
    jlm::rvsdg::CreateOpNode<TruncOperation>({ constOp }, bitType1, bitType2);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundTruncOp = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirTruncOp = ::mlir::dyn_cast<::mlir::arith::TruncIOp>(&op);
      if (mlirTruncOp)
      {
        auto inputBitType = mlirTruncOp.getOperand().getType().dyn_cast<mlir::IntegerType>();
        EXPECT_NE(inputBitType, nullptr);
        EXPECT_EQ(inputBitType.getWidth(), 64);
        auto outputBitType = mlirTruncOp.getResult().getType().dyn_cast<mlir::IntegerType>();
        EXPECT_NE(outputBitType, nullptr);
        EXPECT_EQ(outputBitType.getWidth(), 32);
        foundTruncOp = true;
      }
    }
    EXPECT_TRUE(foundTruncOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);
      bool foundTruncOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedTruncOp = dynamic_cast<const TruncOperation *>(&node.GetOperation());
        if (convertedTruncOp)
        {
          EXPECT_EQ(convertedTruncOp->nresults(), 1);
          EXPECT_EQ(convertedTruncOp->narguments(), 1);
          auto inputBitType = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedTruncOp->argument(0).get());
          EXPECT_EQ(inputBitType->nbits(), 64);
          auto outputBitType =
              jlm::util::assertedCast<const jlm::rvsdg::BitType>(convertedTruncOp->result(0).get());
          EXPECT_EQ(outputBitType->nbits(), 32);
          foundTruncOp = true;
        }
      }
      EXPECT_TRUE(foundTruncOp);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFree)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create() },
        {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto iOStateArgument = lambda->GetFunctionArguments().at(0);
    auto memoryStateArgument = lambda->GetFunctionArguments().at(1);
    auto pointerArgument = lambda->GetFunctionArguments().at(2);

    // Create load operation
    auto freeOp =
        jlm::llvm::FreeOperation::Create(pointerArgument, { memoryStateArgument }, iOStateArgument);
    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();
    auto & mlirOp = mlirLambdaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::jlm::Free>(mlirOp));

    auto mlirFree = mlir::cast<mlir::jlm::Free>(mlirOp);
    EXPECT_EQ(mlirFree.getNumOperands(), 3);
    EXPECT_EQ(mlirFree.getNumResults(), 2);

    auto inputType1 = mlirFree.getOperand(0).getType();
    auto inputType2 = mlirFree.getOperand(1).getType();
    auto inputType3 = mlirFree.getOperand(2).getType();
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(inputType1));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::MemStateEdgeType>(inputType2));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::IOStateEdgeType>(inputType3));

    auto outputType1 = mlirFree.getResult(0).getType();
    auto outputType2 = mlirFree.getResult(1).getType();
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::MemStateEdgeType>(outputType1));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::IOStateEdgeType>(outputType2));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 1);
      auto convertedLambda =
          jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      EXPECT_TRUE(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      EXPECT_EQ(convertedLambda->subregion()->numNodes(), 1);
      EXPECT_TRUE(is<FreeOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedFree = dynamic_cast<const FreeOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      EXPECT_EQ(convertedFree->narguments(), 3);
      EXPECT_EQ(convertedFree->nresults(), 2);

      EXPECT_TRUE(is<jlm::llvm::PointerType>(convertedFree->argument(0)));
      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedFree->argument(1)));
      EXPECT_TRUE(is<jlm::llvm::IOStateType>(convertedFree->argument(2)));

      EXPECT_TRUE(is<jlm::llvm::MemoryStateType>(convertedFree->result(0)));
      EXPECT_TRUE(is<jlm::llvm::IOStateType>(convertedFree->result(1)));
    }
  }
}

TEST(JlmToMlirToJlmTests, TestFunctionGraphImport)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto functionType = jlm::rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    jlm::llvm::GraphImport::Create(
        *graph,
        functionType,
        functionType,
        "test",
        Linkage::externalLinkage);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirOp = omegaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::rvsdg::OmegaArgument>(mlirOp));

    auto mlirOmegaArgument = mlir::cast<mlir::rvsdg::OmegaArgument>(mlirOp);

    auto valueType = mlirOmegaArgument.getValueType();
    auto importedValueType = mlirOmegaArgument.getImportedValue().getType();
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

    auto mlirFunctionType = valueType.dyn_cast<mlir::FunctionType>();
    auto mlirImportedFunctionType = importedValueType.dyn_cast<mlir::FunctionType>();
    EXPECT_NE(mlirFunctionType, nullptr);
    EXPECT_NE(mlirImportedFunctionType, nullptr);
    EXPECT_EQ(mlirFunctionType, mlirImportedFunctionType);
    EXPECT_EQ(mlirFunctionType.getNumInputs(), 3);
    EXPECT_EQ(mlirFunctionType.getNumResults(), 2);
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::IOStateEdgeType>(mlirFunctionType.getInput(0)));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirFunctionType.getInput(1)));
    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirFunctionType.getInput(2)));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::IOStateEdgeType>(mlirFunctionType.getResult(0)));
    EXPECT_TRUE(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirFunctionType.getResult(1)));
    EXPECT_EQ(linkage, "external_linkage");
    EXPECT_EQ(name, "test");

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 0);

      EXPECT_EQ(region->graph()->GetRootRegion().narguments(), 1);
      auto arg = region->graph()->GetRootRegion().argument(0);
      auto imp = dynamic_cast<jlm::llvm::GraphImport *>(arg);
      EXPECT_NE(imp, nullptr);
      EXPECT_EQ(imp->Name(), "test");
      EXPECT_EQ(imp->linkage(), Linkage::externalLinkage);
      EXPECT_EQ(*imp->ValueType(), *functionType);
      EXPECT_EQ(*imp->ImportedType(), *functionType);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestPointerGraphImport)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    jlm::llvm::GraphImport::Create(
        *graph,
        jlm::rvsdg::BitType::Create(32),
        PointerType::Create(),
        "test",
        Linkage::externalLinkage);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirOp = omegaBlock.front();

    EXPECT_TRUE(mlir::isa<mlir::rvsdg::OmegaArgument>(mlirOp));

    auto mlirOmegaArgument = mlir::cast<mlir::rvsdg::OmegaArgument>(mlirOp);

    auto valueType = mlirOmegaArgument.getValueType();
    auto importedValueType = mlirOmegaArgument.getImportedValue().getType();
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

    EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(importedValueType));

    auto mlirIntType = valueType.dyn_cast<mlir::IntegerType>();
    EXPECT_NE(mlirIntType, nullptr);
    EXPECT_EQ(mlirIntType.getWidth(), 32);
    EXPECT_EQ(linkage, "external_linkage");
    EXPECT_EQ(name, "test");

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 0);

      EXPECT_EQ(region->graph()->GetRootRegion().narguments(), 1);
      auto arg = region->graph()->GetRootRegion().argument(0);
      auto imp = dynamic_cast<jlm::llvm::GraphImport *>(arg);
      EXPECT_NE(imp, nullptr);
      EXPECT_EQ(imp->Name(), "test");
      EXPECT_EQ(imp->linkage(), Linkage::externalLinkage);
      EXPECT_EQ(*imp->ValueType(), *jlm::rvsdg::BitType::Create(32));
      EXPECT_EQ(*imp->ImportedType(), *PointerType::Create());
    }
  }
}

// Add IOBarrier test near the end of the file, before the last test registrations
TEST(JlmToMlirToJlmTests, TestIOBarrier)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create a function to contain the test
    auto functionType = jlm::rvsdg::FunctionType::Create({ IOStateType::Create() }, {});

    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));
    auto ioStateArgument = lambda->GetFunctionArguments()[0];

    // Create a value to pass through the barrier
    auto value = &jlm::rvsdg::BitConstantOperation::create(*lambda->subregion(), { 32, 42 });

    // Create the IOBarrier operation
    jlm::rvsdg::CreateOpNode<jlm::llvm::IOBarrierOperation>(
        { value, ioStateArgument },
        jlm::rvsdg::BitType::Create(32));

    // Finalize the lambda
    lambda->finalize({});

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    EXPECT_EQ(omegaRegion.getBlocks().size(), 1);
    auto & omegaBlock = omegaRegion.front();
    auto & mlirLambda = omegaBlock.front();
    auto & mlirLambdaRegion = mlirLambda.getRegion(0);
    auto & mlirLambdaBlock = mlirLambdaRegion.front();

    // Check for lambda operation
    bool foundIOBarrier = false;
    for (auto & lambdaOp : mlirLambdaBlock.getOperations())
    {
      if (auto ioBarrier = mlir::dyn_cast<mlir::jlm::IOBarrier>(&lambdaOp))
      {
        foundIOBarrier = true;

        // Check that the IOBarrier has 2 operands (value and IO state)
        EXPECT_EQ(ioBarrier->getNumOperands(), 2);

        // Check that the first operand is a 32-bit integer
        auto valueType = ioBarrier->getOperand(0).getType().dyn_cast<mlir::IntegerType>();
        EXPECT_NE(valueType, nullptr);
        EXPECT_EQ(valueType.getWidth(), 32);
        EXPECT_TRUE(mlir::isa<mlir::rvsdg::IOStateEdgeType>(ioBarrier->getOperand(1).getType()));

        // Check that the result type matches the input value type
        auto resultType = ioBarrier->getResult(0).getType().dyn_cast<mlir::IntegerType>();
        EXPECT_NE(resultType, nullptr);
        EXPECT_EQ(resultType.getWidth(), 32);
      }
    }
    EXPECT_TRUE(foundIOBarrier);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto convertedRvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &convertedRvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      // Direct access to the lambda node
      EXPECT_EQ(region->numNodes(), 1);
      auto & lambdaNode = *region->Nodes().begin();
      auto lambdaOperation = dynamic_cast<const jlm::rvsdg::LambdaNode *>(&lambdaNode);
      EXPECT_NE(lambdaOperation, nullptr);

      // Find the IOBarrier in the lambda subregion
      bool foundIOBarrier = false;
      for (auto & lambdaNode : lambdaOperation->subregion()->Nodes())
      {
        auto ioBarrierOp = dynamic_cast<const IOBarrierOperation *>(&lambdaNode.GetOperation());
        if (ioBarrierOp)
        {
          foundIOBarrier = true;

          // Check that it has correct number of inputs and outputs
          EXPECT_EQ(ioBarrierOp->nresults(), 1);
          EXPECT_EQ(ioBarrierOp->narguments(), 2);

          // Check that the first input is the 32-bit value
          auto valueType =
              dynamic_cast<const jlm::rvsdg::BitType *>(ioBarrierOp->argument(0).get());
          EXPECT_NE(valueType, nullptr);
          EXPECT_EQ(valueType->nbits(), 32);

          // Check that the second input is an IO state
          auto ioStateType = dynamic_cast<const IOStateType *>(ioBarrierOp->argument(1).get());
          EXPECT_NE(ioStateType, nullptr);

          // Check that the output type matches the input value type
          auto outputType = dynamic_cast<const jlm::rvsdg::BitType *>(ioBarrierOp->result(0).get());
          EXPECT_NE(outputType, nullptr);
          EXPECT_EQ(outputType->nbits(), 32);
        }
      }
      EXPECT_TRUE(foundIOBarrier);
    }
  }
}

TEST(JlmToMlirToJlmTests, TestMalloc)
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto constOp = &jlm::rvsdg::BitConstantOperation::create(graph->GetRootRegion(), { 64, 2 });
    MallocOperation::create(constOp);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    bool foundMallocOp = false;
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirMallocOp = ::mlir::dyn_cast<::mlir::jlm::Malloc>(&op);
      if (mlirMallocOp)
      {
        auto inputBitType = mlirMallocOp.getOperand().getType().dyn_cast<mlir::IntegerType>();
        EXPECT_NE(inputBitType, nullptr);
        EXPECT_EQ(inputBitType.getWidth(), 64);
        EXPECT_TRUE(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirMallocOp.getResult(0).getType()));
        EXPECT_TRUE(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirMallocOp.getResult(1).getType()));
        foundMallocOp = true;
      }
    }
    EXPECT_TRUE(foundMallocOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      EXPECT_EQ(region->numNodes(), 2);
      bool foundMallocOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedMallocOp = dynamic_cast<const MallocOperation *>(&node.GetOperation());
        if (convertedMallocOp)
        {
          EXPECT_EQ(convertedMallocOp->nresults(), 2);
          EXPECT_EQ(convertedMallocOp->narguments(), 1);
          auto inputBitType = jlm::util::assertedCast<const jlm::rvsdg::BitType>(
              convertedMallocOp->argument(0).get());
          EXPECT_EQ(inputBitType->nbits(), 64);
          EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::PointerType>(convertedMallocOp->result(0)));
          EXPECT_TRUE(jlm::rvsdg::is<jlm::llvm::MemoryStateType>(convertedMallocOp->result(1)));
          foundMallocOp = true;
        }
      }
      EXPECT_TRUE(foundMallocOp);
    }
  }
}
