/*
 * Copyright 2024 Halvor Linder Henriksen <halvorlinder@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

static void
TestUndef()
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
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();
    // 1 undef + omegaResult
    assert(omegaBlock.getOperations().size() == 2);
    assert(mlir::isa<mlir::jlm::Undef>(omegaBlock.front()));
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

      assert(region->numNodes() == 1);

      // Get the undef op
      auto convertedUndef =
          dynamic_cast<const UndefValueOperation *>(&region->Nodes().begin()->GetOperation());

      assert(convertedUndef != nullptr);

      auto outputType = convertedUndef->result(0);
      assert(jlm::rvsdg::is<const jlm::rvsdg::BitType>(outputType));
      assert(std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(outputType)->nbits() == 32);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirUndefGen", TestUndef)

static void
TestAlloca()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create a bits node for alloc size
    std::cout << "Bit Constanr" << std::endl;
    auto bits = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);

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
    assert(omegaRegion.getBlocks().size() == 1);
    auto & omegaBlock = omegaRegion.front();

    // Bit-contant + alloca + omegaResult
    assert(omegaBlock.getOperations().size() == 3);

    bool foundAlloca = false;
    for (auto & op : omegaBlock)
    {
      if (mlir::isa<mlir::jlm::Alloca>(op))
      {
        auto mlirAllocaOp = mlir::cast<mlir::jlm::Alloca>(op);
        assert(mlirAllocaOp.getAlignment() == 4);
        assert(mlirAllocaOp.getNumResults() == 2);

        auto valueType = mlir::cast<mlir::IntegerType>(mlirAllocaOp.getValueType());
        assert(valueType);
        assert(valueType.getWidth() == 64);
        foundAlloca = true;
      }
    }
    assert(foundAlloca);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);

      bool foundAlloca = false;
      for (auto & node : region->Nodes())
      {
        if (auto allocaOp = dynamic_cast<const AllocaOperation *>(&node.GetOperation()))
        {
          assert(allocaOp->alignment() == 4);

          assert(jlm::rvsdg::is<jlm::rvsdg::BitType>(allocaOp->ValueType()));
          auto valueBitType =
              dynamic_cast<const jlm::rvsdg::BitType *>(allocaOp->ValueType().get());
          assert(valueBitType->nbits() == 64);

          assert(allocaOp->narguments() == 1);

          assert(jlm::rvsdg::is<jlm::rvsdg::BitType>(allocaOp->argument(0)));
          auto inputBitType =
              dynamic_cast<const jlm::rvsdg::BitType *>(allocaOp->argument(0).get());
          assert(inputBitType->nbits() == 32);

          assert(allocaOp->nresults() == 2);

          assert(jlm::rvsdg::is<PointerType>(allocaOp->result(0)));
          assert(jlm::rvsdg::is<jlm::llvm::MemoryStateType>(allocaOp->result(1)));

          foundAlloca = true;
        }
      }
      assert(foundAlloca);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirAllocaGen", TestAlloca)

static void
TestLoad()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
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

    assert(mlir::isa<mlir::jlm::Load>(mlirOp));

    auto mlirLoad = mlir::cast<mlir::jlm::Load>(mlirOp);
    assert(mlirLoad.getAlignment() == 4);
    assert(mlirLoad.getInputMemStates().size() == 1);
    assert(mlirLoad.getNumOperands() == 2);
    assert(mlirLoad.getNumResults() == 2);

    auto outputType = mlirLoad.getOutput().getType();
    assert(mlir::isa<mlir::IntegerType>(outputType));
    auto integerType = mlir::cast<mlir::IntegerType>(outputType);
    assert(integerType.getWidth() == 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<LoadNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedLoad = convertedLambda->subregion()->Nodes().begin().ptr();
      auto loadOperation =
          dynamic_cast<const LoadNonVolatileOperation *>(&convertedLoad->GetOperation());

      assert(loadOperation->GetAlignment() == 4);
      assert(loadOperation->NumMemoryStates() == 1);

      assert(is<jlm::llvm::PointerType>(convertedLoad->input(0)->Type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedLoad->input(1)->Type()));

      assert(is<jlm::rvsdg::BitType>(convertedLoad->output(0)->Type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedLoad->output(1)->Type()));

      auto outputBitType =
          std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(convertedLoad->output(0)->Type());
      assert(outputBitType->nbits() == 32);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirLoadGen", TestLoad)

static void
TestStore()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
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

    assert(mlir::isa<mlir::jlm::Store>(mlirOp));

    auto mlirStore = mlir::cast<mlir::jlm::Store>(mlirOp);
    assert(mlirStore.getAlignment() == 4);
    assert(mlirStore.getInputMemStates().size() == 1);
    assert(mlirStore.getNumOperands() == 3);

    auto inputType = mlirStore.getValue().getType();
    assert(mlir::isa<mlir::IntegerType>(inputType));
    auto integerType = mlir::cast<mlir::IntegerType>(inputType);
    assert(integerType.getWidth() == 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<StoreNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedStore = convertedLambda->subregion()->Nodes().begin().ptr();
      auto convertedStoreOperation =
          dynamic_cast<const StoreNonVolatileOperation *>(&convertedStore->GetOperation());

      assert(convertedStoreOperation->GetAlignment() == 4);
      assert(convertedStoreOperation->NumMemoryStates() == 1);

      assert(is<jlm::llvm::PointerType>(convertedStore->input(0)->Type()));
      assert(is<jlm::rvsdg::BitType>(convertedStore->input(1)->Type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedStore->input(2)->Type()));

      assert(is<jlm::llvm::MemoryStateType>(convertedStore->output(0)->Type()));

      auto inputBitType =
          std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(convertedStore->input(1)->Type());
      assert(inputBitType->nbits() == 32);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirStoreGen", TestStore)

static void
TestSext()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto bitsArgument = lambda->GetFunctionArguments().at(0);

    // Create sext operation
    auto sextOp = jlm::llvm::SExtOperation::create((size_t)64, bitsArgument);
    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*sextOp);
    assert(node);

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

    assert(mlir::isa<mlir::arith::ExtSIOp>(mlirOp));

    auto mlirSext = mlir::cast<mlir::arith::ExtSIOp>(mlirOp);
    auto inputType = mlirSext.getOperand().getType();
    auto outputType = mlirSext.getType();
    assert(mlir::isa<mlir::IntegerType>(inputType));
    assert(mlir::isa<mlir::IntegerType>(outputType));
    assert(mlir::cast<mlir::IntegerType>(inputType).getWidth() == 32);
    assert(mlir::cast<mlir::IntegerType>(outputType).getWidth() == 64);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<SExtOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSext = dynamic_cast<const SExtOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(convertedSext->ndstbits() == 64);
      assert(convertedSext->nsrcbits() == 32);
      assert(convertedSext->nresults() == 1);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirSextGen", TestSext)

static void
TestSitofp()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
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

    assert(mlir::isa<mlir::arith::SIToFPOp>(mlirOp));

    auto mlirSitofp = mlir::cast<mlir::arith::SIToFPOp>(mlirOp);
    auto inputType = mlirSitofp.getOperand().getType();
    auto outputType = mlirSitofp.getType();
    assert(mlir::isa<mlir::IntegerType>(inputType));
    assert(mlir::cast<mlir::IntegerType>(inputType).getWidth() == 32);
    assert(mlir::isa<mlir::Float64Type>(outputType));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<SIToFPOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSitofp = dynamic_cast<const SIToFPOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(jlm::rvsdg::is<jlm::rvsdg::BitType>(*convertedSitofp->argument(0).get()));
      assert(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedSitofp->result(0).get()));
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirSitofpGen", TestSitofp)

static void
TestConstantFP()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto functionType = jlm::rvsdg::FunctionType::Create({}, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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

    assert(mlir::isa<mlir::arith::ConstantFloatOp>(mlirOp));

    auto mlirConst = mlir::cast<mlir::arith::ConstantFloatOp>(mlirOp);
    assert(mlirConst.value().isExactlyValue(2.0));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();
    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<ConstantFP>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedConst = dynamic_cast<const ConstantFP *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedConst->result(0).get()));
      assert(convertedConst->constant().isExactlyValue(2.0));
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantFPGen", TestConstantFP)

static void
TestFpBinary()
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
          LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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
        assert(mlir::isa<mlir::arith::AddFOp>(mlirOp));
        break;
      case fpop::sub:
        assert(mlir::isa<mlir::arith::SubFOp>(mlirOp));
        break;
      case fpop::mul:
        assert(mlir::isa<mlir::arith::MulFOp>(mlirOp));
        break;
      case fpop::div:
        assert(mlir::isa<mlir::arith::DivFOp>(mlirOp));
        break;
      case fpop::mod:
        assert(mlir::isa<mlir::arith::RemFOp>(mlirOp));
        break;
      default:
        assert(false);
      }

      // Convert the MLIR to RVSDG and check the result
      std::cout << "Converting MLIR to RVSDG" << std::endl;
      std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
      rootBlock->push_back(omega);
      auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
      auto region = &rvsdgModule->Rvsdg().GetRootRegion();
      {
        using namespace jlm::llvm;

        assert(region->numNodes() == 1);
        auto convertedLambda =
            jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
        assert(convertedLambda->subregion()->numNodes() == 1);

        auto node = convertedLambda->subregion()->Nodes().begin().ptr();
        auto convertedFpbin =
            jlm::util::AssertedCast<const FBinaryOperation>(&node->GetOperation());
        assert(convertedFpbin->fpop() == binOp);
        assert(convertedFpbin->nresults() == 1);
        assert(convertedFpbin->narguments() == 2);
      }
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFpBinaryGen", TestFpBinary)

static void
TestFMulAddOp()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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
    assert(mlir::isa<mlir::LLVM::FMulAddOp>(mlirOp));

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto roundTripModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);

    // Assert
    auto region = &roundTripModule->Rvsdg().GetRootRegion();
    assert(region->numNodes() == 1);
    auto convertedLambda =
        jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
    assert(convertedLambda->subregion()->numNodes() == 1);
    const auto arguments = convertedLambda->GetFunctionArguments();
    const auto results = convertedLambda->GetFunctionResults();
    assert(arguments.size() == 3);
    assert(results.size() == 1);

    auto & convertedNode = *convertedLambda->subregion()->Nodes().begin();
    assert(is<jlm::llvm::FMulAddIntrinsicOperation>(&convertedNode));
    assert(convertedNode.input(0)->origin() == arguments[0]);
    assert(convertedNode.input(1)->origin() == arguments[1]);
    assert(convertedNode.input(2)->origin() == arguments[2]);
    assert(results[0]->origin() == convertedNode.output(0));
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFMulAddOp", TestFMulAddOp)

static void
TestGetElementPtr()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

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

    assert(mlir::isa<mlir::LLVM::GEPOp>(op));

    auto mlirGep = mlir::cast<mlir::LLVM::GEPOp>(op);
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirGep.getBase().getType()));
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirGep.getType()));

    assert(mlir::isa<mlir::LLVM::LLVMArrayType>(mlirGep.getElemType()));
    auto mlirArrayType = mlir::cast<mlir::LLVM::LLVMArrayType>(mlirGep.getElemType());

    assert(mlir::isa<mlir::IntegerType>(mlirArrayType.getElementType()));
    assert(mlirArrayType.getNumElements() == 2);

    auto indices = mlirGep.getIndices();
    assert(indices.size() == 2);
    auto index0 = indices[0].dyn_cast<mlir::Value>();
    auto index1 = indices[1].dyn_cast<mlir::Value>();
    assert(index0);
    assert(index1);
    assert(index0.getType().isa<mlir::IntegerType>());
    assert(index1.getType().isa<mlir::IntegerType>());
    assert(index0.getType().getIntOrFloatBitWidth() == 32);
    assert(index1.getType().getIntOrFloatBitWidth() == 32);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->numNodes() == 1);

      auto op = convertedLambda->subregion()->Nodes().begin();
      assert(is<GetElementPtrOperation>(op->GetOperation()));
      auto convertedGep = dynamic_cast<const GetElementPtrOperation *>(&op->GetOperation());

      assert(is<ArrayType>(convertedGep->GetPointeeType()));
      assert(is<PointerType>(convertedGep->result(0)));
      assert(is<jlm::rvsdg::BitType>(convertedGep->argument(1)));
      assert(is<jlm::rvsdg::BitType>(convertedGep->argument(2)));
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirGetElementPtrGen", TestGetElementPtr)

static void
TestDelta()
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
            linkage::external_linkage,
            "section",
            false));

    auto bitConstant = jlm::rvsdg::create_bitconstant(delta1->subregion(), 32, 1);
    delta1->finalize(bitConstant);

    auto delta2 = jlm::rvsdg::DeltaNode::Create(
        &graph->GetRootRegion(),
        jlm::llvm::DeltaOperation::Create(
            bitType,
            "constant-delta",
            linkage::external_linkage,
            "section",
            true));
    auto bitConstant2 = jlm::rvsdg::create_bitconstant(delta2->subregion(), 32, 1);
    delta2->finalize(bitConstant2);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;

    auto & omegaBlock = omega.getRegion().front();
    assert(omegaBlock.getOperations().size() == 3); // 2 delta nodes + 1 omegaresult
    for (auto & op : omegaBlock.getOperations())
    {
      auto mlirDeltaNode = ::mlir::dyn_cast<::mlir::rvsdg::DeltaNode>(&op);
      auto mlirOmegaResult = ::mlir::dyn_cast<::mlir::rvsdg::OmegaResult>(&op);

      assert(mlirDeltaNode || mlirOmegaResult);

      if (mlirOmegaResult)
      {
        break;
      }

      if (mlirDeltaNode.getConstant())
      {
        assert(mlirDeltaNode.getName().str() == "constant-delta");
      }
      else
      {
        assert(mlirDeltaNode.getName().str() == "non-constant-delta");
      }

      assert(mlirDeltaNode.getSection() == "section");
      assert(mlirDeltaNode.getLinkage() == "external_linkage");
      assert(mlirDeltaNode.getType().isa<mlir::LLVM::LLVMPointerType>());
      auto terminator = mlirDeltaNode.getRegion().front().getTerminator();
      assert(terminator);
      assert(terminator->getNumOperands() == 1);
      assert(terminator->getOperand(0).getType().isa<mlir::IntegerType>());
    }

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);
      for (auto & node : region->Nodes())
      {
        auto convertedDelta = jlm::util::AssertedCast<jlm::rvsdg::DeltaNode>(&node);
        assert(convertedDelta->subregion()->numNodes() == 1);
        auto dop = jlm::util::AssertedCast<const jlm::llvm::DeltaOperation>(&node.GetOperation());

        if (convertedDelta->constant())
        {
          assert(dop->name() == "constant-delta");
        }
        else
        {
          assert(dop->name() == "non-constant-delta");
        }

        assert(is<jlm::rvsdg::BitType>(*dop->Type()));
        assert(dop->linkage() == linkage::external_linkage);
        assert(dop->Section() == "section");

        auto op = convertedDelta->subregion()->Nodes().begin();
        assert(is<jlm::llvm::IntegerConstantOperation>(op->GetOperation()));
      }
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirDeltaGen", TestDelta)

static void
TestConstantDataArray()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitConstant1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);
    auto bitConstant2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 2);
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
        assert(mlirConstantDataArray.getNumOperands() == 2);
        assert(mlirConstantDataArray.getOperand(0).getType().isa<mlir::IntegerType>());
        assert(mlirConstantDataArray.getOperand(1).getType().isa<mlir::IntegerType>());
        auto mlirConstantDataArrayResultType =
            mlirConstantDataArray.getResult().getType().dyn_cast<mlir::LLVM::LLVMArrayType>();
        assert(mlirConstantDataArrayResultType);
        assert(mlirConstantDataArrayResultType.getElementType().isa<mlir::IntegerType>());
        assert(mlirConstantDataArrayResultType.getNumElements() == 2);
        foundConstantDataArray = true;
      }
    }
    assert(foundConstantDataArray);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 3);
      bool foundConstantDataArray = false;
      for (auto & node : region->Nodes())
      {
        if (auto constantDataArray = dynamic_cast<const ConstantDataArray *>(&node.GetOperation()))
        {
          foundConstantDataArray = true;
          assert(constantDataArray->nresults() == 1);
          assert(constantDataArray->narguments() == 2);
          auto resultType = constantDataArray->result(0);
          auto arrayType = dynamic_cast<const jlm::llvm::ArrayType *>(resultType.get());
          assert(arrayType);
          assert(is<jlm::rvsdg::BitType>(arrayType->element_type()));
          assert(arrayType->nelements() == 2);
          assert(is<jlm::rvsdg::BitType>(constantDataArray->argument(0)));
          assert(is<jlm::rvsdg::BitType>(constantDataArray->argument(1)));
        }
      }
      assert(foundConstantDataArray);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantDataArrayGen", TestConstantDataArray)

static void
TestConstantAggregateZero()
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
    assert(mlirConstantAggregateZero);
    auto mlirConstantAggregateZeroResultType =
        mlirConstantAggregateZero.getType().dyn_cast<mlir::LLVM::LLVMArrayType>();
    assert(mlirConstantAggregateZeroResultType);
    assert(mlirConstantAggregateZeroResultType.getElementType().isa<mlir::IntegerType>());
    assert(mlirConstantAggregateZeroResultType.getNumElements() == 2);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto const convertedConstantAggregateZero =
          jlm::util::AssertedCast<const ConstantAggregateZeroOperation>(
              &region->Nodes().begin().ptr()->GetOperation());
      assert(convertedConstantAggregateZero->nresults() == 1);
      assert(convertedConstantAggregateZero->narguments() == 0);
      auto resultType = convertedConstantAggregateZero->result(0);
      auto arrayType = dynamic_cast<const jlm::llvm::ArrayType *>(resultType.get());
      assert(arrayType);
      assert(is<jlm::rvsdg::BitType>(arrayType->element_type()));
      assert(arrayType->nelements() == 2);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantAggregateZeroGen", TestConstantAggregateZero)

static void
TestVarArgList()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType = jlm::rvsdg::BitType::Create(32);
    auto bits1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);
    auto bits2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 2);
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
        assert(mlirVarArgOp.getOperands().size() == 2);
        assert(mlirVarArgOp.getOperands()[0].getType().isa<mlir::IntegerType>());
        assert(mlirVarArgOp.getOperands()[1].getType().isa<mlir::IntegerType>());
        assert(mlirVarArgOp.getResult().getType().isa<mlir::jlm::VarargListType>());
        foundVarArgOp = true;
      }
    }
    assert(foundVarArgOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 3);
      bool foundVarArgOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedVarArgOp =
            dynamic_cast<const VariadicArgumentListOperation *>(&node.GetOperation());
        if (convertedVarArgOp)
        {
          assert(convertedVarArgOp->nresults() == 1);
          assert(convertedVarArgOp->narguments() == 2);
          auto resultType = convertedVarArgOp->result(0);
          assert(is<jlm::llvm::VariableArgumentType>(resultType));
          assert(is<jlm::rvsdg::BitType>(convertedVarArgOp->argument(0)));
          assert(is<jlm::rvsdg::BitType>(convertedVarArgOp->argument(1)));
          foundVarArgOp = true;
        }
      }
      assert(foundVarArgOp);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirVarArgListGen", TestVarArgList)

static void
TestFNeg()
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
        assert(inputFloatType);
        assert(inputFloatType.getWidth() == 32);
        auto outputFloatType = mlirFNegOp.getResult().getType().dyn_cast<mlir::FloatType>();
        assert(outputFloatType);
        assert(outputFloatType.getWidth() == 32);
        foundFNegOp = true;
      }
    }
    assert(foundFNegOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);
      bool foundFNegOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedFNegOp = dynamic_cast<const FNegOperation *>(&node.GetOperation());
        if (convertedFNegOp)
        {
          assert(convertedFNegOp->nresults() == 1);
          assert(convertedFNegOp->narguments() == 1);
          auto inputFloatType = jlm::util::AssertedCast<const jlm::llvm::FloatingPointType>(
              convertedFNegOp->argument(0).get());
          assert(inputFloatType->size() == jlm::llvm::fpsize::flt);
          auto outputFloatType = jlm::util::AssertedCast<const jlm::llvm::FloatingPointType>(
              convertedFNegOp->result(0).get());
          assert(outputFloatType->size() == jlm::llvm::fpsize::flt);
          foundFNegOp = true;
        }
      }
      assert(foundFNegOp);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFNegGen", TestFNeg)

static void
TestFPExt()
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
        assert(inputFloatType);
        assert(inputFloatType.getWidth() == 32);
        auto outputFloatType = mlirFPExtOp.getResult().getType().dyn_cast<mlir::FloatType>();
        assert(outputFloatType);
        assert(outputFloatType.getWidth() == 64);
        foundFPExtOp = true;
      }
    }
    assert(foundFPExtOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);
      bool foundFPExtOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedFPExtOp = dynamic_cast<const FPExtOperation *>(&node.GetOperation());
        if (convertedFPExtOp)
        {
          assert(convertedFPExtOp->nresults() == 1);
          assert(convertedFPExtOp->narguments() == 1);
          auto inputFloatType = jlm::util::AssertedCast<const jlm::llvm::FloatingPointType>(
              convertedFPExtOp->argument(0).get());
          assert(inputFloatType->size() == jlm::llvm::fpsize::flt);
          auto outputFloatType = jlm::util::AssertedCast<const jlm::llvm::FloatingPointType>(
              convertedFPExtOp->result(0).get());
          assert(outputFloatType->size() == jlm::llvm::fpsize::dbl);
          foundFPExtOp = true;
        }
      }
      assert(foundFPExtOp);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFPExtGen", TestFPExt)

static void
TestTrunc()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType1 = jlm::rvsdg::BitType::Create(64);
    auto bitType2 = jlm::rvsdg::BitType::Create(32);
    auto constOp = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 64, 2);
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
        assert(inputBitType);
        assert(inputBitType.getWidth() == 64);
        auto outputBitType = mlirTruncOp.getResult().getType().dyn_cast<mlir::IntegerType>();
        assert(outputBitType);
        assert(outputBitType.getWidth() == 32);
        foundTruncOp = true;
      }
    }
    assert(foundTruncOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);
      bool foundTruncOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedTruncOp = dynamic_cast<const TruncOperation *>(&node.GetOperation());
        if (convertedTruncOp)
        {
          assert(convertedTruncOp->nresults() == 1);
          assert(convertedTruncOp->narguments() == 1);
          auto inputBitType = jlm::util::AssertedCast<const jlm::rvsdg::BitType>(
              convertedTruncOp->argument(0).get());
          assert(inputBitType->nbits() == 64);
          auto outputBitType =
              jlm::util::AssertedCast<const jlm::rvsdg::BitType>(convertedTruncOp->result(0).get());
          assert(outputBitType->nbits() == 32);
          foundTruncOp = true;
        }
      }
      assert(foundTruncOp);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirTruncGen", TestTrunc)

static void
TestFree()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
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

    assert(mlir::isa<mlir::jlm::Free>(mlirOp));

    auto mlirFree = mlir::cast<mlir::jlm::Free>(mlirOp);
    assert(mlirFree.getNumOperands() == 3);
    assert(mlirFree.getNumResults() == 2);

    auto inputType1 = mlirFree.getOperand(0).getType();
    auto inputType2 = mlirFree.getOperand(1).getType();
    auto inputType3 = mlirFree.getOperand(2).getType();
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(inputType1));
    assert(mlir::isa<mlir::rvsdg::MemStateEdgeType>(inputType2));
    assert(mlir::isa<mlir::rvsdg::IOStateEdgeType>(inputType3));

    auto outputType1 = mlirFree.getResult(0).getType();
    auto outputType2 = mlirFree.getResult(1).getType();
    assert(mlir::isa<mlir::rvsdg::MemStateEdgeType>(outputType1));
    assert(mlir::isa<mlir::rvsdg::IOStateEdgeType>(outputType2));

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->numNodes() == 1);
      assert(is<FreeOperation>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedFree = dynamic_cast<const FreeOperation *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(convertedFree->narguments() == 3);
      assert(convertedFree->nresults() == 2);

      assert(is<jlm::llvm::PointerType>(convertedFree->argument(0)));
      assert(is<jlm::llvm::MemoryStateType>(convertedFree->argument(1)));
      assert(is<jlm::llvm::IOStateType>(convertedFree->argument(2)));

      assert(is<jlm::llvm::MemoryStateType>(convertedFree->result(0)));
      assert(is<jlm::llvm::IOStateType>(convertedFree->result(1)));
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFreeGen", TestFree)

static void
TestFunctionGraphImport()
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
        linkage::external_linkage);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirOp = omegaBlock.front();

    assert(mlir::isa<mlir::rvsdg::OmegaArgument>(mlirOp));

    auto mlirOmegaArgument = mlir::cast<mlir::rvsdg::OmegaArgument>(mlirOp);

    auto valueType = mlirOmegaArgument.getValueType();
    auto importedValueType = mlirOmegaArgument.getImportedValue().getType();
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

    auto mlirFunctionType = valueType.dyn_cast<mlir::FunctionType>();
    auto mlirImportedFunctionType = importedValueType.dyn_cast<mlir::FunctionType>();
    assert(mlirFunctionType);
    assert(mlirImportedFunctionType);
    assert(mlirFunctionType == mlirImportedFunctionType);
    assert(mlirFunctionType.getNumInputs() == 3);
    assert(mlirFunctionType.getNumResults() == 2);
    assert(mlir::isa<mlir::rvsdg::IOStateEdgeType>(mlirFunctionType.getInput(0)));
    assert(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirFunctionType.getInput(1)));
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirFunctionType.getInput(2)));
    assert(mlir::isa<mlir::rvsdg::IOStateEdgeType>(mlirFunctionType.getResult(0)));
    assert(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirFunctionType.getResult(1)));
    assert(linkage == "external_linkage");
    assert(name == "test");

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 0);

      assert(region->graph()->GetRootRegion().narguments() == 1);
      auto arg = region->graph()->GetRootRegion().argument(0);
      auto imp = dynamic_cast<jlm::llvm::GraphImport *>(arg);
      assert(imp);
      assert(imp->Name() == "test");
      assert(imp->Linkage() == linkage::external_linkage);
      assert(*imp->ValueType() == *functionType);
      assert(*imp->ImportedType() == *functionType);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFunctionGraphImportGen", TestFunctionGraphImport)

static void
TestPointerGraphImport()
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
        linkage::external_linkage);

    // Convert the RVSDG to MLIR
    std::cout << "Convert to MLIR" << std::endl;
    jlm::mlir::JlmToMlirConverter mlirgen;
    auto omega = mlirgen.ConvertModule(*rvsdgModule);

    // Validate the generated MLIR
    std::cout << "Validate MLIR" << std::endl;
    auto & omegaRegion = omega.getRegion();
    auto & omegaBlock = omegaRegion.front();
    auto & mlirOp = omegaBlock.front();

    assert(mlir::isa<mlir::rvsdg::OmegaArgument>(mlirOp));

    auto mlirOmegaArgument = mlir::cast<mlir::rvsdg::OmegaArgument>(mlirOp);

    auto valueType = mlirOmegaArgument.getValueType();
    auto importedValueType = mlirOmegaArgument.getImportedValue().getType();
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(importedValueType));

    auto mlirIntType = valueType.dyn_cast<mlir::IntegerType>();
    assert(mlirIntType);
    assert(mlirIntType.getWidth() == 32);
    assert(linkage == "external_linkage");
    assert(name == "test");

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 0);

      assert(region->graph()->GetRootRegion().narguments() == 1);
      auto arg = region->graph()->GetRootRegion().argument(0);
      auto imp = dynamic_cast<jlm::llvm::GraphImport *>(arg);
      assert(imp);
      assert(imp->Name() == "test");
      assert(imp->Linkage() == linkage::external_linkage);
      assert(*imp->ValueType() == *jlm::rvsdg::BitType::Create(32));
      assert(*imp->ImportedType() == *PointerType::Create());
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirPointerGraphImportGen", TestPointerGraphImport)

// Add IOBarrier test near the end of the file, before the last test registrations
static void
TestIOBarrier()
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
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto ioStateArgument = lambda->GetFunctionArguments()[0];

    // Create a value to pass through the barrier
    auto value = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 42);

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
    assert(omegaRegion.getBlocks().size() == 1);
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
        assert(ioBarrier->getNumOperands() == 2);

        // Check that the first operand is a 32-bit integer
        auto valueType = ioBarrier->getOperand(0).getType().dyn_cast<mlir::IntegerType>();
        assert(valueType);
        assert(valueType.getWidth() == 32);
        assert(mlir::isa<mlir::rvsdg::IOStateEdgeType>(ioBarrier->getOperand(1).getType()));

        // Check that the result type matches the input value type
        auto resultType = ioBarrier->getResult(0).getType().dyn_cast<mlir::IntegerType>();
        assert(resultType);
        assert(resultType.getWidth() == 32);
      }
    }
    assert(foundIOBarrier);

    // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto convertedRvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &convertedRvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      // Direct access to the lambda node
      assert(region->numNodes() == 1);
      auto & lambdaNode = *region->Nodes().begin();
      auto lambdaOperation = dynamic_cast<const jlm::rvsdg::LambdaNode *>(&lambdaNode);
      assert(lambdaOperation);

      // Find the IOBarrier in the lambda subregion
      bool foundIOBarrier = false;
      for (auto & lambdaNode : lambdaOperation->subregion()->Nodes())
      {
        auto ioBarrierOp = dynamic_cast<const IOBarrierOperation *>(&lambdaNode.GetOperation());
        if (ioBarrierOp)
        {
          foundIOBarrier = true;

          // Check that it has correct number of inputs and outputs
          assert(ioBarrierOp->nresults() == 1);
          assert(ioBarrierOp->narguments() == 2);

          // Check that the first input is the 32-bit value
          auto valueType =
              dynamic_cast<const jlm::rvsdg::BitType *>(ioBarrierOp->argument(0).get());
          assert(valueType);
          assert(valueType->nbits() == 32);

          // Check that the second input is an IO state
          auto ioStateType = dynamic_cast<const IOStateType *>(ioBarrierOp->argument(1).get());
          assert(ioStateType);

          // Check that the output type matches the input value type
          auto outputType = dynamic_cast<const jlm::rvsdg::BitType *>(ioBarrierOp->result(0).get());
          assert(outputType);
          assert(outputType->nbits() == 32);
        }
      }
      assert(foundIOBarrier);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirIOBarrierGen", TestIOBarrier)

static void
TestMalloc()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto constOp = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 64, 2);
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
        assert(inputBitType);
        assert(inputBitType.getWidth() == 64);
        assert(mlir::isa<mlir::LLVM::LLVMPointerType>(mlirMallocOp.getResult(0).getType()));
        assert(mlir::isa<mlir::rvsdg::MemStateEdgeType>(mlirMallocOp.getResult(1).getType()));
        foundMallocOp = true;
      }
    }
    assert(foundMallocOp);

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->numNodes() == 2);
      bool foundMallocOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedMallocOp = dynamic_cast<const MallocOperation *>(&node.GetOperation());
        if (convertedMallocOp)
        {
          assert(convertedMallocOp->nresults() == 2);
          assert(convertedMallocOp->narguments() == 1);
          auto inputBitType = jlm::util::AssertedCast<const jlm::rvsdg::BitType>(
              convertedMallocOp->argument(0).get());
          assert(inputBitType->nbits() == 64);
          assert(jlm::rvsdg::is<jlm::llvm::PointerType>(convertedMallocOp->result(0)));
          assert(jlm::rvsdg::is<jlm::llvm::MemoryStateType>(convertedMallocOp->result(1)));
          foundMallocOp = true;
        }
      }
      assert(foundMallocOp);
    }
  }
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirMallocGen", TestMalloc)
