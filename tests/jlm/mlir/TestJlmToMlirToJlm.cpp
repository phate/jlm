/*
 * Copyright 2024 Halvor Linder Henriksen <halvorlinder@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/rvsdg/nullary.hpp>

static int
TestUndef()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create an undef operation
    std::cout << "Undef Operation" << std::endl;
    UndefValueOperation::Create(graph->GetRootRegion(), jlm::rvsdg::bittype::Create(32));

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

      assert(region->nnodes() == 1);

      // Get the undef op
      auto convertedUndef =
          dynamic_cast<const UndefValueOperation *>(&region->Nodes().begin()->GetOperation());

      assert(convertedUndef != nullptr);

      auto outputType = convertedUndef->result(0);
      assert(jlm::rvsdg::is<const jlm::rvsdg::bittype>(outputType));
      assert(std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(outputType)->nbits() == 32);
    }
  }
  return 0;
}

static int
TestAlloca()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    // Create a bits node for alloc size
    std::cout << "Bit Constanr" << std::endl;
    auto bits = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);

    // Create alloca node
    std::cout << "Alloca Operation" << std::endl;
    auto allocaOp = alloca_op(jlm::rvsdg::bittype::Create(64), jlm::rvsdg::bittype::Create(32), 4);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), allocaOp, { bits });

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
    if (!foundAlloca)
    {
      return 1;
    }

    // // Convert the MLIR to RVSDG and check the result
    std::cout << "Converting MLIR to RVSDG" << std::endl;
    std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
    rootBlock->push_back(omega);
    auto rvsdgModule = jlm::mlir::MlirToJlmConverter::CreateAndConvert(rootBlock);
    auto region = &rvsdgModule->Rvsdg().GetRootRegion();

    {
      using namespace jlm::llvm;

      assert(region->nnodes() == 2);

      bool foundAlloca = false;
      for (auto & node : region->Nodes())
      {
        if (auto allocaOp = dynamic_cast<const alloca_op *>(&node.GetOperation()))
        {
          assert(allocaOp->alignment() == 4);

          assert(jlm::rvsdg::is<jlm::rvsdg::bittype>(allocaOp->ValueType()));
          auto valueBitType =
              dynamic_cast<const jlm::rvsdg::bittype *>(allocaOp->ValueType().get());
          assert(valueBitType->nbits() == 64);

          assert(allocaOp->narguments() == 1);

          assert(jlm::rvsdg::is<jlm::rvsdg::bittype>(allocaOp->argument(0)));
          auto inputBitType =
              dynamic_cast<const jlm::rvsdg::bittype *>(allocaOp->argument(0).get());
          assert(inputBitType->nbits() == 32);

          assert(allocaOp->nresults() == 2);

          assert(jlm::rvsdg::is<PointerType>(allocaOp->result(0)));
          assert(jlm::rvsdg::is<jlm::llvm::MemoryStateType>(allocaOp->result(1)));

          foundAlloca = true;
        }
      }
      if (!foundAlloca)
      {
        return 1;
      }
    }
  }
  return 0;
}

static int
TestLoad()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
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
    auto loadType = jlm::rvsdg::bittype::Create(32);
    auto loadOp = jlm::llvm::LoadNonVolatileOperation(loadType, 1, 4);
    auto & subregion = *(lambda->subregion());
    jlm::llvm::LoadNonVolatileNode::Create(
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

    if (!mlir::isa<mlir::jlm::Load>(mlirOp))
      return 1;

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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<LoadNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedLoad = dynamic_cast<const LoadNonVolatileNode *>(
          convertedLambda->subregion()->Nodes().begin().ptr());

      assert(convertedLoad->GetAlignment() == 4);
      assert(convertedLoad->NumMemoryStates() == 1);

      assert(is<jlm::llvm::PointerType>(convertedLoad->input(0)->type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedLoad->input(1)->type()));

      assert(is<jlm::rvsdg::bittype>(convertedLoad->output(0)->type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedLoad->output(1)->type()));

      auto outputBitType =
          dynamic_cast<const jlm::rvsdg::bittype *>(&convertedLoad->output(0)->type());
      assert(outputBitType->nbits() == 32);
    }
  }
  return 0;
}

static int
TestStore()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitsType = jlm::rvsdg::bittype::Create(32);
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
    jlm::llvm::StoreNonVolatileNode::Create(
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

    if (!mlir::isa<mlir::jlm::Store>(mlirOp))
      return 1;

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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<StoreNonVolatileOperation>(
          convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedStore = dynamic_cast<const StoreNonVolatileNode *>(
          convertedLambda->subregion()->Nodes().begin().ptr());

      assert(convertedStore->GetAlignment() == 4);
      assert(convertedStore->NumMemoryStates() == 1);

      assert(is<jlm::llvm::PointerType>(convertedStore->input(0)->type()));
      assert(is<jlm::rvsdg::bittype>(convertedStore->input(1)->type()));
      assert(is<jlm::llvm::MemoryStateType>(convertedStore->input(2)->type()));

      assert(is<jlm::llvm::MemoryStateType>(convertedStore->output(0)->type()));

      auto inputBitType =
          dynamic_cast<const jlm::rvsdg::bittype *>(&convertedStore->input(1)->type());
      assert(inputBitType->nbits() == 32);
    }
  }
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirUndefGen", TestUndef)
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirAllocaGen", TestAlloca)
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirLoadGen", TestLoad)
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirStoreGen", TestStore)
