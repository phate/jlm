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
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/traverser.hpp>

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
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirUndefGen", TestUndef)

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
    assert(foundAlloca);

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
      assert(foundAlloca);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirAllocaGen", TestAlloca)

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
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirLoadGen", TestLoad)

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
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirStoreGen", TestStore)

static int
TestSext()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {

    auto bitsType = jlm::rvsdg::bittype::Create(32);
    auto functionType = jlm::rvsdg::FunctionType::Create({ bitsType }, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto bitsArgument = lambda->GetFunctionArguments().at(0);

    // Create sext operation
    auto sextOp = jlm::llvm::sext_op::create((size_t)64, bitsArgument);
    auto node = jlm::rvsdg::output::GetNode(*sextOp);
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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<sext_op>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSext = dynamic_cast<const sext_op *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(convertedSext->ndstbits() == 64);
      assert(convertedSext->nsrcbits() == 32);
      assert(convertedSext->nresults() == 1);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirSextGen", TestSext)

static int
TestSitofp()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {

    auto bitsType = jlm::rvsdg::bittype::Create(32);
    auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
    auto functionType = jlm::rvsdg::FunctionType::Create({ bitsType }, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto bitsArgument = lambda->GetFunctionArguments().at(0);

    // Create sitofp operation
    auto sitofpOp = jlm::llvm::sitofp_op(bitsType, floatType);
    jlm::rvsdg::SimpleNode::Create(*lambda->subregion(), sitofpOp, { bitsArgument });

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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<sitofp_op>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedSitofp = dynamic_cast<const sitofp_op *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(jlm::rvsdg::is<jlm::rvsdg::bittype>(*convertedSitofp->argument(0).get()));
      assert(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedSitofp->result(0).get()));
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirSitofpGen", TestSitofp)

static int
TestConstantFP()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto functionType = jlm::rvsdg::FunctionType::Create({}, {});
    auto lambda = jlm::rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

    // Create sitofp operation
    auto constOp = ConstantFP(fpsize::dbl, ::llvm::APFloat(2.0));
    jlm::rvsdg::SimpleNode::Create(*lambda->subregion(), constOp, {});

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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->nnodes() == 1);
      assert(is<ConstantFP>(convertedLambda->subregion()->Nodes().begin()->GetOperation()));
      auto convertedConst = dynamic_cast<const ConstantFP *>(
          &convertedLambda->subregion()->Nodes().begin()->GetOperation());

      assert(jlm::rvsdg::is<jlm::llvm::FloatingPointType>(*convertedConst->result(0).get()));
      assert(convertedConst->constant().isExactlyValue(2.0));
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantFPGen", TestConstantFP)

static int
TestFpBinary()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;
  auto binOps = std::vector<fpop>{ fpop::add, fpop::sub, fpop::mul, fpop::div, fpop::mod };
  for (auto binOp : binOps)
  {
    auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
    auto graph = &rvsdgModule->Rvsdg();
    {
      auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
      auto functionType = jlm::rvsdg::FunctionType::Create({ floatType, floatType }, {});
      auto lambda = jlm::rvsdg::LambdaNode::Create(
          graph->GetRootRegion(),
          LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

      auto floatArgument1 = lambda->GetFunctionArguments().at(0);
      auto floatArgument2 = lambda->GetFunctionArguments().at(1);

      jlm::rvsdg::SimpleNode::Create(
          *lambda->subregion(),
          fpbin_op(binOp, floatType),
          { floatArgument1, floatArgument2 });

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

        assert(region->nnodes() == 1);
        auto convertedLambda =
            jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
        assert(convertedLambda->subregion()->nnodes() == 1);

        auto node = convertedLambda->subregion()->Nodes().begin().ptr();
        auto convertedFpbin = jlm::util::AssertedCast<const fpbin_op>(&node->GetOperation());
        assert(convertedFpbin->fpop() == binOp);
        assert(convertedFpbin->nresults() == 1);
        assert(convertedFpbin->narguments() == 2);
      }
    }
  }

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFpBinaryGen", TestFpBinary)
