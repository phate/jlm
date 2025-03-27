/*
 * Copyright 2024 Halvor Linder Henriksen <halvorlinder@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
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

static int
TestGetElementPtr()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto pointerType = PointerType::Create();
    auto bitType = jlm::rvsdg::bittype::Create(32);

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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(convertedLambda->subregion()->nnodes() == 1);

      auto op = convertedLambda->subregion()->Nodes().begin();
      assert(is<GetElementPtrOperation>(op->GetOperation()));
      auto convertedGep = dynamic_cast<const GetElementPtrOperation *>(&op->GetOperation());

      assert(is<ArrayType>(convertedGep->GetPointeeType()));
      assert(is<PointerType>(convertedGep->result(0)));
      assert(is<jlm::rvsdg::bittype>(convertedGep->argument(1)));
      assert(is<jlm::rvsdg::bittype>(convertedGep->argument(2)));
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirGetElementPtrGen", TestGetElementPtr)

static int
TestDelta()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();
  {
    auto bitType = jlm::rvsdg::bittype::Create(32);

    auto delta1 = delta::node::Create(
        &graph->GetRootRegion(),
        bitType,
        "non-constant-delta",
        linkage::external_linkage,
        "section",
        false);

    auto bitConstant = jlm::rvsdg::create_bitconstant(delta1->subregion(), 32, 1);
    delta1->finalize(bitConstant);

    auto delta2 = delta::node::Create(
        &graph->GetRootRegion(),
        bitType,
        "constant-delta",
        linkage::external_linkage,
        "section",
        true);
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

      assert(region->nnodes() == 2);
      for (auto & node : region->Nodes())
      {
        auto convertedDelta = jlm::util::AssertedCast<delta::node>(&node);
        assert(convertedDelta->subregion()->nnodes() == 1);

        if (convertedDelta->constant())
        {
          assert(convertedDelta->name() == "constant-delta");
        }
        else
        {
          assert(convertedDelta->name() == "non-constant-delta");
        }

        assert(is<jlm::rvsdg::bittype>(convertedDelta->type()));
        assert(convertedDelta->linkage() == linkage::external_linkage);
        assert(convertedDelta->Section() == "section");

        auto op = convertedDelta->subregion()->Nodes().begin();
        assert(is<jlm::rvsdg::bitconstant_op>(op->GetOperation()));
      }
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirDeltaGen", TestDelta)

static int
TestConstantDataArray()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitConstant1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);
    auto bitConstant2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 2);
    auto bitType = jlm::rvsdg::bittype::Create(32);
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

      assert(region->nnodes() == 3);
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
          assert(is<jlm::rvsdg::bittype>(arrayType->element_type()));
          assert(arrayType->nelements() == 2);
          assert(is<jlm::rvsdg::bittype>(constantDataArray->argument(0)));
          assert(is<jlm::rvsdg::bittype>(constantDataArray->argument(1)));
        }
      }
      assert(foundConstantDataArray);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantDataArrayGen", TestConstantDataArray)

static int
TestConstantAggregateZero()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType = jlm::rvsdg::bittype::Create(32);
    auto arrayType = jlm::llvm::ArrayType::Create(bitType, 2);
    jlm::llvm::ConstantAggregateZero::Create(graph->GetRootRegion(), arrayType);

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

      assert(region->nnodes() == 1);
      auto const convertedConstantAggregateZero =
          jlm::util::AssertedCast<const ConstantAggregateZero>(
              &region->Nodes().begin().ptr()->GetOperation());
      assert(convertedConstantAggregateZero->nresults() == 1);
      assert(convertedConstantAggregateZero->narguments() == 0);
      auto resultType = convertedConstantAggregateZero->result(0);
      auto arrayType = dynamic_cast<const jlm::llvm::ArrayType *>(resultType.get());
      assert(arrayType);
      assert(is<jlm::rvsdg::bittype>(arrayType->element_type()));
      assert(arrayType->nelements() == 2);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirConstantAggregateZeroGen", TestConstantAggregateZero)

static int
TestVarArgList()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType = jlm::rvsdg::bittype::Create(32);
    auto bits1 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 1);
    auto bits2 = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 32, 2);
    jlm::llvm::valist_op::Create(graph->GetRootRegion(), { bits1, bits2 });

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

      assert(region->nnodes() == 3);
      bool foundVarArgOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedVarArgOp = dynamic_cast<const valist_op *>(&node.GetOperation());
        if (convertedVarArgOp)
        {
          assert(convertedVarArgOp->nresults() == 1);
          assert(convertedVarArgOp->narguments() == 2);
          auto resultType = convertedVarArgOp->result(0);
          assert(is<jlm::llvm::VariableArgumentType>(resultType));
          assert(is<jlm::rvsdg::bittype>(convertedVarArgOp->argument(0)));
          assert(is<jlm::rvsdg::bittype>(convertedVarArgOp->argument(1)));
          foundVarArgOp = true;
        }
      }
      assert(foundVarArgOp);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirVarArgListGen", TestVarArgList)

static int
TestFNeg()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto floatType = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::flt);
    auto constOp = ConstantFP(floatType, ::llvm::APFloat(2.0));
    auto & constNode = jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), constOp, {});
    auto fnegOp = FNegOperation(jlm::llvm::fpsize::flt);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), fnegOp, { constNode.output(0) });

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

      assert(region->nnodes() == 2);
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
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFNegGen", TestFNeg)

static int
TestFPExt()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto floatType1 = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::flt);
    auto floatType2 = jlm::llvm::FloatingPointType::Create(jlm::llvm::fpsize::dbl);
    auto constOp = ConstantFP(floatType1, ::llvm::APFloat(2.0));
    auto & constNode = jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), constOp, {});
    auto fpextOp = FPExtOperation(floatType1, floatType2);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), fpextOp, { constNode.output(0) });

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

      assert(region->nnodes() == 2);
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
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFPExtGen", TestFPExt)

static int
TestTrunc()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    auto bitType1 = jlm::rvsdg::bittype::Create(64);
    auto bitType2 = jlm::rvsdg::bittype::Create(32);
    auto constOp = jlm::rvsdg::create_bitconstant(&graph->GetRootRegion(), 64, 2);
    auto truncOp = TruncOperation(bitType1, bitType2);
    jlm::rvsdg::SimpleNode::Create(graph->GetRootRegion(), truncOp, { constOp });

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

      assert(region->nnodes() == 2);
      bool foundTruncOp = false;
      for (auto & node : region->Nodes())
      {
        auto convertedTruncOp = dynamic_cast<const TruncOperation *>(&node.GetOperation());
        if (convertedTruncOp)
        {
          assert(convertedTruncOp->nresults() == 1);
          assert(convertedTruncOp->narguments() == 1);
          auto inputBitType = jlm::util::AssertedCast<const jlm::rvsdg::bittype>(
              convertedTruncOp->argument(0).get());
          assert(inputBitType->nbits() == 64);
          auto outputBitType =
              jlm::util::AssertedCast<const jlm::rvsdg::bittype>(convertedTruncOp->result(0).get());
          assert(outputBitType->nbits() == 32);
          foundTruncOp = true;
        }
      }
      assert(foundTruncOp);
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirTruncGen", TestTrunc)

static int
TestFree()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
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

      assert(region->nnodes() == 1);
      auto convertedLambda =
          jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
      assert(is<jlm::rvsdg::LambdaOperation>(convertedLambda));

      assert(convertedLambda->subregion()->nnodes() == 1);
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
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFreeGen", TestFree)

static int
TestFunctionGraphImport()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
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
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

    auto mlirFunctionType = valueType.dyn_cast<mlir::FunctionType>();
    assert(mlirFunctionType);
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

      assert(region->nnodes() == 0);

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
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirFunctionGraphImportGen", TestFunctionGraphImport)

static int
TestPointerGraphImport()
{
  using namespace jlm::llvm;
  using namespace mlir::rvsdg;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &rvsdgModule->Rvsdg();

  {
    jlm::llvm::GraphImport::Create(
        *graph,
        jlm::rvsdg::bittype::Create(32),
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
    auto linkage = mlirOmegaArgument.getLinkage();
    auto name = mlirOmegaArgument.getName();

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

      assert(region->nnodes() == 0);

      assert(region->graph()->GetRootRegion().narguments() == 1);
      auto arg = region->graph()->GetRootRegion().argument(0);
      auto imp = dynamic_cast<jlm::llvm::GraphImport *>(arg);
      assert(imp);
      assert(imp->Name() == "test");
      assert(imp->Linkage() == linkage::external_linkage);
      assert(*imp->ValueType() == *jlm::rvsdg::bittype::Create(32));
      assert(*imp->ImportedType() == *PointerType::Create());
    }
  }
  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/mlir/TestMlirPointerGraphImportGen", TestPointerGraphImport)
