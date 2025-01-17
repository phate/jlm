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
