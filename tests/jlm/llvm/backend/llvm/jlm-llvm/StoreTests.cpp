/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>
#include <test-util.hpp>

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <llvm/IR/Instructions.h>

static int
StoreConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::bittype::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), jlm::rvsdg::bittype::Create(64), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(argument::create("address", pointerType));
  auto valueArgument = cfg->entry()->append_argument(argument::create("value", bit64Type));
  auto memoryStateArgument =
      cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  size_t alignment = 4;
  auto storeTac = basicBlock->append_last(StoreNonVolatileOperation::Create(
      addressArgument,
      valueArgument,
      memoryStateArgument,
      alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(storeTac->result(0));

  auto f = function_node::create(ipgModule.ipgraph(), "f", functionType, linkage::external_linkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = jlm2llvm::convert(ipgModule, ctx);
  jlm::tests::print(*llvmModule);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto storeInstruction = ::llvm::dyn_cast<::llvm::StoreInst>(&instruction);
    assert(storeInstruction != nullptr);
    assert(storeInstruction->isVolatile() == false);
    assert(storeInstruction->getAlign().value() == alignment);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/StoreTests-StoreConversion", StoreConversion)

static int
StoreVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::bittype::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        jlm::rvsdg::bittype::Create(64),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(argument::create("address", pointerType));
  auto valueArgument = cfg->entry()->append_argument(argument::create("value", bit64Type));
  auto ioStateArgument = cfg->entry()->append_argument(argument::create("ioState", ioStateType));
  auto memoryStateArgument =
      cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  size_t alignment = 4;
  auto storeTac = basicBlock->append_last(StoreVolatileOperation::Create(
      addressArgument,
      valueArgument,
      ioStateArgument,
      memoryStateArgument,
      alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(storeTac->result(0));
  cfg->exit()->append_result(storeTac->result(1));

  auto f = function_node::create(ipgModule.ipgraph(), "f", functionType, linkage::external_linkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = jlm2llvm::convert(ipgModule, ctx);
  jlm::tests::print(*llvmModule);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto storeInstruction = ::llvm::dyn_cast<::llvm::StoreInst>(&instruction);
    assert(storeInstruction != nullptr);
    assert(storeInstruction->isVolatile() == true);
    assert(storeInstruction->getAlign().value() == alignment);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/backend/llvm/jlm-llvm/StoreTests-StoreVolatileConversion",
    StoreVolatileConversion)
