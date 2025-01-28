/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

static int
MemCpyConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::bittype::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        jlm::rvsdg::bittype::Create(64),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto destinationArgument =
      cfg->entry()->append_argument(argument::create("destination", pointerType));
  auto sourceArgument = cfg->entry()->append_argument(argument::create("source", pointerType));
  auto lengthArgument = cfg->entry()->append_argument(argument::create("length", bit64Type));
  auto memoryStateArgument =
      cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  auto memCpyTac = basicBlock->append_last(MemCpyNonVolatileOperation::create(
      destinationArgument,
      sourceArgument,
      lengthArgument,
      { memoryStateArgument }));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(memCpyTac->result(0));

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

    auto memCpyInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    assert(memCpyInstruction != nullptr);
    assert(memCpyInstruction->getIntrinsicID() == ::llvm::Intrinsic::memcpy);
    assert(memCpyInstruction->isVolatile() == false);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/backend/llvm/jlm-llvm/MemCpyTests-MemCpyConversion",
    MemCpyConversion)

static int
MemCpyVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::bittype::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        jlm::rvsdg::bittype::Create(64),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto & destinationArgument =
      *cfg->entry()->append_argument(argument::create("destination", pointerType));
  auto & sourceArgument = *cfg->entry()->append_argument(argument::create("source", pointerType));
  auto & lengthArgument = *cfg->entry()->append_argument(argument::create("length", bit64Type));
  auto & ioStateArgument = *cfg->entry()->append_argument(argument::create("ioState", ioStateType));
  auto & memoryStateArgument =
      *cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  auto memCpyTac = basicBlock->append_last(MemCpyVolatileOperation::CreateThreeAddressCode(
      destinationArgument,
      sourceArgument,
      lengthArgument,
      ioStateArgument,
      { &memoryStateArgument }));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(memCpyTac->result(0));
  cfg->exit()->append_result(memCpyTac->result(1));

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

    auto memCpyInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    assert(memCpyInstruction != nullptr);
    assert(memCpyInstruction->getIntrinsicID() == ::llvm::Intrinsic::memcpy);
    assert(memCpyInstruction->isVolatile() == true);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/backend/llvm/jlm-llvm/MemCpyTests-MemCpyVolatileConversion",
    MemCpyVolatileConversion)
