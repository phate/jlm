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

static int
LoadConversion()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType;
  MemoryStateType memoryStateType;
  jlm::rvsdg::bittype bit64Type(64);
  FunctionType functionType({ &pointerType, &memoryStateType }, { &bit64Type, &memoryStateType });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(argument::create("address", pointerType));
  auto memoryStateArgument =
      cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  size_t alignment = 4;
  auto loadTac = basicBlock->append_last(
      LoadNonVolatileOperation::Create(addressArgument, memoryStateArgument, bit64Type, alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(loadTac->result(0));
  cfg->exit()->append_result(loadTac->result(1));

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

    auto loadInstruction = ::llvm::dyn_cast<::llvm::LoadInst>(&instruction);
    assert(loadInstruction != nullptr);
    assert(loadInstruction->isVolatile() == false);
    assert(loadInstruction->getAlign().value() == alignment);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/LoadTests-LoadConversion", LoadConversion)

static int
LoadVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  PointerType pointerType;
  iostatetype ioStateType;
  MemoryStateType memoryStateType;
  jlm::rvsdg::bittype bit64Type(64);
  FunctionType functionType(
      { &pointerType, &ioStateType, &memoryStateType },
      { &bit64Type, &ioStateType, &memoryStateType });

  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto cfg = cfg::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(argument::create("address", pointerType));
  auto ioStateArgument = cfg->entry()->append_argument(argument::create("ioState", ioStateType));
  auto memoryStateArgument =
      cfg->entry()->append_argument(argument::create("memoryState", memoryStateType));

  auto basicBlock = basic_block::create(*cfg);
  size_t alignment = 4;
  auto loadTac = basicBlock->append_last(LoadVolatileOperation::Create(
      addressArgument,
      ioStateArgument,
      memoryStateArgument,
      bit64Type,
      alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(loadTac->result(0));
  cfg->exit()->append_result(loadTac->result(1));
  cfg->exit()->append_result(loadTac->result(2));

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

    auto loadInstruction = ::llvm::dyn_cast<::llvm::LoadInst>(&instruction);
    assert(loadInstruction != nullptr);
    assert(loadInstruction->isVolatile() == true);
    assert(loadInstruction->getAlign().value() == alignment);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/backend/llvm/jlm-llvm/LoadTests-LoadVolatileConversion",
    LoadVolatileConversion)
