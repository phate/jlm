/*
 * Copyright 2024, 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

TEST(IpGraphToLlvmConverterTests, LoadConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::BitType::Create(64), MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto addressArgument =
      cfg->entry()->append_argument(Argument::create("address", PointerType::Create()));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", MemoryStateType::Create()));

  auto basicBlock = BasicBlock::create(*cfg);
  size_t alignment = 4;
  auto loadTac = basicBlock->append_last(LoadNonVolatileOperation::Create(
      addressArgument,
      memoryStateArgument,
      jlm::rvsdg::BitType::Create(64),
      alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(loadTac->result(0));
  cfg->exit()->append_result(loadTac->result(1));

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto loadInstruction = ::llvm::dyn_cast<::llvm::LoadInst>(&instruction);
    EXPECT_NE(loadInstruction, nullptr);
    EXPECT_FALSE(loadInstruction->isVolatile());
    EXPECT_EQ(loadInstruction->getAlign().value(), alignment);
  }
}

TEST(IpGraphToLlvmConverterTests, LoadVolatileConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::BitType::Create(64), IOStateType::Create(), MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(Argument::create("address", pointerType));
  auto ioStateArgument = cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
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

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto loadInstruction = ::llvm::dyn_cast<::llvm::LoadInst>(&instruction);
    EXPECT_NE(loadInstruction, nullptr);
    EXPECT_TRUE(loadInstruction->isVolatile());
    EXPECT_EQ(loadInstruction->getAlign().value(), alignment);
  }
}

TEST(IpGraphToLlvmConverterTests, MemCpyConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        jlm::rvsdg::BitType::Create(64),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto destinationArgument =
      cfg->entry()->append_argument(Argument::create("destination", pointerType));
  auto sourceArgument = cfg->entry()->append_argument(Argument::create("source", pointerType));
  auto lengthArgument = cfg->entry()->append_argument(Argument::create("length", bit64Type));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
  auto memCpyTac = basicBlock->append_last(MemCpyNonVolatileOperation::create(
      destinationArgument,
      sourceArgument,
      lengthArgument,
      { memoryStateArgument }));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(memCpyTac->result(0));

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto memCpyInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    EXPECT_NE(memCpyInstruction, nullptr);
    EXPECT_EQ(memCpyInstruction->getIntrinsicID(), ::llvm::Intrinsic::memcpy);
    EXPECT_FALSE(memCpyInstruction->isVolatile());
  }
}

TEST(IpGraphToLlvmConverterTests, MemCpyVolatileConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        jlm::rvsdg::BitType::Create(64),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto & destinationArgument =
      *cfg->entry()->append_argument(Argument::create("destination", pointerType));
  auto & sourceArgument = *cfg->entry()->append_argument(Argument::create("source", pointerType));
  auto & lengthArgument = *cfg->entry()->append_argument(Argument::create("length", bit64Type));
  auto & ioStateArgument = *cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
  auto & memoryStateArgument =
      *cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
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

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto memCpyInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    EXPECT_NE(memCpyInstruction, nullptr);
    EXPECT_EQ(memCpyInstruction->getIntrinsicID(), ::llvm::Intrinsic::memcpy);
    EXPECT_TRUE(memCpyInstruction->isVolatile());
  }
}

TEST(IpGraphToLlvmConverterTests, MemSetConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit8Type = jlm::rvsdg::BitType::Create(8);
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { pointerType, bit8Type, bit64Type, memoryStateType },
      { memoryStateType });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto destinationArgument =
      cfg->entry()->append_argument(Argument::create("destination", pointerType));
  auto valueArgument = cfg->entry()->append_argument(Argument::create("value", bit8Type));
  auto lengthArgument = cfg->entry()->append_argument(Argument::create("length", bit64Type));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
  auto memsetTac = basicBlock->append_last(MemSetNonVolatileOperation::createTac(
      *destinationArgument,
      *valueArgument,
      *lengthArgument,
      { memoryStateArgument }));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(memsetTac->result(0));

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto memsetInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    EXPECT_NE(memsetInstruction, nullptr);
    EXPECT_EQ(memsetInstruction->getIntrinsicID(), ::llvm::Intrinsic::memset);
    EXPECT_FALSE(memsetInstruction->isVolatile());
  }
}

TEST(IpGraphToLlvmConverterTests, StoreConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(), jlm::rvsdg::BitType::Create(64), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(Argument::create("address", pointerType));
  auto valueArgument = cfg->entry()->append_argument(Argument::create("value", bit64Type));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
  size_t alignment = 4;
  auto storeTac = basicBlock->append_last(StoreNonVolatileOperation::Create(
      addressArgument,
      valueArgument,
      memoryStateArgument,
      alignment));

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());
  cfg->exit()->append_result(storeTac->result(0));

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto storeInstruction = ::llvm::dyn_cast<::llvm::StoreInst>(&instruction);
    EXPECT_NE(storeInstruction, nullptr);
    EXPECT_FALSE(storeInstruction->isVolatile());
    EXPECT_EQ(storeInstruction->getAlign().value(), alignment);
  }
}

TEST(IpGraphToLlvmConverterTests, StoreVolatileConversion)
{
  using namespace jlm::llvm;

  // Arrange
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { PointerType::Create(),
        jlm::rvsdg::BitType::Create(64),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(ipgModule);
  auto addressArgument = cfg->entry()->append_argument(Argument::create("address", pointerType));
  auto valueArgument = cfg->entry()->append_argument(Argument::create("value", bit64Type));
  auto ioStateArgument = cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
  auto memoryStateArgument =
      cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

  auto basicBlock = BasicBlock::create(*cfg);
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

  auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    auto storeInstruction = ::llvm::dyn_cast<::llvm::StoreInst>(&instruction);
    EXPECT_NE(storeInstruction, nullptr);
    EXPECT_TRUE(storeInstruction->isVolatile());
    EXPECT_EQ(storeInstruction->getAlign().value(), alignment);
  }
}

TEST(IpGraphToLlvmConverterTests, FMulAddConversion)
{
  using namespace jlm::llvm;

  // Arrange
  const auto pointerType = PointerType::Create();
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  auto doubleType = FloatingPointType::Create(fpsize::dbl);
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { doubleType, doubleType, doubleType, IOStateType::Create(), MemoryStateType::Create() },
      { doubleType, IOStateType::Create(), MemoryStateType::Create() });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");
  {
    auto cfg = ControlFlowGraph::create(ipgModule);
    auto & multiplierArgument =
        *cfg->entry()->append_argument(Argument::create("multiplier", doubleType));
    auto & multiplicandArgument =
        *cfg->entry()->append_argument(Argument::create("multiplicand", doubleType));
    auto & summandArgument =
        *cfg->entry()->append_argument(Argument::create("summand", doubleType));
    const auto ioStateArgument =
        cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
    const auto memoryStateArgument =
        cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

    auto basicBlock = BasicBlock::create(*cfg);
    auto fMulAddTac = basicBlock->append_last(FMulAddIntrinsicOperation::CreateTac(
        multiplierArgument,
        multiplicandArgument,
        summandArgument));

    cfg->exit()->divert_inedges(basicBlock);
    basicBlock->add_outedge(cfg->exit());
    cfg->exit()->append_result(fMulAddTac->result(0));
    cfg->exit()->append_result(ioStateArgument);
    cfg->exit()->append_result(memoryStateArgument);

    auto f = FunctionNode::create(ipgModule.ipgraph(), "f", functionType, Linkage::externalLinkage);
    f->add_cfg(std::move(cfg));
  }

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  const auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  // Assert
  {
    const auto llvmFunction = llvmModule->getFunction("f");
    auto & basicBlock = llvmFunction->back();
    auto & instruction = basicBlock.front();

    const auto fMulAddInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
    EXPECT_NE(fMulAddInstruction, nullptr);
    EXPECT_EQ(fMulAddInstruction->getIntrinsicID(), ::llvm::Intrinsic::fmuladd);
  }
}

TEST(IpGraphToLlvmConverterTests, IntegerConstant)
{
  const char * bs = "0100000000"
                    "0000000000"
                    "0000000000"
                    "0000000000"
                    "0000000000"
                    "0000000000"
                    "00001";

  using namespace jlm::llvm;

  auto ft = jlm::rvsdg::FunctionType::Create({}, { jlm::rvsdg::BitType::Create(65) });

  jlm::rvsdg::BitValueRepresentation vr(bs);

  InterProceduralGraphModule im(jlm::util::FilePath(""), "", "");

  auto cfg = ControlFlowGraph::create(im);
  auto bb = BasicBlock::create(*cfg);
  bb->append_last(ThreeAddressCode::create(std::make_unique<IntegerConstantOperation>(vr), {}));
  auto c = bb->last()->result(0);

  cfg->exit()->divert_inedges(bb);
  bb->add_outedge(cfg->exit());
  cfg->exit()->append_result(c);

  auto f = FunctionNode::create(im.ipgraph(), "f", ft, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(im, stdout);

  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(im, ctx);

  llvmModule->print(llvm::errs(), nullptr);
}

TEST(IpGraphToLlvmConverterTests, Malloc)
{
  auto setup = []()
  {
    using namespace jlm::llvm;

    auto memoryStateType = MemoryStateType::Create();
    auto pointerType = PointerType::Create();
    auto ioStateType = IOStateType::Create();
    auto im = InterProceduralGraphModule::create(jlm::util::FilePath(""), "", "");

    auto cfg = ControlFlowGraph::create(*im);
    auto bb = BasicBlock::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    auto size =
        cfg->entry()->append_argument(Argument::create("size", jlm::rvsdg::BitType::Create(64)));
    auto ioState =
        cfg->entry()->append_argument(Argument::create("ioState", IOStateType::Create()));

    bb->append_last(MallocOperation::createTac(size, ioState));

    cfg->exit()->append_result(bb->last()->result(0));
    cfg->exit()->append_result(bb->last()->result(1));
    cfg->exit()->append_result(bb->last()->result(2));

    auto functionType = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::BitType::Create(64), ioStateType },
        { pointerType, ioStateType, memoryStateType });
    auto f = FunctionNode::create(im->ipgraph(), "f", functionType, Linkage::externalLinkage);
    f->add_cfg(std::move(cfg));

    return im;
  };

  auto verify = [](const llvm::Module & m)
  {
    using namespace llvm;

    auto f = m.getFunction("f");
    auto & bb = f->getEntryBlock();

    EXPECT_EQ(bb.sizeWithoutDebug(), 2);
    EXPECT_EQ(bb.getFirstNonPHI()->getOpcode(), llvm::Instruction::Call);
    EXPECT_EQ(bb.getTerminator()->getOpcode(), llvm::Instruction::Ret);
  };

  auto im = setup();
  print(*im, stdout);

  llvm::LLVMContext ctx;
  auto llvmModule = jlm::llvm::IpGraphToLlvmConverter::CreateAndConvertModule(*im, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  verify(*llvmModule);
}

TEST(IpGraphToLlvmConverterTests, Free)
{
  auto setup = []()
  {
    using namespace jlm::llvm;

    auto iot = IOStateType::Create();
    auto mt = MemoryStateType::Create();
    auto pt = PointerType::Create();

    auto ipgmod = InterProceduralGraphModule::create(jlm::util::FilePath(""), "", "");

    auto ft = jlm::rvsdg::FunctionType::Create(
        { PointerType::Create(), MemoryStateType::Create(), IOStateType::Create() },
        { MemoryStateType::Create(), IOStateType::Create() });
    auto f = FunctionNode::create(ipgmod->ipgraph(), "f", ft, Linkage::externalLinkage);

    auto cfg = ControlFlowGraph::create(*ipgmod);
    auto arg0 = cfg->entry()->append_argument(Argument::create("pointer", pt));
    auto arg1 = cfg->entry()->append_argument(Argument::create("memstate", mt));
    auto arg2 = cfg->entry()->append_argument(Argument::create("iostate", iot));

    auto bb = BasicBlock::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    bb->append_last(FreeOperation::Create(arg0, { arg1 }, arg2));

    cfg->exit()->append_result(bb->last()->result(0));
    cfg->exit()->append_result(bb->last()->result(1));

    f->add_cfg(std::move(cfg));

    return ipgmod;
  };

  auto verify = [](const llvm::Module & module)
  {
    using namespace llvm;

    auto f = module.getFunction("f");
    auto & bb = f->getEntryBlock();

    EXPECT_EQ(bb.sizeWithoutDebug(), 2);
    EXPECT_EQ(bb.getFirstNonPHI()->getOpcode(), Instruction::Call);
    EXPECT_EQ(bb.getTerminator()->getOpcode(), Instruction::Ret);
  };

  auto ipgmod = setup();
  print(*ipgmod, stdout);

  llvm::LLVMContext ctx;
  auto llvmModule = jlm::llvm::IpGraphToLlvmConverter::CreateAndConvertModule(*ipgmod, ctx);
  llvmModule->print(llvm::errs(), nullptr);

  verify(*llvmModule);
}

TEST(IpGraphToLlvmConverterTests, IgnoreMemoryState)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  InterProceduralGraphModule m(jlm::util::FilePath(""), "", "");

  std::unique_ptr<ControlFlowGraph> cfg(new ControlFlowGraph(m));
  auto bb = BasicBlock::create(*cfg);
  cfg->exit()->divert_inedges(bb);
  bb->add_outedge(cfg->exit());

  bb->append_last(UndefValueOperation::Create(mt, "s1"));
  auto s1 = bb->last()->result(0);

  cfg->exit()->append_result(s1);

  auto ft = FunctionType::Create({}, { mt });
  auto f = FunctionNode::create(m.ipgraph(), "f", ft, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  llvm::LLVMContext ctx;
  IpGraphToLlvmConverter::CreateAndConvertModule(m, ctx);
}

TEST(IpGraphToLlvmConverterTests, SelectWithState)
{
  using namespace jlm::llvm;

  auto vt = jlm::rvsdg::TestType::createValueType();
  auto pt = PointerType::Create();
  auto mt = MemoryStateType::Create();
  InterProceduralGraphModule m(jlm::util::FilePath(""), "", "");

  std::unique_ptr<ControlFlowGraph> cfg(new ControlFlowGraph(m));
  auto bb = BasicBlock::create(*cfg);
  cfg->exit()->divert_inedges(bb);
  bb->add_outedge(cfg->exit());

  auto p = cfg->entry()->append_argument(Argument::create("p", jlm::rvsdg::BitType::Create(1)));
  auto s1 = cfg->entry()->append_argument(Argument::create("s1", mt));
  auto s2 = cfg->entry()->append_argument(Argument::create("s2", mt));

  bb->append_last(SelectOperation::create(p, s1, s2));
  auto s3 = bb->last()->result(0);

  cfg->exit()->append_result(s3);
  cfg->exit()->append_result(s3);

  auto ft = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), MemoryStateType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create(), MemoryStateType::Create() });
  auto f = FunctionNode::create(m.ipgraph(), "f", ft, Linkage::externalLinkage);
  f->add_cfg(std::move(cfg));

  print(m, stdout);

  llvm::LLVMContext ctx;
  IpGraphToLlvmConverter::CreateAndConvertModule(m, ctx);
}

TEST(IpGraphToLlvmConverterTests, TestAttributeKindConversion)
{
  typedef jlm::llvm::Attribute::kind ak;

  int begin = static_cast<int>(ak::None);
  int end = static_cast<int>(ak::EndAttrKinds);
  for (int attributeKind = begin; attributeKind != end; attributeKind++)
  {
    jlm::llvm::IpGraphToLlvmConverter::ConvertAttributeKind(static_cast<ak>(attributeKind));
  }
}

TEST(IpGraphToLlvmConverterTests, CallingConvConversion)
{
  /**
   * Tests that function nodes and call operations in the IpGraph are converted to
   * LLVM function declarations, definitions and calls with the correct calling conventions.
   *
   * The tested IpGraph corresponds to the one created in the CallingConvConversion test
   * in LlvmModuleConversionTests.cpp
   */
  using namespace jlm::llvm;

  // Arrange
  auto bit64Type = jlm::rvsdg::BitType::Create(64);
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { bit64Type, ioStateType, memoryStateType },
      { bit64Type, ioStateType, memoryStateType });

  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto imported = FunctionNode::create(
      ipgModule.ipgraph(),
      "imported",
      functionType,
      Linkage::externalLinkage,
      CallingConvention::Fast,
      {});
  auto importedVariable = ipgModule.create_variable(imported);

  auto callee = FunctionNode::create(
      ipgModule.ipgraph(),
      "callee",
      functionType,
      Linkage::externalLinkage,
      CallingConvention::Cold,
      {});
  auto calleeVariable = ipgModule.create_variable(callee);

  // Create the body of the callee
  {
    auto cfg = ControlFlowGraph::create(ipgModule);
    auto valueArgument = cfg->entry()->append_argument(Argument::create("value", bit64Type));
    auto ioStateArgument = cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
    auto memoryStateArgument =
        cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

    auto basicBlock = BasicBlock::create(*cfg);
    cfg->exit()->divert_inedges(basicBlock);
    basicBlock->add_outedge(cfg->exit());
    cfg->exit()->append_result(valueArgument);
    cfg->exit()->append_result(ioStateArgument);
    cfg->exit()->append_result(memoryStateArgument);

    callee->add_cfg(std::move(cfg));
  }

  auto caller = FunctionNode::create(
      ipgModule.ipgraph(),
      "caller",
      functionType,
      Linkage::externalLinkage,
      CallingConvention::Tail,
      {});
  ipgModule.create_variable(caller);

  // Create the body of the caller
  {
    auto cfg = ControlFlowGraph::create(ipgModule);
    auto valueArgument = cfg->entry()->append_argument(Argument::create("value", bit64Type));
    auto ioStateArgument = cfg->entry()->append_argument(Argument::create("ioState", ioStateType));
    auto memoryStateArgument =
        cfg->entry()->append_argument(Argument::create("memoryState", memoryStateType));

    auto basicBlock = BasicBlock::create(*cfg);
    auto importedCall = basicBlock->append_last(CallOperation::create(
        importedVariable,
        functionType,
        CallingConvention::Fast,
        AttributeList::createEmptyList(),
        { valueArgument, ioStateArgument, memoryStateArgument }));
    auto calleeCall = basicBlock->append_last(CallOperation::create(
        calleeVariable,
        functionType,
        CallingConvention::Cold,
        AttributeList::createEmptyList(),
        { importedCall->result(0), importedCall->result(1), importedCall->result(2) }));

    cfg->exit()->divert_inedges(basicBlock);
    basicBlock->add_outedge(cfg->exit());
    cfg->exit()->append_result(calleeCall->result(0));
    cfg->exit()->append_result(calleeCall->result(1));
    cfg->exit()->append_result(calleeCall->result(2));

    caller->add_cfg(std::move(cfg));
  }

  print(ipgModule, stdout);

  // Act
  llvm::LLVMContext ctx;
  auto llvmModule = IpGraphToLlvmConverter::CreateAndConvertModule(ipgModule, ctx);

  // Assert
  {
    auto importedFunction = llvmModule->getFunction("imported");
    auto calleeFunction = llvmModule->getFunction("callee");
    auto callerFunction = llvmModule->getFunction("caller");

    ASSERT_NE(importedFunction, nullptr);
    ASSERT_NE(calleeFunction, nullptr);
    ASSERT_NE(callerFunction, nullptr);

    EXPECT_TRUE(importedFunction->empty());
    EXPECT_FALSE(calleeFunction->empty());
    EXPECT_FALSE(callerFunction->empty());

    EXPECT_EQ(importedFunction->getCallingConv(), ::llvm::CallingConv::Fast);
    EXPECT_EQ(calleeFunction->getCallingConv(), ::llvm::CallingConv::Cold);
    EXPECT_EQ(callerFunction->getCallingConv(), ::llvm::CallingConv::Tail);

    std::vector<const ::llvm::CallInst *> callInstructions;
    for (const auto & instruction : callerFunction->getEntryBlock())
    {
      if (const auto callInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction))
        callInstructions.push_back(callInstruction);
    }

    ASSERT_EQ(callInstructions.size(), 2u);
    ASSERT_NE(callInstructions[0]->getCalledFunction(), nullptr);
    ASSERT_NE(callInstructions[1]->getCalledFunction(), nullptr);

    EXPECT_EQ(callInstructions[0]->getCalledFunction()->getName(), "imported");
    EXPECT_EQ(callInstructions[0]->getCallingConv(), ::llvm::CallingConv::Fast);
    EXPECT_EQ(callInstructions[1]->getCalledFunction()->getName(), "callee");
    EXPECT_EQ(callInstructions[1]->getCallingConv(), ::llvm::CallingConv::Cold);
  }
}
