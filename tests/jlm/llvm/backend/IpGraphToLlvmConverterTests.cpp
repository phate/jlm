/*
 * Copyright 2024, 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
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
  llvmModule->dump();

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
  llvmModule->dump();

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
  llvmModule->dump();

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
  llvmModule->dump();

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
  llvmModule->dump();

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
  llvmModule->dump();

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
  llvmModule->dump();

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

  llvmModule->dump();
}

TEST(IpGraphToLlvmConverterTests, Malloc)
{
  auto setup = []()
  {
    using namespace jlm::llvm;

    auto mt = MemoryStateType::Create();
    auto pt = PointerType::Create();
    auto im = InterProceduralGraphModule::create(jlm::util::FilePath(""), "", "");

    auto cfg = ControlFlowGraph::create(*im);
    auto bb = BasicBlock::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    auto size =
        cfg->entry()->append_argument(Argument::create("size", jlm::rvsdg::BitType::Create(64)));

    bb->append_last(MallocOperation::create(size));

    cfg->exit()->append_result(bb->last()->result(0));
    cfg->exit()->append_result(bb->last()->result(1));

    auto ft = jlm::rvsdg::FunctionType::Create(
        { jlm::rvsdg::BitType::Create(64) },
        { PointerType::Create(), MemoryStateType::Create() });
    auto f = FunctionNode::create(im->ipgraph(), "f", ft, Linkage::externalLinkage);
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
  llvmModule->dump();

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
  llvmModule->dump();

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
