/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

namespace jlm::tests
{

std::unique_ptr<jlm::llvm::RvsdgModule>
StoreTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  auto fcttype =
      rvsdg::FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto d = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto c = alloca_op::create(pointerType, csize, 4);
  auto b = alloca_op::create(pointerType, csize, 4);
  auto a = alloca_op::create(pointerType, csize, 4);

  auto merge_d = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ d[1], fct->GetFunctionArguments()[0] });
  auto merge_c =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ c[1], merge_d }));
  auto merge_b =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ b[1], merge_c }));
  auto merge_a =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ a[1], merge_b }));

  auto a_amp_b = StoreNonVolatileNode::Create(a[0], b[0], { merge_a }, 4);
  auto b_amp_c = StoreNonVolatileNode::Create(b[0], c[0], { a_amp_b[0] }, 4);
  auto c_amp_d = StoreNonVolatileNode::Create(c[0], d[0], { b_amp_c[0] }, 4);

  fct->finalize({ c_amp_d[0] });

  GraphExport::Create(*fct->output(), "f");

  /* extract nodes */

  this->lambda = fct;

  this->size = jlm::rvsdg::output::GetNode(*csize);

  this->alloca_a = jlm::rvsdg::output::GetNode(*a[0]);
  this->alloca_b = jlm::rvsdg::output::GetNode(*b[0]);
  this->alloca_c = jlm::rvsdg::output::GetNode(*c[0]);
  this->alloca_d = jlm::rvsdg::output::GetNode(*d[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
StoreTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  auto fcttype =
      rvsdg::FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ a[1], fct->GetFunctionArguments()[0] });
  auto merge_b =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ b[1], merge_a }));
  auto merge_x =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ x[1], merge_b }));
  auto merge_y =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ y[1], merge_x }));
  auto merge_p =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ p[1], merge_y }));

  auto x_amp_a = StoreNonVolatileNode::Create(x[0], a[0], { merge_p }, 4);
  auto y_amp_b = StoreNonVolatileNode::Create(y[0], b[0], { x_amp_a[0] }, 4);
  auto p_amp_x = StoreNonVolatileNode::Create(p[0], x[0], { y_amp_b[0] }, 4);
  auto p_amp_y = StoreNonVolatileNode::Create(p[0], y[0], { p_amp_x[0] }, 4);

  fct->finalize({ p_amp_y[0] });

  GraphExport::Create(*fct->output(), "f");

  /* extract nodes */

  this->lambda = fct;

  this->size = jlm::rvsdg::output::GetNode(*csize);

  this->alloca_a = jlm::rvsdg::output::GetNode(*a[0]);
  this->alloca_b = jlm::rvsdg::output::GetNode(*b[0]);
  this->alloca_x = jlm::rvsdg::output::GetNode(*x[0]);
  this->alloca_y = jlm::rvsdg::output::GetNode(*y[0]);
  this->alloca_p = jlm::rvsdg::output::GetNode(*p[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath("LoadTest1.c"), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto ld1 = LoadNonVolatileNode::Create(
      fct->GetFunctionArguments()[0],
      { fct->GetFunctionArguments()[1] },
      pointerType,
      4);
  auto ld2 = LoadNonVolatileNode::Create(ld1[0], { ld1[1] }, jlm::rvsdg::bittype::Create(32), 4);

  fct->finalize(ld2);

  GraphExport::Create(*fct->output(), "f");

  /* extract nodes */

  this->lambda = fct;

  this->load_p = jlm::rvsdg::output::GetNode(*ld1[0]);
  this->load_x = jlm::rvsdg::output::GetNode(*ld2[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype =
      rvsdg::FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ a[1], fct->GetFunctionArguments()[0] });
  auto merge_b =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ b[1], merge_a }));
  auto merge_x =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ x[1], merge_b }));
  auto merge_y =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ y[1], merge_x }));
  auto merge_p =
      MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ p[1], merge_y }));

  auto x_amp_a = StoreNonVolatileNode::Create(x[0], a[0], { merge_p }, 4);
  auto y_amp_b = StoreNonVolatileNode::Create(y[0], b[0], x_amp_a, 4);
  auto p_amp_x = StoreNonVolatileNode::Create(p[0], x[0], y_amp_b, 4);

  auto ld1 = LoadNonVolatileNode::Create(p[0], p_amp_x, pointerType, 4);
  auto ld2 = LoadNonVolatileNode::Create(ld1[0], { ld1[1] }, pointerType, 4);
  auto y_star_p = StoreNonVolatileNode::Create(y[0], ld2[0], { ld2[1] }, 4);

  fct->finalize({ y_star_p[0] });

  GraphExport::Create(*fct->output(), "f");

  /* extract nodes */

  this->lambda = fct;

  this->size = jlm::rvsdg::output::GetNode(*csize);

  this->alloca_a = jlm::rvsdg::output::GetNode(*a[0]);
  this->alloca_b = jlm::rvsdg::output::GetNode(*b[0]);
  this->alloca_x = jlm::rvsdg::output::GetNode(*x[0]);
  this->alloca_y = jlm::rvsdg::output::GetNode(*y[0]);
  this->alloca_p = jlm::rvsdg::output::GetNode(*p[0]);

  this->load_x = jlm::rvsdg::output::GetNode(*ld1[0]);
  this->load_a = jlm::rvsdg::output::GetNode(*ld2[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadFromUndefTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto memoryStateType = MemoryStateType::Create();
  auto functionType = rvsdg::FunctionType::Create(
      { MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  Lambda_ = rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));

  auto undefValue = UndefValueOperation::Create(*Lambda_->subregion(), pointerType);
  auto loadResults = LoadNonVolatileNode::Create(
      undefValue,
      { Lambda_->GetFunctionArguments()[0] },
      jlm::rvsdg::bittype::Create(32),
      4);

  Lambda_->finalize(loadResults);
  GraphExport::Create(*Lambda_->output(), "f");

  /*
   * Extract nodes
   */
  UndefValueNode_ = jlm::rvsdg::output::GetNode(*undefValue);

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
GetElementPtrTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto & declaration = module->AddStructTypeDeclaration(StructType::Declaration::Create(
      { jlm::rvsdg::bittype::Create(32), jlm::rvsdg::bittype::Create(32) }));
  auto structType = StructType::Create(false, declaration);

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto one = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 1);

  auto gepx = GetElementPtrOperation::Create(
      fct->GetFunctionArguments()[0],
      { zero, zero },
      structType,
      pointerType);
  auto ldx = LoadNonVolatileNode::Create(
      gepx,
      { fct->GetFunctionArguments()[1] },
      jlm::rvsdg::bittype::Create(32),
      4);

  auto gepy = GetElementPtrOperation::Create(
      fct->GetFunctionArguments()[0],
      { zero, one },
      structType,
      pointerType);
  auto ldy = LoadNonVolatileNode::Create(gepy, { ldx[1] }, jlm::rvsdg::bittype::Create(32), 4);

  auto sum = jlm::rvsdg::bitadd_op::create(32, ldx[0], ldy[0]);

  fct->finalize({ sum, ldy[1] });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;

  this->getElementPtrX = jlm::rvsdg::output::GetNode(*gepx);
  this->getElementPtrY = jlm::rvsdg::output::GetNode(*gepy);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
BitCastTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create({ PointerType::Create() }, { PointerType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto cast = bitcast_op::create(fct->GetFunctionArguments()[0], pointerType);

  fct->finalize({ cast });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->bitCast = jlm::rvsdg::output::GetNode(*cast);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
Bits2PtrTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto setupBit2PtrFunction = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64), IOStateType::Create(), MemoryStateType::Create() },
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "bit2ptr", linkage::external_linkage));
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto cast = bits2ptr_op::create(valueArgument, pt);

    lambda->finalize({ cast, iOStateArgument, memoryStateArgument });

    return std::make_tuple(lambda, jlm::rvsdg::output::GetNode(*cast));
  };

  auto setupTestFunction = [&](rvsdg::output * b2p)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64), IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto cvbits2ptr = lambda->AddContextVar(*b2p).inner;

    auto & call = CallNode::CreateNode(
        cvbits2ptr,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*b2p).GetOperation().Type(),
        { valueArgument, iOStateArgument, memoryStateArgument });

    lambda->finalize({ call.GetIoStateOutput(), call.GetMemoryStateOutput() });
    GraphExport::Create(*lambda->output(), "testfct");

    return std::make_tuple(lambda, &call);
  };

  auto [lambdaBits2Ptr, bitsToPtrNode] = setupBit2PtrFunction();
  auto [lambdaTest, callNode] = setupTestFunction(lambdaBits2Ptr->output());

  // Assign nodes
  this->LambdaBits2Ptr_ = lambdaBits2Ptr;
  this->LambdaTest_ = lambdaTest;

  this->BitsToPtrNode_ = bitsToPtrNode;

  this->CallNode_ = callNode;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ConstantPointerNullTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto constantPointerNullResult =
      ConstantPointerNullOperation::Create(fct->subregion(), pointerType);
  auto st = StoreNonVolatileNode::Create(
      fct->GetFunctionArguments()[0],
      constantPointerNullResult,
      { fct->GetFunctionArguments()[1] },
      4);

  fct->finalize({ st[0] });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->constantPointerNullNode = jlm::rvsdg::output::GetNode(*constantPointerNullResult);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
CallTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupF = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto pointerArgument1 = lambda->GetFunctionArguments()[0];
    auto pointerArgument2 = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto ld1 = LoadNonVolatileNode::Create(
        pointerArgument1,
        { memoryStateArgument },
        jlm::rvsdg::bittype::Create(32),
        4);
    auto ld2 = LoadNonVolatileNode::Create(
        pointerArgument2,
        { ld1[1] },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({ sum, iOStateArgument, ld2[1] });

    return lambda;
  };

  auto SetupG = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "g", linkage::external_linkage));
    auto pointerArgument1 = lambda->GetFunctionArguments()[0];
    auto pointerArgument2 = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto ld1 = LoadNonVolatileNode::Create(
        pointerArgument1,
        { memoryStateArgument },
        jlm::rvsdg::bittype::Create(32),
        4);
    auto ld2 = LoadNonVolatileNode::Create(
        pointerArgument2,
        { ld1[1] },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto diff = jlm::rvsdg::bitsub_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({ diff, iOStateArgument, ld2[1] });

    return lambda;
  };

  auto SetupH = [&](rvsdg::LambdaNode * f, rvsdg::LambdaNode * g)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "h", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvf = lambda->AddContextVar(*f->output()).inner;
    auto cvg = lambda->AddContextVar(*g->output()).inner;

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto x = alloca_op::create(jlm::rvsdg::bittype::Create(32), size, 4);
    auto y = alloca_op::create(jlm::rvsdg::bittype::Create(32), size, 4);
    auto z = alloca_op::create(jlm::rvsdg::bittype::Create(32), size, 4);

    auto mx = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ x[1], memoryStateArgument }));
    auto my = MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ y[1], mx }));
    auto mz = MemoryStateMergeOperation::Create(std::vector<jlm::rvsdg::output *>({ z[1], my }));

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 7);

    auto stx = StoreNonVolatileNode::Create(x[0], five, { mz }, 4);
    auto sty = StoreNonVolatileNode::Create(y[0], six, { stx[0] }, 4);
    auto stz = StoreNonVolatileNode::Create(z[0], seven, { sty[0] }, 4);

    auto & callF = CallNode::CreateNode(
        cvf,
        f->GetOperation().Type(),
        { x[0], y[0], iOStateArgument, stz[0] });
    auto & callG = CallNode::CreateNode(
        cvg,
        g->GetOperation().Type(),
        { z[0], z[0], callF.GetIoStateOutput(), callF.GetMemoryStateOutput() });

    auto sum = jlm::rvsdg::bitadd_op::create(32, callF.Result(0), callG.Result(0));

    lambda->finalize({ sum, callG.GetIoStateOutput(), callG.GetMemoryStateOutput() });
    GraphExport::Create(*lambda->output(), "h");

    auto allocaX = jlm::rvsdg::output::GetNode(*x[0]);
    auto allocaY = jlm::rvsdg::output::GetNode(*y[0]);
    auto allocaZ = jlm::rvsdg::output::GetNode(*z[0]);

    return std::make_tuple(lambda, allocaX, allocaY, allocaZ, &callF, &callG);
  };

  auto lambdaF = SetupF();
  auto lambdaG = SetupG();
  auto [lambdaH, allocaX, allocaY, allocaZ, callF, callG] = SetupH(lambdaF, lambdaG);

  /*
   * Assign nodes
   */
  this->lambda_f = lambdaF;
  this->lambda_g = lambdaG;
  this->lambda_h = lambdaH;

  this->alloca_x = allocaX;
  this->alloca_y = allocaY;
  this->alloca_z = allocaZ;

  this->CallF_ = callF;
  this->CallG_ = callG;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
CallTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupCreate = [&]()
  {
    auto pt32 = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() },
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "create", linkage::external_linkage));
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto prod = jlm::rvsdg::bitmul_op::create(32, valueArgument, four);

    auto alloc = malloc_op::create(prod);
    auto cast = bitcast_op::create(alloc[0], pt32);
    auto mx = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ alloc[1], memoryStateArgument }));

    lambda->finalize({ cast, iOStateArgument, mx });

    auto mallocNode = jlm::rvsdg::output::GetNode(*alloc[0]);
    return std::make_tuple(lambda, mallocNode);
  };

  auto SetupDestroy = [&]()
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "destroy", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto cast = bitcast_op::create(pointerArgument, pointerType);
    auto freeResults = FreeOperation::Create(cast, { memoryStateArgument }, iOStateArgument);

    lambda->finalize({ freeResults[1], freeResults[0] });

    auto freeNode = jlm::rvsdg::output::GetNode(*freeResults[0]);
    return std::make_tuple(lambda, freeNode);
  };

  auto SetupTest = [&](rvsdg::LambdaNode * lambdaCreate, rvsdg::LambdaNode * lambdaDestroy)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto create_cv = lambda->AddContextVar(*lambdaCreate->output()).inner;
    auto destroy_cv = lambda->AddContextVar(*lambdaDestroy->output()).inner;

    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 7);

    auto & create1 = CallNode::CreateNode(
        create_cv,
        lambdaCreate->GetOperation().Type(),
        { six, iOStateArgument, memoryStateArgument });
    auto & create2 = CallNode::CreateNode(
        create_cv,
        lambdaCreate->GetOperation().Type(),
        { seven, create1.GetIoStateOutput(), create1.GetMemoryStateOutput() });

    auto & destroy1 = CallNode::CreateNode(
        destroy_cv,
        lambdaDestroy->GetOperation().Type(),
        { create1.Result(0), create2.GetIoStateOutput(), create2.GetMemoryStateOutput() });
    auto & destroy2 = CallNode::CreateNode(
        destroy_cv,
        lambdaDestroy->GetOperation().Type(),
        { create2.Result(0), destroy1.GetIoStateOutput(), destroy1.GetMemoryStateOutput() });

    lambda->finalize({ destroy2.GetIoStateOutput(), destroy2.GetMemoryStateOutput() });
    GraphExport::Create(*lambda->output(), "test");

    return std::make_tuple(lambda, &create1, &create2, &destroy1, &destroy2);
  };

  auto [lambdaCreate, mallocNode] = SetupCreate();
  auto [lambdaDestroy, freeNode] = SetupDestroy();
  auto [lambdaTest, callCreate1, callCreate2, callDestroy1, callDestroy2] =
      SetupTest(lambdaCreate, lambdaDestroy);

  /*
   * Assign nodes
   */
  this->lambda_create = lambdaCreate;
  this->lambda_destroy = lambdaDestroy;
  this->lambda_test = lambdaTest;

  this->malloc = mallocNode;
  this->free = freeNode;

  this->CallCreate1_ = callCreate1;
  this->CallCreate2_ = callCreate2;

  this->CallDestroy1_ = callCreate1;
  this->CallDestroy2_ = callCreate2;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
IndirectCallTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto constantFunctionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupConstantFunction = [&](ssize_t n, const std::string & name)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(constantFunctionType, name, linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto SetupIndirectCallFunction = [&]()
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "indcall", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto functionOfPointer =
        rvsdg::CreateOpNode<PointerToFunctionOperation>({ pointerArgument }, constantFunctionType)
            .output(0);
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto & call = CallNode::CreateNode(
        functionOfPointer,
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupTestFunction =
      [&](rvsdg::output * fctindcall, rvsdg::output * fctthree, rvsdg::output * fctfour)
  {
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto fctindcall_cv = lambda->AddContextVar(*fctindcall).inner;
    auto fctfour_cv = lambda->AddContextVar(*fctfour).inner;
    auto fctthree_cv = lambda->AddContextVar(*fctthree).inner;

    auto & call_four = CallNode::CreateNode(
        fctindcall_cv,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fctindcall).GetOperation().Type(),
        { fctfour_cv, iOStateArgument, memoryStateArgument });
    auto & call_three = CallNode::CreateNode(
        fctindcall_cv,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fctindcall).GetOperation().Type(),
        { fctthree_cv, call_four.GetIoStateOutput(), call_four.GetMemoryStateOutput() });

    auto add = jlm::rvsdg::bitadd_op::create(32, call_four.Result(0), call_three.Result(0));

    auto lambdaOutput =
        lambda->finalize({ add, call_three.GetIoStateOutput(), call_three.GetMemoryStateOutput() });
    GraphExport::Create(*lambda->output(), "test");

    return std::make_tuple(lambdaOutput, &call_three, &call_four);
  };

  auto fctfour = SetupConstantFunction(4, "four");
  auto fctthree = SetupConstantFunction(3, "three");
  auto [fctindcall, callIndirectFunction] = SetupIndirectCallFunction();
  auto [fcttest, callFunctionThree, callFunctionFour] = SetupTestFunction(
      fctindcall,
      rvsdg::CreateOpNode<FunctionToPointerOperation>({ fctthree }, constantFunctionType).output(0),
      rvsdg::CreateOpNode<FunctionToPointerOperation>({ fctfour }, constantFunctionType).output(0));

  /*
   * Assign
   */
  this->LambdaThree_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fctthree);
  this->LambdaFour_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fctfour);
  this->LambdaIndcall_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fctindcall);
  this->LambdaTest_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fcttest);

  this->CallIndcall_ = callIndirectFunction;
  this->CallThree_ = callFunctionThree;
  this->CallFour_ = callFunctionFour;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
IndirectCallTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto constantFunctionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupG1 = [&]()
  {
    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "g1",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 1);

    return delta->finalize(constant);
  };

  auto SetupG2 = [&]()
  {
    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "g2",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 2);

    return delta->finalize(constant);
  };

  auto SetupConstantFunction = [&](ssize_t n, const std::string & name)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(constantFunctionType, name, linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto functionIType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto SetupI = [&]()
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionIType, "i", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto & call = CallNode::CreateNode(
        rvsdg::CreateOpNode<PointerToFunctionOperation>({ pointerArgument }, constantFunctionType)
            .output(0),
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupIndirectCallFunction = [&](ssize_t n,
                                       const std::string & name,
                                       rvsdg::output & functionI,
                                       rvsdg::output & argumentFunction)
  {
    auto pointerType = PointerType::Create();

    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, name, linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto functionICv = lambda->AddContextVar(functionI).inner;
    auto argumentFunctionCv = lambda->AddContextVar(argumentFunction).inner;
    auto argumentFunctionPtr = rvsdg::CreateOpNode<FunctionToPointerOperation>(
                                   { argumentFunctionCv },
                                   constantFunctionType)
                                   .output(0);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);
    auto storeNode =
        StoreNonVolatileNode::Create(pointerArgument, five, { memoryStateArgument }, 4);

    auto & call = CallNode::CreateNode(
        functionICv,
        functionIType,
        { argumentFunctionPtr, iOStateArgument, storeNode[0] });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupTestFunction = [&](rvsdg::output & functionX,
                               rvsdg::output & functionY,
                               delta::output & globalG1,
                               delta::output & globalG2)
  {
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto functionXCv = lambda->AddContextVar(functionX).inner;
    auto functionYCv = lambda->AddContextVar(functionY).inner;
    auto globalG1Cv = lambda->AddContextVar(globalG1).inner;
    auto globalG2Cv = lambda->AddContextVar(globalG2).inner;

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pxAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);
    auto pyAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);

    auto pxMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ pxAlloca[1], memoryStateArgument });
    auto pyMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pyAlloca[1], pxMerge }));

    auto & callX = CallNode::CreateNode(
        functionXCv,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(functionX).GetOperation().Type(),
        { pxAlloca[0], iOStateArgument, pyMerge });

    auto & callY = CallNode::CreateNode(
        functionYCv,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(functionY).GetOperation().Type(),
        { pyAlloca[0], callX.GetIoStateOutput(), callX.GetMemoryStateOutput() });

    auto loadG1 = LoadNonVolatileNode::Create(
        globalG1Cv,
        { callY.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(32),
        4);
    auto loadG2 =
        LoadNonVolatileNode::Create(globalG2Cv, { loadG1[1] }, jlm::rvsdg::bittype::Create(32), 4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, callX.Result(0), callY.Result(0));
    sum = jlm::rvsdg::bitadd_op::create(32, sum, loadG1[0]);
    sum = jlm::rvsdg::bitadd_op::create(32, sum, loadG2[0]);

    auto lambdaOutput =
        lambda->finalize({ sum, callY.GetIoStateOutput(), callY.GetMemoryStateOutput() });
    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(
        lambdaOutput,
        &callX,
        &callY,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pxAlloca[0])),
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pyAlloca[0])));
  };

  auto SetupTest2Function = [&](rvsdg::output & functionX)
  {
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test2", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pzAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);
    auto pzMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ pzAlloca[1], memoryStateArgument });

    auto functionXCv = lambda->AddContextVar(functionX).inner;

    auto & callX = CallNode::CreateNode(
        functionXCv,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(functionX).GetOperation().Type(),
        { pzAlloca[0], iOStateArgument, pzMerge });

    auto lambdaOutput = lambda->finalize(callX.Results());
    GraphExport::Create(*lambdaOutput, "test2");

    return std::make_tuple(
        lambdaOutput,
        &callX,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pzAlloca[0])));
  };

  auto deltaG1 = SetupG1();
  auto deltaG2 = SetupG2();
  auto lambdaThree = SetupConstantFunction(3, "three");
  auto lambdaFour = SetupConstantFunction(4, "four");
  auto [lambdaI, indirectCall] = SetupI();
  auto [lambdaX, callIWithThree] = SetupIndirectCallFunction(5, "x", *lambdaI, *lambdaThree);
  auto [lambdaY, callIWithFour] = SetupIndirectCallFunction(6, "y", *lambdaI, *lambdaFour);
  auto [lambdaTest, testCallX, callY, allocaPx, allocaPy] =
      SetupTestFunction(*lambdaX, *lambdaY, *deltaG1, *deltaG2);
  auto [lambdaTest2, test2CallX, allocaPz] = SetupTest2Function(*lambdaX);

  /*
   * Assign
   */
  this->DeltaG1_ = deltaG1->node();
  this->DeltaG2_ = deltaG2->node();
  this->LambdaThree_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaThree);
  this->LambdaFour_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaFour);
  this->LambdaI_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaI);
  this->LambdaX_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaX);
  this->LambdaY_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaY);
  this->LambdaTest_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaTest);
  this->LambdaTest2_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaTest2);

  this->IndirectCall_ = indirectCall;
  this->CallIWithThree_ = callIWithThree;
  this->CallIWithFour_ = callIWithFour;
  this->TestCallX_ = testCallX;
  this->Test2CallX_ = test2CallX;
  this->CallY_ = callY;

  this->AllocaPx_ = allocaPx;
  this->AllocaPy_ = allocaPy;
  this->AllocaPz_ = allocaPz;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ExternalCallTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionGType = rvsdg::FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

  auto SetupFunctionGDeclaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        functionGType,
        functionGType,
        "g",
        linkage::external_linkage);
  };

  auto SetupFunctionF = [&](jlm::rvsdg::RegionArgument * functionG)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto pathArgument = lambda->GetFunctionArguments()[0];
    auto modeArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto functionGCv = lambda->AddContextVar(*functionG).inner;

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaPath = alloca_op::create(pointerType, size, 4);
    auto allocaMode = alloca_op::create(pointerType, size, 4);

    auto mergePath = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaPath[1], memoryStateArgument });
    auto mergeMode = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ allocaMode[1], mergePath }));

    auto storePath = StoreNonVolatileNode::Create(allocaPath[0], pathArgument, { mergeMode }, 4);
    auto storeMode = StoreNonVolatileNode::Create(allocaMode[0], modeArgument, { storePath[0] }, 4);

    auto loadPath = LoadNonVolatileNode::Create(allocaPath[0], storeMode, pointerType, 4);
    auto loadMode = LoadNonVolatileNode::Create(allocaMode[0], { loadPath[1] }, pointerType, 4);

    auto & callG = CallNode::CreateNode(
        functionGCv,
        functionGType,
        { loadPath[0], loadMode[0], iOStateArgument, loadMode[1] });

    lambda->finalize(callG.Results());
    GraphExport::Create(*lambda->output(), "f");

    return std::make_tuple(lambda, &callG);
  };

  this->ExternalGArgument_ = SetupFunctionGDeclaration();
  auto [lambdaF, callG] = SetupFunctionF(ExternalGArgument_);

  this->LambdaF_ = lambdaF;
  this->CallG_ = callG;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ExternalCallTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(StructType::Declaration::Create(
      { rvsdg::bittype::Create(32), PointerType::Create(), PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, structDeclaration);
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  VariableArgumentType varArgType;
  auto lambdaLlvmLifetimeStartType = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaLlvmLifetimeEndType = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaFType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaGType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      {
          IOStateType::Create(),
          MemoryStateType::Create(),
      });

  auto llvmLifetimeStart = &GraphImport::Create(
      rvsdg,
      lambdaLlvmLifetimeStartType,
      lambdaLlvmLifetimeStartType,
      "llvm.lifetime.start.p0",
      linkage::external_linkage);
  auto llvmLifetimeEnd = &GraphImport::Create(
      rvsdg,
      lambdaLlvmLifetimeEndType,
      lambdaLlvmLifetimeEndType,
      "llvm.lifetime.end.p0",
      linkage::external_linkage);
  ExternalFArgument_ =
      &GraphImport::Create(rvsdg, lambdaFType, lambdaFType, "f", linkage::external_linkage);

  // Setup function g()
  LambdaG_ = rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(lambdaGType, "g", linkage::external_linkage));
  auto iOStateArgument = LambdaG_->GetFunctionArguments()[0];
  auto memoryStateArgument = LambdaG_->GetFunctionArguments()[1];
  auto llvmLifetimeStartArgument = LambdaG_->AddContextVar(*llvmLifetimeStart).inner;
  auto llvmLifetimeEndArgument = LambdaG_->AddContextVar(*llvmLifetimeEnd).inner;
  auto lambdaFArgument = LambdaG_->AddContextVar(*ExternalFArgument_).inner;

  auto twentyFour = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 64, 24);

  auto allocaResults = alloca_op::create(structType, twentyFour, 16);
  auto memoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ allocaResults[1], memoryStateArgument });

  auto & callLLvmLifetimeStart = CallNode::CreateNode(
      llvmLifetimeStartArgument,
      lambdaLlvmLifetimeStartType,
      { twentyFour, allocaResults[0], iOStateArgument, memoryState });

  CallF_ = &CallNode::CreateNode(
      lambdaFArgument,
      lambdaFType,
      { allocaResults[0],
        callLLvmLifetimeStart.GetIoStateOutput(),
        callLLvmLifetimeStart.GetMemoryStateOutput() });

  auto zero = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 64, 0);
  auto one = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 1);
  auto two = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 2);

  auto gepResult1 =
      GetElementPtrOperation::Create(allocaResults[0], { zero, one }, structType, pointerType);
  auto loadResults1 =
      LoadNonVolatileNode::Create(gepResult1, { CallF_->GetMemoryStateOutput() }, pointerType, 8);
  auto loadResults2 =
      LoadNonVolatileNode::Create(loadResults1[0], { loadResults1[1] }, pointerType, 8);

  auto gepResult2 =
      GetElementPtrOperation::Create(allocaResults[0], { zero, two }, structType, pointerType);
  auto loadResults3 = LoadNonVolatileNode::Create(gepResult2, { loadResults2[1] }, pointerType, 8);
  auto loadResults4 =
      LoadNonVolatileNode::Create(loadResults1[0], { loadResults3[1] }, pointerType, 8);

  auto storeResults1 =
      StoreNonVolatileNode::Create(loadResults1[0], loadResults4[0], { loadResults4[1] }, 8);

  auto loadResults5 = LoadNonVolatileNode::Create(gepResult2, { storeResults1[0] }, pointerType, 8);
  auto storeResults2 =
      StoreNonVolatileNode::Create(loadResults5[0], loadResults2[0], { loadResults5[1] }, 8);

  auto & callLLvmLifetimeEnd = CallNode::CreateNode(
      llvmLifetimeEndArgument,
      lambdaLlvmLifetimeEndType,
      { twentyFour, allocaResults[0], CallF_->GetIoStateOutput(), storeResults2[0] });

  LambdaG_->finalize(callLLvmLifetimeEnd.Results());

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
GammaTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pt = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        PointerType::Create(),
        PointerType::Create(),
        PointerType::Create(),
        PointerType::Create(),
        MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto biteq = jlm::rvsdg::biteq_op::create(32, fct->GetFunctionArguments()[0], zero);
  auto predicate = jlm::rvsdg::match(1, { { 0, 1 } }, 0, 2, biteq);

  auto gammanode = jlm::rvsdg::GammaNode::create(predicate, 2);
  auto p1ev = gammanode->AddEntryVar(fct->GetFunctionArguments()[1]);
  auto p2ev = gammanode->AddEntryVar(fct->GetFunctionArguments()[2]);
  auto p3ev = gammanode->AddEntryVar(fct->GetFunctionArguments()[3]);
  auto p4ev = gammanode->AddEntryVar(fct->GetFunctionArguments()[4]);

  auto tmp1 = gammanode->AddExitVar({ p1ev.branchArgument[0], p3ev.branchArgument[1] });
  auto tmp2 = gammanode->AddExitVar({ p2ev.branchArgument[0], p4ev.branchArgument[1] });

  auto ld1 = LoadNonVolatileNode::Create(
      tmp1.output,
      { fct->GetFunctionArguments()[5] },
      jlm::rvsdg::bittype::Create(32),
      4);
  auto ld2 =
      LoadNonVolatileNode::Create(tmp2.output, { ld1[1] }, jlm::rvsdg::bittype::Create(32), 4);
  auto sum = jlm::rvsdg::bitadd_op::create(32, ld1[0], ld2[0]);

  fct->finalize({ sum, ld2[1] });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->gamma = gammanode;

  return module;
}

std::unique_ptr<llvm::RvsdgModule>
GammaTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto SetupLambdaF = [&]()
  {
    auto SetupGamma = [](rvsdg::output * predicate,
                         rvsdg::output * xAddress,
                         rvsdg::output * yAddress,
                         rvsdg::output * zAddress,
                         rvsdg::output * memoryState)
    {
      auto gammaNode = rvsdg::GammaNode::create(predicate, 2);

      auto gammaInputX = gammaNode->AddEntryVar(xAddress);
      auto gammaInputY = gammaNode->AddEntryVar(yAddress);
      auto gammaInputZ = gammaNode->AddEntryVar(zAddress);
      auto gammaInputMemoryState = gammaNode->AddEntryVar(memoryState);

      // gamma subregion 0
      auto loadXResults = LoadNonVolatileNode::Create(
          gammaInputX.branchArgument[0],
          { gammaInputMemoryState.branchArgument[0] },
          jlm::rvsdg::bittype::Create(32),
          4);

      auto one = rvsdg::create_bitconstant(gammaNode->subregion(0), 32, 1);
      auto storeZRegion0Results =
          StoreNonVolatileNode::Create(gammaInputZ.branchArgument[0], one, { loadXResults[1] }, 4);

      // gamma subregion 1
      auto loadYResults = LoadNonVolatileNode::Create(
          gammaInputY.branchArgument[1],
          { gammaInputMemoryState.branchArgument[1] },
          jlm::rvsdg::bittype::Create(32),
          4);

      auto two = rvsdg::create_bitconstant(gammaNode->subregion(1), 32, 2);
      auto storeZRegion1Results =
          StoreNonVolatileNode::Create(gammaInputZ.branchArgument[1], two, { loadYResults[1] }, 4);

      // finalize gamma
      auto gammaOutputA = gammaNode->AddExitVar({ loadXResults[0], loadYResults[0] });
      auto gammaOutputMemoryState =
          gammaNode->AddExitVar({ storeZRegion0Results[0], storeZRegion1Results[0] });

      return std::make_tuple(gammaOutputA.output, gammaOutputMemoryState.output);
    };

    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto pointerType = PointerType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { rvsdg::bittype::Create(32),
          PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto cArgument = lambda->GetFunctionArguments()[0];
    auto xArgument = lambda->GetFunctionArguments()[1];
    auto yArgument = lambda->GetFunctionArguments()[2];
    auto iOStateArgument = lambda->GetFunctionArguments()[3];
    auto memoryStateArgument = lambda->GetFunctionArguments()[4];

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaZResults = alloca_op::create(pointerType, size, 4);

    auto memoryState = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaZResults[1], memoryStateArgument });

    auto nullPointer = ConstantPointerNullOperation::Create(lambda->subregion(), pointerType);
    auto storeZResults =
        StoreNonVolatileNode::Create(allocaZResults[0], nullPointer, { memoryState }, 4);

    auto zero = rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto bitEq = rvsdg::biteq_op::create(32, cArgument, zero);
    auto predicate = rvsdg::match(1, { { 0, 1 } }, 0, 2, bitEq);

    auto [gammaOutputA, gammaOutputMemoryState] =
        SetupGamma(predicate, xArgument, yArgument, allocaZResults[0], memoryState);

    auto loadZResults = LoadNonVolatileNode::Create(
        allocaZResults[0],
        { gammaOutputMemoryState },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, gammaOutputA, loadZResults[0]);

    lambda->finalize({ sum, iOStateArgument, loadZResults[1] });

    return std::make_tuple(
        lambda->output(),
        &rvsdg::AssertGetOwnerNode<jlm::rvsdg::GammaNode>(*gammaOutputA),
        rvsdg::output::GetNode(*allocaZResults[0]));
  };

  auto SetupLambdaGH = [&](rvsdg::output & lambdaF,
                           int64_t cValue,
                           int64_t xValue,
                           int64_t yValue,
                           const char * functionName)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto pointerType = PointerType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, functionName, linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];
    auto lambdaFArgument = lambda->AddContextVar(lambdaF).inner;

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaXResults = alloca_op::create(rvsdg::bittype::Create(32), size, 4);
    auto allocaYResults = alloca_op::create(pointerType, size, 4);

    auto memoryState = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaXResults[1], memoryStateArgument });
    memoryState = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ allocaYResults[1], memoryState }));

    auto predicate = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, cValue);
    auto x = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, xValue);
    auto y = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, yValue);

    auto storeXResults =
        StoreNonVolatileNode::Create(allocaXResults[0], x, { allocaXResults[1] }, 4);

    auto storeYResults =
        StoreNonVolatileNode::Create(allocaYResults[0], y, { storeXResults[0] }, 4);

    auto & call = CallNode::CreateNode(
        lambdaFArgument,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(lambdaF).GetOperation().Type(),
        { predicate, allocaXResults[0], allocaYResults[0], iOStateArgument, storeYResults[0] });

    lambda->finalize(call.Results());
    GraphExport::Create(*lambda->output(), functionName);

    return std::make_tuple(
        lambda->output(),
        &call,
        rvsdg::output::GetNode(*allocaXResults[0]),
        rvsdg::output::GetNode(*allocaYResults[1]));
  };

  auto [lambdaF, gammaNode, allocaZ] = SetupLambdaF();
  auto [lambdaG, callFromG, allocaXFromG, allocaYFromG] = SetupLambdaGH(*lambdaF, 0, 1, 2, "g");
  auto [lambdaH, callFromH, allocaXFromH, allocaYFromH] = SetupLambdaGH(*lambdaF, 1, 3, 4, "h");

  // Assign nodes
  this->LambdaF_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaF);
  this->LambdaG_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaG);
  this->LambdaH_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaH);

  this->Gamma_ = gammaNode;

  this->CallFromG_ = callFromG;
  this->CallFromH_ = callFromH;

  this->AllocaXFromG_ = allocaXFromG;
  this->AllocaYFromG_ = allocaYFromG;
  this->AllocaXFromH_ = allocaXFromH;
  this->AllocaYFromH_ = allocaYFromH;
  this->AllocaZ_ = allocaZ;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ThetaTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        PointerType::Create(),
        jlm::rvsdg::bittype::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);

  auto thetanode = jlm::rvsdg::ThetaNode::create(fct->subregion());

  auto n = thetanode->AddLoopVar(zero);
  auto l = thetanode->AddLoopVar(fct->GetFunctionArguments()[0]);
  auto a = thetanode->AddLoopVar(fct->GetFunctionArguments()[1]);
  auto c = thetanode->AddLoopVar(fct->GetFunctionArguments()[2]);
  auto s = thetanode->AddLoopVar(fct->GetFunctionArguments()[3]);

  auto gepnode = GetElementPtrOperation::Create(
      a.pre,
      { n.pre },
      jlm::rvsdg::bittype::Create(32),
      pointerType);
  auto store = StoreNonVolatileNode::Create(gepnode, c.pre, { s.pre }, 4);

  auto one = jlm::rvsdg::create_bitconstant(thetanode->subregion(), 32, 1);
  auto sum = jlm::rvsdg::bitadd_op::create(32, n.pre, one);
  auto cmp = jlm::rvsdg::bitult_op::create(32, sum, l.pre);
  auto predicate = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  n.post->divert_to(sum);
  s.post->divert_to(store[0]);
  thetanode->set_predicate(predicate);

  fct->finalize({ s.output });
  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->theta = thetanode;
  this->gep = jlm::rvsdg::output::GetNode(*gepnode);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
DeltaTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupGlobalF = [&]()
  {
    auto dfNode = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "f",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(dfNode->subregion(), 32, 0);

    return dfNode->finalize(constant);
  };

  auto SetupFunctionG = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "g", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto ld = LoadNonVolatileNode::Create(
        pointerArgument,
        { memoryStateArgument },
        jlm::rvsdg::bittype::Create(32),
        4);

    return lambda->finalize({ ld[0], iOStateArgument, ld[1] });
  };

  auto SetupFunctionH = [&](delta::output * f, rvsdg::output * g)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "h", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvf = lambda->AddContextVar(*f).inner;
    auto cvg = lambda->AddContextVar(*g).inner;

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNonVolatileNode::Create(cvf, five, { memoryStateArgument }, 4);
    auto & callG = CallNode::CreateNode(
        cvg,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*g).GetOperation().Type(),
        { cvf, iOStateArgument, st[0] });

    auto lambdaOutput = lambda->finalize(callG.Results());
    GraphExport::Create(*lambda->output(), "h");

    return std::make_tuple(lambdaOutput, &callG, jlm::rvsdg::output::GetNode(*five));
  };

  auto f = SetupGlobalF();
  auto g = SetupFunctionG();
  auto [h, callFunctionG, constantFive] = SetupFunctionH(f, g);

  /*
   * Assign nodes
   */
  this->lambda_g = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*g);
  this->lambda_h = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*h);

  this->delta_f = f->node();

  this->CallG_ = callFunctionG;
  this->constantFive = constantFive;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
DeltaTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupD1 = [&]()
  {
    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "d1",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 0);

    return delta->finalize(constant);
  };

  auto SetupD2 = [&]()
  {
    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "d2",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 0);

    return delta->finalize(constant);
  };

  auto SetupF1 = [&](delta::output * d1)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f1", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvd1 = lambda->AddContextVar(*d1).inner;
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto st = StoreNonVolatileNode::Create(cvd1, b2, { memoryStateArgument }, 4);

    return lambda->finalize({ iOStateArgument, st[0] });
  };

  auto SetupF2 = [&](rvsdg::output * f1, delta::output * d1, delta::output * d2)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f2", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvd1 = lambda->AddContextVar(*d1).inner;
    auto cvd2 = lambda->AddContextVar(*d2).inner;
    auto cvf1 = lambda->AddContextVar(*f1).inner;

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto b42 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 42);
    auto st = StoreNonVolatileNode::Create(cvd1, b5, { memoryStateArgument }, 4);
    auto & call = CallNode::CreateNode(
        cvf1,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f1).GetOperation().Type(),
        { iOStateArgument, st[0] });
    st = StoreNonVolatileNode::Create(cvd2, b42, { call.GetMemoryStateOutput() }, 4);

    auto lambdaOutput = lambda->finalize(call.Results());
    GraphExport::Create(*lambdaOutput, "f2");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto d1 = SetupD1();
  auto d2 = SetupD2();
  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  // Assign nodes
  this->lambda_f1 = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f1);
  this->lambda_f2 = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f2);

  this->delta_d1 = d1->node();
  this->delta_d2 = d2->node();

  this->CallF1_ = callF1;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
DeltaTest3::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupG1 = [&]()
  {
    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "g1",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 1);

    return delta->finalize(constant);
  };

  auto SetupG2 = [&](delta::output & g1)
  {
    auto pointerType = PointerType::Create();

    auto delta = delta::node::Create(
        &graph->GetRootRegion(),
        pointerType,
        "g2",
        linkage::external_linkage,
        "",
        false);

    auto g1Argument = delta->add_ctxvar(&g1);

    return delta->finalize(g1Argument);
  };

  auto SetupF = [&](delta::output & g1, delta::output & g2)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(16), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];
    auto g1CtxVar = lambda->AddContextVar(g1).inner;
    auto g2CtxVar = lambda->AddContextVar(g2).inner;

    auto loadResults =
        LoadNonVolatileNode::Create(g2CtxVar, { memoryStateArgument }, PointerType::Create(), 8);
    auto storeResults =
        StoreNonVolatileNode::Create(g2CtxVar, loadResults[0], { loadResults[1] }, 8);

    loadResults =
        LoadNonVolatileNode::Create(g1CtxVar, storeResults, jlm::rvsdg::bittype::Create(32), 8);
    auto truncResult = trunc_op::create(16, loadResults[0]);

    return lambda->finalize({ truncResult, iOStateArgument, loadResults[1] });
  };

  auto SetupTest = [&](rvsdg::output & lambdaF)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto lambdaFArgument = lambda->AddContextVar(lambdaF).inner;

    auto & call = CallNode::CreateNode(
        lambdaFArgument,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(lambdaF).GetOperation().Type(),
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize({ call.GetIoStateOutput(), call.GetMemoryStateOutput() });
    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto g1 = SetupG1();
  auto g2 = SetupG2(*g1);
  auto f = SetupF(*g1, *g2);
  auto [test, callF] = SetupTest(*f);

  /*
   * Assign nodes
   */
  this->LambdaF_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f);
  this->LambdaTest_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*test);

  this->DeltaG1_ = g1->node();
  this->DeltaG2_ = g2->node();

  this->CallF_ = callF;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ImportTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupF1 = [&](jlm::rvsdg::output * d1)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f1", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvd1 = lambda->AddContextVar(*d1).inner;

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNonVolatileNode::Create(cvd1, b5, { memoryStateArgument }, 4);

    return lambda->finalize({ iOStateArgument, st[0] });
  };

  auto SetupF2 = [&](rvsdg::output * f1, jlm::rvsdg::output * d1, jlm::rvsdg::output * d2)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f2", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto cvd1 = lambda->AddContextVar(*d1).inner;
    auto cvd2 = lambda->AddContextVar(*d2).inner;
    auto cvf1 = lambda->AddContextVar(*f1).inner;
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto b21 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 21);
    auto st = StoreNonVolatileNode::Create(cvd1, b2, { memoryStateArgument }, 4);
    auto & call = CallNode::CreateNode(
        cvf1,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f1).GetOperation().Type(),
        { iOStateArgument, st[0] });
    st = StoreNonVolatileNode::Create(cvd2, b21, { call.GetMemoryStateOutput() }, 4);

    auto lambdaOutput = lambda->finalize(call.Results());
    GraphExport::Create(*lambda->output(), "f2");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto d1 = &GraphImport::Create(
      *graph,
      jlm::rvsdg::bittype::Create(32),
      PointerType::Create(),
      "d1",
      linkage::external_linkage);
  auto d2 = &GraphImport::Create(
      *graph,
      jlm::rvsdg::bittype::Create(32),
      PointerType::Create(),
      "d2",
      linkage::external_linkage);

  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  // Assign nodes
  this->lambda_f1 = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f1);
  this->lambda_f2 = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*f2);

  this->CallF1_ = callF1;

  this->import_d1 = d1;
  this->import_d2 = d2;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
PhiTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto pbit64 = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto fibFunctionType = rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(64),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  auto SetupFib = [&]()
  {
    auto pt = PointerType::Create();

    jlm::llvm::phi::builder pb;
    pb.begin(&graph->GetRootRegion());
    auto fibrv = pb.add_recvar(fibFunctionType);

    auto lambda = rvsdg::LambdaNode::Create(
        *pb.subregion(),
        llvm::LlvmLambdaOperation::Create(fibFunctionType, "fib", linkage::external_linkage));
    auto valueArgument = lambda->GetFunctionArguments()[0];
    auto pointerArgument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];
    auto ctxVarFib = lambda->AddContextVar(*fibrv->argument()).inner;

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 2);
    auto bitult = jlm::rvsdg::bitult_op::create(64, valueArgument, two);
    auto predicate = jlm::rvsdg::match(1, { { 0, 1 } }, 0, 2, bitult);

    auto gammaNode = jlm::rvsdg::GammaNode::create(predicate, 2);
    auto nev = gammaNode->AddEntryVar(valueArgument);
    auto resultev = gammaNode->AddEntryVar(pointerArgument);
    auto fibev = gammaNode->AddEntryVar(ctxVarFib);
    auto gIIoState = gammaNode->AddEntryVar(iOStateArgument);
    auto gIMemoryState = gammaNode->AddEntryVar(memoryStateArgument);

    /* gamma subregion 0 */
    auto one = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jlm::rvsdg::bitsub_op::create(64, nev.branchArgument[0], one);
    auto & callFibm1 = CallNode::CreateNode(
        fibev.branchArgument[0],
        fibFunctionType,
        { nm1,
          resultev.branchArgument[0],
          gIIoState.branchArgument[0],
          gIMemoryState.branchArgument[0] });

    two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jlm::rvsdg::bitsub_op::create(64, nev.branchArgument[0], two);
    auto & callFibm2 = CallNode::CreateNode(
        fibev.branchArgument[0],
        fibFunctionType,
        { nm2,
          resultev.branchArgument[0],
          callFibm1.GetIoStateOutput(),
          callFibm1.GetMemoryStateOutput() });

    auto gepnm1 = GetElementPtrOperation::Create(
        resultev.branchArgument[0],
        { nm1 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm1 = LoadNonVolatileNode::Create(
        gepnm1,
        { callFibm2.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(64),
        8);

    auto gepnm2 = GetElementPtrOperation::Create(
        resultev.branchArgument[0],
        { nm2 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm2 =
        LoadNonVolatileNode::Create(gepnm2, { ldnm1[1] }, jlm::rvsdg::bittype::Create(64), 8);

    auto sum = jlm::rvsdg::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->AddExitVar({ sum, nev.branchArgument[1] });
    auto gOIoState =
        gammaNode->AddExitVar({ callFibm2.GetIoStateOutput(), gIIoState.branchArgument[1] });
    auto gOMemoryState = gammaNode->AddExitVar({ ldnm2[1], gIMemoryState.branchArgument[1] });

    auto gepn = GetElementPtrOperation::Create(
        pointerArgument,
        { valueArgument },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto store = StoreNonVolatileNode::Create(gepn, sumex.output, { gOMemoryState.output }, 8);

    auto lambdaOutput = lambda->finalize({ gOIoState.output, store[0] });

    fibrv->result()->divert_to(lambdaOutput);
    auto phiNode = pb.end();

    return std::make_tuple(phiNode, lambdaOutput, gammaNode, &callFibm1, &callFibm2);
  };

  auto SetupTestFunction = [&](phi::node * phiNode)
  {
    auto at = ArrayType::Create(jlm::rvsdg::bittype::Create(64), 10);
    auto pbit64 = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];
    auto fibcv = lambda->AddContextVar(*phiNode->output(0)).inner;

    auto ten = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 10);
    auto allocaResults = alloca_op::create(at, ten, 16);
    auto state = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaResults[1], memoryStateArgument });

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 0);
    auto gep = GetElementPtrOperation::Create(allocaResults[0], { zero, zero }, at, pbit64);

    auto & call =
        CallNode::CreateNode(fibcv, fibFunctionType, { ten, gep, iOStateArgument, state });

    auto lambdaOutput = lambda->finalize(call.Results());
    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::output::GetNode(*allocaResults[0]));
  };

  auto [phiNode, fibfct, gammaNode, callFib1, callFib2] = SetupFib();
  auto [testfct, callFib, alloca] = SetupTestFunction(phiNode);

  // Assign nodes
  this->lambda_fib = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*fibfct);
  this->lambda_test = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*testfct);

  this->gamma = gammaNode;
  this->phi = phiNode;

  this->CallFibm1_ = callFib1;
  this->CallFibm2_ = callFib2;

  this->CallFib_ = callFib;

  this->alloca = alloca;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
PhiTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto pointerType = PointerType::Create();

  auto constantFunctionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto recursiveFunctionType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto functionIType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto recFunctionType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto SetupEight = [&]()
  {
    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(
            constantFunctionType,
            "eight",
            linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto SetupI = [&]()
  {
    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionIType, "i", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto functionArgument =
        rvsdg::CreateOpNode<PointerToFunctionOperation>({ pointerArgument }, constantFunctionType)
            .output(0);
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto & call = CallNode::CreateNode(
        functionArgument,
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupA =
      [&](jlm::rvsdg::Region & region, phi::rvargument & functionB, phi::rvargument & functionD)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        region,
        llvm::LlvmLambdaOperation::Create(recFunctionType, "a", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto functionBCv = lambda->AddContextVar(functionB).inner;
    auto functionDCv = lambda->AddContextVar(functionD).inner;

    auto one = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 1);
    auto storeNode = StoreNonVolatileNode::Create(pointerArgument, one, { memoryStateArgument }, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto paAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto paMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ paAlloca[1], storeNode[0] }));

    auto & callB = CallNode::CreateNode(
        functionBCv,
        recFunctionType,
        { paAlloca[0], iOStateArgument, paMerge });

    auto & callD = CallNode::CreateNode(
        functionDCv,
        recFunctionType,
        { paAlloca[0], callB.GetIoStateOutput(), callB.GetMemoryStateOutput() });

    auto sum = jlm::rvsdg::bitadd_op::create(32, callB.Result(0), callD.Result(0));

    auto lambdaOutput =
        lambda->finalize({ sum, callD.GetIoStateOutput(), callD.GetMemoryStateOutput() });

    return std::make_tuple(
        lambdaOutput,
        &callB,
        &callD,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*paAlloca[0])));
  };

  auto SetupB = [&](jlm::rvsdg::Region & region,
                    phi::cvargument & functionI,
                    phi::rvargument & functionC,
                    phi::cvargument & functionEight)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        region,
        llvm::LlvmLambdaOperation::Create(recFunctionType, "b", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto functionICv = lambda->AddContextVar(functionI).inner;
    auto functionCCv = lambda->AddContextVar(functionC).inner;
    auto functionEightCv = lambda->AddContextVar(functionEight).inner;

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto storeNode = StoreNonVolatileNode::Create(pointerArgument, two, { memoryStateArgument }, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pbAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto pbMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pbAlloca[1], storeNode[0] }));

    auto & callI = CallNode::CreateNode(
        functionICv,
        functionIType,
        { rvsdg::CreateOpNode<FunctionToPointerOperation>({ functionEightCv }, constantFunctionType)
              .output(0),
          iOStateArgument,
          pbMerge });

    auto & callC = CallNode::CreateNode(
        functionCCv,
        recFunctionType,
        { pbAlloca[0], callI.GetIoStateOutput(), callI.GetMemoryStateOutput() });

    auto sum = jlm::rvsdg::bitadd_op::create(32, callI.Result(0), callC.Result(0));

    auto lambdaOutput =
        lambda->finalize({ sum, callC.GetIoStateOutput(), callC.GetMemoryStateOutput() });

    return std::make_tuple(
        lambdaOutput,
        &callI,
        &callC,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pbAlloca[0])));
  };

  auto SetupC = [&](jlm::rvsdg::Region & region, phi::rvargument & functionA)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        region,
        llvm::LlvmLambdaOperation::Create(recFunctionType, "c", linkage::external_linkage));
    auto xArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto functionACv = lambda->AddContextVar(functionA).inner;

    auto three = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 3);
    auto storeNode = StoreNonVolatileNode::Create(xArgument, three, { memoryStateArgument }, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pcAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto pcMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pcAlloca[1], storeNode[0] }));

    auto & callA = CallNode::CreateNode(
        functionACv,
        recFunctionType,
        { pcAlloca[0], iOStateArgument, pcMerge });

    auto loadX = LoadNonVolatileNode::Create(
        xArgument,
        { callA.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, callA.Result(0), loadX[0]);

    auto lambdaOutput = lambda->finalize({ sum, callA.GetIoStateOutput(), loadX[1] });

    return std::make_tuple(
        lambdaOutput,
        &callA,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pcAlloca[0])));
  };

  auto SetupD = [&](jlm::rvsdg::Region & region, phi::rvargument & functionA)
  {
    auto lambda = rvsdg::LambdaNode::Create(
        region,
        llvm::LlvmLambdaOperation::Create(recFunctionType, "d", linkage::external_linkage));
    auto xArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto functionACv = lambda->AddContextVar(functionA).inner;

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto storeNode = StoreNonVolatileNode::Create(xArgument, four, { memoryStateArgument }, 4);

    auto pdAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto pdMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pdAlloca[1], storeNode[0] }));

    auto & callA = CallNode::CreateNode(
        functionACv,
        recFunctionType,
        { pdAlloca[0], iOStateArgument, pdMerge });

    auto lambdaOutput = lambda->finalize(callA.Results());

    return std::make_tuple(
        lambdaOutput,
        &callA,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(jlm::rvsdg::output::GetNode(*pdAlloca[0])));
  };

  auto SetupPhi = [&](rvsdg::output & lambdaEight, rvsdg::output & lambdaI)
  {
    jlm::llvm::phi::builder phiBuilder;
    phiBuilder.begin(&graph->GetRootRegion());
    auto lambdaARv = phiBuilder.add_recvar(recFunctionType);
    auto lambdaBRv = phiBuilder.add_recvar(recFunctionType);
    auto lambdaCRv = phiBuilder.add_recvar(recFunctionType);
    auto lambdaDRv = phiBuilder.add_recvar(recFunctionType);
    auto lambdaEightCv = phiBuilder.add_ctxvar(&lambdaEight);
    auto lambdaICv = phiBuilder.add_ctxvar(&lambdaI);

    auto [lambdaAOutput, callB, callD, paAlloca] =
        SetupA(*phiBuilder.subregion(), *lambdaBRv->argument(), *lambdaDRv->argument());

    auto [lambdaBOutput, callI, callC, pbAlloca] =
        SetupB(*phiBuilder.subregion(), *lambdaICv, *lambdaCRv->argument(), *lambdaEightCv);

    auto [lambdaCOutput, callAFromC, pcAlloca] =
        SetupC(*phiBuilder.subregion(), *lambdaARv->argument());

    auto [lambdaDOutput, callAFromD, pdAlloca] =
        SetupD(*phiBuilder.subregion(), *lambdaARv->argument());

    lambdaARv->result()->divert_to(lambdaAOutput);
    lambdaBRv->result()->divert_to(lambdaBOutput);
    lambdaCRv->result()->divert_to(lambdaCOutput);
    lambdaDRv->result()->divert_to(lambdaDOutput);

    phiBuilder.end();

    return std::make_tuple(
        lambdaARv,
        lambdaBRv,
        lambdaCRv,
        lambdaDRv,
        callB,
        callD,
        callI,
        callC,
        callAFromC,
        callAFromD,
        paAlloca,
        pbAlloca,
        pcAlloca,
        pdAlloca);
  };

  auto SetupTest = [&](phi::rvoutput & functionA)
  {
    auto pointerType = PointerType::Create();

    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        graph->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto functionACv = lambda->AddContextVar(functionA).inner;

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pTestAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto pTestMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pTestAlloca[1], memoryStateArgument }));

    auto & callA = CallNode::CreateNode(
        functionACv,
        recFunctionType,
        { pTestAlloca[0], iOStateArgument, pTestMerge });

    auto lambdaOutput = lambda->finalize(callA.Results());
    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(
        lambdaOutput,
        &callA,
        jlm::util::AssertedCast<jlm::rvsdg::SimpleNode>(
            jlm::rvsdg::output::GetNode(*pTestAlloca[0])));
  };

  auto lambdaEight = SetupEight();
  auto [lambdaI, indirectCall] = SetupI();

  auto
      [lambdaA,
       lambdaB,
       lambdaC,
       lambdaD,
       callB,
       callD,
       callI,
       callC,
       callAFromC,
       callAFromD,
       paAlloca,
       pbAlloca,
       pcAlloca,
       pdAlloca] = SetupPhi(*lambdaEight, *lambdaI);

  auto [lambdaTest, callAFromTest, pTestAlloca] = SetupTest(*lambdaA);

  /*
   * Assign nodes
   */
  this->LambdaEight_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaEight);
  this->LambdaI_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaI);
  this->LambdaA_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaA->result()->origin());
  this->LambdaB_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaB->result()->origin());
  this->LambdaC_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaC->result()->origin());
  this->LambdaD_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaD->result()->origin());
  this->LambdaTest_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaTest);

  this->CallAFromTest_ = callAFromTest;
  this->CallAFromC_ = callAFromC;
  this->CallAFromD_ = callAFromD;
  this->CallB_ = callB;
  this->CallC_ = callC;
  this->CallD_ = callD;
  this->CallI_ = callI;
  this->IndirectCall_ = indirectCall;

  this->PTestAlloca_ = pTestAlloca;
  this->PaAlloca_ = paAlloca;
  this->PbAlloca_ = pbAlloca;
  this->PcAlloca_ = pcAlloca;
  this->PdAlloca_ = pdAlloca;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
PhiWithDeltaTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, structDeclaration);
  auto arrayType = ArrayType::Create(structType, 2);

  jlm::llvm::phi::builder pb;
  pb.begin(&rvsdg.GetRootRegion());
  auto myArrayRecVar = pb.add_recvar(pointerType);

  auto delta = delta::node::Create(
      pb.subregion(),
      arrayType,
      "myArray",
      linkage::external_linkage,
      "",
      false);
  auto myArrayArgument = delta->add_ctxvar(myArrayRecVar->argument());

  auto aggregateZero = ConstantAggregateZero::Create(*delta->subregion(), structType);
  auto & constantStruct =
      ConstantStruct::Create(*delta->subregion(), { myArrayArgument }, structType);
  auto constantArray = ConstantArray::Create({ aggregateZero, &constantStruct });

  auto deltaOutput = delta->finalize(constantArray);
  Delta_ = deltaOutput->node();
  myArrayRecVar->result()->divert_to(deltaOutput);

  auto phiNode = pb.end();
  GraphExport::Create(*phiNode->output(0), "myArray");

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
ExternalMemoryTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto ft = rvsdg::FunctionType::Create(
      { PointerType::Create(), PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  /**
   * Setup function f.
   */
  LambdaF = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(ft, "f", linkage::external_linkage));
  auto x = LambdaF->GetFunctionArguments()[0];
  auto y = LambdaF->GetFunctionArguments()[1];
  auto state = LambdaF->GetFunctionArguments()[2];

  auto one = jlm::rvsdg::create_bitconstant(LambdaF->subregion(), 32, 1);
  auto two = jlm::rvsdg::create_bitconstant(LambdaF->subregion(), 32, 2);

  auto storeOne = StoreNonVolatileNode::Create(x, one, { state }, 4);
  auto storeTwo = StoreNonVolatileNode::Create(y, two, { storeOne[0] }, 4);

  LambdaF->finalize(storeTwo);
  GraphExport::Create(*LambdaF->output(), "f");

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
EscapedMemoryTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto SetupDeltaA = [&]()
  {
    auto deltaNode = delta::node::Create(
        &rvsdg->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "a",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(deltaNode->subregion(), 32, 1);

    return deltaNode->finalize(constant);
  };

  auto SetupDeltaB = [&]()
  {
    auto deltaNode = delta::node::Create(
        &rvsdg->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "b",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(deltaNode->subregion(), 32, 2);

    return deltaNode->finalize(constant);
  };

  auto SetupDeltaX = [&](delta::output & deltaA)
  {
    auto pointerType = PointerType::Create();

    auto deltaNode = delta::node::Create(
        &rvsdg->GetRootRegion(),
        pointerType,
        "x",
        linkage::external_linkage,
        "",
        false);

    auto contextVariableA = deltaNode->add_ctxvar(&deltaA);

    return deltaNode->finalize(contextVariableA);
  };

  auto SetupDeltaY = [&](delta::output & deltaX)
  {
    auto pointerType = PointerType::Create();

    auto deltaNode = delta::node::Create(
        &rvsdg->GetRootRegion(),
        pointerType,
        "y",
        linkage::external_linkage,
        "",
        false);

    auto contextVariableX = deltaNode->add_ctxvar(&deltaX);

    auto deltaOutput = deltaNode->finalize(contextVariableX);
    GraphExport::Create(*deltaOutput, "y");

    return deltaOutput;
  };

  auto SetupLambdaTest = [&](delta::output & deltaB)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto pointerArgument = lambda->GetFunctionArguments()[0];
    auto iOStateArgument = lambda->GetFunctionArguments()[1];
    auto memoryStateArgument = lambda->GetFunctionArguments()[2];

    auto contextVariableB = lambda->AddContextVar(deltaB).inner;

    auto loadResults1 =
        LoadNonVolatileNode::Create(pointerArgument, { memoryStateArgument }, pointerType, 4);
    auto loadResults2 = LoadNonVolatileNode::Create(
        loadResults1[0],
        { loadResults1[1] },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto storeResults =
        StoreNonVolatileNode::Create(contextVariableB, five, { loadResults2[1] }, 4);

    auto lambdaOutput = lambda->finalize({ loadResults2[0], iOStateArgument, storeResults[0] });

    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(
        lambdaOutput,
        jlm::util::AssertedCast<LoadNonVolatileNode>(
            jlm::rvsdg::output::GetNode(*loadResults1[0])));
  };

  auto deltaA = SetupDeltaA();
  auto deltaB = SetupDeltaB();
  auto deltaX = SetupDeltaX(*deltaA);
  auto deltaY = SetupDeltaY(*deltaX);
  auto [lambdaTest, loadNode1] = SetupLambdaTest(*deltaB);

  /*
   * Assign nodes
   */
  this->LambdaTest = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaTest);

  this->DeltaA = deltaA->node();
  this->DeltaB = deltaB->node();
  this->DeltaX = deltaX->node();
  this->DeltaY = deltaY->node();

  this->LoadNode1 = loadNode1;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
EscapedMemoryTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto externalFunction1Type = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  auto externalFunction2Type = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

  auto SetupExternalFunction1Declaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunction1Type,
        externalFunction1Type,
        "ExternalFunction1",
        linkage::external_linkage);
  };

  auto SetupExternalFunction2Declaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunction2Type,
        externalFunction2Type,
        "ExternalFunction2",
        linkage::external_linkage);
  };

  auto SetupReturnAddressFunction = [&]()
  {
    PointerType p8;
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(
            functionType,
            "ReturnAddress",
            linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto eight = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    auto mallocResults = malloc_op::create(eight);
    auto mergeResults = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ memoryStateArgument, mallocResults[1] }));

    auto lambdaOutput = lambda->finalize({ mallocResults[0], iOStateArgument, mergeResults });

    GraphExport::Create(*lambdaOutput, "ReturnAddress");

    return std::make_tuple(lambdaOutput, jlm::rvsdg::output::GetNode(*mallocResults[0]));
  };

  auto SetupCallExternalFunction1 = [&](jlm::rvsdg::RegionArgument * externalFunction1Argument)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(
            functionType,
            "CallExternalFunction1",
            linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto externalFunction1 = lambda->AddContextVar(*externalFunction1Argument).inner;

    auto eight = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    auto mallocResults = malloc_op::create(eight);
    auto mergeResult = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ memoryStateArgument, mallocResults[1] }));

    auto & call = CallNode::CreateNode(
        externalFunction1,
        externalFunction1Type,
        { mallocResults[0], iOStateArgument, mergeResult });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "CallExternalFunction1");

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::output::GetNode(*mallocResults[0]));
  };

  auto SetupCallExternalFunction2 = [&](jlm::rvsdg::RegionArgument * externalFunction2Argument)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(
            functionType,
            "CallExternalFunction2",
            linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto externalFunction2 = lambda->AddContextVar(*externalFunction2Argument).inner;

    auto & call = CallNode::CreateNode(
        externalFunction2,
        externalFunction2Type,
        { iOStateArgument, memoryStateArgument });

    auto loadResults = LoadNonVolatileNode::Create(
        call.Result(0),
        { call.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto lambdaOutput =
        lambda->finalize({ loadResults[0], call.GetIoStateOutput(), loadResults[1] });

    GraphExport::Create(*lambdaOutput, "CallExternalFunction2");

    return std::make_tuple(
        lambdaOutput,
        &call,
        jlm::util::AssertedCast<jlm::llvm::LoadNonVolatileNode>(
            jlm::rvsdg::output::GetNode(*loadResults[0])));
  };

  auto externalFunction1 = SetupExternalFunction1Declaration();
  auto externalFunction2 = SetupExternalFunction2Declaration();
  auto [returnAddressFunction, returnAddressMalloc] = SetupReturnAddressFunction();
  auto [callExternalFunction1, externalFunction1Call, callExternalFunction1Malloc] =
      SetupCallExternalFunction1(externalFunction1);
  auto [callExternalFunction2, externalFunction2Call, loadNode] =
      SetupCallExternalFunction2(externalFunction2);

  /*
   * Assign nodes
   */
  this->ReturnAddressFunction =
      &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*returnAddressFunction);
  this->CallExternalFunction1 =
      &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*callExternalFunction1);
  this->CallExternalFunction2 =
      &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*callExternalFunction2);

  this->ExternalFunction1Call = externalFunction1Call;
  this->ExternalFunction2Call = externalFunction2Call;

  this->ReturnAddressMalloc = returnAddressMalloc;
  this->CallExternalFunction1Malloc = callExternalFunction1Malloc;

  this->ExternalFunction1Import = externalFunction1;
  this->ExternalFunction2Import = externalFunction2;

  this->LoadNode = loadNode;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
EscapedMemoryTest3::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto externalFunctionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

  auto SetupExternalFunctionDeclaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunctionType,
        externalFunctionType,
        "externalFunction",
        linkage::external_linkage);
  };

  auto SetupGlobal = [&]()
  {
    auto delta = delta::node::Create(
        &rvsdg->GetRootRegion(),
        jlm::rvsdg::bittype::Create(32),
        "global",
        linkage::external_linkage,
        "",
        false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 4);

    auto deltaOutput = delta->finalize(constant);

    GraphExport::Create(*deltaOutput, "global");

    return deltaOutput;
  };

  auto SetupTestFunction = [&](jlm::rvsdg::RegionArgument * externalFunctionArgument)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto externalFunction = lambda->AddContextVar(*externalFunctionArgument).inner;

    auto & call = CallNode::CreateNode(
        externalFunction,
        externalFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto loadResults = LoadNonVolatileNode::Create(
        call.Result(0),
        { call.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(32),
        4);

    auto lambdaOutput =
        lambda->finalize({ loadResults[0], call.GetIoStateOutput(), loadResults[1] });

    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(
        lambdaOutput,
        &call,
        jlm::util::AssertedCast<jlm::llvm::LoadNonVolatileNode>(
            jlm::rvsdg::output::GetNode(*loadResults[0])));
  };

  auto importExternalFunction = SetupExternalFunctionDeclaration();
  auto deltaGlobal = SetupGlobal();
  auto [lambdaTest, callExternalFunction, loadNode] = SetupTestFunction(importExternalFunction);

  // Assign nodes
  this->LambdaTest = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaTest);
  this->DeltaGlobal = deltaGlobal->node();
  this->ImportExternalFunction = importExternalFunction;
  this->CallExternalFunction = callExternalFunction;
  this->LoadNode = loadNode;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
MemcpyTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto arrayType = ArrayType::Create(jlm::rvsdg::bittype::Create(32), 5);

  auto SetupLocalArray = [&]()
  {
    auto delta = delta::node::Create(
        &rvsdg->GetRootRegion(),
        arrayType,
        "localArray",
        linkage::external_linkage,
        "",
        false);

    auto zero = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 0);
    auto one = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 1);
    auto two = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 2);
    auto three = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 3);
    auto four = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 4);

    auto constantDataArray = ConstantDataArray::Create({ zero, one, two, three, four });

    auto deltaOutput = delta->finalize(constantDataArray);

    GraphExport::Create(*deltaOutput, "localArray");

    return deltaOutput;
  };

  auto SetupGlobalArray = [&]()
  {
    auto delta = delta::node::Create(
        &rvsdg->GetRootRegion(),
        arrayType,
        "globalArray",
        linkage::external_linkage,
        "",
        false);

    auto constantAggregateZero = ConstantAggregateZero::Create(*delta->subregion(), arrayType);

    auto deltaOutput = delta->finalize(constantAggregateZero);

    GraphExport::Create(*deltaOutput, "globalArray");

    return deltaOutput;
  };

  auto SetupFunctionF = [&](delta::output & globalArray)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto globalArrayArgument = lambda->AddContextVar(globalArray).inner;

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);

    auto gep = GetElementPtrOperation::Create(
        globalArrayArgument,
        { zero, two },
        arrayType,
        PointerType::Create());

    auto storeResults = StoreNonVolatileNode::Create(gep, six, { memoryStateArgument }, 8);

    auto loadResults =
        LoadNonVolatileNode::Create(gep, { storeResults[0] }, jlm::rvsdg::bittype::Create(32), 8);

    auto lambdaOutput = lambda->finalize({ loadResults[0], iOStateArgument, loadResults[1] });

    GraphExport::Create(*lambdaOutput, "f");

    return lambdaOutput;
  };

  auto SetupFunctionG =
      [&](delta::output & localArray, delta::output & globalArray, rvsdg::output & lambdaF)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "g", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto localArrayArgument = lambda->AddContextVar(localArray).inner;
    auto globalArrayArgument = lambda->AddContextVar(globalArray).inner;
    auto functionFArgument = lambda->AddContextVar(lambdaF).inner;

    auto bcLocalArray = bitcast_op::create(localArrayArgument, PointerType::Create());
    auto bcGlobalArray = bitcast_op::create(globalArrayArgument, PointerType::Create());

    auto twenty = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 20);

    auto memcpyResults = MemCpyNonVolatileOperation::create(
        bcGlobalArray,
        bcLocalArray,
        twenty,
        { memoryStateArgument });

    auto & call = CallNode::CreateNode(
        functionFArgument,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(lambdaF).GetOperation().Type(),
        { iOStateArgument, memcpyResults[0] });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "g");

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::output::GetNode(*memcpyResults[0]));
  };

  auto localArray = SetupLocalArray();
  auto globalArray = SetupGlobalArray();
  auto lambdaF = SetupFunctionF(*globalArray);
  auto [lambdaG, callF, memcpyNode] = SetupFunctionG(*localArray, *globalArray, *lambdaF);

  /*
   * Assign nodes
   */
  this->LambdaF_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaF);
  this->LambdaG_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaG);
  this->LocalArray_ = localArray->node();
  this->GlobalArray_ = globalArray->node();
  this->CallF_ = callF;
  this->Memcpy_ = memcpyNode;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
MemcpyTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto arrayType = ArrayType::Create(PointerType::Create(), 32);
  auto & structBDeclaration =
      rvsdgModule->AddStructTypeDeclaration(StructType::Declaration::Create({ arrayType }));
  auto structTypeB = StructType::Create("structTypeB", false, structBDeclaration);

  auto SetupFunctionG = [&]()
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "g", linkage::internal_linkage));
    auto s1Argument = lambda->GetFunctionArguments()[0];
    auto s2Argument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto c0 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto c128 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 128);

    auto gepS21 = GetElementPtrOperation::Create(s2Argument, { c0, c0 }, structTypeB, pointerType);
    auto gepS22 = GetElementPtrOperation::Create(gepS21, { c0, c0 }, arrayType, pointerType);
    auto ldS2 = LoadNonVolatileNode::Create(gepS22, { memoryStateArgument }, pointerType, 8);

    auto gepS11 = GetElementPtrOperation::Create(s1Argument, { c0, c0 }, structTypeB, pointerType);
    auto gepS12 = GetElementPtrOperation::Create(gepS11, { c0, c0 }, arrayType, pointerType);
    auto ldS1 = LoadNonVolatileNode::Create(gepS12, { ldS2[1] }, pointerType, 8);

    auto memcpyResults = MemCpyNonVolatileOperation::create(ldS2[0], ldS1[0], c128, { ldS1[1] });

    auto lambdaOutput = lambda->finalize({ iOStateArgument, memcpyResults[0] });

    return std::make_tuple(lambdaOutput, jlm::rvsdg::output::GetNode(*memcpyResults[0]));
  };

  auto SetupFunctionF = [&](rvsdg::output & functionF)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          IOStateType::Create(),
          MemoryStateType::Create() },
        { IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg->GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
    auto s1Argument = lambda->GetFunctionArguments()[0];
    auto s2Argument = lambda->GetFunctionArguments()[1];
    auto iOStateArgument = lambda->GetFunctionArguments()[2];
    auto memoryStateArgument = lambda->GetFunctionArguments()[3];

    auto functionFArgument = lambda->AddContextVar(functionF).inner;

    auto c0 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);

    auto gepS1 = GetElementPtrOperation::Create(s1Argument, { c0, c0 }, structTypeB, pointerType);
    auto ldS1 = LoadNonVolatileNode::Create(gepS1, { memoryStateArgument }, pointerType, 8);

    auto gepS2 = GetElementPtrOperation::Create(s2Argument, { c0, c0 }, structTypeB, pointerType);
    auto ldS2 = LoadNonVolatileNode::Create(gepS2, { ldS1[1] }, pointerType, 8);

    auto & call = CallNode::CreateNode(
        functionFArgument,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(functionF).GetOperation().Type(),
        { ldS1[0], ldS2[0], iOStateArgument, ldS2[1] });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "f");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto [lambdaG, memcpyNode] = SetupFunctionG();
  auto [lambdaF, callG] = SetupFunctionF(*lambdaG);

  this->LambdaF_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaF);
  this->LambdaG_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaG);
  this->CallG_ = callG;
  this->Memcpy_ = memcpyNode;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
MemcpyTest3::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto & declaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, declaration);

  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  Lambda_ = rvsdg::LambdaNode::Create(
      rvsdg->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", linkage::internal_linkage));
  auto pArgument = Lambda_->GetFunctionArguments()[0];
  auto iOStateArgument = Lambda_->GetFunctionArguments()[1];
  auto memoryStateArgument = Lambda_->GetFunctionArguments()[2];

  auto eight = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, 8);
  auto zero = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 0);
  auto minusFive = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, -5);
  auto three = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, 3);

  auto allocaResults = alloca_op::create(structType, eight, 8);
  auto memoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ allocaResults[1], memoryStateArgument });

  auto memcpyResults =
      MemCpyNonVolatileOperation::create(allocaResults[0], pArgument, eight, { memoryState });

  auto gep1 =
      GetElementPtrOperation::Create(allocaResults[0], { zero, zero }, structType, pointerType);
  auto ld = LoadNonVolatileNode::Create(gep1, { memcpyResults[0] }, pointerType, 8);

  auto gep2 =
      GetElementPtrOperation::Create(allocaResults[0], { minusFive }, structType, pointerType);

  memcpyResults = MemCpyNonVolatileOperation::create(ld[0], gep2, three, { ld[1] });

  auto lambdaOutput = Lambda_->finalize({ iOStateArgument, memcpyResults[0] });

  GraphExport::Create(*lambdaOutput, "f");

  Alloca_ = rvsdg::output::GetNode(*allocaResults[0]);
  Memcpy_ = rvsdg::output::GetNode(*memcpyResults[0]);

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LinkedListTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto & declaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("list", false, declaration);

  auto SetupDeltaMyList = [&]()
  {
    auto delta = delta::node::Create(
        &rvsdg.GetRootRegion(),
        pointerType,
        "MyList",
        linkage::external_linkage,
        "",
        false);

    auto constantPointerNullResult =
        ConstantPointerNullOperation::Create(delta->subregion(), pointerType);

    auto deltaOutput = delta->finalize(constantPointerNullResult);
    GraphExport::Create(*deltaOutput, "myList");

    return deltaOutput;
  };

  auto SetupFunctionNext = [&](delta::output & myList)
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "next", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto myListArgument = lambda->AddContextVar(myList).inner;

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto alloca = alloca_op::create(pointerType, size, 4);
    auto mergedMemoryState = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ alloca[1], memoryStateArgument });

    auto load1 = LoadNonVolatileNode::Create(myListArgument, { mergedMemoryState }, pointerType, 4);
    auto store1 = StoreNonVolatileNode::Create(alloca[0], load1[0], { load1[1] }, 4);

    auto load2 = LoadNonVolatileNode::Create(alloca[0], { store1[0] }, pointerType, 4);
    auto gep = GetElementPtrOperation::Create(load2[0], { zero, zero }, structType, pointerType);

    auto load3 = LoadNonVolatileNode::Create(gep, { load2[1] }, pointerType, 4);
    auto store2 = StoreNonVolatileNode::Create(alloca[0], load3[0], { load3[1] }, 4);

    auto load4 = LoadNonVolatileNode::Create(alloca[0], { store2[0] }, pointerType, 4);

    auto lambdaOutput = lambda->finalize({ load4[0], iOStateArgument, load4[1] });
    GraphExport::Create(*lambdaOutput, "next");

    return std::make_tuple(jlm::rvsdg::output::GetNode(*alloca[0]), lambdaOutput);
  };

  auto deltaMyList = SetupDeltaMyList();
  auto [alloca, lambdaNext] = SetupFunctionNext(*deltaMyList);

  /*
   * Assign nodes
   */
  this->DeltaMyList_ = deltaMyList->node();
  this->LambdaNext_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaNext);
  this->Alloca_ = alloca;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
AllMemoryNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype =
      rvsdg::FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  // Create imported symbol "imported"
  Import_ = &GraphImport::Create(
      *graph,
      rvsdg::bittype::Create(32),
      PointerType::Create(),
      "imported",
      linkage::external_linkage);

  // Create global variable "global"
  Delta_ = delta::node::Create(
      &graph->GetRootRegion(),
      pointerType,
      "global",
      linkage::external_linkage,
      "",
      false);
  auto constantPointerNullResult =
      ConstantPointerNullOperation::Create(Delta_->subregion(), pointerType);
  Delta_->finalize(constantPointerNullResult);

  // Start of function "f"
  Lambda_ = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));
  auto entryMemoryState = Lambda_->GetFunctionArguments()[0];
  auto deltaContextVar = Lambda_->AddContextVar(*Delta_->output()).inner;
  auto importContextVar = Lambda_->AddContextVar(*Import_).inner;

  // Create alloca node
  auto allocaSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 1);
  auto allocaOutputs = alloca_op::create(pointerType, allocaSize, 8);
  Alloca_ = jlm::rvsdg::output::GetNode(*allocaOutputs[0]);

  auto afterAllocaMemoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ entryMemoryState, allocaOutputs[1] });

  // Create malloc node
  auto mallocSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 4);
  auto mallocOutputs = malloc_op::create(mallocSize);
  Malloc_ = jlm::rvsdg::output::GetNode(*mallocOutputs[0]);

  auto afterMallocMemoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ afterAllocaMemoryState, mallocOutputs[1] });

  // Store the result of malloc into the alloca'd memory
  auto storeAllocaOutputs = StoreNonVolatileNode::Create(
      allocaOutputs[0],
      mallocOutputs[0],
      { afterMallocMemoryState },
      8);

  // load the value in the alloca again
  auto loadAllocaOutputs =
      LoadNonVolatileNode::Create(allocaOutputs[0], { storeAllocaOutputs[0] }, pointerType, 8);

  // Load the value of the imported symbol "imported"
  auto loadImportedOutputs = LoadNonVolatileNode::Create(
      importContextVar,
      { loadAllocaOutputs[1] },
      jlm::rvsdg::bittype::Create(32),
      4);

  // Store the loaded value from imported, into the address loaded from the alloca (aka. the malloc
  // result)
  auto storeImportedOutputs = StoreNonVolatileNode::Create(
      loadAllocaOutputs[0],
      loadImportedOutputs[0],
      { loadImportedOutputs[1] },
      4);

  // store the loaded alloca value in the global variable
  auto storeOutputs = StoreNonVolatileNode::Create(
      deltaContextVar,
      loadAllocaOutputs[0],
      { storeImportedOutputs[0] },
      8);

  Lambda_->finalize({ storeOutputs[0] });

  GraphExport::Create(*Delta_->output(), "global");
  GraphExport::Create(*Lambda_->output(), "f");

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
NAllocaNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype =
      rvsdg::FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  Function_ = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", linkage::external_linkage));

  auto allocaSize = jlm::rvsdg::create_bitconstant(Function_->subregion(), 32, 1);

  jlm::rvsdg::output * latestMemoryState = Function_->GetFunctionArguments()[0];

  for (size_t i = 0; i < NumAllocaNodes_; i++)
  {
    auto allocaOutputs = alloca_op::create(jlm::rvsdg::bittype::Create(32), allocaSize, 4);
    auto allocaNode = jlm::rvsdg::output::GetNode(*allocaOutputs[0]);

    AllocaNodes_.push_back(allocaNode);

    // Update latestMemoryState to include the alloca memory state output
    latestMemoryState =
        MemoryStateMergeOperation::Create(std::vector{ latestMemoryState, allocaOutputs[1] });
  }

  Function_->finalize({ latestMemoryState });

  GraphExport::Create(*Function_->output(), "f");

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
EscapingLocalFunctionTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto uint32Type = rvsdg::bittype::Create(32);
  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto localFuncType = rvsdg::FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create() });
  auto exportedFuncType = rvsdg::FunctionType::Create(
      { MemoryStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(util::filepath(""), "", "");
  const auto graph = &module->Rvsdg();

  Global_ = delta::node::Create(
      &graph->GetRootRegion(),
      uint32Type,
      "global",
      linkage::internal_linkage,
      "",
      false);
  const auto constantZero = rvsdg::create_bitconstant(Global_->subregion(), 32, 0);
  const auto deltaOutput = Global_->finalize(constantZero);

  LocalFunc_ = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(localFuncType, "localFunction", linkage::internal_linkage));

  LocalFuncParam_ = LocalFunc_->GetFunctionArguments()[0];

  const auto allocaSize = rvsdg::create_bitconstant(LocalFunc_->subregion(), 32, 1);
  const auto allocaOutputs = alloca_op::create(uint32Type, allocaSize, 4);
  LocalFuncParamAllocaNode_ = rvsdg::output::GetNode(*allocaOutputs[0]);

  // Merge function's input Memory State and alloca node's memory state
  rvsdg::output * mergedMemoryState = MemoryStateMergeOperation::Create(
      std::vector<rvsdg::output *>{ LocalFunc_->GetFunctionArguments()[1], allocaOutputs[1] });

  // Store the function parameter into the alloca node
  auto storeOutputs =
      StoreNonVolatileNode::Create(allocaOutputs[0], LocalFuncParam_, { mergedMemoryState }, 4);

  // Bring in deltaOuput as a context variable
  const auto deltaOutputCtxVar = LocalFunc_->AddContextVar(*deltaOutput).inner;

  // Return &global
  LocalFunc_->finalize({ deltaOutputCtxVar, storeOutputs[0] });

  LocalFuncRegister_ =
      rvsdg::CreateOpNode<FunctionToPointerOperation>({ LocalFunc_->output() }, localFuncType)
          .output(0);

  ExportedFunc_ = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(
          exportedFuncType,
          "exportedFunc",
          linkage::external_linkage));

  const auto localFuncCtxVar = ExportedFunc_->AddContextVar(*LocalFuncRegister_).inner;

  // Return &localFunc, pass memory state directly through
  ExportedFunc_->finalize({ localFuncCtxVar, ExportedFunc_->GetFunctionArguments()[0] });

  GraphExport::Create(*ExportedFunc_->output(), "exportedFunc");

  return module;
}

std::unique_ptr<llvm::RvsdgModule>
FreeNullTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto functionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  LambdaMain_ = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "main", linkage::external_linkage));
  auto iOStateArgument = LambdaMain_->GetFunctionArguments()[0];
  auto memoryStateArgument = LambdaMain_->GetFunctionArguments()[1];

  auto constantPointerNullResult =
      ConstantPointerNullOperation::Create(LambdaMain_->subregion(), PointerType::Create());

  auto FreeResults =
      FreeOperation::Create(constantPointerNullResult, { memoryStateArgument }, iOStateArgument);

  LambdaMain_->finalize({ FreeResults[1], FreeResults[0] });

  GraphExport::Create(*LambdaMain_->output(), "main");

  return module;
}

std::unique_ptr<llvm::RvsdgModule>
LambdaCallArgumentMismatch::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto functionType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });
  auto variableArgumentType = VariableArgumentType::Create();
  auto functionTypeCall = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(32),
        variableArgumentType,
        IOStateType::Create(),
        MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto setupLambdaG = [&]()
  {
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionType, "g", linkage::internal_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];

    auto five = rvsdg::create_bitconstant(lambda->subregion(), 32, 5);

    return lambda->finalize({ five, iOStateArgument, memoryStateArgument });
  };

  auto setupLambdaMain = [&](rvsdg::output & lambdaG)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = IOStateType::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionTypeMain = rvsdg::FunctionType::Create(
        { IOStateType::Create(), MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

    auto lambda = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(functionTypeMain, "main", linkage::external_linkage));
    auto iOStateArgument = lambda->GetFunctionArguments()[0];
    auto memoryStateArgument = lambda->GetFunctionArguments()[1];
    auto lambdaGArgument = lambda->AddContextVar(lambdaG).inner;

    auto one = rvsdg::create_bitconstant(lambda->subregion(), 32, 1);
    auto six = rvsdg::create_bitconstant(lambda->subregion(), 32, 6);

    auto vaList = valist_op::Create(*lambda->subregion(), {});

    auto allocaResults = alloca_op::create(rvsdg::bittype::Create(32), one, 4);

    auto memoryState = MemoryStateMergeOperation::Create(
        std::vector<rvsdg::output *>{ memoryStateArgument, allocaResults[1] });

    auto storeResults = StoreNonVolatileNode::Create(allocaResults[0], six, { memoryState }, 4);

    auto loadResults =
        LoadNonVolatileNode::Create(allocaResults[0], storeResults, rvsdg::bittype::Create(32), 4);

    auto & call = CallNode::CreateNode(
        lambdaGArgument,
        functionTypeCall,
        { loadResults[0], vaList, iOStateArgument, loadResults[1] });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "main");

    return std::make_tuple(lambdaOutput, &call);
  };

  LambdaG_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*setupLambdaG());

  // Formal arguments and call arguments do not match. Force conversion through pointer
  // to hide the mismatch, the call operator would complain otherwise.
  // The semantic of this is llvm-specific.
  auto ptr =
      jlm::rvsdg::CreateOpNode<FunctionToPointerOperation>({ LambdaG_->output() }, functionType)
          .output(0);
  auto fn =
      jlm::rvsdg::CreateOpNode<PointerToFunctionOperation>({ ptr }, functionTypeCall).output(0);
  auto [lambdaMainOutput, call] = setupLambdaMain(*fn);
  LambdaMain_ = &rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(*lambdaMainOutput);
  Call_ = call;

  return rvsdgModule;
}

std::unique_ptr<llvm::RvsdgModule>
VariadicFunctionTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto varArgType = VariableArgumentType::Create();
  auto lambdaHType = rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        varArgType,
        IOStateType::Create(),
        MemoryStateType::Create() },
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaFType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaGType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });

  // Setup h()
  ImportH_ = &GraphImport::Create(rvsdg, lambdaHType, lambdaHType, "h", linkage::external_linkage);

  // Setup f()
  {
    LambdaF_ = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(lambdaFType, "f", linkage::internal_linkage));
    auto iArgument = LambdaF_->GetFunctionArguments()[0];
    auto iOStateArgument = LambdaF_->GetFunctionArguments()[1];
    auto memoryStateArgument = LambdaF_->GetFunctionArguments()[2];
    auto lambdaHArgument = LambdaF_->AddContextVar(*ImportH_).inner;

    auto one = jlm::rvsdg::create_bitconstant(LambdaF_->subregion(), 32, 1);
    auto three = jlm::rvsdg::create_bitconstant(LambdaF_->subregion(), 32, 3);

    auto varArgList = valist_op::Create(*LambdaF_->subregion(), { iArgument });

    CallH_ = &CallNode::CreateNode(
        lambdaHArgument,
        lambdaHType,
        { one, varArgList, iOStateArgument, memoryStateArgument });

    auto storeResults = StoreNonVolatileNode::Create(
        CallH_->Result(0),
        three,
        { CallH_->GetMemoryStateOutput() },
        4);

    LambdaF_->finalize({ CallH_->GetIoStateOutput(), storeResults[0] });
  }

  // Setup g()
  {
    LambdaG_ = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(lambdaGType, "g", linkage::external_linkage));
    auto iOStateArgument = LambdaG_->GetFunctionArguments()[0];
    auto memoryStateArgument = LambdaG_->GetFunctionArguments()[1];
    auto lambdaFArgument = LambdaG_->AddContextVar(*LambdaF_->output()).inner;

    auto one = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 1);
    auto five = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 5);

    auto allocaResults = alloca_op::create(jlm::rvsdg::bittype::Create(32), one, 4);
    auto merge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaResults[1], memoryStateArgument });
    AllocaNode_ = rvsdg::output::GetNode(*allocaResults[0]);

    auto storeResults = StoreNonVolatileNode::Create(allocaResults[0], five, { merge }, 4);

    auto & callF = CallNode::CreateNode(
        lambdaFArgument,
        lambdaFType,
        { allocaResults[0], iOStateArgument, storeResults[0] });

    LambdaG_->finalize(callF.Results());
  }

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
VariadicFunctionTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ rvsdg::bittype::Create(32),
                                        rvsdg::bittype::Create(32),
                                        PointerType::Create(),
                                        PointerType::Create() }));
  auto structType = StructType::Create("struct.__va_list_tag", false, structDeclaration);
  auto arrayType = ArrayType::Create(structType, 1);
  auto iOStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto varArgType = VariableArgumentType::Create();
  auto lambdaLlvmLifetimeStartType = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaLlvmLifetimeEndType = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        IOStateType::Create(),
        MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaVaStartType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaVaEndType = rvsdg::FunctionType::Create(
      { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() },
      { IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaFstType = rvsdg::FunctionType::Create(
      { rvsdg::bittype::Create(32), varArgType, IOStateType::Create(), MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });
  auto lambdaGType = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto llvmLifetimeStart = &GraphImport::Create(
      rvsdg,
      lambdaLlvmLifetimeStartType,
      lambdaLlvmLifetimeStartType,
      "llvm.lifetime.start.p0",
      linkage::external_linkage);
  auto llvmLifetimeEnd = &GraphImport::Create(
      rvsdg,
      lambdaLlvmLifetimeEndType,
      lambdaLlvmLifetimeEndType,
      "llvm.lifetime.end.p0",
      linkage::external_linkage);
  auto llvmVaStart = &GraphImport::Create(
      rvsdg,
      lambdaVaStartType,
      lambdaVaStartType,
      "llvm.va_start",
      linkage::external_linkage);
  auto llvmVaEnd = &GraphImport::Create(
      rvsdg,
      lambdaVaEndType,
      lambdaVaEndType,
      "llvm.va_end",
      linkage::external_linkage);

  // Setup function fst()
  {
    LambdaFst_ = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(lambdaFstType, "fst", linkage::internal_linkage));
    auto iOStateArgument = LambdaFst_->GetFunctionArguments()[2];
    auto memoryStateArgument = LambdaFst_->GetFunctionArguments()[3];
    auto llvmLifetimeStartArgument = LambdaFst_->AddContextVar(*llvmLifetimeStart).inner;
    auto llvmLifetimeEndArgument = LambdaFst_->AddContextVar(*llvmLifetimeEnd).inner;
    auto llvmVaStartArgument = LambdaFst_->AddContextVar(*llvmVaStart).inner;
    auto llvmVaEndArgument = LambdaFst_->AddContextVar(*llvmVaEnd).inner;

    auto one = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 32, 1);
    auto twentyFour = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 64, 24);
    auto fortyOne = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 32, 41);

    auto allocaResults = alloca_op::create(arrayType, one, 16);
    auto memoryState = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>{ allocaResults[1], memoryStateArgument });
    AllocaNode_ = rvsdg::output::GetNode(*allocaResults[0]);

    auto & callLLvmLifetimeStart = CallNode::CreateNode(
        llvmLifetimeStartArgument,
        lambdaLlvmLifetimeStartType,
        { twentyFour, allocaResults[0], iOStateArgument, memoryState });
    auto & callVaStart = CallNode::CreateNode(
        llvmVaStartArgument,
        lambdaVaStartType,
        { allocaResults[0],
          callLLvmLifetimeStart.GetIoStateOutput(),
          callLLvmLifetimeStart.GetMemoryStateOutput() });

    auto loadResults = LoadNonVolatileNode::Create(
        allocaResults[0],
        { callVaStart.GetMemoryStateOutput() },
        rvsdg::bittype::Create(32),
        16);
    auto icmpResult = rvsdg::bitult_op::create(32, loadResults[0], fortyOne);
    auto matchResult = rvsdg::match_op::Create(*icmpResult, { { 1, 1 } }, 0, 2);

    auto gammaNode = rvsdg::GammaNode::create(matchResult, 2);
    auto gammaVaAddress = gammaNode->AddEntryVar(allocaResults[0]);
    auto gammaLoadResult = gammaNode->AddEntryVar(loadResults[0]);
    auto gammaMemoryState = gammaNode->AddEntryVar(loadResults[1]);

    // gamma subregion 0
    auto zero = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 0);
    auto two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 32, 2);
    auto eight = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 8);
    auto gepResult1 = GetElementPtrOperation::Create(
        gammaVaAddress.branchArgument[0],
        { zero, two },
        structType,
        pointerType);
    auto loadResultsGamma0 = LoadNonVolatileNode::Create(
        gepResult1,
        { gammaMemoryState.branchArgument[0] },
        pointerType,
        8);
    auto gepResult2 = GetElementPtrOperation::Create(
        loadResultsGamma0[0],
        { eight },
        rvsdg::bittype::Create(8),
        pointerType);
    auto storeResultsGamma0 =
        StoreNonVolatileNode::Create(gepResult1, gepResult2, { loadResultsGamma0[1] }, 8);

    // gamma subregion 1
    zero = jlm::rvsdg::create_bitconstant(gammaNode->subregion(1), 64, 0);
    auto eightBit32 = jlm::rvsdg::create_bitconstant(gammaNode->subregion(1), 32, 8);
    auto three = jlm::rvsdg::create_bitconstant(gammaNode->subregion(1), 32, 3);
    gepResult1 = GetElementPtrOperation::Create(
        gammaVaAddress.branchArgument[1],
        { zero, three },
        structType,
        pointerType);
    auto loadResultsGamma1 = LoadNonVolatileNode::Create(
        gepResult1,
        { gammaMemoryState.branchArgument[1] },
        pointerType,
        16);
    auto & zextResult =
        zext_op::Create(*gammaLoadResult.branchArgument[1], rvsdg::bittype::Create(64));
    gepResult2 = GetElementPtrOperation::Create(
        loadResultsGamma1[0],
        { &zextResult },
        rvsdg::bittype::Create(8),
        pointerType);
    auto addResult = rvsdg::bitadd_op::create(32, gammaLoadResult.branchArgument[1], eightBit32);
    auto storeResultsGamma1 = StoreNonVolatileNode::Create(
        gammaVaAddress.branchArgument[1],
        addResult,
        { loadResultsGamma1[1] },
        16);

    auto gammaAddress = gammaNode->AddExitVar({ loadResultsGamma0[0], gepResult2 });
    auto gammaOutputMemoryState =
        gammaNode->AddExitVar({ storeResultsGamma0[0], storeResultsGamma1[0] });

    loadResults = LoadNonVolatileNode::Create(
        gammaAddress.output,
        { gammaOutputMemoryState.output },
        rvsdg::bittype::Create(32),
        4);
    auto & callVaEnd = CallNode::CreateNode(
        llvmVaEndArgument,
        lambdaVaEndType,
        { allocaResults[0], callVaStart.GetIoStateOutput(), loadResults[1] });
    auto & callLLvmLifetimeEnd = CallNode::CreateNode(
        llvmLifetimeEndArgument,
        lambdaLlvmLifetimeEndType,
        { twentyFour,
          allocaResults[0],
          callVaEnd.GetIoStateOutput(),
          callVaEnd.GetMemoryStateOutput() });

    LambdaFst_->finalize({ loadResults[0],
                           callLLvmLifetimeEnd.GetIoStateOutput(),
                           callLLvmLifetimeEnd.GetMemoryStateOutput() });
  }

  // Setup function g()
  {
    LambdaG_ = rvsdg::LambdaNode::Create(
        rvsdg.GetRootRegion(),
        llvm::LlvmLambdaOperation::Create(lambdaGType, "g", linkage::external_linkage));
    auto iOStateArgument = LambdaG_->GetFunctionArguments()[0];
    auto memoryStateArgument = LambdaG_->GetFunctionArguments()[1];
    auto lambdaFstArgument = LambdaG_->AddContextVar(*LambdaFst_->output()).inner;

    auto zero = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 0);
    auto one = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 1);
    auto two = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 2);
    auto three = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 3);

    auto vaListResult = valist_op::Create(*LambdaG_->subregion(), { zero, one, two });

    auto & callFst = CallNode::CreateNode(
        lambdaFstArgument,
        lambdaFstType,
        { three, vaListResult, iOStateArgument, memoryStateArgument });

    LambdaG_->finalize(callFst.Results());
  }

  return rvsdgModule;
}

}
