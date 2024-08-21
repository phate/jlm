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
  auto fcttype = FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto d = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto c = alloca_op::create(pointerType, csize, 4);
  auto b = alloca_op::create(pointerType, csize, 4);
  auto a = alloca_op::create(pointerType, csize, 4);

  auto merge_d = MemoryStateMergeOperation::Create({ d[1], fct->fctargument(0) });
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

  this->size = jlm::rvsdg::node_output::node(csize);

  this->alloca_a = jlm::rvsdg::node_output::node(a[0]);
  this->alloca_b = jlm::rvsdg::node_output::node(b[0]);
  this->alloca_c = jlm::rvsdg::node_output::node(c[0]);
  this->alloca_d = jlm::rvsdg::node_output::node(d[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
StoreTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemoryStateMergeOperation::Create({ a[1], fct->fctargument(0) });
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

  this->size = jlm::rvsdg::node_output::node(csize);

  this->alloca_a = jlm::rvsdg::node_output::node(a[0]);
  this->alloca_b = jlm::rvsdg::node_output::node(b[0]);
  this->alloca_x = jlm::rvsdg::node_output::node(x[0]);
  this->alloca_y = jlm::rvsdg::node_output::node(y[0]);
  this->alloca_p = jlm::rvsdg::node_output::node(p[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath("LoadTest1.c"), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto ld1 =
      LoadNonVolatileNode::Create(fct->fctargument(0), { fct->fctargument(1) }, pointerType, 4);
  auto ld2 = LoadNonVolatileNode::Create(ld1[0], { ld1[1] }, jlm::rvsdg::bittype::Create(32), 4);

  fct->finalize(ld2);

  GraphExport::Create(*fct->output(), "f");

  /* extract nodes */

  this->lambda = fct;

  this->load_p = jlm::rvsdg::node_output::node(ld1[0]);
  this->load_x = jlm::rvsdg::node_output::node(ld2[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bittype::Create(32), csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemoryStateMergeOperation::Create({ a[1], fct->fctargument(0) });
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

  this->size = jlm::rvsdg::node_output::node(csize);

  this->alloca_a = jlm::rvsdg::node_output::node(a[0]);
  this->alloca_b = jlm::rvsdg::node_output::node(b[0]);
  this->alloca_x = jlm::rvsdg::node_output::node(x[0]);
  this->alloca_y = jlm::rvsdg::node_output::node(y[0]);
  this->alloca_p = jlm::rvsdg::node_output::node(p[0]);

  this->load_x = jlm::rvsdg::node_output::node(ld1[0]);
  this->load_a = jlm::rvsdg::node_output::node(ld2[0]);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadFromUndefTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto memoryStateType = MemoryStateType::Create();
  auto functionType = FunctionType::Create(
      { MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  Lambda_ = lambda::node::create(rvsdg.root(), functionType, "f", linkage::external_linkage);

  auto undefValue = UndefValueOperation::Create(*Lambda_->subregion(), pointerType);
  auto loadResults = LoadNonVolatileNode::Create(
      undefValue,
      { Lambda_->fctargument(0) },
      jlm::rvsdg::bittype::Create(32),
      4);

  Lambda_->finalize(loadResults);
  GraphExport::Create(*Lambda_->output(), "f");

  /*
   * Extract nodes
   */
  UndefValueNode_ = jlm::rvsdg::node_output::node(undefValue);

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
GetElementPtrTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto & declaration = module->AddStructTypeDeclaration(StructType::Declaration::Create(
      { jlm::rvsdg::bittype::Create(32), jlm::rvsdg::bittype::Create(32) }));
  auto structType = StructType::Create(false, declaration);

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto one = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 1);

  auto gepx =
      GetElementPtrOperation::Create(fct->fctargument(0), { zero, zero }, structType, pointerType);
  auto ldx = LoadNonVolatileNode::Create(
      gepx,
      { fct->fctargument(1) },
      jlm::rvsdg::bittype::Create(32),
      4);

  auto gepy =
      GetElementPtrOperation::Create(fct->fctargument(0), { zero, one }, structType, pointerType);
  auto ldy = LoadNonVolatileNode::Create(gepy, { ldx[1] }, jlm::rvsdg::bittype::Create(32), 4);

  auto sum = jlm::rvsdg::bitadd_op::create(32, ldx[0], ldy[0]);

  fct->finalize({ sum, ldy[1] });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;

  this->getElementPtrX = jlm::rvsdg::node_output::node(gepx);
  this->getElementPtrY = jlm::rvsdg::node_output::node(gepy);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
BitCastTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create({ PointerType::Create() }, { PointerType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto cast = bitcast_op::create(fct->fctargument(0), pointerType);

  fct->finalize({ cast });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->bitCast = jlm::rvsdg::node_output::node(cast);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
Bits2PtrTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto setupBit2PtrFunction = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64), iostatetype::Create(), MemoryStateType::Create() },
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "bit2ptr", linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto cast = bits2ptr_op::create(valueArgument, pt);

    lambda->finalize({ cast, iOStateArgument, memoryStateArgument });

    return std::make_tuple(lambda, jlm::rvsdg::node_output::node(cast));
  };

  auto setupTestFunction = [&](lambda::output * b2p)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { jlm::rvsdg::bittype::Create(64), iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto cvbits2ptr = lambda->add_ctxvar(b2p);

    auto & call = CallNode::CreateNode(
        cvbits2ptr,
        b2p->node()->Type(),
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
  auto fcttype = FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto constantPointerNullResult =
      ConstantPointerNullOperation::Create(fct->subregion(), pointerType);
  auto st = StoreNonVolatileNode::Create(
      fct->fctargument(0),
      constantPointerNullResult,
      { fct->fctargument(1) },
      4);

  fct->finalize({ st[0] });

  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->constantPointerNullNode = jlm::rvsdg::node_output::node(constantPointerNullResult);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
CallTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupF = [&]()
  {
    auto pt = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "f", linkage::external_linkage);
    auto pointerArgument1 = lambda->fctargument(0);
    auto pointerArgument2 = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

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
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "g", linkage::external_linkage);
    auto pointerArgument1 = lambda->fctargument(0);
    auto pointerArgument2 = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

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

  auto SetupH = [&](lambda::node * f, lambda::node * g)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "h", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvf = lambda->add_ctxvar(f->output());
    auto cvg = lambda->add_ctxvar(g->output());

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

    auto & callF = CallNode::CreateNode(cvf, f->Type(), { x[0], y[0], iOStateArgument, stz[0] });
    auto & callG = CallNode::CreateNode(
        cvg,
        g->Type(),
        { z[0], z[0], callF.GetIoStateOutput(), callF.GetMemoryStateOutput() });

    auto sum = jlm::rvsdg::bitadd_op::create(32, callF.Result(0), callG.Result(0));

    lambda->finalize({ sum, callG.GetIoStateOutput(), callG.GetMemoryStateOutput() });
    GraphExport::Create(*lambda->output(), "h");

    auto allocaX = jlm::rvsdg::node_output::node(x[0]);
    auto allocaY = jlm::rvsdg::node_output::node(y[0]);
    auto allocaZ = jlm::rvsdg::node_output::node(z[0]);

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

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupCreate = [&]()
  {
    auto pt32 = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() },
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "create", linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto prod = jlm::rvsdg::bitmul_op::create(32, valueArgument, four);

    auto alloc = malloc_op::create(prod);
    auto cast = bitcast_op::create(alloc[0], pt32);
    auto mx = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ alloc[1], memoryStateArgument }));

    lambda->finalize({ cast, iOStateArgument, mx });

    auto mallocNode = jlm::rvsdg::node_output::node(alloc[0]);
    return std::make_tuple(lambda, mallocNode);
  };

  auto SetupDestroy = [&]()
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "destroy", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto cast = bitcast_op::create(pointerArgument, pointerType);
    auto freeResults = FreeOperation::Create(cast, { memoryStateArgument }, iOStateArgument);

    lambda->finalize({ freeResults[1], freeResults[0] });

    auto freeNode = jlm::rvsdg::node_output::node(freeResults[0]);
    return std::make_tuple(lambda, freeNode);
  };

  auto SetupTest = [&](lambda::node * lambdaCreate, lambda::node * lambdaDestroy)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto create_cv = lambda->add_ctxvar(lambdaCreate->output());
    auto destroy_cv = lambda->add_ctxvar(lambdaDestroy->output());

    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 7);

    auto & create1 = CallNode::CreateNode(
        create_cv,
        lambdaCreate->Type(),
        { six, iOStateArgument, memoryStateArgument });
    auto & create2 = CallNode::CreateNode(
        create_cv,
        lambdaCreate->Type(),
        { seven, create1.GetIoStateOutput(), create1.GetMemoryStateOutput() });

    auto & destroy1 = CallNode::CreateNode(
        destroy_cv,
        lambdaDestroy->Type(),
        { create1.Result(0), create2.GetIoStateOutput(), create2.GetMemoryStateOutput() });
    auto & destroy2 = CallNode::CreateNode(
        destroy_cv,
        lambdaDestroy->Type(),
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

  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto constantFunctionType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupConstantFunction = [&](ssize_t n, const std::string & name)
  {
    auto lambda =
        lambda::node::create(graph->root(), constantFunctionType, name, linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto SetupIndirectCallFunction = [&]()
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "indcall", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto & call = CallNode::CreateNode(
        pointerArgument,
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupTestFunction =
      [&](lambda::output * fctindcall, lambda::output * fctthree, lambda::output * fctfour)
  {
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto fctindcall_cv = lambda->add_ctxvar(fctindcall);
    auto fctfour_cv = lambda->add_ctxvar(fctfour);
    auto fctthree_cv = lambda->add_ctxvar(fctthree);

    auto & call_four = CallNode::CreateNode(
        fctindcall_cv,
        fctindcall->node()->Type(),
        { fctfour_cv, iOStateArgument, memoryStateArgument });
    auto & call_three = CallNode::CreateNode(
        fctindcall_cv,
        fctindcall->node()->Type(),
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
  auto [fcttest, callFunctionThree, callFunctionFour] =
      SetupTestFunction(fctindcall, fctthree, fctfour);

  /*
   * Assign
   */
  this->LambdaThree_ = fctthree->node();
  this->LambdaFour_ = fctfour->node();
  this->LambdaIndcall_ = fctindcall->node();
  this->LambdaTest_ = fcttest->node();

  this->CallIndcall_ = callIndirectFunction;
  this->CallThree_ = callFunctionThree;
  this->CallFour_ = callFunctionFour;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
IndirectCallTest2::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto constantFunctionType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });
  auto pointerType = PointerType::Create();

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupG1 = [&]()
  {
    auto delta = delta::node::Create(
        graph->root(),
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
        graph->root(),
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
    auto lambda =
        lambda::node::create(graph->root(), constantFunctionType, name, linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto SetupI = [&]()
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "i", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto & call = CallNode::CreateNode(
        pointerArgument,
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupIndirectCallFunction = [&](ssize_t n,
                                       const std::string & name,
                                       lambda::output & functionI,
                                       lambda::output & argumentFunction)
  {
    auto pointerType = PointerType::Create();

    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, name, linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto functionICv = lambda->add_ctxvar(&functionI);
    auto argumentFunctionCv = lambda->add_ctxvar(&argumentFunction);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);
    auto storeNode =
        StoreNonVolatileNode::Create(pointerArgument, five, { memoryStateArgument }, 4);

    auto & call = CallNode::CreateNode(
        functionICv,
        functionI.node()->Type(),
        { argumentFunctionCv, iOStateArgument, storeNode[0] });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupTestFunction = [&](lambda::output & functionX,
                               lambda::output & functionY,
                               delta::output & globalG1,
                               delta::output & globalG2)
  {
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto functionXCv = lambda->add_ctxvar(&functionX);
    auto functionYCv = lambda->add_ctxvar(&functionY);
    auto globalG1Cv = lambda->add_ctxvar(&globalG1);
    auto globalG2Cv = lambda->add_ctxvar(&globalG2);

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pxAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);
    auto pyAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);

    auto pxMerge = MemoryStateMergeOperation::Create({ pxAlloca[1], memoryStateArgument });
    auto pyMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pyAlloca[1], pxMerge }));

    auto & callX = CallNode::CreateNode(
        functionXCv,
        functionX.node()->Type(),
        { pxAlloca[0], iOStateArgument, pyMerge });

    auto & callY = CallNode::CreateNode(
        functionYCv,
        functionY.node()->Type(),
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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pxAlloca[0])),
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pyAlloca[0])));
  };

  auto SetupTest2Function = [&](lambda::output & functionX)
  {
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test2", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pzAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), constantSize, 4);
    auto pzMerge = MemoryStateMergeOperation::Create({ pzAlloca[1], memoryStateArgument });

    auto functionXCv = lambda->add_ctxvar(&functionX);

    auto & callX = CallNode::CreateNode(
        functionXCv,
        functionX.node()->Type(),
        { pzAlloca[0], iOStateArgument, pzMerge });

    auto lambdaOutput = lambda->finalize(callX.Results());
    GraphExport::Create(*lambdaOutput, "test2");

    return std::make_tuple(
        lambdaOutput,
        &callX,
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pzAlloca[0])));
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
  this->LambdaThree_ = lambdaThree->node();
  this->LambdaFour_ = lambdaFour->node();
  this->LambdaI_ = lambdaI->node();
  this->LambdaX_ = lambdaX->node();
  this->LambdaY_ = lambdaY->node();
  this->LambdaTest_ = lambdaTest->node();
  this->LambdaTest2_ = lambdaTest2->node();

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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionGType = FunctionType::Create(
      { PointerType::Create(),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

  auto SetupFunctionGDeclaration = [&]()
  {
    return &GraphImport::Create(*rvsdg, functionGType, "g", linkage::external_linkage);
  };

  auto SetupFunctionF = [&](jlm::rvsdg::argument * functionG)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
    auto pathArgument = lambda->fctargument(0);
    auto modeArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

    auto functionGCv = lambda->add_ctxvar(functionG);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaPath = alloca_op::create(pointerType, size, 4);
    auto allocaMode = alloca_op::create(pointerType, size, 4);

    auto mergePath = MemoryStateMergeOperation::Create({ allocaPath[1], memoryStateArgument });
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

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(StructType::Declaration::Create(
      { rvsdg::bittype::Create(32), PointerType::Create(), PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, structDeclaration);
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  varargtype varArgType;
  auto lambdaLlvmLifetimeStartType = FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaLlvmLifetimeEndType = FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaFType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaGType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      {
          iostatetype::Create(),
          MemoryStateType::Create(),
      });

  auto llvmLifetimeStart =
      &GraphImport::Create(rvsdg, pointerType, "llvm.lifetime.start.p0", linkage::external_linkage);
  auto llvmLifetimeEnd =
      &GraphImport::Create(rvsdg, pointerType, "llvm.lifetime.end.p0", linkage::external_linkage);
  ExternalFArgument_ = &GraphImport::Create(rvsdg, pointerType, "f", linkage::external_linkage);

  // Setup function g()
  LambdaG_ = lambda::node::create(rvsdg.root(), lambdaGType, "g", linkage::external_linkage);
  auto iOStateArgument = LambdaG_->fctargument(0);
  auto memoryStateArgument = LambdaG_->fctargument(1);
  auto llvmLifetimeStartArgument = LambdaG_->add_ctxvar(llvmLifetimeStart);
  auto llvmLifetimeEndArgument = LambdaG_->add_ctxvar(llvmLifetimeEnd);
  auto lambdaFArgument = LambdaG_->add_ctxvar(ExternalFArgument_);

  auto twentyFour = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 64, 24);

  auto allocaResults = alloca_op::create(structType, twentyFour, 16);
  auto memoryState = MemoryStateMergeOperation::Create({ allocaResults[1], memoryStateArgument });

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
  auto fcttype = FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        PointerType::Create(),
        PointerType::Create(),
        PointerType::Create(),
        PointerType::Create(),
        MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto biteq = jlm::rvsdg::biteq_op::create(32, fct->fctargument(0), zero);
  auto predicate = jlm::rvsdg::match(1, { { 0, 1 } }, 0, 2, biteq);

  auto gammanode = jlm::rvsdg::gamma_node::create(predicate, 2);
  auto p1ev = gammanode->add_entryvar(fct->fctargument(1));
  auto p2ev = gammanode->add_entryvar(fct->fctargument(2));
  auto p3ev = gammanode->add_entryvar(fct->fctargument(3));
  auto p4ev = gammanode->add_entryvar(fct->fctargument(4));

  auto tmp1 = gammanode->add_exitvar({ p1ev->argument(0), p3ev->argument(1) });
  auto tmp2 = gammanode->add_exitvar({ p2ev->argument(0), p4ev->argument(1) });

  auto ld1 = LoadNonVolatileNode::Create(
      tmp1,
      { fct->fctargument(5) },
      jlm::rvsdg::bittype::Create(32),
      4);
  auto ld2 = LoadNonVolatileNode::Create(tmp2, { ld1[1] }, jlm::rvsdg::bittype::Create(32), 4);
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

  auto nf = rvsdg->node_normal_form(typeid(rvsdg::operation));
  nf->set_mutable(false);

  auto SetupLambdaF = [&]()
  {
    auto SetupGamma = [](rvsdg::output * predicate,
                         rvsdg::output * xAddress,
                         rvsdg::output * yAddress,
                         rvsdg::output * zAddress,
                         rvsdg::output * memoryState)
    {
      auto gammaNode = rvsdg::gamma_node::create(predicate, 2);

      auto gammaInputX = gammaNode->add_entryvar(xAddress);
      auto gammaInputY = gammaNode->add_entryvar(yAddress);
      auto gammaInputZ = gammaNode->add_entryvar(zAddress);
      auto gammaInputMemoryState = gammaNode->add_entryvar(memoryState);

      // gamma subregion 0
      auto loadXResults = LoadNonVolatileNode::Create(
          gammaInputX->argument(0),
          { gammaInputMemoryState->argument(0) },
          jlm::rvsdg::bittype::Create(32),
          4);

      auto one = rvsdg::create_bitconstant(gammaNode->subregion(0), 32, 1);
      auto storeZRegion0Results =
          StoreNonVolatileNode::Create(gammaInputZ->argument(0), one, { loadXResults[1] }, 4);

      // gamma subregion 1
      auto loadYResults = LoadNonVolatileNode::Create(
          gammaInputY->argument(1),
          { gammaInputMemoryState->argument(1) },
          jlm::rvsdg::bittype::Create(32),
          4);

      auto two = rvsdg::create_bitconstant(gammaNode->subregion(1), 32, 2);
      auto storeZRegion1Results =
          StoreNonVolatileNode::Create(gammaInputZ->argument(1), two, { loadYResults[1] }, 4);

      // finalize gamma
      auto gammaOutputA = gammaNode->add_exitvar({ loadXResults[0], loadYResults[0] });
      auto gammaOutputMemoryState =
          gammaNode->add_exitvar({ storeZRegion0Results[0], storeZRegion1Results[0] });

      return std::make_tuple(gammaOutputA, gammaOutputMemoryState);
    };

    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto pointerType = PointerType::Create();
    auto functionType = FunctionType::Create(
        { rvsdg::bittype::Create(32),
          PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
    auto cArgument = lambda->fctargument(0);
    auto xArgument = lambda->fctargument(1);
    auto yArgument = lambda->fctargument(2);
    auto iOStateArgument = lambda->fctargument(3);
    auto memoryStateArgument = lambda->fctargument(4);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaZResults = alloca_op::create(pointerType, size, 4);

    auto memoryState =
        MemoryStateMergeOperation::Create({ allocaZResults[1], memoryStateArgument });

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
        gammaOutputA->node(),
        rvsdg::node_output::node(allocaZResults[0]));
  };

  auto SetupLambdaGH = [&](lambda::output & lambdaF,
                           int64_t cValue,
                           int64_t xValue,
                           int64_t yValue,
                           const char * functionName)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto pointerType = PointerType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(rvsdg->root(), functionType, functionName, linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto lambdaFArgument = lambda->add_ctxvar(&lambdaF);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaXResults = alloca_op::create(rvsdg::bittype::Create(32), size, 4);
    auto allocaYResults = alloca_op::create(pointerType, size, 4);

    auto memoryState =
        MemoryStateMergeOperation::Create({ allocaXResults[1], memoryStateArgument });
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
        lambdaF.node()->Type(),
        { predicate, allocaXResults[0], allocaYResults[0], iOStateArgument, storeYResults[0] });

    lambda->finalize(call.Results());
    GraphExport::Create(*lambda->output(), functionName);

    return std::make_tuple(
        lambda->output(),
        &call,
        rvsdg::node_output::node(allocaXResults[0]),
        rvsdg::node_output::node(allocaYResults[1]));
  };

  auto [lambdaF, gammaNode, allocaZ] = SetupLambdaF();
  auto [lambdaG, callFromG, allocaXFromG, allocaYFromG] = SetupLambdaGH(*lambdaF, 0, 1, 2, "g");
  auto [lambdaH, callFromH, allocaXFromH, allocaYFromH] = SetupLambdaGH(*lambdaF, 1, 3, 4, "h");

  // Assign nodes
  this->LambdaF_ = lambdaF->node();
  this->LambdaG_ = lambdaG->node();
  this->LambdaH_ = lambdaH->node();

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
  auto fcttype = FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        PointerType::Create(),
        jlm::rvsdg::bittype::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);

  auto thetanode = jlm::rvsdg::theta_node::create(fct->subregion());

  auto n = thetanode->add_loopvar(zero);
  auto l = thetanode->add_loopvar(fct->fctargument(0));
  auto a = thetanode->add_loopvar(fct->fctargument(1));
  auto c = thetanode->add_loopvar(fct->fctargument(2));
  auto s = thetanode->add_loopvar(fct->fctargument(3));

  auto gepnode = GetElementPtrOperation::Create(
      a->argument(),
      { n->argument() },
      jlm::rvsdg::bittype::Create(32),
      pointerType);
  auto store = StoreNonVolatileNode::Create(gepnode, c->argument(), { s->argument() }, 4);

  auto one = jlm::rvsdg::create_bitconstant(thetanode->subregion(), 32, 1);
  auto sum = jlm::rvsdg::bitadd_op::create(32, n->argument(), one);
  auto cmp = jlm::rvsdg::bitult_op::create(32, sum, l->argument());
  auto predicate = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  n->result()->divert_to(sum);
  s->result()->divert_to(store[0]);
  thetanode->set_predicate(predicate);

  fct->finalize({ s });
  GraphExport::Create(*fct->output(), "f");

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->theta = thetanode;
  this->gep = jlm::rvsdg::node_output::node(gepnode);

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
DeltaTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupGlobalF = [&]()
  {
    auto dfNode = delta::node::Create(
        graph->root(),
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
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "g", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto ld = LoadNonVolatileNode::Create(
        pointerArgument,
        { memoryStateArgument },
        jlm::rvsdg::bittype::Create(32),
        4);

    return lambda->finalize({ ld[0], iOStateArgument, ld[1] });
  };

  auto SetupFunctionH = [&](delta::output * f, lambda::output * g)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "h", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvf = lambda->add_ctxvar(f);
    auto cvg = lambda->add_ctxvar(g);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNonVolatileNode::Create(cvf, five, { memoryStateArgument }, 4);
    auto & callG = CallNode::CreateNode(cvg, g->node()->Type(), { cvf, iOStateArgument, st[0] });

    auto lambdaOutput = lambda->finalize(callG.Results());
    GraphExport::Create(*lambda->output(), "h");

    return std::make_tuple(lambdaOutput, &callG, jlm::rvsdg::node_output::node(five));
  };

  auto f = SetupGlobalF();
  auto g = SetupFunctionG();
  auto [h, callFunctionG, constantFive] = SetupFunctionH(f, g);

  /*
   * Assign nodes
   */
  this->lambda_g = g->node();
  this->lambda_h = h->node();

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

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupD1 = [&]()
  {
    auto delta = delta::node::Create(
        graph->root(),
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
        graph->root(),
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
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "f1", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto st = StoreNonVolatileNode::Create(cvd1, b2, { memoryStateArgument }, 4);

    return lambda->finalize({ iOStateArgument, st[0] });
  };

  auto SetupF2 = [&](lambda::output * f1, delta::output * d1, delta::output * d2)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "f2", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto b42 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 42);
    auto st = StoreNonVolatileNode::Create(cvd1, b5, { memoryStateArgument }, 4);
    auto & call = CallNode::CreateNode(cvf1, f1->node()->Type(), { iOStateArgument, st[0] });
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
  this->lambda_f1 = f1->node();
  this->lambda_f2 = f2->node();

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

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupG1 = [&]()
  {
    auto delta = delta::node::Create(
        graph->root(),
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

    auto delta =
        delta::node::Create(graph->root(), pointerType, "g2", linkage::external_linkage, "", false);

    auto g1Argument = delta->add_ctxvar(&g1);

    return delta->finalize(g1Argument);
  };

  auto SetupF = [&](delta::output & g1, delta::output & g2)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(16), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(graph->root(), functionType, "f", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto g1CtxVar = lambda->add_ctxvar(&g1);
    auto g2CtxVar = lambda->add_ctxvar(&g2);

    auto loadResults =
        LoadNonVolatileNode::Create(g2CtxVar, { memoryStateArgument }, PointerType::Create(), 8);
    auto storeResults =
        StoreNonVolatileNode::Create(g2CtxVar, loadResults[0], { loadResults[1] }, 8);

    loadResults =
        LoadNonVolatileNode::Create(g1CtxVar, storeResults, jlm::rvsdg::bittype::Create(32), 8);
    auto truncResult = trunc_op::create(16, loadResults[0]);

    return lambda->finalize({ truncResult, iOStateArgument, loadResults[1] });
  };

  auto SetupTest = [&](lambda::output & lambdaF)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto lambdaFArgument = lambda->add_ctxvar(&lambdaF);

    auto & call = CallNode::CreateNode(
        lambdaFArgument,
        lambdaF.node()->Type(),
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
  this->LambdaF_ = f->node();
  this->LambdaTest_ = test->node();

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

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupF1 = [&](jlm::rvsdg::output * d1)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "f1", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvd1 = lambda->add_ctxvar(d1);

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNonVolatileNode::Create(cvd1, b5, { memoryStateArgument }, 4);

    return lambda->finalize({ iOStateArgument, st[0] });
  };

  auto SetupF2 = [&](lambda::output * f1, jlm::rvsdg::output * d1, jlm::rvsdg::output * d2)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "f2", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto b21 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 21);
    auto st = StoreNonVolatileNode::Create(cvd1, b2, { memoryStateArgument }, 4);
    auto & call = CallNode::CreateNode(cvf1, f1->node()->Type(), { iOStateArgument, st[0] });
    st = StoreNonVolatileNode::Create(cvd2, b21, { call.GetMemoryStateOutput() }, 4);

    auto lambdaOutput = lambda->finalize(call.Results());
    GraphExport::Create(*lambda->output(), "f2");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto d1 = &GraphImport::Create(
      *graph,
      jlm::rvsdg::bittype::Create(32),
      "d1",
      linkage::external_linkage);
  auto d2 = &GraphImport::Create(
      *graph,
      jlm::rvsdg::bittype::Create(32),
      "d2",
      linkage::external_linkage);

  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  // Assign nodes
  this->lambda_f1 = f1->node();
  this->lambda_f2 = f2->node();

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

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pbit64 = PointerType::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto fibFunctionType = FunctionType::Create(
      { jlm::rvsdg::bittype::Create(64),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  auto SetupFib = [&]()
  {
    auto pt = PointerType::Create();

    jlm::llvm::phi::builder pb;
    pb.begin(graph->root());
    auto fibrv = pb.add_recvar(pt);

    auto lambda =
        lambda::node::create(pb.subregion(), fibFunctionType, "fib", linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto pointerArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto ctxVarFib = lambda->add_ctxvar(fibrv->argument());

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 2);
    auto bitult = jlm::rvsdg::bitult_op::create(64, valueArgument, two);
    auto predicate = jlm::rvsdg::match(1, { { 0, 1 } }, 0, 2, bitult);

    auto gammaNode = jlm::rvsdg::gamma_node::create(predicate, 2);
    auto nev = gammaNode->add_entryvar(valueArgument);
    auto resultev = gammaNode->add_entryvar(pointerArgument);
    auto fibev = gammaNode->add_entryvar(ctxVarFib);
    auto gIIoState = gammaNode->add_entryvar(iOStateArgument);
    auto gIMemoryState = gammaNode->add_entryvar(memoryStateArgument);

    /* gamma subregion 0 */
    auto one = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jlm::rvsdg::bitsub_op::create(64, nev->argument(0), one);
    auto & callFibm1 = CallNode::CreateNode(
        fibev->argument(0),
        fibFunctionType,
        { nm1, resultev->argument(0), gIIoState->argument(0), gIMemoryState->argument(0) });

    two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jlm::rvsdg::bitsub_op::create(64, nev->argument(0), two);
    auto & callFibm2 = CallNode::CreateNode(
        fibev->argument(0),
        fibFunctionType,
        { nm2,
          resultev->argument(0),
          callFibm1.GetIoStateOutput(),
          callFibm1.GetMemoryStateOutput() });

    auto gepnm1 = GetElementPtrOperation::Create(
        resultev->argument(0),
        { nm1 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm1 = LoadNonVolatileNode::Create(
        gepnm1,
        { callFibm2.GetMemoryStateOutput() },
        jlm::rvsdg::bittype::Create(64),
        8);

    auto gepnm2 = GetElementPtrOperation::Create(
        resultev->argument(0),
        { nm2 },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto ldnm2 =
        LoadNonVolatileNode::Create(gepnm2, { ldnm1[1] }, jlm::rvsdg::bittype::Create(64), 8);

    auto sum = jlm::rvsdg::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->add_exitvar({ sum, nev->argument(1) });
    auto gOIoState =
        gammaNode->add_exitvar({ callFibm2.GetIoStateOutput(), gIIoState->argument(1) });
    auto gOMemoryState = gammaNode->add_exitvar({ ldnm2[1], gIMemoryState->argument(1) });

    auto gepn = GetElementPtrOperation::Create(
        pointerArgument,
        { valueArgument },
        jlm::rvsdg::bittype::Create(64),
        pbit64);
    auto store = StoreNonVolatileNode::Create(gepn, sumex, { gOMemoryState }, 8);

    auto lambdaOutput = lambda->finalize({ gOIoState, store[0] });

    fibrv->result()->divert_to(lambdaOutput);
    auto phiNode = pb.end();

    return std::make_tuple(phiNode, lambdaOutput, gammaNode, &callFibm1, &callFibm2);
  };

  auto SetupTestFunction = [&](phi::node * phiNode)
  {
    auto at = arraytype::Create(jlm::rvsdg::bittype::Create(64), 10);
    auto pbit64 = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto fibcv = lambda->add_ctxvar(phiNode->output(0));

    auto ten = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 10);
    auto allocaResults = alloca_op::create(at, ten, 16);
    auto state = MemoryStateMergeOperation::Create({ allocaResults[1], memoryStateArgument });

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 0);
    auto gep = GetElementPtrOperation::Create(allocaResults[0], { zero, zero }, at, pbit64);

    auto & call =
        CallNode::CreateNode(fibcv, fibFunctionType, { ten, gep, iOStateArgument, state });

    auto lambdaOutput = lambda->finalize(call.Results());
    GraphExport::Create(*lambdaOutput, "test");

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::node_output::node(allocaResults[0]));
  };

  auto [phiNode, fibfct, gammaNode, callFib1, callFib2] = SetupFib();
  auto [testfct, callFib, alloca] = SetupTestFunction(phiNode);

  // Assign nodes
  this->lambda_fib = fibfct->node();
  this->lambda_test = testfct->node();

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

  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto pointerType = PointerType::Create();

  auto constantFunctionType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

  auto recursiveFunctionType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

  auto functionIType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

  auto recFunctionType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupEight = [&]()
  {
    auto lambda = lambda::node::create(
        graph->root(),
        constantFunctionType,
        "eight",
        linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    return lambda->finalize({ constant, iOStateArgument, memoryStateArgument });
  };

  auto SetupI = [&]()
  {
    auto lambda =
        lambda::node::create(graph->root(), functionIType, "i", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto & call = CallNode::CreateNode(
        pointerArgument,
        constantFunctionType,
        { iOStateArgument, memoryStateArgument });

    auto lambdaOutput = lambda->finalize(call.Results());

    return std::make_tuple(lambdaOutput, &call);
  };

  auto SetupA =
      [&](jlm::rvsdg::region & region, phi::rvargument & functionB, phi::rvargument & functionD)
  {
    auto lambda = lambda::node::create(&region, recFunctionType, "a", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto functionBCv = lambda->add_ctxvar(&functionB);
    auto functionDCv = lambda->add_ctxvar(&functionD);

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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(paAlloca[0])));
  };

  auto SetupB = [&](jlm::rvsdg::region & region,
                    phi::cvargument & functionI,
                    phi::rvargument & functionC,
                    phi::cvargument & functionEight)
  {
    auto lambda = lambda::node::create(&region, recFunctionType, "b", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto functionICv = lambda->add_ctxvar(&functionI);
    auto functionCCv = lambda->add_ctxvar(&functionC);
    auto functionEightCv = lambda->add_ctxvar(&functionEight);

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto storeNode = StoreNonVolatileNode::Create(pointerArgument, two, { memoryStateArgument }, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pbAlloca = alloca_op::create(jlm::rvsdg::bittype::Create(32), four, 4);
    auto pbMerge = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ pbAlloca[1], storeNode[0] }));

    auto & callI = CallNode::CreateNode(
        functionICv,
        functionIType,
        { functionEightCv, iOStateArgument, pbMerge });

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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pbAlloca[0])));
  };

  auto SetupC = [&](jlm::rvsdg::region & region, phi::rvargument & functionA)
  {
    auto lambda = lambda::node::create(&region, recFunctionType, "c", linkage::external_linkage);
    auto xArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto functionACv = lambda->add_ctxvar(&functionA);

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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pcAlloca[0])));
  };

  auto SetupD = [&](jlm::rvsdg::region & region, phi::rvargument & functionA)
  {
    auto lambda = lambda::node::create(&region, recFunctionType, "d", linkage::external_linkage);
    auto xArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto functionACv = lambda->add_ctxvar(&functionA);

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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pdAlloca[0])));
  };

  auto SetupPhi = [&](lambda::output & lambdaEight, lambda::output & lambdaI)
  {
    jlm::llvm::phi::builder phiBuilder;
    phiBuilder.begin(graph->root());
    auto lambdaARv = phiBuilder.add_recvar(pointerType);
    auto lambdaBRv = phiBuilder.add_recvar(pointerType);
    auto lambdaCRv = phiBuilder.add_recvar(pointerType);
    auto lambdaDRv = phiBuilder.add_recvar(pointerType);
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

    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(graph->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto functionACv = lambda->add_ctxvar(&functionA);

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
        jlm::util::AssertedCast<jlm::rvsdg::simple_node>(
            jlm::rvsdg::node_output::node(pTestAlloca[0])));
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
  this->LambdaEight_ = lambdaEight->node();
  this->LambdaI_ = lambdaI->node();
  this->LambdaA_ = jlm::util::AssertedCast<lambda::node>(
      jlm::rvsdg::node_output::node(lambdaA->result()->origin()));
  this->LambdaB_ = jlm::util::AssertedCast<lambda::node>(
      jlm::rvsdg::node_output::node(lambdaB->result()->origin()));
  this->LambdaC_ = jlm::util::AssertedCast<lambda::node>(
      jlm::rvsdg::node_output::node(lambdaC->result()->origin()));
  this->LambdaD_ = jlm::util::AssertedCast<lambda::node>(
      jlm::rvsdg::node_output::node(lambdaD->result()->origin()));
  this->LambdaTest_ = lambdaTest->node();

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

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, structDeclaration);
  auto arrayType = arraytype::Create(structType, 2);

  jlm::llvm::phi::builder pb;
  pb.begin(rvsdg.root());
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
  auto ft = FunctionType::Create(
      { PointerType::Create(), PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  /**
   * Setup function f.
   */
  LambdaF = lambda::node::create(graph->root(), ft, "f", linkage::external_linkage);
  auto x = LambdaF->fctargument(0);
  auto y = LambdaF->fctargument(1);
  auto state = LambdaF->fctargument(2);

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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupDeltaA = [&]()
  {
    auto deltaNode = delta::node::Create(
        rvsdg->root(),
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
        rvsdg->root(),
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

    auto deltaNode =
        delta::node::Create(rvsdg->root(), pointerType, "x", linkage::external_linkage, "", false);

    auto contextVariableA = deltaNode->add_ctxvar(&deltaA);

    return deltaNode->finalize(contextVariableA);
  };

  auto SetupDeltaY = [&](delta::output & deltaX)
  {
    auto pointerType = PointerType::Create();

    auto deltaNode =
        delta::node::Create(rvsdg->root(), pointerType, "y", linkage::external_linkage, "", false);

    auto contextVariableX = deltaNode->add_ctxvar(&deltaX);

    auto deltaOutput = deltaNode->finalize(contextVariableX);
    GraphExport::Create(*deltaOutput, "y");

    return deltaOutput;
  };

  auto SetupLambdaTest = [&](delta::output & deltaB)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(rvsdg->root(), functionType, "test", linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);

    auto contextVariableB = lambda->add_ctxvar(&deltaB);

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
            jlm::rvsdg::node_output::node(loadResults1[0])));
  };

  auto deltaA = SetupDeltaA();
  auto deltaB = SetupDeltaB();
  auto deltaX = SetupDeltaX(*deltaA);
  auto deltaY = SetupDeltaY(*deltaX);
  auto [lambdaTest, loadNode1] = SetupLambdaTest(*deltaB);

  /*
   * Assign nodes
   */
  this->LambdaTest = lambdaTest->node();

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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();

  auto externalFunction1Type = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  auto externalFunction2Type = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

  auto SetupExternalFunction1Declaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunction1Type,
        "ExternalFunction1",
        linkage::external_linkage);
  };

  auto SetupExternalFunction2Declaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunction2Type,
        "ExternalFunction2",
        linkage::external_linkage);
  };

  auto SetupReturnAddressFunction = [&]()
  {
    PointerType p8;
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(
        rvsdg->root(),
        functionType,
        "ReturnAddress",
        linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto eight = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    auto mallocResults = malloc_op::create(eight);
    auto mergeResults = MemoryStateMergeOperation::Create(
        std::vector<jlm::rvsdg::output *>({ memoryStateArgument, mallocResults[1] }));

    auto lambdaOutput = lambda->finalize({ mallocResults[0], iOStateArgument, mergeResults });

    GraphExport::Create(*lambdaOutput, "ReturnAddress");

    return std::make_tuple(lambdaOutput, jlm::rvsdg::node_output::node(mallocResults[0]));
  };

  auto SetupCallExternalFunction1 = [&](jlm::rvsdg::argument * externalFunction1Argument)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(
        rvsdg->root(),
        functionType,
        "CallExternalFunction1",
        linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto externalFunction1 = lambda->add_ctxvar(externalFunction1Argument);

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

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::node_output::node(mallocResults[0]));
  };

  auto SetupCallExternalFunction2 = [&](jlm::rvsdg::argument * externalFunction2Argument)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(
        rvsdg->root(),
        functionType,
        "CallExternalFunction2",
        linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto externalFunction2 = lambda->add_ctxvar(externalFunction2Argument);

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
            jlm::rvsdg::node_output::node(loadResults[0])));
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
  this->ReturnAddressFunction = returnAddressFunction->node();
  this->CallExternalFunction1 = callExternalFunction1->node();
  this->CallExternalFunction2 = callExternalFunction2->node();

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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto externalFunctionType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

  auto SetupExternalFunctionDeclaration = [&]()
  {
    return &GraphImport::Create(
        *rvsdg,
        externalFunctionType,
        "externalFunction",
        linkage::external_linkage);
  };

  auto SetupGlobal = [&]()
  {
    auto delta = delta::node::Create(
        rvsdg->root(),
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

  auto SetupTestFunction = [&](jlm::rvsdg::argument * externalFunctionArgument)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(rvsdg->root(), functionType, "test", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto externalFunction = lambda->add_ctxvar(externalFunctionArgument);

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
            jlm::rvsdg::node_output::node(loadResults[0])));
  };

  auto importExternalFunction = SetupExternalFunctionDeclaration();
  auto deltaGlobal = SetupGlobal();
  auto [lambdaTest, callExternalFunction, loadNode] = SetupTestFunction(importExternalFunction);

  // Assign nodes
  this->LambdaTest = lambdaTest->node();
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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto arrayType = arraytype::Create(jlm::rvsdg::bittype::Create(32), 5);

  auto SetupLocalArray = [&]()
  {
    auto delta = delta::node::Create(
        rvsdg->root(),
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
        rvsdg->root(),
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
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto globalArrayArgument = lambda->add_ctxvar(&globalArray);

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
      [&](delta::output & localArray, delta::output & globalArray, lambda::output & lambdaF)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { jlm::rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "g", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto localArrayArgument = lambda->add_ctxvar(&localArray);
    auto globalArrayArgument = lambda->add_ctxvar(&globalArray);
    auto functionFArgument = lambda->add_ctxvar(&lambdaF);

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
        lambdaF.node()->Type(),
        { iOStateArgument, memcpyResults[0] });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "g");

    return std::make_tuple(lambdaOutput, &call, jlm::rvsdg::node_output::node(memcpyResults[0]));
  };

  auto localArray = SetupLocalArray();
  auto globalArray = SetupGlobalArray();
  auto lambdaF = SetupFunctionF(*globalArray);
  auto [lambdaG, callF, memcpyNode] = SetupFunctionG(*localArray, *globalArray, *lambdaF);

  /*
   * Assign nodes
   */
  this->LambdaF_ = lambdaF->node();
  this->LambdaG_ = lambdaG->node();
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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto arrayType = arraytype::Create(PointerType::Create(), 32);
  auto & structBDeclaration =
      rvsdgModule->AddStructTypeDeclaration(StructType::Declaration::Create({ arrayType }));
  auto structTypeB = StructType::Create("structTypeB", false, structBDeclaration);

  auto SetupFunctionG = [&]()
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "g", linkage::internal_linkage);
    auto s1Argument = lambda->fctargument(0);
    auto s2Argument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

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

    return std::make_tuple(lambdaOutput, jlm::rvsdg::node_output::node(memcpyResults[0]));
  };

  auto SetupFunctionF = [&](lambda::output & functionF)
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { PointerType::Create(),
          PointerType::Create(),
          iostatetype::Create(),
          MemoryStateType::Create() },
        { iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
    auto s1Argument = lambda->fctargument(0);
    auto s2Argument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);

    auto functionFArgument = lambda->add_ctxvar(&functionF);

    auto c0 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);

    auto gepS1 = GetElementPtrOperation::Create(s1Argument, { c0, c0 }, structTypeB, pointerType);
    auto ldS1 = LoadNonVolatileNode::Create(gepS1, { memoryStateArgument }, pointerType, 8);

    auto gepS2 = GetElementPtrOperation::Create(s2Argument, { c0, c0 }, structTypeB, pointerType);
    auto ldS2 = LoadNonVolatileNode::Create(gepS2, { ldS1[1] }, pointerType, 8);

    auto & call = CallNode::CreateNode(
        functionFArgument,
        functionF.node()->Type(),
        { ldS1[0], ldS2[0], iOStateArgument, ldS2[1] });

    auto lambdaOutput = lambda->finalize(call.Results());

    GraphExport::Create(*lambdaOutput, "f");

    return std::make_tuple(lambdaOutput, &call);
  };

  auto [lambdaG, memcpyNode] = SetupFunctionG();
  auto [lambdaF, callG] = SetupFunctionF(*lambdaG);

  this->LambdaF_ = lambdaF->node();
  this->LambdaG_ = lambdaG->node();
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

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto & declaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("myStruct", false, declaration);

  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto functionType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  Lambda_ = lambda::node::create(rvsdg->root(), functionType, "f", linkage::internal_linkage);
  auto pArgument = Lambda_->fctargument(0);
  auto iOStateArgument = Lambda_->fctargument(1);
  auto memoryStateArgument = Lambda_->fctargument(2);

  auto eight = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, 8);
  auto zero = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 0);
  auto minusFive = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, -5);
  auto three = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 64, 3);

  auto allocaResults = alloca_op::create(structType, eight, 8);
  auto memoryState = MemoryStateMergeOperation::Create({ allocaResults[1], memoryStateArgument });

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

  Alloca_ = rvsdg::node_output::node(allocaResults[0]);
  Memcpy_ = rvsdg::node_output::node(memcpyResults[0]);

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LinkedListTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto & declaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ PointerType::Create() }));
  auto structType = StructType::Create("list", false, declaration);

  auto SetupDeltaMyList = [&]()
  {
    auto delta = delta::node::Create(
        rvsdg.root(),
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
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(rvsdg.root(), functionType, "next", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto myListArgument = lambda->add_ctxvar(&myList);

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto alloca = alloca_op::create(pointerType, size, 4);
    auto mergedMemoryState = MemoryStateMergeOperation::Create({ alloca[1], memoryStateArgument });

    auto load1 = LoadNonVolatileNode::Create(myListArgument, { mergedMemoryState }, pointerType, 4);
    auto store1 = StoreNonVolatileNode::Create(alloca[0], load1[0], { load1[1] }, 4);

    auto load2 = LoadNonVolatileNode::Create(alloca[0], { store1[0] }, pointerType, 4);
    auto gep = GetElementPtrOperation::Create(load2[0], { zero, zero }, structType, pointerType);

    auto load3 = LoadNonVolatileNode::Create(gep, { load2[1] }, pointerType, 4);
    auto store2 = StoreNonVolatileNode::Create(alloca[0], load3[0], { load3[1] }, 4);

    auto load4 = LoadNonVolatileNode::Create(alloca[0], { store2[0] }, pointerType, 4);

    auto lambdaOutput = lambda->finalize({ load4[0], iOStateArgument, load4[1] });
    GraphExport::Create(*lambdaOutput, "next");

    return std::make_tuple(jlm::rvsdg::node_output::node(alloca[0]), lambdaOutput);
  };

  auto deltaMyList = SetupDeltaMyList();
  auto [alloca, lambdaNext] = SetupFunctionNext(*deltaMyList);

  /*
   * Assign nodes
   */
  this->DeltaMyList_ = deltaMyList->node();
  this->LambdaNext_ = lambdaNext->node();
  this->Alloca_ = alloca;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
AllMemoryNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto mt = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  // Create imported symbol "imported"
  Import_ = &GraphImport::Create(
      *graph,
      rvsdg::bittype::Create(32),
      "imported",
      linkage::external_linkage);

  // Create global variable "global"
  Delta_ = delta::node::Create(
      graph->root(),
      pointerType,
      "global",
      linkage::external_linkage,
      "",
      false);
  auto constantPointerNullResult =
      ConstantPointerNullOperation::Create(Delta_->subregion(), pointerType);
  Delta_->finalize(constantPointerNullResult);

  // Start of function "f"
  Lambda_ = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);
  auto entryMemoryState = Lambda_->fctargument(0);
  auto deltaContextVar = Lambda_->add_ctxvar(Delta_->output());
  auto importContextVar = Lambda_->add_ctxvar(Import_);

  // Create alloca node
  auto allocaSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 1);
  auto allocaOutputs = alloca_op::create(pointerType, allocaSize, 8);
  Alloca_ = jlm::rvsdg::node_output::node(allocaOutputs[0]);

  auto afterAllocaMemoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::output *>{ entryMemoryState, allocaOutputs[1] });

  // Create malloc node
  auto mallocSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 4);
  auto mallocOutputs = malloc_op::create(mallocSize);
  Malloc_ = jlm::rvsdg::node_output::node(mallocOutputs[0]);

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
  auto fcttype = FunctionType::Create({ MemoryStateType::Create() }, { MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  Function_ = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto allocaSize = jlm::rvsdg::create_bitconstant(Function_->subregion(), 32, 1);

  jlm::rvsdg::output * latestMemoryState = Function_->fctargument(0);

  for (size_t i = 0; i < NumAllocaNodes_; i++)
  {
    auto allocaOutputs = alloca_op::create(jlm::rvsdg::bittype::Create(32), allocaSize, 4);
    auto allocaNode = jlm::rvsdg::node_output::node(allocaOutputs[0]);

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
  auto localFuncType = FunctionType::Create(
      { PointerType::Create(), MemoryStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create() });
  auto exportedFuncType = FunctionType::Create(
      { MemoryStateType::Create() },
      { PointerType::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(util::filepath(""), "", "");
  const auto graph = &module->Rvsdg();
  graph->node_normal_form(typeid(rvsdg::operation))->set_mutable(false);

  Global_ = delta::node::Create(
      graph->root(),
      uint32Type,
      "global",
      linkage::internal_linkage,
      "",
      false);
  const auto constantZero = rvsdg::create_bitconstant(Global_->subregion(), 32, 0);
  const auto deltaOutput = Global_->finalize(constantZero);

  LocalFunc_ = lambda::node::create(
      graph->root(),
      localFuncType,
      "localFunction",
      linkage::internal_linkage);

  LocalFuncParam_ = LocalFunc_->fctargument(0);

  const auto allocaSize = rvsdg::create_bitconstant(LocalFunc_->subregion(), 32, 1);
  const auto allocaOutputs = alloca_op::create(uint32Type, allocaSize, 4);
  LocalFuncParamAllocaNode_ = rvsdg::node_output::node(allocaOutputs[0]);

  // Merge function's input Memory State and alloca node's memory state
  rvsdg::output * mergedMemoryState = MemoryStateMergeOperation::Create(
      std::vector<rvsdg::output *>{ LocalFunc_->fctargument(1), allocaOutputs[1] });

  // Store the function parameter into the alloca node
  auto storeOutputs =
      StoreNonVolatileNode::Create(allocaOutputs[0], LocalFuncParam_, { mergedMemoryState }, 4);

  // Bring in deltaOuput as a context variable
  const auto deltaOutputCtxVar = LocalFunc_->add_ctxvar(deltaOutput);

  // Return &global
  LocalFunc_->finalize({ deltaOutputCtxVar, storeOutputs[0] });

  LocalFuncRegister_ = LocalFunc_->output();

  ExportedFunc_ = lambda::node::create(
      graph->root(),
      exportedFuncType,
      "exportedFunc",
      linkage::external_linkage);

  const auto localFuncCtxVar = ExportedFunc_->add_ctxvar(LocalFuncRegister_);

  // Return &localFunc, pass memory state directly through
  ExportedFunc_->finalize({ localFuncCtxVar, ExportedFunc_->fctargument(0) });

  GraphExport::Create(*ExportedFunc_->output(), "exportedFunc");

  return module;
}

std::unique_ptr<llvm::RvsdgModule>
FreeNullTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto functionType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  LambdaMain_ =
      lambda::node::create(graph->root(), functionType, "main", linkage::external_linkage);
  auto iOStateArgument = LambdaMain_->fctargument(0);
  auto memoryStateArgument = LambdaMain_->fctargument(1);

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
  rvsdg.node_normal_form(typeid(rvsdg::operation))->set_mutable(false);

  auto setupLambdaG = [&]()
  {
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto functionType = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda = lambda::node::create(rvsdg.root(), functionType, "g", linkage::internal_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);

    auto five = rvsdg::create_bitconstant(lambda->subregion(), 32, 5);

    return lambda->finalize({ five, iOStateArgument, memoryStateArgument });
  };

  auto setupLambdaMain = [&](lambda::output & lambdaG)
  {
    auto pointerType = PointerType::Create();
    auto iOStateType = iostatetype::Create();
    auto memoryStateType = MemoryStateType::Create();
    auto variableArgumentType = varargtype::Create();
    auto functionTypeMain = FunctionType::Create(
        { iostatetype::Create(), MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });
    auto functionTypeCall = FunctionType::Create(
        { rvsdg::bittype::Create(32),
          variableArgumentType,
          iostatetype::Create(),
          MemoryStateType::Create() },
        { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

    auto lambda =
        lambda::node::create(rvsdg.root(), functionTypeMain, "main", linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto lambdaGArgument = lambda->add_ctxvar(&lambdaG);

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

  LambdaG_ = setupLambdaG()->node();
  auto [lambdaMainOutput, call] = setupLambdaMain(*LambdaG_->output());
  LambdaMain_ = lambdaMainOutput->node();
  Call_ = call;

  return rvsdgModule;
}

std::unique_ptr<llvm::RvsdgModule>
VariadicFunctionTest1::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  rvsdg.node_normal_form(typeid(rvsdg::operation))->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto varArgType = varargtype::Create();
  auto lambdaHType = FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32),
        varArgType,
        iostatetype::Create(),
        MemoryStateType::Create() },
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaFType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaGType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });

  // Setup h()
  ImportH_ = &GraphImport::Create(rvsdg, lambdaHType, "h", linkage::external_linkage);

  // Setup f()
  {
    LambdaF_ = lambda::node::create(rvsdg.root(), lambdaFType, "f", linkage::internal_linkage);
    auto iArgument = LambdaF_->fctargument(0);
    auto iOStateArgument = LambdaF_->fctargument(1);
    auto memoryStateArgument = LambdaF_->fctargument(2);
    auto lambdaHArgument = LambdaF_->add_ctxvar(ImportH_);

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
    LambdaG_ = lambda::node::create(rvsdg.root(), lambdaGType, "g", linkage::external_linkage);
    auto iOStateArgument = LambdaG_->fctargument(0);
    auto memoryStateArgument = LambdaG_->fctargument(1);
    auto lambdaFArgument = LambdaG_->add_ctxvar(LambdaF_->output());

    auto one = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 1);
    auto five = jlm::rvsdg::create_bitconstant(LambdaG_->subregion(), 32, 5);

    auto allocaResults = alloca_op::create(jlm::rvsdg::bittype::Create(32), one, 4);
    auto merge = MemoryStateMergeOperation::Create({ allocaResults[1], memoryStateArgument });
    AllocaNode_ = rvsdg::node_output::node(allocaResults[0]);

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

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto pointerType = PointerType::Create();
  auto & structDeclaration = rvsdgModule->AddStructTypeDeclaration(
      StructType::Declaration::Create({ rvsdg::bittype::Create(32),
                                        rvsdg::bittype::Create(32),
                                        PointerType::Create(),
                                        PointerType::Create() }));
  auto structType = StructType::Create("struct.__va_list_tag", false, structDeclaration);
  auto arrayType = arraytype::Create(structType, 1);
  auto iOStateType = iostatetype::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto varArgType = varargtype::Create();
  auto lambdaLlvmLifetimeStartType = FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaLlvmLifetimeEndType = FunctionType::Create(
      { rvsdg::bittype::Create(64),
        PointerType::Create(),
        iostatetype::Create(),
        MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaVaStartType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaVaEndType = FunctionType::Create(
      { PointerType::Create(), iostatetype::Create(), MemoryStateType::Create() },
      { iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaFstType = FunctionType::Create(
      { rvsdg::bittype::Create(32), varArgType, iostatetype::Create(), MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });
  auto lambdaGType = FunctionType::Create(
      { iostatetype::Create(), MemoryStateType::Create() },
      { rvsdg::bittype::Create(32), iostatetype::Create(), MemoryStateType::Create() });

  auto llvmLifetimeStart =
      &GraphImport::Create(rvsdg, pointerType, "llvm.lifetime.start.p0", linkage::external_linkage);
  auto llvmLifetimeEnd =
      &GraphImport::Create(rvsdg, pointerType, "llvm.lifetime.end.p0", linkage::external_linkage);
  auto llvmVaStart =
      &GraphImport::Create(rvsdg, pointerType, "llvm.va_start", linkage::external_linkage);
  auto llvmVaEnd =
      &GraphImport::Create(rvsdg, pointerType, "llvm.va_end", linkage::external_linkage);

  // Setup function fst()
  {
    LambdaFst_ =
        lambda::node::create(rvsdg.root(), lambdaFstType, "fst", linkage::internal_linkage);
    auto iOStateArgument = LambdaFst_->fctargument(2);
    auto memoryStateArgument = LambdaFst_->fctargument(3);
    auto llvmLifetimeStartArgument = LambdaFst_->add_ctxvar(llvmLifetimeStart);
    auto llvmLifetimeEndArgument = LambdaFst_->add_ctxvar(llvmLifetimeEnd);
    auto llvmVaStartArgument = LambdaFst_->add_ctxvar(llvmVaStart);
    auto llvmVaEndArgument = LambdaFst_->add_ctxvar(llvmVaEnd);

    auto one = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 32, 1);
    auto twentyFour = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 64, 24);
    auto fortyOne = jlm::rvsdg::create_bitconstant(LambdaFst_->subregion(), 32, 41);

    auto allocaResults = alloca_op::create(arrayType, one, 16);
    auto memoryState = MemoryStateMergeOperation::Create({ allocaResults[1], memoryStateArgument });
    AllocaNode_ = rvsdg::node_output::node(allocaResults[0]);

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

    auto gammaNode = rvsdg::gamma_node::create(matchResult, 2);
    auto gammaVaAddress = gammaNode->add_entryvar(allocaResults[0]);
    auto gammaLoadResult = gammaNode->add_entryvar(loadResults[0]);
    auto gammaMemoryState = gammaNode->add_entryvar(loadResults[1]);

    // gamma subregion 0
    auto zero = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 0);
    auto two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 32, 2);
    auto eight = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 8);
    auto gepResult1 = GetElementPtrOperation::Create(
        gammaVaAddress->argument(0),
        { zero, two },
        structType,
        pointerType);
    auto loadResultsGamma0 =
        LoadNonVolatileNode::Create(gepResult1, { gammaMemoryState->argument(0) }, pointerType, 8);
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
        gammaVaAddress->argument(1),
        { zero, three },
        structType,
        pointerType);
    auto loadResultsGamma1 =
        LoadNonVolatileNode::Create(gepResult1, { gammaMemoryState->argument(1) }, pointerType, 16);
    auto & zextResult = zext_op::Create(*gammaLoadResult->argument(1), rvsdg::bittype::Create(64));
    gepResult2 = GetElementPtrOperation::Create(
        loadResultsGamma1[0],
        { &zextResult },
        rvsdg::bittype::Create(8),
        pointerType);
    auto addResult = rvsdg::bitadd_op::create(32, gammaLoadResult->argument(1), eightBit32);
    auto storeResultsGamma1 = StoreNonVolatileNode::Create(
        gammaVaAddress->argument(1),
        addResult,
        { loadResultsGamma1[1] },
        16);

    auto gammaAddress = gammaNode->add_exitvar({ loadResultsGamma0[0], gepResult2 });
    auto gammaOutputMemoryState =
        gammaNode->add_exitvar({ storeResultsGamma0[0], storeResultsGamma1[0] });

    loadResults = LoadNonVolatileNode::Create(
        gammaAddress,
        { gammaOutputMemoryState },
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
    LambdaG_ = lambda::node::create(rvsdg.root(), lambdaGType, "g", linkage::external_linkage);
    auto iOStateArgument = LambdaG_->fctargument(0);
    auto memoryStateArgument = LambdaG_->fctargument(1);
    auto lambdaFstArgument = LambdaG_->add_ctxvar(LambdaFst_->output());

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
