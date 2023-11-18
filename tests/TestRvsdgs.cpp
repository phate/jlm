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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto d = alloca_op::create(jlm::rvsdg::bit32, csize, 4);
  auto c = alloca_op::create(pointerType, csize, 4);
  auto b = alloca_op::create(pointerType, csize, 4);
  auto a = alloca_op::create(pointerType, csize, 4);

  auto merge_d = MemStateMergeOperator::Create({d[1], fct->fctargument(0)});
  auto merge_c = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({c[1], merge_d}));
  auto merge_b = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({b[1], merge_c}));
  auto merge_a = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({a[1], merge_b}));

  auto a_amp_b = StoreNode::Create(a[0], b[0], {merge_a}, 4);
  auto b_amp_c = StoreNode::Create(b[0], c[0], {a_amp_b[0]}, 4);
  auto c_amp_d = StoreNode::Create(c[0], d[0], {b_amp_c[0]}, 4);

  fct->finalize({c_amp_d[0]});

  graph->add_export(fct->output(), {pointerType, "f"});

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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bit32, csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bit32, csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemStateMergeOperator::Create({a[1], fct->fctargument(0)});
  auto merge_b = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({b[1], merge_a}));
  auto merge_x = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({x[1], merge_b}));
  auto merge_y = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({y[1], merge_x}));
  auto merge_p = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({p[1], merge_y}));

  auto x_amp_a = StoreNode::Create(x[0], a[0], {merge_p}, 4);
  auto y_amp_b = StoreNode::Create(y[0], b[0], {x_amp_a[0]}, 4);
  auto p_amp_x = StoreNode::Create(p[0], x[0], {y_amp_b[0]}, 4);
  auto p_amp_y = StoreNode::Create(p[0], y[0], {p_amp_x[0]}, 4);

  fct->finalize({p_amp_y[0]});

  graph->add_export(fct->output(), {pointerType, "f"});

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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&pointerType, &mt}, {&jlm::rvsdg::bit32, &mt});

  auto module = RvsdgModule::Create(jlm::util::filepath("LoadTest1.c"), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto ld1 = LoadNode::Create(fct->fctargument(0), {fct->fctargument(1)}, pointerType, 4);
  auto ld2 = LoadNode::Create(ld1[0], {ld1[1]}, jlm::rvsdg::bit32, 4);

  fct->finalize(ld2);

  graph->add_export(fct->output(), {pointerType, "f"});

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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jlm::rvsdg::bit32, csize, 4);
  auto b = alloca_op::create(jlm::rvsdg::bit32, csize, 4);
  auto x = alloca_op::create(pointerType, csize, 4);
  auto y = alloca_op::create(pointerType, csize, 4);
  auto p = alloca_op::create(pointerType, csize, 4);

  auto merge_a = MemStateMergeOperator::Create({a[1], fct->fctargument(0)});
  auto merge_b = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({b[1], merge_a}));
  auto merge_x = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({x[1], merge_b}));
  auto merge_y = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({y[1], merge_x}));
  auto merge_p = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({p[1], merge_y}));

  auto x_amp_a = StoreNode::Create(x[0], a[0], {merge_p}, 4);
  auto y_amp_b = StoreNode::Create(y[0], b[0], x_amp_a, 4);
  auto p_amp_x = StoreNode::Create(p[0], x[0], y_amp_b, 4);

  auto ld1 = LoadNode::Create(p[0], p_amp_x, pointerType, 4);
  auto ld2 = LoadNode::Create(ld1[0], {ld1[1]}, pointerType, 4);
  auto y_star_p = StoreNode::Create(y[0], ld2[0], {ld2[1]}, 4);

  fct->finalize({y_star_p[0]});

  graph->add_export(fct->output(), {pointerType, "f"});

  /* extract nodes */

  this->lambda = fct;

  this->size = jlm::rvsdg::node_output::node(csize);

  this->alloca_a = jlm::rvsdg::node_output::node(a[0]);
  this->alloca_b = jlm::rvsdg::node_output::node(b[0]);
  this->alloca_x = jlm::rvsdg::node_output::node(x[0]);
  this->alloca_y = jlm::rvsdg::node_output::node(y[0]);
  this->alloca_p = jlm::rvsdg::node_output::node(p[0]);

  this->load_x = jlm::rvsdg::node_output::node(ld1[0]);
  this->load_a = jlm::rvsdg::node_output::node(ld2[0]);;

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
LoadFromUndefTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  MemoryStateType memoryStateType;
  FunctionType functionType(
    {&memoryStateType},
    {&jlm::rvsdg::bit32, &memoryStateType});
  PointerType pointerType;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  Lambda_ = lambda::node::create(
    rvsdg.root(),
    functionType,
    "f",
    linkage::external_linkage);

  auto undefValue = UndefValueOperation::Create(*Lambda_->subregion(), pointerType);
  auto loadResults = LoadNode::Create(undefValue, {Lambda_->fctargument(0)}, jlm::rvsdg::bit32, 4);

  Lambda_->finalize(loadResults);
  rvsdg.add_export(Lambda_->output(), {pointerType, "f"});

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

  auto dcl = jlm::rvsdg::rcddeclaration::create({&jlm::rvsdg::bit32, &jlm::rvsdg::bit32});
  jlm::rvsdg::rcdtype rt(dcl.get());

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&pointerType, &mt}, {&jlm::rvsdg::bit32, &mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto one = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 1);

  auto gepx = GetElementPtrOperation::Create(fct->fctargument(0), {zero, zero}, rt, pointerType);
  auto ldx = LoadNode::Create(gepx, {fct->fctargument(1)}, jlm::rvsdg::bit32, 4);

  auto gepy = GetElementPtrOperation::Create(fct->fctargument(0), {zero, one}, rt, pointerType);
  auto ldy = LoadNode::Create(gepy, {ldx[1]}, jlm::rvsdg::bit32, 4);

  auto sum = jlm::rvsdg::bitadd_op::create(32, ldx[0], ldy[0]);

  fct->finalize({sum, ldy[1]});

  graph->add_export(fct->output(), {pointerType, "f"});

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

  PointerType pointerType;
  FunctionType fcttype({&pointerType}, {&pointerType});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto cast = bitcast_op::create(fct->fctargument(0), pointerType);

  fct->finalize({cast});

  graph->add_export(fct->output(), {pointerType, "f"});

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
    PointerType pt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&jlm::rvsdg::bit64, &iOStateType, &memoryStateType, &loopStateType},
      {&pt, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "bit2ptr",
      linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto cast = bits2ptr_op::create(valueArgument, pt);

    lambda->finalize({cast, iOStateArgument, memoryStateArgument, loopStateArgument});

    return std::make_tuple(
      lambda,
      jlm::rvsdg::node_output::node(cast));
  };

  auto setupTestFunction = [&](lambda::output * b2p)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&jlm::rvsdg::bit64, &iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto cvbits2ptr = lambda->add_ctxvar(b2p);

    auto callResults = CallNode::Create(
      cvbits2ptr,
      b2p->node()->type(),
      {valueArgument, iOStateArgument, memoryStateArgument, loopStateArgument});

    lambda->finalize({callResults[1], callResults[2], callResults[3]});
    graph->add_export(lambda->output(), {PointerType(), "testfct"});

    return std::make_tuple(
      lambda,
      util::AssertedCast<CallNode>(rvsdg::node_output::node(callResults[0])));
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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&pointerType, &mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto constantPointerNullResult = ConstantPointerNullOperation::Create(fct->subregion(), pointerType);
  auto st = StoreNode::Create(
    fct->fctargument(0),
    constantPointerNullResult,
    {fct->fctargument(1)},
    4);

  fct->finalize({st[0]});

  graph->add_export(fct->output(), {pointerType, "f"});

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

    PointerType pt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pt, &pt, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto pointerArgument1 = lambda->fctargument(0);
    auto pointerArgument2 = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);

    auto ld1 = LoadNode::Create(pointerArgument1, {memoryStateArgument}, jlm::rvsdg::bit32, 4);
    auto ld2 = LoadNode::Create(pointerArgument2, {ld1[1]}, jlm::rvsdg::bit32, 4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({sum, iOStateArgument, ld2[1], loopStateArgument});

    return lambda;
  };

  auto SetupG = [&]()
  {
    PointerType pt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pt, &pt, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "g",
      linkage::external_linkage);
    auto pointerArgument1 = lambda->fctargument(0);
    auto pointerArgument2 = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);

    auto ld1 = LoadNode::Create(pointerArgument1, {memoryStateArgument}, jlm::rvsdg::bit32, 4);
    auto ld2 = LoadNode::Create(pointerArgument2, {ld1[1]}, jlm::rvsdg::bit32, 4);

    auto diff = jlm::rvsdg::bitsub_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({diff, iOStateArgument, ld2[1], loopStateArgument});

    return lambda;
  };

  auto SetupH = [&](lambda::node * f, lambda::node * g)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "h",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvf = lambda->add_ctxvar(f->output());
    auto cvg = lambda->add_ctxvar(g->output());

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto x = alloca_op::create(jlm::rvsdg::bit32, size, 4);
    auto y = alloca_op::create(jlm::rvsdg::bit32, size, 4);
    auto z = alloca_op::create(jlm::rvsdg::bit32, size, 4);

    auto mx = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({x[1], memoryStateArgument}));
    auto my = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({y[1], mx}));
    auto mz = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({z[1], my}));

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 7);

    auto stx = StoreNode::Create(x[0], five, {mz}, 4);
    auto sty = StoreNode::Create(y[0], six, {stx[0]}, 4);
    auto stz = StoreNode::Create(z[0], seven, {sty[0]}, 4);

    auto callFResults = CallNode::Create(
      cvf,
      f->type(),
      {x[0], y[0], iOStateArgument, stz[0], loopStateArgument});
    auto callGResults = CallNode::Create(
      cvg,
      g->type(),
      {z[0], z[0], callFResults[1], callFResults[2], callFResults[3]});

    auto sum = jlm::rvsdg::bitadd_op::create(32, callFResults[0], callGResults[0]);

    lambda->finalize({sum, callGResults[1], callGResults[2], callGResults[3]});
    graph->add_export(lambda->output(), {PointerType(), "h"});

    auto allocaX = jlm::rvsdg::node_output::node(x[0]);
    auto allocaY = jlm::rvsdg::node_output::node(y[0]);
    auto allocaZ = jlm::rvsdg::node_output::node(z[0]);
    auto callF = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callFResults[0]));
    auto callG = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callGResults[0]));

    return std::make_tuple(lambda, allocaX, allocaY, allocaZ, callF, callG);
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
    PointerType pt32;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType},
      {&pt32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "create",
      linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto prod = jlm::rvsdg::bitmul_op::create(32, valueArgument, four);

    auto alloc = malloc_op::create(prod);
    auto cast = bitcast_op::create(alloc[0], pt32);
    auto mx = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>(
      {alloc[1], memoryStateArgument}));

    lambda->finalize({cast, iOStateArgument, mx, loopStateArgument});

    auto mallocNode = jlm::rvsdg::node_output::node(alloc[0]);
    return std::make_tuple(lambda, mallocNode);
  };

  auto SetupDestroy = [&]()
  {
    PointerType pointerType;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "destroy",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto cast = bitcast_op::create(pointerArgument, pointerType);
    auto freeResults = free_op::create(cast, {memoryStateArgument}, iOStateArgument);

    lambda->finalize({freeResults[1], freeResults[0], loopStateArgument});

    auto freeNode = jlm::rvsdg::node_output::node(freeResults[0]);
    return std::make_tuple(lambda, freeNode);
  };

  auto SetupTest = [&](lambda::node * lambdaCreate, lambda::node * lambdaDestroy)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto create_cv = lambda->add_ctxvar(lambdaCreate->output());
    auto destroy_cv = lambda->add_ctxvar(lambdaDestroy->output());

    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 7);

    auto create1 = CallNode::Create(
      create_cv,
      lambdaCreate->type(),
      {six, iOStateArgument, memoryStateArgument, loopStateArgument});
    auto create2 = CallNode::Create(
      create_cv,
      lambdaCreate->type(),
      {seven, create1[1], create1[2], create1[3]});

    auto destroy1 = CallNode::Create(
      destroy_cv,
      lambdaDestroy->type(),
      {create1[0], create2[1], create2[2], create2[3]});
    auto destroy2 = CallNode::Create(
      destroy_cv,
      lambdaDestroy->type(),
      {create2[0], destroy1[0], destroy1[1], destroy1[2]});

    lambda->finalize(destroy2);
    graph->add_export(lambda->output(), {PointerType(), "test"});

    auto callCreate1Node = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(create1[0]));
    auto callCreate2Node = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(create2[0]));
    auto callDestroy1Node = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(destroy1[0]));
    auto callDestroy2Node = jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(destroy2[0]));

    return std::make_tuple(lambda, callCreate1Node, callCreate2Node, callDestroy1Node, callDestroy2Node);
  };

  auto [lambdaCreate, mallocNode] = SetupCreate();
  auto [lambdaDestroy, freeNode] = SetupDestroy();
  auto [lambdaTest, callCreate1, callCreate2, callDestroy1, callDestroy2] = SetupTest(lambdaCreate, lambdaDestroy);

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

  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType constantFunctionType(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});
  PointerType pointerType;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupConstantFunction = [&](ssize_t n, const std::string & name)
  {
    auto lambda = lambda::node::create(
      graph->root(),
      constantFunctionType,
      name,
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({constant, iOStateArgument, memoryStateArgument, loopStateArgument});
  };

  auto SetupIndirectCallFunction = [&]()
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "indcall",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto call = CallNode::Create(
      pointerArgument,
      constantFunctionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto lambdaOutput = lambda->finalize(call);

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call[0])));
  };

  auto SetupTestFunction = [&](
    lambda::output * fctindcall,
    lambda::output * fctthree,
    lambda::output * fctfour)
  {
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto fctindcall_cv = lambda->add_ctxvar(fctindcall);
    auto fctfour_cv = lambda->add_ctxvar(fctfour);
    auto fctthree_cv = lambda->add_ctxvar(fctthree);

    auto call_four = CallNode::Create(
      fctindcall_cv,
      fctindcall->node()->type(),
      {fctfour_cv, iOStateArgument, memoryStateArgument, loopStateArgument});
    auto call_three = CallNode::Create(
      fctindcall_cv,
      fctindcall->node()->type(),
      {fctthree_cv, call_four[1], call_four[2], call_four[3]});

    auto add = jlm::rvsdg::bitadd_op::create(32, call_four[0], call_three[0]);

    auto lambdaOutput = lambda->finalize({add, call_three[1], call_three[2], call_three[3]});
    graph->add_export(lambda->output(), {pointerType, "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call_three[0])),
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call_four[0])));
  };

  auto fctfour = SetupConstantFunction(4, "four");
  auto fctthree = SetupConstantFunction(3, "three");
  auto [fctindcall, callIndirectFunction] = SetupIndirectCallFunction();
  auto [fcttest, callFunctionThree, callFunctionFour] = SetupTestFunction(fctindcall, fctthree, fctfour);

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

  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType constantFunctionType(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});
  PointerType pointerType;

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto SetupG1 = [&]()
  {
    auto delta = delta::node::Create(
      graph->root(),
      jlm::rvsdg::bit32,
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
      jlm::rvsdg::bit32,
      "g2",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 2);

    return delta->finalize(constant);
  };

  auto SetupConstantFunction = [&](ssize_t n, const std::string & name)
  {
    auto lambda = lambda::node::create(
      graph->root(),
      constantFunctionType,
      name,
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);

    return lambda->finalize({constant, iOStateArgument, memoryStateArgument, loopStateArgument});
  };

  auto SetupI = [&]()
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "i",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto call = CallNode::Create(
      pointerArgument,
      constantFunctionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto lambdaOutput = lambda->finalize(call);

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call[0])));
  };

  auto SetupIndirectCallFunction = [&](
    ssize_t n,
    const std::string & name,
    lambda::output & functionI,
    lambda::output & argumentFunction)
  {
    PointerType pointerType;

    FunctionType functionType(
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      name,
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto functionICv = lambda->add_ctxvar(&functionI);
    auto argumentFunctionCv = lambda->add_ctxvar(&argumentFunction);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, n);
    auto storeNode = StoreNode::Create(pointerArgument, five, {memoryStateArgument}, 4);

    auto call = CallNode::Create(
      functionICv,
      functionI.node()->type(),
      {argumentFunctionCv, iOStateArgument, storeNode[0], loopStateArgument});

    auto lambdaOutput = lambda->finalize(call);

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call[0])));
  };

  auto SetupTestFunction = [&](
    lambda::output & functionX,
    lambda::output & functionY,
    delta::output & globalG1,
    delta::output & globalG2)
  {
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto functionXCv = lambda->add_ctxvar(&functionX);
    auto functionYCv = lambda->add_ctxvar(&functionY);
    auto globalG1Cv = lambda->add_ctxvar(&globalG1);
    auto globalG2Cv = lambda->add_ctxvar(&globalG2);

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pxAlloca = alloca_op::create(jlm::rvsdg::bit32, constantSize, 4);
    auto pyAlloca = alloca_op::create(jlm::rvsdg::bit32, constantSize, 4);

    auto pxMerge = MemStateMergeOperator::Create({pxAlloca[1], memoryStateArgument});
    auto pyMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({pyAlloca[1], pxMerge}));

    auto callX = CallNode::Create(
      functionXCv,
      functionX.node()->type(),
      {pxAlloca[0], iOStateArgument, pyMerge, loopStateArgument});

    auto callY = CallNode::Create(
      functionYCv,
      functionY.node()->type(),
      {pyAlloca[0], iOStateArgument, callX[2], loopStateArgument});

    auto loadG1 = LoadNode::Create(globalG1Cv, {callY[2]}, jlm::rvsdg::bit32, 4);
    auto loadG2 = LoadNode::Create(globalG2Cv, {loadG1[1]}, jlm::rvsdg::bit32, 4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, callX[0], callY[0]);
    sum = jlm::rvsdg::bitadd_op::create(32, sum, loadG1[0]);
    sum = jlm::rvsdg::bitadd_op::create(32, sum, loadG2[0]);

    auto lambdaOutput = lambda->finalize({sum, callY[1], callY[2], callY[3]});
    graph->add_export(lambdaOutput, {PointerType(), "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callX[0])),
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callY[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pxAlloca[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pyAlloca[0])));
  };

  auto SetupTest2Function = [&](
    lambda::output & functionX)
  {
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test2",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto constantSize = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto pzAlloca = alloca_op::create(jlm::rvsdg::bit32, constantSize, 4);
    auto pzMerge = MemStateMergeOperator::Create({pzAlloca[1], memoryStateArgument});

    auto functionXCv = lambda->add_ctxvar(&functionX);

    auto callX = CallNode::Create(
      functionXCv,
      functionX.node()->type(),
      {pzAlloca[0], iOStateArgument, pzMerge, loopStateArgument});

    auto lambdaOutput = lambda->finalize(callX);
    graph->add_export(lambdaOutput, {PointerType(), "test2"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callX[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pzAlloca[0])));
  };

  auto deltaG1 = SetupG1();
  auto deltaG2 = SetupG2();
  auto lambdaThree = SetupConstantFunction(3, "three");
  auto lambdaFour = SetupConstantFunction(4, "four");
  auto [lambdaI, indirectCall] = SetupI();
  auto [lambdaX, callIWithThree] = SetupIndirectCallFunction(5, "x", *lambdaI, *lambdaThree);
  auto [lambdaY, callIWithFour] = SetupIndirectCallFunction(6, "y", *lambdaI, *lambdaFour);
  auto [lambdaTest, testCallX, callY, allocaPx, allocaPy] = SetupTestFunction(*lambdaX, *lambdaY, *deltaG1, *deltaG2);
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
ExternalCallTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto nf = rvsdg->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  PointerType pointerType;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType functionGType(
    {&pointerType, &pointerType, &iOStateType, &memoryStateType, &loopStateType},
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupFunctionGDeclaration = [&]()
  {
    return rvsdg->add_import(impport(
      functionGType,
      "g",
      linkage::external_linkage));
  };

  auto SetupFunctionF = [&](jlm::rvsdg::argument * functionG)
  {
    PointerType pointerType;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pointerType, &pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto pathArgument = lambda->fctargument(0);
    auto modeArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);

    auto functionGCv = lambda->add_ctxvar(functionG);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaPath = alloca_op::create(pointerType, size, 4);
    auto allocaMode = alloca_op::create(pointerType, size, 4);

    auto mergePath = MemStateMergeOperator::Create({allocaPath[1], memoryStateArgument});
    auto mergeMode = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({allocaMode[1], mergePath}));

    auto storePath = StoreNode::Create(allocaPath[0], pathArgument, {mergeMode}, 4);
    auto storeMode = StoreNode::Create(allocaMode[0], modeArgument, {storePath[0]}, 4);

    auto loadPath = LoadNode::Create(allocaPath[0], storeMode, pointerType, 4);
    auto loadMode = LoadNode::Create(allocaMode[0], {loadPath[1]}, pointerType, 4);

    auto callGResults = CallNode::Create(
      functionGCv,
      functionGType,
      {loadPath[0], loadMode[0], iOStateArgument, loadMode[1], loopStateArgument});

    lambda->finalize(callGResults);
    rvsdg->add_export(lambda->output(), {pointerType, "f"});

    return std::make_tuple(
      lambda,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callGResults[0])));
  };

  auto externalFunction = SetupFunctionGDeclaration();
  auto [lambdaF, callG] = SetupFunctionF(externalFunction);

  this->LambdaF_ = lambdaF;
  this->CallG_ = callG;

  return rvsdgModule;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
GammaTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  MemoryStateType mt;
  PointerType pt;
  FunctionType fcttype(
    {&jlm::rvsdg::bit32, &pt, &pt, &pt, &pt, &mt},
    {&jlm::rvsdg::bit32, &mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 0);
  auto biteq = jlm::rvsdg::biteq_op::create(32, fct->fctargument(0), zero);
  auto predicate = jlm::rvsdg::match(1, {{0, 1}}, 0, 2, biteq);

  auto gammanode = jlm::rvsdg::gamma_node::create(predicate, 2);
  auto p1ev = gammanode->add_entryvar(fct->fctargument(1));
  auto p2ev = gammanode->add_entryvar(fct->fctargument(2));
  auto p3ev = gammanode->add_entryvar(fct->fctargument(3));
  auto p4ev = gammanode->add_entryvar(fct->fctargument(4));

  auto tmp1 = gammanode->add_exitvar({p1ev->argument(0), p3ev->argument(1)});
  auto tmp2 = gammanode->add_exitvar({p2ev->argument(0), p4ev->argument(1)});

  auto ld1 = LoadNode::Create(tmp1, {fct->fctargument(5)}, jlm::rvsdg::bit32, 4);
  auto ld2 = LoadNode::Create(tmp2, {ld1[1]}, jlm::rvsdg::bit32, 4);
  auto sum = jlm::rvsdg::bitadd_op::create(32, ld1[0], ld2[0]);

  fct->finalize({sum, ld2[1]});

  graph->add_export(fct->output(), {PointerType(), "f"});

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
    auto SetupGamma = [](
      rvsdg::output * predicate,
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
      auto loadXResults = LoadNode::Create(
        gammaInputX->argument(0),
        {gammaInputMemoryState->argument(0)},
        jlm::rvsdg::bit32,
        4);

      auto one = rvsdg::create_bitconstant(gammaNode->subregion(0), 32, 1);
      auto storeZRegion0Results = StoreNode::Create(
        gammaInputZ->argument(0),
        one,
        {loadXResults[1]},
        4);

      // gamma subregion 1
      auto loadYResults = LoadNode::Create(
        gammaInputY->argument(1),
        {gammaInputMemoryState->argument(1)},
        jlm::rvsdg::bit32,
        4);

      auto two = rvsdg::create_bitconstant(gammaNode->subregion(1), 32, 2);
      auto storeZRegion1Results = StoreNode::Create(
        gammaInputZ->argument(1),
        two,
        {loadYResults[1]},
        4);

      // finalize gamma
      auto gammaOutputA = gammaNode->add_exitvar({loadXResults[0], loadYResults[0]});
      auto gammaOutputMemoryState = gammaNode->add_exitvar({storeZRegion0Results[0], storeZRegion1Results[0]});

      return std::make_tuple(
        gammaOutputA,
        gammaOutputMemoryState);
    };

    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    PointerType pointerType;
    FunctionType functionType(
      {&rvsdg::bit32, &pointerType, &pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(rvsdg->root(), functionType, "f", linkage::external_linkage);
    auto cArgument = lambda->fctargument(0);
    auto xArgument = lambda->fctargument(1);
    auto yArgument = lambda->fctargument(2);
    auto iOStateArgument = lambda->fctargument(3);
    auto memoryStateArgument = lambda->fctargument(4);
    auto loopStateArgument = lambda->fctargument(5);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaZResults = alloca_op::create(pointerType, size, 4);

    auto memoryState = MemStateMergeOperator::Create({allocaZResults[1], memoryStateArgument});

    auto nullPointer = ConstantPointerNullOperation::Create(lambda->subregion(), pointerType);
    auto storeZResults = StoreNode::Create(allocaZResults[0], nullPointer, {memoryState}, 4);

    auto zero = rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto bitEq = rvsdg::biteq_op::create(32, cArgument, zero);
    auto predicate = rvsdg::match(1, {{0, 1}}, 0, 2, bitEq);

    auto [gammaOutputA, gammaOutputMemoryState] = SetupGamma(
      predicate,
      xArgument,
      yArgument,
      allocaZResults[0],
      memoryState);

    auto loadZResults = LoadNode::Create(
      allocaZResults[0],
      {gammaOutputMemoryState},
      jlm::rvsdg::bit32,
      4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, gammaOutputA, loadZResults[0]);

    lambda->finalize({sum, iOStateArgument, loadZResults[1], loopStateArgument});

    return std::make_tuple(
      lambda->output(),
      gammaOutputA->node(),
      rvsdg::node_output::node(allocaZResults[0]));
  };

  auto SetupLambdaGH = [&](
    lambda::output & lambdaF,
    int64_t cValue,
    int64_t xValue,
    int64_t yValue,
    const char * functionName)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    PointerType pointerType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(rvsdg->root(), functionType, functionName, linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);
    auto lambdaFArgument = lambda->add_ctxvar(&lambdaF);

    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto allocaXResults = alloca_op::create(rvsdg::bit32, size, 4);
    auto allocaYResults = alloca_op::create(pointerType, size, 4);

    auto memoryState = MemStateMergeOperator::Create({allocaXResults[1], memoryStateArgument});
    memoryState = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>({allocaYResults[1], memoryState}));

    auto predicate = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, cValue);
    auto x = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, xValue);
    auto y = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, yValue);

    auto storeXResults = StoreNode::Create(
      allocaXResults[0],
      x,
      {allocaXResults[1]},
      4);

    auto storeYResults = StoreNode::Create(
      allocaYResults[0],
      y,
      {storeXResults[0]},
      4);

    auto callResults = CallNode::Create(
      lambdaFArgument,
      lambdaF.node()->type(),
      {predicate, allocaXResults[0], allocaYResults[0], iOStateArgument, storeYResults[0], loopStateArgument});

    lambda->finalize(callResults);
    rvsdg->add_export(lambda->output(), {PointerType(), functionName});

    return std::make_tuple(
      lambda->output(),
      util::AssertedCast<CallNode>(rvsdg::node_output::node(callResults[0])),
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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&jlm::rvsdg::bit32, &pointerType, &jlm::rvsdg::bit32, &mt}, {&mt});

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

  auto gepnode = GetElementPtrOperation::Create(a->argument(), {n->argument()}, jlm::rvsdg::bit32, pointerType);
  auto store = StoreNode::Create(gepnode, c->argument(), {s->argument()}, 4);

  auto one = jlm::rvsdg::create_bitconstant(thetanode->subregion(), 32, 1);
  auto sum = jlm::rvsdg::bitadd_op::create(32, n->argument(), one);
  auto cmp = jlm::rvsdg::bitult_op::create(32, sum, l->argument());
  auto predicate = jlm::rvsdg::match(1, {{1, 1}}, 0, 2, cmp);

  n->result()->divert_to(sum);
  s->result()->divert_to(store[0]);
  thetanode->set_predicate(predicate);

  fct->finalize({s});
  graph->add_export(fct->output(), {PointerType(), "f"});

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
      jlm::rvsdg::bit32,
      "f",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(dfNode->subregion(), 32, 0);

    return  dfNode->finalize(constant);
  };

  auto SetupFunctionG = [&]()
  {
    PointerType pt;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pt, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "g",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto ld = LoadNode::Create(pointerArgument, {memoryStateArgument}, jlm::rvsdg::bit32, 4);

    return lambda->finalize({ld[0], iOStateArgument, ld[1], loopStateArgument});
  };

  auto SetupFunctionH = [&](delta::output * f, lambda::output * g)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "h",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvf = lambda->add_ctxvar(f);
    auto cvg = lambda->add_ctxvar(g);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNode::Create(cvf, five, {memoryStateArgument}, 4);
    auto callg = CallNode::Create(
      cvg,
      g->node()->type(),
      {cvf, iOStateArgument, st[0], loopStateArgument});

    auto lambdaOutput = lambda->finalize(callg);
    graph->add_export(lambda->output(), {PointerType(), "h"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callg[0])),
      jlm::rvsdg::node_output::node(five));
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
      jlm::rvsdg::bit32,
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
      jlm::rvsdg::bit32,
      "d2",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 0);

    return delta->finalize(constant);
  };

  auto SetupF1 = [&](delta::output * d1)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f1",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto st = StoreNode::Create(cvd1, b2, {memoryStateArgument}, 4);

    return lambda->finalize({iOStateArgument, st[0], loopStateArgument});
  };

  auto SetupF2 = [&](lambda::output * f1, delta::output * d1, delta::output * d2)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f2",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto b42 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 42);
    auto st = StoreNode::Create(cvd1, b5, {memoryStateArgument}, 4);
    auto callResults = CallNode::Create(
      cvf1,
      f1->node()->type(),
      {iOStateArgument, st[0], loopStateArgument});
    st = StoreNode::Create(cvd2, b42, {callResults[1]}, 4);

    auto lambdaOutput = lambda->finalize(callResults);
    graph->add_export(lambdaOutput, {PointerType(), "f2"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])));
  };

  auto d1 = SetupD1();
  auto d2 = SetupD2();
  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  /*
   * Assign nodes
   */
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
      jlm::rvsdg::bit32,
      "g1",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 1);

    return delta->finalize(constant);
  };

  auto SetupG2 = [&](delta::output & g1)
  {
    PointerType pointerType;

    auto delta = delta::node::Create(
      graph->root(),
      pointerType,
      "g2",
      linkage::external_linkage,
      "",
      false);

    auto g1Argument = delta->add_ctxvar(&g1);

    return delta->finalize(g1Argument);
  };

  auto SetupF = [&](
    delta::output & g1,
    delta::output & g2)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit16, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);
    auto g1CtxVar = lambda->add_ctxvar(&g1);
    auto g2CtxVar = lambda->add_ctxvar(&g2);

    auto loadResults = LoadNode::Create(g2CtxVar, {memoryStateArgument}, PointerType(), 8);
    auto storeResults = StoreNode::Create(g2CtxVar, loadResults[0], {loadResults[1]}, 8);

    loadResults = LoadNode::Create(g1CtxVar, storeResults, jlm::rvsdg::bit32, 8);
    auto truncResult = trunc_op::create(16, loadResults[0]);

    return lambda->finalize({truncResult, iOStateArgument, loadResults[1], loopStateArgument});
  };

  auto SetupTest = [&](lambda::output & lambdaF)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto lambdaFArgument = lambda->add_ctxvar(&lambdaF);

    auto callResults = CallNode::Create(
      lambdaFArgument,
      lambdaF.node()->type(),
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto lambdaOutput = lambda->finalize({callResults[1], callResults[2], callResults[3]});
    graph->add_export(lambdaOutput, {PointerType(), "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])));
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
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f1",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvd1 = lambda->add_ctxvar(d1);

    auto b5 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = StoreNode::Create(cvd1, b5, {memoryStateArgument}, 4);

    return lambda->finalize({iOStateArgument, st[0], loopStateArgument});
  };

  auto SetupF2 = [&](lambda::output * f1, jlm::rvsdg::output * d1, jlm::rvsdg::output * d2)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "f2",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto b2 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto b21 = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 21);
    auto st = StoreNode::Create(cvd1, b2, {memoryStateArgument}, 4);
    auto callResults = CallNode::Create(
      cvf1,
      f1->node()->type(),
      {iOStateArgument, st[0], loopStateArgument});
    st = StoreNode::Create(cvd2, b21, {callResults[1]}, 4);

    auto lambdaOutput = lambda->finalize(callResults);
    graph->add_export(lambda->output(), {PointerType(), "f2"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])));
  };

  auto d1 = graph->add_import(impport(jlm::rvsdg::bit32, "d1", linkage::external_linkage));
  auto d2 = graph->add_import(impport(jlm::rvsdg::bit32, "d2", linkage::external_linkage));

  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  /*
   * Assign nodes
   */
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

  PointerType pbit64;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType fibFunctionType(
    {&jlm::rvsdg::bit64, &pbit64, &iOStateType, &memoryStateType, &loopStateType},
    {&iOStateType, &memoryStateType, &loopStateType});

  auto SetupFib = [&]()
  {
    PointerType pt;

    jlm::llvm::phi::builder pb;
    pb.begin(graph->root());
    auto fibrv = pb.add_recvar(pt);

    auto lambda = lambda::node::create(
      pb.subregion(),
      fibFunctionType,
      "fib",
      linkage::external_linkage);
    auto valueArgument = lambda->fctargument(0);
    auto pointerArgument = lambda->fctargument(1);
    auto iOStateArgument = lambda->fctargument(2);
    auto memoryStateArgument = lambda->fctargument(3);
    auto loopStateArgument = lambda->fctargument(4);
    auto ctxVarFib = lambda->add_ctxvar(fibrv->argument());

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 2);
    auto bitult = jlm::rvsdg::bitult_op::create(64, valueArgument, two);
    auto predicate = jlm::rvsdg::match(1, {{0, 1}}, 0, 2, bitult);

    auto gammaNode = jlm::rvsdg::gamma_node::create(predicate, 2);
    auto nev = gammaNode->add_entryvar(valueArgument);
    auto resultev = gammaNode->add_entryvar(pointerArgument);
    auto fibev = gammaNode->add_entryvar(ctxVarFib);
    auto gIIoState = gammaNode->add_entryvar(iOStateArgument);
    auto gIMemoryState = gammaNode->add_entryvar(memoryStateArgument);
    auto gILoopState = gammaNode->add_entryvar(loopStateArgument);

    /* gamma subregion 0 */
    auto one = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jlm::rvsdg::bitsub_op::create(64, nev->argument(0), one);
    auto callfibm1Results = CallNode::Create(
      fibev->argument(0),
      fibFunctionType,
      {nm1, resultev->argument(0), gIIoState->argument(0), gIMemoryState->argument(0), gILoopState->argument(0)});

    two = jlm::rvsdg::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jlm::rvsdg::bitsub_op::create(64, nev->argument(0), two);
    auto callfibm2Results = CallNode::Create(
      fibev->argument(0),
      fibFunctionType,
      {nm2, resultev->argument(0), callfibm1Results[0], callfibm1Results[1], callfibm1Results[2]});

    auto gepnm1 = GetElementPtrOperation::Create(resultev->argument(0), {nm1}, jlm::rvsdg::bit64, pbit64);
    auto ldnm1 = LoadNode::Create(gepnm1, {callfibm2Results[1]}, jlm::rvsdg::bit64, 8);

    auto gepnm2 = GetElementPtrOperation::Create(resultev->argument(0), {nm2}, jlm::rvsdg::bit64, pbit64);
    auto ldnm2 = LoadNode::Create(gepnm2, {ldnm1[1]}, jlm::rvsdg::bit64, 8);

    auto sum = jlm::rvsdg::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->add_exitvar({sum, nev->argument(1)});
    auto gOIoState = gammaNode->add_exitvar({callfibm2Results[0], gIIoState->argument(1)});
    auto gOMemoryState = gammaNode->add_exitvar({ldnm2[1], gIMemoryState->argument(1)});
    auto gOLoopState = gammaNode->add_exitvar({callfibm2Results[2], gILoopState->argument(1)});

    auto gepn = GetElementPtrOperation::Create(pointerArgument, {valueArgument}, jlm::rvsdg::bit64, pbit64);
    auto store = StoreNode::Create(gepn, sumex, {gOMemoryState}, 8);

    auto lambdaOutput = lambda->finalize({gOIoState, store[0], gOLoopState});

    fibrv->result()->divert_to(lambdaOutput);
    auto phiNode = pb.end();

    return std::make_tuple(
      phiNode,
      lambdaOutput,
      gammaNode,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callfibm1Results[0])),
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callfibm2Results[0])));
  };

  auto SetupTestFunction = [&](phi::node * phiNode)
  {
    arraytype at(jlm::rvsdg::bit64, 10);
    PointerType pbit64;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);
    auto fibcv = lambda->add_ctxvar(phiNode->output(0));

    auto ten = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 10);
    auto allocaResults = alloca_op::create(at, ten, 16);
    auto state = MemStateMergeOperator::Create({allocaResults[1], memoryStateArgument});

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 64, 0);
    auto gep = GetElementPtrOperation::Create(allocaResults[0], {zero, zero}, at, pbit64);

    auto callResults = CallNode::Create(
      fibcv,
      fibFunctionType,
      {ten, gep, iOStateArgument, state, loopStateArgument});

    auto lambdaOutput = lambda->finalize(callResults);
    graph->add_export(lambdaOutput, {PointerType(), "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])),
      jlm::rvsdg::node_output::node(allocaResults[0]));
  };

  auto [phiNode, fibfct, gammaNode, callFib1, callFib2] = SetupFib();
  auto [testfct, callFib, alloca] = SetupTestFunction(phiNode);

  /*
   * Assign nodes
   */
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

  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;

  PointerType pointerType;

  FunctionType constantFunctionType(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

  FunctionType recursiveFunctionType(
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

  FunctionType functionIType(
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

  FunctionType recFunctionType(
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
    {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

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
    auto loopStateArgument = lambda->fctargument(2);

    auto constant = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    return lambda->finalize({constant, iOStateArgument, memoryStateArgument, loopStateArgument});
  };

  auto SetupI = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      functionIType,
      "i",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto call = CallNode::Create(
      pointerArgument,
      constantFunctionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto lambdaOutput = lambda->finalize(call);

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(call[0])));
  };

  auto SetupA = [&](
    jlm::rvsdg::region & region,
    phi::rvargument & functionB,
    phi::rvargument & functionD)
  {
    auto lambda = lambda::node::create(
      &region,
      recFunctionType,
      "a",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto functionBCv = lambda->add_ctxvar(&functionB);
    auto functionDCv = lambda->add_ctxvar(&functionD);

    auto one = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 1);
    auto storeNode = StoreNode::Create(pointerArgument, one, {memoryStateArgument}, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto paAlloca = alloca_op::create(jlm::rvsdg::bit32, four, 4);
    auto paMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output*>({paAlloca[1], storeNode[0]}));

    auto callB = CallNode::Create(
      functionBCv,
      recFunctionType,
      {paAlloca[0], iOStateArgument, paMerge, loopStateArgument});

    auto callD = CallNode::Create(
      functionDCv,
      recFunctionType,
      {paAlloca[0], callB[1], callB[2], callB[3]});

    auto sum = jlm::rvsdg::bitadd_op::create(32, callB[0], callD[0]);

    auto lambdaOutput = lambda->finalize({sum, callD[1], callD[2], callD[3]});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callB[0])),
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callD[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(paAlloca[0])));
  };

  auto SetupB = [&](
    jlm::rvsdg::region & region,
    phi::cvargument & functionI,
    phi::rvargument & functionC,
    phi::cvargument & functionEight)
  {
    auto lambda = lambda::node::create(
      &region,
      recFunctionType,
      "b",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto functionICv = lambda->add_ctxvar(&functionI);
    auto functionCCv = lambda->add_ctxvar(&functionC);
    auto functionEightCv = lambda->add_ctxvar(&functionEight);

    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto storeNode = StoreNode::Create(pointerArgument, two, {memoryStateArgument}, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pbAlloca = alloca_op::create(jlm::rvsdg::bit32, four, 4);
    auto pbMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output*>({pbAlloca[1], storeNode[0]}));

    auto callI = CallNode::Create(
      functionICv,
      functionIType,
      {functionEightCv, iOStateArgument, pbMerge, loopStateArgument});

    auto callC = CallNode::Create(
      functionCCv,
      recFunctionType,
      {pbAlloca[0], callI[1], callI[2], callI[3]});

    auto sum = jlm::rvsdg::bitadd_op::create(32, callI[0], callC[0]);

    auto lambdaOutput = lambda->finalize({sum, callC[1], callC[2], callC[3]});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callI[0])),
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callC[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pbAlloca[0])));
  };

  auto SetupC = [&](
    jlm::rvsdg::region & region,
    phi::rvargument & functionA)
  {
    auto lambda = lambda::node::create(
      &region,
      recFunctionType,
      "c",
      linkage::external_linkage);
    auto xArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto functionACv = lambda->add_ctxvar(&functionA);

    auto three = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 3);
    auto storeNode = StoreNode::Create(xArgument, three, {memoryStateArgument}, 4);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pcAlloca = alloca_op::create(jlm::rvsdg::bit32, four, 4);
    auto pcMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output*>({pcAlloca[1], storeNode[0]}));

    auto callA = CallNode::Create(
      functionACv,
      recFunctionType,
      {pcAlloca[0], iOStateArgument, pcMerge, loopStateArgument});

    auto loadX = LoadNode::Create(xArgument, {callA[2]}, jlm::rvsdg::bit32, 4);

    auto sum = jlm::rvsdg::bitadd_op::create(32, callA[0], loadX[0]);

    auto lambdaOutput = lambda->finalize({sum, callA[1], callA[2], callA[3]});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callA[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pcAlloca[0])));
  };

  auto SetupD = [&](
    jlm::rvsdg::region & region,
    phi::rvargument & functionA)
  {
    auto lambda = lambda::node::create(
      &region,
      recFunctionType,
      "d",
      linkage::external_linkage);
    auto xArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto functionACv = lambda->add_ctxvar(&functionA);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto storeNode = StoreNode::Create(xArgument, four, {memoryStateArgument}, 4);

    auto pdAlloca = alloca_op::create(jlm::rvsdg::bit32, four, 4);
    auto pdMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output*>({pdAlloca[1], storeNode[0]}));

    auto callA = CallNode::Create(
      functionACv,
      recFunctionType,
      {pdAlloca[0], iOStateArgument, pdMerge, loopStateArgument});

    auto lambdaOutput = lambda->finalize(callA);

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callA[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pdAlloca[0])));
  };

  auto SetupPhi = [&](
    lambda::output & lambdaEight,
    lambda::output & lambdaI)
  {
    jlm::llvm::phi::builder phiBuilder;
    phiBuilder.begin(graph->root());
    auto lambdaARv = phiBuilder.add_recvar(pointerType);
    auto lambdaBRv = phiBuilder.add_recvar(pointerType);
    auto lambdaCRv = phiBuilder.add_recvar(pointerType);
    auto lambdaDRv = phiBuilder.add_recvar(pointerType);
    auto lambdaEightCv = phiBuilder.add_ctxvar(&lambdaEight);
    auto lambdaICv = phiBuilder.add_ctxvar(&lambdaI);

    auto [lambdaAOutput, callB, callD, paAlloca] = SetupA(
      *phiBuilder.subregion(),
      *lambdaBRv->argument(),
      *lambdaDRv->argument());

    auto [lambdaBOutput, callI, callC, pbAlloca] = SetupB(
      *phiBuilder.subregion(),
      *lambdaICv,
      *lambdaCRv->argument(),
      *lambdaEightCv);

    auto [lambdaCOutput, callAFromC, pcAlloca] = SetupC(
      *phiBuilder.subregion(),
      *lambdaARv->argument());

    auto [lambdaDOutput, callAFromD, pdAlloca] = SetupD(
      *phiBuilder.subregion(),
      *lambdaARv->argument());

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
    PointerType pointerType;

    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto functionACv = lambda->add_ctxvar(&functionA);

    auto four = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);
    auto pTestAlloca = alloca_op::create(jlm::rvsdg::bit32, four, 4);
    auto pTestMerge = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output*>({pTestAlloca[1], memoryStateArgument}));

    auto callA = CallNode::Create(
      functionACv,
      recFunctionType,
      {pTestAlloca[0], iOStateArgument, pTestMerge, loopStateArgument});

    auto lambdaOutput = lambda->finalize(callA);
    graph->add_export(lambdaOutput, {PointerType(), "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callA[0])),
      jlm::util::AssertedCast<jlm::rvsdg::simple_node>(jlm::rvsdg::node_output::node(pTestAlloca[0])));
  };

  auto lambdaEight = SetupEight();
  auto [lambdaI, indirectCall] = SetupI();

  auto [
    lambdaA,
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
    pdAlloca
  ] = SetupPhi(
    *lambdaEight,
    *lambdaI);

  auto [lambdaTest, callAFromTest, pTestAlloca] = SetupTest(*lambdaA);

  /*
   * Assign nodes
   */
  this->LambdaEight_ = lambdaEight->node();
  this->LambdaI_ = lambdaI->node();
  this->LambdaA_ = jlm::util::AssertedCast<lambda::node>(jlm::rvsdg::node_output::node(lambdaA->result()->origin()));
  this->LambdaB_ = jlm::util::AssertedCast<lambda::node>(jlm::rvsdg::node_output::node(lambdaB->result()->origin()));
  this->LambdaC_ = jlm::util::AssertedCast<lambda::node>(jlm::rvsdg::node_output::node(lambdaC->result()->origin()));
  this->LambdaD_ = jlm::util::AssertedCast<lambda::node>(jlm::rvsdg::node_output::node(lambdaD->result()->origin()));
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
ExternalMemoryTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType ft({&pointerType, &pointerType, &mt}, {&mt});

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

  auto storeOne = StoreNode::Create(x, one, {state}, 4);
  auto storeTwo = StoreNode::Create(y, two, {storeOne[0]}, 4);

  LambdaF->finalize(storeTwo);
  graph->add_export(LambdaF->output(), {pointerType, "f"});

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
      jlm::rvsdg::bit32,
      "a",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(deltaNode->subregion(), 32, 1);

    return  deltaNode->finalize(constant);
  };

  auto SetupDeltaB = [&]()
  {
    auto deltaNode = delta::node::Create(
      rvsdg->root(),
      jlm::rvsdg::bit32,
      "b",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(deltaNode->subregion(), 32, 2);

    return  deltaNode->finalize(constant);
  };

  auto SetupDeltaX = [&](delta::output & deltaA)
  {
    PointerType pointerType;

    auto deltaNode = delta::node::Create(
      rvsdg->root(),
      pointerType,
      "x",
      linkage::external_linkage,
      "",
      false);

    auto contextVariableA = deltaNode->add_ctxvar(&deltaA);

    return  deltaNode->finalize(contextVariableA);
  };

  auto SetupDeltaY = [&](delta::output & deltaX)
  {
    PointerType pointerType;

    auto deltaNode = delta::node::Create(
      rvsdg->root(),
      pointerType,
      "y",
      linkage::external_linkage,
      "",
      false);

    auto contextVariableX = deltaNode->add_ctxvar(&deltaX);

    auto deltaOutput = deltaNode->finalize(contextVariableX);
    rvsdg->add_export(deltaOutput, {pointerType, "y"});

    return  deltaOutput;
  };

  auto SetupLambdaTest = [&](delta::output & deltaB)
  {
    PointerType pointerType;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto pointerArgument = lambda->fctargument(0);
    auto iOStateArgument = lambda->fctargument(1);
    auto memoryStateArgument = lambda->fctargument(2);
    auto loopStateArgument = lambda->fctargument(3);

    auto contextVariableB = lambda->add_ctxvar(&deltaB);

    auto loadResults1 = LoadNode::Create(pointerArgument, {memoryStateArgument}, pointerType, 4);
    auto loadResults2 = LoadNode::Create(loadResults1[0], {loadResults1[1]}, jlm::rvsdg::bit32, 4);

    auto five = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 5);
    auto storeResults = StoreNode::Create(contextVariableB, five, {loadResults2[1]}, 4);

    auto lambdaOutput = lambda->finalize({loadResults2[0], iOStateArgument, storeResults[0], loopStateArgument});

    rvsdg->add_export(lambdaOutput, {pointerType, "test"});

    return std::make_tuple(lambdaOutput, jlm::util::AssertedCast<LoadNode>(jlm::rvsdg::node_output::node(loadResults1[0])));
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

  PointerType pointerType;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;

  FunctionType externalFunction1Type(
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType},
    {&iOStateType, &memoryStateType, &loopStateType});

  FunctionType externalFunction2Type(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupExternalFunction1Declaration = [&]()
  {
    return rvsdg->add_import(impport(
      externalFunction1Type,
      "ExternalFunction1",
      linkage::external_linkage));
  };

  auto SetupExternalFunction2Declaration = [&]()
  {
    return rvsdg->add_import(impport(
      externalFunction2Type,
      "ExternalFunction2",
      linkage::external_linkage));
  };

  auto SetupReturnAddressFunction = [&]()
  {
    PointerType p8;
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&p8, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "ReturnAddress",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto eight = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    auto mallocResults = malloc_op::create(eight);
    auto mergeResults = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>(
      {memoryStateArgument, mallocResults[1]}));

    auto lambdaOutput = lambda->finalize({mallocResults[0], iOStateArgument, mergeResults, loopStateArgument});

    rvsdg->add_export(lambdaOutput, {pointerType, "ReturnAddress"});

    return std::make_tuple(
      lambdaOutput,
      jlm::rvsdg::node_output::node(mallocResults[0]));
  };

  auto SetupCallExternalFunction1 = [&](jlm::rvsdg::argument * externalFunction1Argument)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "CallExternalFunction1",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto externalFunction1 = lambda->add_ctxvar(externalFunction1Argument);

    auto eight = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 8);

    auto mallocResults = malloc_op::create(eight);
    auto mergeResult = MemStateMergeOperator::Create(std::vector<jlm::rvsdg::output *>(
      {memoryStateArgument, mallocResults[1]}));

    auto callResults = CallNode::Create(
      externalFunction1,
      externalFunction1Type,
      {mallocResults[0], iOStateArgument, mergeResult, loopStateArgument});

    auto lambdaOutput = lambda->finalize(callResults);

    rvsdg->add_export(lambdaOutput, {pointerType, "CallExternalFunction1"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])),
      jlm::rvsdg::node_output::node(mallocResults[0]));
  };

  auto SetupCallExternalFunction2 = [&](jlm::rvsdg::argument * externalFunction2Argument)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "CallExternalFunction2",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto externalFunction2 = lambda->add_ctxvar(externalFunction2Argument);

    auto callResults = CallNode::Create(
      externalFunction2,
      externalFunction2Type,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto loadResults = LoadNode::Create(callResults[0], {callResults[2]}, jlm::rvsdg::bit32, 4);

    auto lambdaOutput = lambda->finalize({loadResults[0], callResults[1], loadResults[1], callResults[3]});

    rvsdg->add_export(lambdaOutput, {pointerType, "CallExternalFunction2"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])),
      jlm::util::AssertedCast<jlm::llvm::LoadNode>(jlm::rvsdg::node_output::node(loadResults[0])));
  };

  auto externalFunction1 = SetupExternalFunction1Declaration();
  auto externalFunction2 = SetupExternalFunction2Declaration();
  auto [returnAddressFunction, returnAddressMalloc] = SetupReturnAddressFunction();
  auto [callExternalFunction1, externalFunction1Call, callExternalFunction1Malloc] = SetupCallExternalFunction1(externalFunction1);
  auto [callExternalFunction2, externalFunction2Call, loadNode] = SetupCallExternalFunction2(externalFunction2);

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

  PointerType pointerType;
  iostatetype iOStateType;
  MemoryStateType memoryStateType;
  loopstatetype loopStateType;
  FunctionType externalFunctionType(
    {&iOStateType, &memoryStateType, &loopStateType},
    {&pointerType, &iOStateType, &memoryStateType, &loopStateType});

  auto SetupExternalFunctionDeclaration = [&]()
  {
    return rvsdg->add_import(impport(
      externalFunctionType,
      "externalFunction",
      linkage::external_linkage));
  };

  auto SetupGlobal = [&]()
  {
    auto delta = delta::node::Create(
      rvsdg->root(),
      jlm::rvsdg::bit32,
      "global",
      linkage::external_linkage,
      "",
      false);

    auto constant = jlm::rvsdg::create_bitconstant(delta->subregion(), 32, 4);

    auto deltaOutput = delta->finalize(constant);

    rvsdg->add_export(deltaOutput, {pointerType, "global"});

    return deltaOutput;
  };

  auto SetupTestFunction = [&](jlm::rvsdg::argument * externalFunctionArgument)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "test",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto externalFunction = lambda->add_ctxvar(externalFunctionArgument);

    auto callResults = CallNode::Create(
      externalFunction,
      externalFunctionType,
      {iOStateArgument, memoryStateArgument, loopStateArgument});

    auto loadResults = LoadNode::Create(callResults[0], {callResults[2]}, jlm::rvsdg::bit32, 4);

    auto lambdaOutput = lambda->finalize({loadResults[0], callResults[1], loadResults[1], callResults[3]});

    rvsdg->add_export(lambdaOutput, {pointerType, "test"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])),
      jlm::util::AssertedCast<jlm::llvm::LoadNode>(jlm::rvsdg::node_output::node(loadResults[0])));
  };

  auto importExternalFunction = SetupExternalFunctionDeclaration();
  auto deltaGlobal = SetupGlobal();
  auto [lambdaTest, callExternalFunction, loadNode] = SetupTestFunction(importExternalFunction);

  /*
   * Assign nodes
   */
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

  arraytype arrayType(jlm::rvsdg::bit32, 5);

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

    auto constantDataArray = ConstantDataArray::Create({zero, one, two, three, four});

    auto deltaOutput = delta->finalize(constantDataArray);

    rvsdg->add_export(deltaOutput, {PointerType(), "localArray"});

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

      rvsdg->add_export(deltaOutput, {PointerType(), "globalArray"});

      return deltaOutput;
  };

  auto SetupFunctionF = [&](delta::output & globalArray)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "f",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto globalArrayArgument = lambda->add_ctxvar(&globalArray);

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto two = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 2);
    auto six = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 6);

    auto gep = GetElementPtrOperation::Create(
      globalArrayArgument,
      {zero, two},
      arrayType,
      PointerType());

    auto storeResults = StoreNode::Create(gep, six, {memoryStateArgument}, 8);

    auto loadResults = LoadNode::Create(gep, {storeResults[0]}, jlm::rvsdg::bit32, 8);

    auto lambdaOutput = lambda->finalize({loadResults[0], iOStateArgument, loadResults[1], loopStateArgument});

    rvsdg->add_export(lambdaOutput, {PointerType(), "f"});

    return lambdaOutput;
  };

  auto SetupFunctionG = [&](
    delta::output & localArray,
    delta::output & globalArray,
    lambda::output & lambdaF)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&jlm::rvsdg::bit32, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg->root(),
      functionType,
      "g",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto localArrayArgument = lambda->add_ctxvar(&localArray);
    auto globalArrayArgument = lambda->add_ctxvar(&globalArray);
    auto functionFArgument = lambda->add_ctxvar(&lambdaF);

    auto bcLocalArray = bitcast_op::create(localArrayArgument, PointerType());
    auto bcGlobalArray = bitcast_op::create(globalArrayArgument, PointerType());

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 1, 0);
    auto twenty = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 20);

    auto memcpyResults = Memcpy::create(
      bcGlobalArray,
      bcLocalArray,
      twenty,
      zero,
      {memoryStateArgument});

    auto callResults = CallNode::Create(
      functionFArgument,
      lambdaF.node()->type(),
      {iOStateArgument, memcpyResults[0], loopStateArgument});

    auto lambdaOutput = lambda->finalize(callResults);

    rvsdg->add_export(lambdaOutput, {PointerType(), "g"});

    return std::make_tuple(
      lambdaOutput,
      jlm::util::AssertedCast<CallNode>(jlm::rvsdg::node_output::node(callResults[0])),
      jlm::rvsdg::node_output::node(memcpyResults[0]));
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
LinkedListTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto nf = rvsdg.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto declaration = jlm::rvsdg::rcddeclaration::create({});
  auto structType = StructType::Create("list", false, *declaration);
  PointerType pointerType;
  declaration->append(pointerType);

  auto SetupDeltaMyList = [&]()
  {
    auto delta = delta::node::Create(
      rvsdg.root(),
      pointerType,
      "MyList",
      linkage::external_linkage,
      "",
      false);

    auto constantPointerNullResult = ConstantPointerNullOperation::Create(delta->subregion(), pointerType);

    auto deltaOutput = delta->finalize(constantPointerNullResult);
    rvsdg.add_export(deltaOutput, {PointerType(), "myList"});

    return deltaOutput;
  };

  auto SetupFunctionNext = [&](delta::output & myList)
  {
    iostatetype iOStateType;
    MemoryStateType memoryStateType;
    loopstatetype loopStateType;
    FunctionType functionType(
      {&iOStateType, &memoryStateType, &loopStateType},
      {&pointerType, &iOStateType, &memoryStateType, &loopStateType});

    auto lambda = lambda::node::create(
      rvsdg.root(),
      functionType,
      "next",
      linkage::external_linkage);
    auto iOStateArgument = lambda->fctargument(0);
    auto memoryStateArgument = lambda->fctargument(1);
    auto loopStateArgument = lambda->fctargument(2);

    auto myListArgument = lambda->add_ctxvar(&myList);

    auto zero = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 0);
    auto size = jlm::rvsdg::create_bitconstant(lambda->subregion(), 32, 4);

    auto alloca = alloca_op::create(pointerType, size, 4);
    auto mergedMemoryState = MemStateMergeOperator::Create({alloca[1], memoryStateArgument});

    auto load1 = LoadNode::Create(myListArgument, {mergedMemoryState}, pointerType, 4);
    auto store1 = StoreNode::Create(alloca[0], load1[0], {load1[1]}, 4);

    auto load2 = LoadNode::Create(alloca[0], {store1[0]}, pointerType, 4);
    auto gep = GetElementPtrOperation::Create(
      load2[0],
      {zero, zero},
      *structType,
      pointerType);

    auto load3 = LoadNode::Create(gep, {load2[1]}, pointerType, 4);
    auto store2 = StoreNode::Create(alloca[0], load3[0], {load3[1]}, 4);

    auto load4 = LoadNode::Create(alloca[0], {store2[0]}, pointerType, 4);

    auto lambdaOutput = lambda->finalize({load4[0], iOStateArgument, load4[1], loopStateArgument});
    rvsdg.add_export(lambdaOutput, {pointerType, "next"});

    return std::make_tuple(
      jlm::rvsdg::node_output::node(alloca[0]),
      lambdaOutput);
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

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  // Create imported symbol "imported"
  Import_ = graph->add_import(impport(jlm::rvsdg::bit32, "imported", linkage::external_linkage));

  // Create global variable "global"
  Delta_ = delta::node::Create(
          graph->root(),
          pointerType,
          "global",
          linkage::external_linkage,
          "",
          false);
  auto constantPointerNullResult = ConstantPointerNullOperation::Create(Delta_->subregion(), pointerType);
  Delta_->finalize(constantPointerNullResult);

  // Start of function "f"
  Lambda_ = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);
  auto entryMemoryState = Lambda_->fctargument(0);
  auto deltaContextVar = Lambda_->add_ctxvar(Delta_->output());
  auto importContextVar = Lambda_->add_ctxvar(Import_);

  // Create alloca node
  auto allocaSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 8);
  auto allocaOutputs = alloca_op::create(pointerType, allocaSize, 8);
  Alloca_ = jlm::rvsdg::node_output::node(allocaOutputs[0]);

  auto afterAllocaMemoryState = MemStateMergeOperator::Create(
          std::vector<jlm::rvsdg::output *>{entryMemoryState, allocaOutputs[1]});

  // Create malloc node
  auto mallocSize = jlm::rvsdg::create_bitconstant(Lambda_->subregion(), 32, 4);
  auto mallocOutputs = malloc_op::create(mallocSize);
  Malloc_ = jlm::rvsdg::node_output::node(mallocOutputs[0]);

  auto afterMallocMemoryState = MemStateMergeOperator::Create(
          std::vector<jlm::rvsdg::output *>{afterAllocaMemoryState, mallocOutputs[1]});

  // Store the result of malloc into the alloca'd memory
  auto storeAllocaOutputs = StoreNode::Create(allocaOutputs[0], mallocOutputs[0], {afterMallocMemoryState}, 8);

  // load the value in the alloca again
  auto loadAllocaOutputs = LoadNode::Create(allocaOutputs[0], {storeAllocaOutputs[0]}, pointerType, 8);

  // Load the value of the imported symbol "imported"
  auto loadImportedOutputs = LoadNode::Create(importContextVar, {loadAllocaOutputs[1]}, jlm::rvsdg::bit32, 4);

  // Store the loaded value from imported, into the address loaded from the alloca (aka. the malloc result)
  auto storeImportedOutputs = StoreNode::Create(loadAllocaOutputs[0], loadImportedOutputs[0], {loadImportedOutputs[1]}, 4);

  // store the loaded alloca value in the global variable
  auto storeOutputs = StoreNode::Create(deltaContextVar, loadAllocaOutputs[0], {storeImportedOutputs[0]}, 8);

  Lambda_->finalize({storeOutputs[0]});

  graph->add_export(Delta_->output(), {pointerType, "global"});
  graph->add_export(Lambda_->output(), {pointerType, "f"});

  return module;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
NAllocaNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;

  MemoryStateType mt;
  PointerType pointerType;
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto allocaSize = jlm::rvsdg::create_bitconstant(fct->subregion(), 32, 4);

  jlm::rvsdg::output * latestMemoryState = fct->fctargument(0);

  for (size_t i = 0; i < NumAllocaNodes_; i++) {
    auto alloca_outputs = alloca_op::create(jlm::rvsdg::bit32, allocaSize, 4);
    auto alloca_node = jlm::rvsdg::node_output::node(alloca_outputs[0]);

    AllocaNodes_.push_back(alloca_node);

    // Update latestMemoryState to include the alloca memory state output
    latestMemoryState = MemStateMergeOperator::Create(
                            std::vector<jlm::rvsdg::output*>{latestMemoryState, alloca_outputs[1]});
  }

  fct->finalize({latestMemoryState});

  graph->add_export(fct->output(), {pointerType, "f"});

  return module;
}

}