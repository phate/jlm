/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

std::unique_ptr<jlm::RvsdgModule>
StoreTest1::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  auto ppt = ptrtype::create(*pt);
  auto pppt = ptrtype::create(*ppt);
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jive::create_bitconstant(fct->subregion(), 32, 4);

  auto d = alloca_op::create(jive::bit32, csize, 4);
  auto c = alloca_op::create(*pt, csize, 4);
  auto b = alloca_op::create(*ppt, csize, 4);
  auto a = alloca_op::create(*pppt, csize, 4);

  auto merge_d = MemStateMergeOperator::Create({d[1], fct->fctargument(0)});
  auto merge_c = MemStateMergeOperator::Create(std::vector<jive::output *>({c[1], merge_d}));
  auto merge_b = MemStateMergeOperator::Create(std::vector<jive::output *>({b[1], merge_c}));
  auto merge_a = MemStateMergeOperator::Create(std::vector<jive::output *>({a[1], merge_b}));

  auto a_amp_b = store_op::create(a[0], b[0], {merge_a}, 4);
  auto b_amp_c = store_op::create(b[0], c[0], {a_amp_b[0]}, 4);
  auto c_amp_d = store_op::create(c[0], d[0], {b_amp_c[0]}, 4);

  fct->finalize({c_amp_d[0]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /* extract nodes */

  this->lambda = fct;

  this->size = jive::node_output::node(csize);

  this->alloca_a = jive::node_output::node(a[0]);
  this->alloca_b = jive::node_output::node(b[0]);
  this->alloca_c = jive::node_output::node(c[0]);
  this->alloca_d = jive::node_output::node(d[0]);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
StoreTest2::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  auto ppt = ptrtype::create(*pt);
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jive::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jive::bit32, csize, 4);
  auto b = alloca_op::create(jive::bit32, csize, 4);
  auto x = alloca_op::create(*pt, csize, 4);
  auto y = alloca_op::create(*pt, csize, 4);
  auto p = alloca_op::create(*ppt, csize, 4);

  auto merge_a = MemStateMergeOperator::Create({a[1], fct->fctargument(0)});
  auto merge_b = MemStateMergeOperator::Create(std::vector<jive::output *>({b[1], merge_a}));
  auto merge_x = MemStateMergeOperator::Create(std::vector<jive::output *>({x[1], merge_b}));
  auto merge_y = MemStateMergeOperator::Create(std::vector<jive::output *>({y[1], merge_x}));
  auto merge_p = MemStateMergeOperator::Create(std::vector<jive::output *>({p[1], merge_y}));

  auto x_amp_a = store_op::create(x[0], a[0], {merge_p}, 4);
  auto y_amp_b = store_op::create(y[0], b[0], {x_amp_a[0]}, 4);
  auto p_amp_x = store_op::create(p[0], x[0], {y_amp_b[0]}, 4);
  auto p_amp_y = store_op::create(p[0], y[0], {p_amp_x[0]}, 4);

  fct->finalize({p_amp_y[0]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /* extract nodes */

  this->lambda = fct;

  this->size = jive::node_output::node(csize);

  this->alloca_a = jive::node_output::node(a[0]);
  this->alloca_b = jive::node_output::node(b[0]);
  this->alloca_x = jive::node_output::node(x[0]);
  this->alloca_y = jive::node_output::node(y[0]);
  this->alloca_p = jive::node_output::node(p[0]);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
LoadTest1::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  auto ppt = ptrtype::create(*pt);
  FunctionType fcttype({ppt.get(), &mt}, {&jive::bit32, &mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto ld1 = load_op::create(fct->fctargument(0), {fct->fctargument(1)}, 4);
  auto ld2 = load_op::create(ld1[0], {ld1[1]}, 4);

  fct->finalize(ld2);

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /* extract nodes */

  this->lambda = fct;

  this->load_p = jive::node_output::node(ld1[0]);
  this->load_x = jive::node_output::node(ld2[0]);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
LoadTest2::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  auto ppt = ptrtype::create(*pt);
  FunctionType fcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto csize = jive::create_bitconstant(fct->subregion(), 32, 4);

  auto a = alloca_op::create(jive::bit32, csize, 4);
  auto b = alloca_op::create(jive::bit32, csize, 4);
  auto x = alloca_op::create(*pt, csize, 4);
  auto y = alloca_op::create(*pt, csize, 4);
  auto p = alloca_op::create(*ppt, csize, 4);

  auto merge_a = MemStateMergeOperator::Create({a[1], fct->fctargument(0)});
  auto merge_b = MemStateMergeOperator::Create(std::vector<jive::output *>({b[1], merge_a}));
  auto merge_x = MemStateMergeOperator::Create(std::vector<jive::output *>({x[1], merge_b}));
  auto merge_y = MemStateMergeOperator::Create(std::vector<jive::output *>({y[1], merge_x}));
  auto merge_p = MemStateMergeOperator::Create(std::vector<jive::output *>({p[1], merge_y}));

  auto x_amp_a = store_op::create(x[0], a[0], {merge_p}, 4);
  auto y_amp_b = store_op::create(y[0], b[0], x_amp_a, 4);
  auto p_amp_x = store_op::create(p[0], x[0], y_amp_b, 4);

  auto ld1 = load_op::create(p[0], p_amp_x, 4);
  auto ld2 = load_op::create(ld1[0], {ld1[1]}, 4);
  auto y_star_p = store_op::create(y[0], ld2[0], {ld2[1]}, 4);

  fct->finalize({y_star_p[0]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /* extract nodes */

  this->lambda = fct;

  this->size = jive::node_output::node(csize);

  this->alloca_a = jive::node_output::node(a[0]);
  this->alloca_b = jive::node_output::node(b[0]);
  this->alloca_x = jive::node_output::node(x[0]);
  this->alloca_y = jive::node_output::node(y[0]);
  this->alloca_p = jive::node_output::node(p[0]);

  this->load_x = jive::node_output::node(ld1[0]);
  this->load_a = jive::node_output::node(ld2[0]);;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
GetElementPtrTest::SetupRvsdg()
{
  using namespace jlm;

  auto dcl = jive::rcddeclaration::create({&jive::bit32, &jive::bit32});
  jive::rcdtype rt(dcl.get());

  MemoryStateType mt;
  auto pt = ptrtype::create(rt);
  auto pbt = ptrtype::create(jive::bit32);
  FunctionType fcttype({pt.get(), &mt}, {&jive::bit32, &mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jive::create_bitconstant(fct->subregion(), 32, 0);
  auto one = jive::create_bitconstant(fct->subregion(), 32, 1);

  auto gepx = getelementptr_op::create(fct->fctargument(0), {zero, zero}, *pbt);
  auto ldx = load_op::create(gepx, {fct->fctargument(1)}, 4);

  auto gepy = getelementptr_op::create(fct->fctargument(0), {zero, one}, *pbt);
  auto ldy = load_op::create(gepy, {ldx[1]}, 4);

  auto sum = jive::bitadd_op::create(32, ldx[0], ldy[0]);

  fct->finalize({sum, ldy[1]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /*
   * Assign nodes
   */
  this->lambda = fct;

  this->getElementPtrX = jive::node_output::node(gepx);
  this->getElementPtrY = jive::node_output::node(gepy);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
BitCastTest::SetupRvsdg()
{
  using namespace jlm;

  auto pbt16 = ptrtype::create(jive::bit16);
  auto pbt32 = ptrtype::create(jive::bit32);
  FunctionType fcttype({pbt32.get()}, {pbt16.get()});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto cast = bitcast_op::create(fct->fctargument(0), *pbt16);

  fct->finalize({cast});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->bitCast = jive::node_output::node(cast);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
Bits2PtrTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit8);
  FunctionType fctbits2ptrtype({&jive::bit64, &mt}, {pt.get(), &mt});
  FunctionType fcttesttype({&jive::bit64, &mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupBit2PtrFunction = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      fctbits2ptrtype,
      "bit2ptr",
      linkage::external_linkage);
    auto cast = bits2ptr_op::create(lambda->fctargument(0), *pt);
    lambda->finalize({cast, lambda->fctargument(1)});

    return std::make_tuple(lambda, jive::node_output::node(cast));
  };

  auto SetupTestFunction = [&](lambda::output * b2p)
  {
    auto lambda = lambda::node::create(graph->root(), fcttesttype, "test",
                                        linkage::external_linkage);
    auto cvbits2ptr = lambda->add_ctxvar(b2p);

    auto results = CallNode::Create(
      cvbits2ptr,
      {lambda->fctargument(0),
       lambda->fctargument(1)});

    lambda->finalize({results[1]});
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "testfct"});

    return std::make_tuple(lambda, jive::node_output::node(results[0]));
  };

  auto [bits2ptrFunction, castNode] = SetupBit2PtrFunction();
  auto [testfct, callNode] = SetupTestFunction(bits2ptrFunction->output());

  /*
   * Assign nodes
   */
  this->lambda_bits2ptr = bits2ptrFunction;
  this->lambda_test = testfct;

  this->bits2ptr = castNode;

  this->call = callNode;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
ConstantPointerNullTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  auto ppt = ptrtype::create(*pt);
  FunctionType fcttype({ppt.get(), &mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto cnull = ptr_constant_null_op::create(fct->subregion(), *pt);
  auto st = store_op::create(fct->fctargument(0), cnull, {fct->fctargument(1)}, 4);

  fct->finalize({st[0]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->null = jive::node_output::node(cnull);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
CallTest1::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  FunctionType ft1({pt.get(), pt.get(), &mt}, {&jive::bit32, &mt});
  FunctionType ft2({&mt}, {&jive::bit32, &mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupF = [&]()
  {
    auto lambda = lambda::node::create(graph->root(), ft1, "f", linkage::external_linkage);

    auto ld1 = load_op::create(lambda->fctargument(0), {lambda->fctargument(2)}, 4);
    auto ld2 = load_op::create(lambda->fctargument(1), {ld1[1]}, 4);

    auto sum = jive::bitadd_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({sum, ld2[1]});

    return lambda;
  };

  auto SetupG = [&]()
  {
    auto lambda = lambda::node::create(graph->root(), ft1, "g", linkage::external_linkage);

    auto ld1 = load_op::create(lambda->fctargument(0), {lambda->fctargument(2)}, 4);
    auto ld2 = load_op::create(lambda->fctargument(1), {ld1[1]}, 4);

    auto diff = jive::bitsub_op::create(32, ld1[0], ld2[0]);

    lambda->finalize({diff, ld2[1]});

    return lambda;
  };

  auto SetupH = [&](lambda::node * f, lambda::node * g)
  {
    auto lambda = lambda::node::create(graph->root(), ft2, "h", linkage::external_linkage);

    auto cvf = lambda->add_ctxvar(f->output());
    auto cvg = lambda->add_ctxvar(g->output());

    auto size = jive::create_bitconstant(lambda->subregion(), 32, 4);

    auto x = alloca_op::create(jive::bit32, size, 4);
    auto y = alloca_op::create(jive::bit32, size, 4);
    auto z = alloca_op::create(jive::bit32, size, 4);

    auto mx = MemStateMergeOperator::Create(std::vector<jive::output *>({x[1], lambda->fctargument(0)}));
    auto my = MemStateMergeOperator::Create(std::vector<jive::output *>({y[1], mx}));
    auto mz = MemStateMergeOperator::Create(std::vector<jive::output *>({z[1], my}));

    auto five = jive::create_bitconstant(lambda->subregion(), 32, 5);
    auto six = jive::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jive::create_bitconstant(lambda->subregion(), 32, 7);

    auto stx = store_op::create(x[0], five, {mz}, 4);
    auto sty = store_op::create(y[0], six, {stx[0]}, 4);
    auto stz = store_op::create(z[0], seven, {sty[0]}, 4);

    auto callFResults = CallNode::Create(cvf, {x[0], y[0], stz[0]});
    auto callGResults = CallNode::Create(cvg, {z[0], z[0], callFResults[1]});

    auto sum = jive::bitadd_op::create(32, callFResults[0], callGResults[0]);

    lambda->finalize({sum, callGResults[1]});
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "h"});

    auto allocaX = jive::node_output::node(x[0]);
    auto allocaY = jive::node_output::node(y[0]);
    auto allocaZ = jive::node_output::node(z[0]);
    auto callF = jive::node_output::node(callFResults[0]);
    auto callG = jive::node_output::node(callGResults[0]);

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

  this->callF = callF;
  this->callG = callG;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
CallTest2::SetupRvsdg()
{
  using namespace jlm;

  iostatetype iot;
  MemoryStateType mt;
  auto pbit8 = ptrtype::create(jive::bit8);
  auto pbit32 = ptrtype::create(jive::bit32);

  FunctionType create_type({&jive::bit32, &mt, &iot}, {pbit32.get(), &mt, &iot});
  FunctionType destroy_type({pbit32.get(), &mt, &iot}, {&mt, &iot});
  FunctionType test_type({&mt, &iot}, {&mt, &iot});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupCreate = [&]()
  {
    auto lambda = lambda::node::create(graph->root(), create_type, "create",
                                       linkage::external_linkage);

    auto four = jive::create_bitconstant(lambda->subregion(), 32, 4);
    auto prod = jive::bitmul_op::create(32, lambda->fctargument(0), four);

    auto alloc = malloc_op::create(prod);
    auto cast = bitcast_op::create(alloc[0], *pbit32);
    auto mx = MemStateMergeOperator::Create(std::vector<jive::output *>(
      {alloc[1], lambda->fctargument(1)}));

    lambda->finalize({cast, mx, lambda->fctargument(2)});

    auto mallocNode = jive::node_output::node(alloc[0]);
    return std::make_tuple(lambda, mallocNode);
  };

  auto SetupDestroy = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      destroy_type,
      "destroy",
      linkage::external_linkage);

    auto cast = bitcast_op::create(lambda->fctargument(0), *pbit8);
    auto freeResults = free_op::create(cast, {lambda->fctargument(1)}, lambda->fctargument(2));

    lambda->finalize(freeResults);

    auto freeNode = jive::node_output::node(freeResults[0]);
    return std::make_tuple(lambda, freeNode);
  };

  auto SetupTest = [&](lambda::node * lambdaCreate, lambda::node * lambdaDestroy)
  {
    auto lambda = lambda::node::create(graph->root(), test_type, "test", linkage::external_linkage);
    auto create_cv = lambda->add_ctxvar(lambdaCreate->output());
    auto destroy_cv = lambda->add_ctxvar(lambdaDestroy->output());

    auto six = jive::create_bitconstant(lambda->subregion(), 32, 6);
    auto seven = jive::create_bitconstant(lambda->subregion(), 32, 7);

    auto create1 = CallNode::Create(
      create_cv,
      {six, lambda->fctargument(0), lambda->fctargument(1)});
    auto create2 = CallNode::Create(create_cv, {seven, create1[1], create1[2]});

    auto destroy1 = CallNode::Create(destroy_cv, {create1[0], create2[1], create2[2]});
    auto destroy2 = CallNode::Create(destroy_cv, {create2[0], destroy1[0], destroy1[1]});

    lambda->finalize(destroy2);
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "test"});

    auto callCreate1Node = jive::node_output::node(create1[0]);
    auto callCreate2Node = jive::node_output::node(create2[0]);
    auto callDestroy1Node = jive::node_output::node(destroy1[0]);
    auto callDestroy2Node = jive::node_output::node(destroy2[0]);

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

  this->call_create1 = callCreate1;
  this->call_create2 = callCreate2;

  this->call_destroy1 = callCreate1;
  this->call_destroy2 = callCreate2;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
IndirectCallTest::SetupRvsdg()
{
  using namespace jlm;

  iostatetype iot;
  MemoryStateType mt;

  FunctionType four_type({&mt, &iot}, {&jive::bit32, &mt, &iot});
  auto pfct = ptrtype::create(four_type);

  FunctionType test_type({&mt, &iot}, {&jive::bit32, &mt, &iot});


  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupFunctionFour = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      four_type,
      "four",
      linkage::external_linkage);

    auto four = jive::create_bitconstant(lambda->subregion(), 32, 4);

    return lambda->finalize({four, lambda->fctargument(0), lambda->fctargument(1)});
  };

  auto SetupFunctionThree = [&]()
  {
    auto lambda = lambda::node::create(
      graph->root(),
      four_type,
      "three",
      linkage::external_linkage);

    auto three = jive::create_bitconstant(lambda->subregion(), 32, 3);

    return lambda->finalize({three, lambda->fctargument(0), lambda->fctargument(1)});
  };

  auto SetupIndirectCallFunction = [&]()
  {
    FunctionType functionType(
      {pfct.get(), &mt, &iot},
      {&jive::bit32, &mt, &iot});

    auto lambda = lambda::node::create(
      graph->root(),
      functionType,
      "indcall",
      linkage::external_linkage);

    auto call = CallNode::Create(
      lambda->fctargument(0),
      {lambda->fctargument(1), lambda->fctargument(2)});

    auto lambdaOutput = lambda->finalize(call);

    return std::make_tuple(lambdaOutput, jive::node_output::node(call[0]));
  };

  auto SetupTestFunction = [&](
    lambda::output * fctindcall,
    lambda::output * fctthree,
    lambda::output * fctfour)
  {
    auto lambda = lambda::node::create(
      graph->root(),
      test_type,
      "test",
      linkage::external_linkage);

    auto fctindcall_cv = lambda->add_ctxvar(fctindcall);
    auto fctfour_cv = lambda->add_ctxvar(fctfour);
    auto fctthree_cv = lambda->add_ctxvar(fctthree);

    auto call_four = CallNode::Create(
      fctindcall_cv,
      {fctfour_cv, lambda->fctargument(0), lambda->fctargument(1)});
    auto call_three = CallNode::Create(
      fctindcall_cv,
      {fctthree_cv, call_four[1], call_four[2]});

    auto add = jive::bitadd_op::create(32, call_four[0], call_three[0]);

    auto lambdaOutput = lambda->finalize({add, call_three[1], call_three[2]});
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "test"});

    return std::make_tuple(lambdaOutput, jive::node_output::node(call_three[0]), jive::node_output::node(call_four[0]));
  };

  auto fctfour = SetupFunctionFour();
  auto fctthree = SetupFunctionThree();
  auto [fctindcall, callIndirectFunction] = SetupIndirectCallFunction();
  auto [fcttest, callFunctionThree, callFunctionFour] = SetupTestFunction(fctindcall, fctthree, fctfour);

  /*
   * Assign
   */
  this->lambda_three = fctthree->node();
  this->lambda_four = fctfour->node();
  this->lambda_indcall = fctindcall->node();
  this->lambda_test = fcttest->node();

  this->call_fctindcall = callIndirectFunction;
  this->call_fctthree = callFunctionThree;
  this->call_fctfour = callFunctionFour;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
GammaTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  FunctionType fcttype(
    {&jive::bit32, pt.get(), pt.get(), pt.get(), pt.get(), &mt},
    {&jive::bit32, &mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jive::create_bitconstant(fct->subregion(), 32, 0);
  auto biteq = jive::biteq_op::create(32, fct->fctargument(0), zero);
  auto predicate = jive::match(1, {{0, 1}}, 0, 2, biteq);

  auto gammanode = jive::gamma_node::create(predicate, 2);
  auto p1ev = gammanode->add_entryvar(fct->fctargument(1));
  auto p2ev = gammanode->add_entryvar(fct->fctargument(2));
  auto p3ev = gammanode->add_entryvar(fct->fctargument(3));
  auto p4ev = gammanode->add_entryvar(fct->fctargument(4));

  auto tmp1 = gammanode->add_exitvar({p1ev->argument(0), p3ev->argument(1)});
  auto tmp2 = gammanode->add_exitvar({p2ev->argument(0), p4ev->argument(1)});

  auto ld1 = load_op::create(tmp1, {fct->fctargument(5)}, 4);
  auto ld2 = load_op::create(tmp2, {ld1[1]}, 4);
  auto sum = jive::bitadd_op::create(32, ld1[0], ld2[0]);

  fct->finalize({sum, ld2[1]});

  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->gamma = gammanode;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
ThetaTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  FunctionType fcttype({&jive::bit32, pt.get(), &jive::bit32, &mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto zero = jive::create_bitconstant(fct->subregion(), 32, 0);

  auto thetanode = jive::theta_node::create(fct->subregion());

  auto n = thetanode->add_loopvar(zero);
  auto l = thetanode->add_loopvar(fct->fctargument(0));
  auto a = thetanode->add_loopvar(fct->fctargument(1));
  auto c = thetanode->add_loopvar(fct->fctargument(2));
  auto s = thetanode->add_loopvar(fct->fctargument(3));

  auto gepnode = getelementptr_op::create(a->argument(), {n->argument()}, *pt);
  auto store = store_op::create(gepnode, c->argument(), {s->argument()}, 4);

  auto one = jive::create_bitconstant(thetanode->subregion(), 32, 1);
  auto sum = jive::bitadd_op::create(32, n->argument(), one);
  auto cmp = jive::bitult_op::create(32, sum, l->argument());
  auto predicate = jive::match(1, {{1, 1}}, 0, 2, cmp);

  n->result()->divert_to(sum);
  s->result()->divert_to(store[0]);
  thetanode->set_predicate(predicate);

  fct->finalize({s});
  graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

  /*
   * Assign nodes
   */
  this->lambda = fct;
  this->theta = thetanode;
  this->gep = jive::node_output::node(gepnode);

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
DeltaTest1::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  FunctionType fctgtype({pt.get(), &mt}, {&jive::bit32, &mt});
  FunctionType fcthtype({&mt}, {&jive::bit32, &mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);


  auto SetupGlobalF = [&]()
  {
    auto dfNode = delta::node::create(
      graph->root(),
      ptrtype(jive::bit32),
      "f",
      linkage::external_linkage,
      false);

    auto constant = jive::create_bitconstant(dfNode->subregion(), 32, 0);

    return  dfNode->finalize(constant);
  };

  auto SetupFunctionG = [&]()
  {
    auto lambda = lambda::node::create(graph->root(), fctgtype, "g", linkage::external_linkage);
    auto ld = load_op::create(lambda->fctargument(0), {lambda->fctargument(1)}, 4);
    return lambda->finalize(ld);
  };

  auto SetupFunctionH = [&](delta::output * f, lambda::output * g)
  {
    auto lambda = lambda::node::create(graph->root(), fcthtype, "h", linkage::external_linkage);
    auto cvf = lambda->add_ctxvar(f);
    auto cvg = lambda->add_ctxvar(g);

    auto five = jive::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = store_op::create(cvf, five, {lambda->fctargument(0)}, 4);
    auto callg = CallNode::Create(cvg, {cvf, st[0]});

    auto lambdaOutput = lambda->finalize(callg);
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "h"});

    return std::make_tuple(lambdaOutput, jive::node_output::node(callg[0]), jive::node_output::node(five));
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

  this->call_g = callFunctionG;
  this->constantFive = constantFive;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
DeltaTest2::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  FunctionType ft({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupD1 = [&]()
  {
    auto delta = delta::node::create(
      graph->root(),
      ptrtype(jive::bit32),
      "d1",
      linkage::external_linkage,
      false);

    auto constant = jive::create_bitconstant(delta->subregion(), 32, 0);

    return delta->finalize(constant);
  };

  auto SetupD2 = [&]()
  {
    auto delta = delta::node::create(
      graph->root(),
      ptrtype(jive::bit32),
      "d2",
      linkage::external_linkage,
      false);

    auto constant = jive::create_bitconstant(delta->subregion(), 32, 0);

    return delta->finalize(constant);
  };

  auto SetupF1 = [&](delta::output * d1)
  {
    auto lambda = lambda::node::create(graph->root(), ft, "f1", linkage::external_linkage);
    auto cvd1 = lambda->add_ctxvar(d1);
    auto b2 = jive::create_bitconstant(lambda->subregion(), 32, 2);
    auto st = store_op::create(cvd1, b2, {lambda->fctargument(0)}, 4);

    return lambda->finalize(st);
  };

  auto SetupF2 = [&](lambda::output * f1, delta::output * d1, delta::output * d2)
  {
    auto lambda = lambda::node::create(graph->root(), ft, "f2", linkage::external_linkage);
    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto b5 = jive::create_bitconstant(lambda->subregion(), 32, 5);
    auto b42 = jive::create_bitconstant(lambda->subregion(), 32, 42);
    auto st = store_op::create(cvd1, b5, {lambda->fctargument(0)}, 4);
    auto callf1 = CallNode::Create(cvf1, st);
    st = store_op::create(cvd2, b42, callf1, 4);

    auto lambdaOutput = lambda->finalize(st);
    graph->add_export(lambdaOutput, {ptrtype(lambda->type()), "f2"});

    return std::make_tuple(lambdaOutput, jive::node_output::node(callf1[0]));
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

  this->call_f1 = callF1;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
ImportTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  FunctionType ft({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupF1 = [&](jive::output * d1)
  {
    auto lambda = lambda::node::create(graph->root(), ft, "f1", linkage::external_linkage);
    auto cvd1 = lambda->add_ctxvar(d1);
    auto b5 = jive::create_bitconstant(lambda->subregion(), 32, 5);
    auto st = store_op::create(cvd1, b5, {lambda->fctargument(0)}, 4);

    return lambda->finalize(st);
  };

  auto SetupF2 = [&](lambda::output * f1, jive::output * d1, jive::output * d2)
  {
    auto lambda = lambda::node::create(graph->root(), ft, "f2", linkage::external_linkage);
    auto cvd1 = lambda->add_ctxvar(d1);
    auto cvd2 = lambda->add_ctxvar(d2);
    auto cvf1 = lambda->add_ctxvar(f1);
    auto b2 = jive::create_bitconstant(lambda->subregion(), 32, 2);
    auto b21 = jive::create_bitconstant(lambda->subregion(), 32, 21);
    auto st = store_op::create(cvd1, b2, {lambda->fctargument(0)}, 4);
    auto callf1 = CallNode::Create(cvf1, st);
    st = store_op::create(cvd2, b21, callf1, 4);

    auto lambdaOutput = lambda->finalize(st);
    graph->add_export(lambda->output(), {ptrtype(lambda->type()), "f2"});

    return std::make_tuple(lambdaOutput, jive::node_output::node(callf1[0]));
  };

  auto d1 = graph->add_import(impport(ptrtype(jive::bit32), "d1", linkage::external_linkage));
  auto d2 = graph->add_import(impport(ptrtype(jive::bit32), "d2", linkage::external_linkage));

  auto f1 = SetupF1(d1);
  auto [f2, callF1] = SetupF2(f1, d1, d2);

  /*
   * Assign nodes
   */
  this->lambda_f1 = f1->node();
  this->lambda_f2 = f2->node();

  this->call_f1 = callF1;

  this->import_d1 = d1;
  this->import_d2 = d2;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
PhiTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;

  arraytype at(jive::bit64, 10);
  ptrtype pat(at);

  ptrtype pbit64(jive::bit64);

  FunctionType fibfcttype(
    {&jive::bit64, &pbit64, &mt},
    {&mt});
  ptrtype pfibfcttype(fibfcttype);

  FunctionType testfcttype({&mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  auto SetupFib = [&]()
  {
    jlm::phi::builder pb;
    pb.begin(graph->root());
    auto fibrv = pb.add_recvar(pfibfcttype);

    auto fibfct = lambda::node::create(pb.subregion(), fibfcttype, "fib",
                                       linkage::external_linkage);
    fibfct->add_ctxvar(fibrv->argument());

    auto two = jive::create_bitconstant(fibfct->subregion(), 64, 2);
    auto bitult = jive::bitult_op::create(64, fibfct->fctargument(0), two);
    auto predicate = jive::match(1, {{0, 1}}, 0, 2, bitult);

    auto gammaNode = jive::gamma_node::create(predicate, 2);
    auto nev = gammaNode->add_entryvar(fibfct->fctargument(0));
    auto resultev = gammaNode->add_entryvar(fibfct->fctargument(1));
    auto fibev = gammaNode->add_entryvar(fibfct->cvargument(0));
    auto stateev = gammaNode->add_entryvar(fibfct->fctargument(2));

    /* gamma subregion 0 */
    auto one = jive::create_bitconstant(gammaNode->subregion(0), 64, 1);
    auto nm1 = jive::bitsub_op::create(64, nev->argument(0), one);
    auto callfibm1Results = CallNode::Create(
      fibev->argument(0),
      {nm1, resultev->argument(0),
       stateev->argument(0)});

    two = jive::create_bitconstant(gammaNode->subregion(0), 64, 2);
    auto nm2 = jive::bitsub_op::create(64, nev->argument(0), two);
    auto callfibm2Results = CallNode::Create(
      fibev->argument(0),
      {nm2, resultev->argument(0),
       callfibm1Results[0]});

    auto gepnm1 = getelementptr_op::create(resultev->argument(0), {nm1}, pbit64);
    auto ldnm1 = load_op::create(gepnm1, {callfibm2Results[0]}, 8);

    auto gepnm2 = getelementptr_op::create(resultev->argument(0), {nm2}, pbit64);
    auto ldnm2 = load_op::create(gepnm2, {ldnm1[1]}, 8);

    auto sum = jive::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

    /* gamma subregion 1 */
    /* Nothing needs to be done */

    auto sumex = gammaNode->add_exitvar({sum, nev->argument(1)});
    auto stateex = gammaNode->add_exitvar({ldnm2[1], stateev->argument(1)});

    auto gepn = getelementptr_op::create(fibfct->fctargument(1), {fibfct->fctargument(0)}, pbit64);
    auto store = store_op::create(gepn, sumex, {stateex}, 8);

    auto lambdaOutput = fibfct->finalize({store[0]});

    fibrv->result()->divert_to(lambdaOutput);
    auto phiNode = pb.end();

    return std::make_tuple(
      phiNode,
      lambdaOutput,
      gammaNode,
      jive::node_output::node(callfibm1Results[0]),
      jive::node_output::node(callfibm2Results[0]));
  };

  auto SetupTestFunction = [&](phi::node * phiNode)
  {
    auto lambda = lambda::node::create(
      graph->root(),
      testfcttype,
      "test",
      linkage::external_linkage);
    auto fibcv = lambda->add_ctxvar(phiNode->output(0));

    auto ten = jive::create_bitconstant(lambda->subregion(), 64, 10);
    auto allocaResults = alloca_op::create(at, ten, 16);
    auto state = MemStateMergeOperator::Create({allocaResults[1], lambda->fctargument(0)});

    auto zero = jive::create_bitconstant(lambda->subregion(), 64, 0);
    auto gep = getelementptr_op::create(allocaResults[0], {zero, zero}, pbit64);

    auto call = CallNode::Create(fibcv, {ten, gep, state});

    auto lambdaOutput = lambda->finalize({call[0]});
    graph->add_export(lambdaOutput, {ptrtype(testfcttype), "test"});

    return std::make_tuple(
      lambdaOutput,
      jive::node_output::node(call[0]),
      jive::node_output::node(allocaResults[0]));
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

  this->callfibm1 = callFib1;
  this->callfibm2 = callFib2;

  this->callfib = callFib;

  this->alloca = alloca;

  return module;
}

std::unique_ptr<jlm::RvsdgModule>
ExternalMemoryTest::SetupRvsdg()
{
  using namespace jlm;

  MemoryStateType mt;
  auto pt = ptrtype::create(jive::bit32);
  FunctionType ft({pt.get(), pt.get(), &mt}, {&mt});

  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  /**
   * Setup function f.
   */
  LambdaF = lambda::node::create(graph->root(), ft, "f", linkage::external_linkage);
  auto x = LambdaF->fctargument(0);
  auto y = LambdaF->fctargument(1);
  auto state = LambdaF->fctargument(2);

  auto one = jive::create_bitconstant(LambdaF->subregion(), 32, 1);
  auto two = jive::create_bitconstant(LambdaF->subregion(), 32, 2);

  auto storeOne = store_op::create(x, one, {state}, 4);
  auto storeTwo = store_op::create(y, two, {storeOne[0]}, 4);

  LambdaF->finalize(storeTwo);
  graph->add_export(LambdaF->output(), {ptrtype(LambdaF->type()), "f"});

  return module;
}