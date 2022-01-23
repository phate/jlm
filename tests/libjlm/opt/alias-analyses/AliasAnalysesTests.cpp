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

/* extract nodes */

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

  /* extract nodes */

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

  /* bit2ptr function */
  auto bits2ptrfct = lambda::node::create(graph->root(), fctbits2ptrtype, "bit2ptr",
                                          linkage::external_linkage);
  auto cast = bits2ptr_op::create(bits2ptrfct->fctargument(0), *pt);
  auto b2p = bits2ptrfct->finalize({cast, bits2ptrfct->fctargument(1)});

  /* test function */
  auto testfct = lambda::node::create(graph->root(), fcttesttype, "test",
                                      linkage::external_linkage);
  auto cvbits2ptr = testfct->add_ctxvar(b2p);

  auto results = call_op::create(cvbits2ptr, {testfct->fctargument(0), testfct->fctargument(1)});

  testfct->finalize({results[1]});
  graph->add_export(testfct->output(), {ptrtype(testfct->type()), "testfct"});

  /* extract nodes */

  this->lambda_bits2ptr = bits2ptrfct;
  this->lambda_test = testfct;

  this->bits2ptr = jive::node_output::node(cast);

  this->call = jive::node_output::node(results[0]);

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

  /* extract nodes */
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

/* function f */
  auto f = lambda::node::create(graph->root(), ft1, "f", linkage::external_linkage);

  auto ld1 = load_op::create(f->fctargument(0), {f->fctargument(2)}, 4);
  auto ld2 = load_op::create(f->fctargument(1), {ld1[1]}, 4);

  auto sum = jive::bitadd_op::create(32, ld1[0], ld2[0]);

  f->finalize({sum, ld2[1]});

/* function g */
  auto g = lambda::node::create(graph->root(), ft1, "g", linkage::external_linkage);

  ld1 = load_op::create(g->fctargument(0), {g->fctargument(2)}, 4);
  ld2 = load_op::create(g->fctargument(1), {ld1[1]}, 4);

  auto diff = jive::bitsub_op::create(32, ld1[0], ld2[0]);

  g->finalize({diff, ld2[1]});

/* function h */
  auto h = lambda::node::create(graph->root(), ft2, "h", linkage::external_linkage);

  auto cvf = h->add_ctxvar(f->output());
  auto cvg = h->add_ctxvar(g->output());

  auto size = jive::create_bitconstant(h->subregion(), 32, 4);

  auto x = alloca_op::create(jive::bit32, size, 4);
  auto y = alloca_op::create(jive::bit32, size, 4);
  auto z = alloca_op::create(jive::bit32, size, 4);

  auto mx = MemStateMergeOperator::Create(std::vector<jive::output *>({x[1], h->fctargument(0)}));
  auto my = MemStateMergeOperator::Create(std::vector<jive::output *>({y[1], mx}));
  auto mz = MemStateMergeOperator::Create(std::vector<jive::output *>({z[1], my}));

  auto five = jive::create_bitconstant(h->subregion(), 32, 5);
  auto six = jive::create_bitconstant(h->subregion(), 32, 6);
  auto seven = jive::create_bitconstant(h->subregion(), 32, 7);

  auto stx = store_op::create(x[0], five, {mz}, 4);
  auto sty = store_op::create(y[0], six, {stx[0]}, 4);
  auto stz = store_op::create(z[0], seven, {sty[0]}, 4);

  auto callf = call_op::create(cvf, {x[0], y[0], stz[0]});
  auto callg = call_op::create(cvg, {z[0], z[0], callf[1]});

  sum = jive::bitadd_op::create(32, callf[0], callg[0]);

  h->finalize({sum, callg[1]});
  graph->add_export(h->output(), {ptrtype(h->type()), "h"});

/* extract nodes */

  this->lambda_f = f;
  this->lambda_g = g;
  this->lambda_h = h;

  this->alloca_x = jive::node_output::node(x[0]);
  this->alloca_y = jive::node_output::node(y[0]);
  this->alloca_z = jive::node_output::node(z[0]);

  this->callF = jive::node_output::node(callf[0]);
  this->callG = jive::node_output::node(callg[0]);

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

  /* function create */
  auto create = lambda::node::create(graph->root(), create_type, "create",
                                     linkage::external_linkage);

  auto four = jive::create_bitconstant(create->subregion(), 32, 4);
  auto prod = jive::bitmul_op::create(32, create->fctargument(0), four);

  auto alloc = malloc_op::create(prod);
  auto cast = bitcast_op::create(alloc[0], *pbit32);
  auto mx = MemStateMergeOperator::Create(std::vector<jive::output *>(
    {alloc[1], create->fctargument(1)}));

  create->finalize({cast, mx, create->fctargument(2)});

  /* function destroy */
  auto destroy = lambda::node::create(graph->root(), destroy_type, "destroy",
                                      linkage::external_linkage);

  cast = bitcast_op::create(destroy->fctargument(0), *pbit8);
  auto freenode = free_op::create(cast, {destroy->fctargument(1)}, destroy->fctargument(2));
  destroy->finalize(freenode);

  /* function test */
  auto test = lambda::node::create(graph->root(), test_type, "test", linkage::external_linkage);
  auto create_cv = test->add_ctxvar(create->output());
  auto destroy_cv = test->add_ctxvar(destroy->output());

  auto six = jive::create_bitconstant(test->subregion(), 32, 6);
  auto seven = jive::create_bitconstant(test->subregion(), 32, 7);

  auto create1 = call_op::create(
    create_cv,
    {six, test->fctargument(0), test->fctargument(1)});
  auto create2 = call_op::create(create_cv, {seven, create1[1], create1[2]});

  auto destroy1 = call_op::create(destroy_cv, {create1[0], create2[1], create2[2]});
  auto destroy2 = call_op::create(destroy_cv, {create2[0], destroy1[0], destroy1[1]});

  test->finalize(destroy2);
  graph->add_export(test->output(), {ptrtype(test->type()), "test"});

  /* extract nodes */

  this->lambda_create = create;
  this->lambda_destroy = destroy;
  this->lambda_test = test;

  this->malloc = jive::node_output::node(alloc[0]);
  this->free = jive::node_output::node(freenode[0]);

  this->call_create1 = jive::node_output::node(create1[0]);
  this->call_create2 = jive::node_output::node(create2[0]);

  this->call_destroy1 = jive::node_output::node(destroy1[0]);
  this->call_destroy2 = jive::node_output::node(destroy2[0]);

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
  FunctionType indcall_type({pfct.get(), &mt, &iot}, {&jive::bit32, &mt, &iot});

  FunctionType test_type({&mt, &iot}, {&jive::bit32, &mt, &iot});


  auto module = RvsdgModule::Create(filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jive::operation));
  nf->set_mutable(false);

  /* function four */
  auto fctfour = lambda::node::create(graph->root(), four_type, "four",
                                      linkage::external_linkage);

  auto four = jive::create_bitconstant(fctfour->subregion(), 32, 4);

  fctfour->finalize({four, fctfour->fctargument(0), fctfour->fctargument(1)});

  /* function three */
  auto fctthree = lambda::node::create(graph->root(), four_type, "three",
                                       linkage::external_linkage);

  auto three = jive::create_bitconstant(fctthree->subregion(), 32, 3);

  fctthree->finalize({three, fctthree->fctargument(0), fctthree->fctargument(1)});

  /* function call */
  auto fctindcall = lambda::node::create(graph->root(), indcall_type, "indcall",
                                         linkage::external_linkage);

  auto call = call_op::create(fctindcall->fctargument(0),
                              {fctindcall->fctargument(1), fctindcall->fctargument(2)});

  fctindcall->finalize(call);

  /* function test */
  auto fcttest = lambda::node::create(graph->root(), test_type, "test",
                                      linkage::external_linkage);
  auto fctindcall_cv = fcttest->add_ctxvar(fctindcall->output());
  auto fctfour_cv = fcttest->add_ctxvar(fctfour->output());
  auto fctthree_cv = fcttest->add_ctxvar(fctthree->output());

  auto call_four = call_op::create(fctindcall_cv,
                                   {fctfour_cv, fcttest->fctargument(0), fcttest->fctargument(1)});
  auto call_three = call_op::create(fctindcall_cv,
                                    {fctthree_cv, call_four[1], call_four[2]});

  auto add = jive::bitadd_op::create(32, call_four[0], call_three[0]);

  fcttest->finalize({add, call_three[1], call_three[2]});
  graph->add_export(fcttest->output(), {ptrtype(fcttest->type()), "test"});

  /* extract nodes */

  this->lambda_three = fctthree;
  this->lambda_four = fctfour;
  this->lambda_indcall = fctindcall;
  this->lambda_test = fcttest;

  this->call_fctindcall = jive::node_output::node(call[0]);
  this->call_fctthree = jive::node_output::node(call_three[0]);
  this->call_fctfour = jive::node_output::node(call_four[0]);

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

  /* extract nodes */

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

  /* extract nodes */

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

  /* global f */
  auto dfNode = delta::node::create(
    graph->root(),
    ptrtype(jive::bit32),
    "f",
    linkage::external_linkage,
    false);
  auto f = dfNode->finalize(jive::create_bitconstant(dfNode->subregion(), 32, 0));

  /* function g */
  auto g = lambda::node::create(graph->root(), fctgtype, "g", linkage::external_linkage);
  auto ld = load_op::create(g->fctargument(0), {g->fctargument(1)}, 4);
  g->finalize(ld);

  /* function h */
  auto h = lambda::node::create(graph->root(), fcthtype, "h", linkage::external_linkage);
  auto cvf = h->add_ctxvar(f);
  auto cvg = h->add_ctxvar(g->output());

  auto five = jive::create_bitconstant(h->subregion(), 32, 5);
  auto st = store_op::create(cvf, five, {h->fctargument(0)}, 4);
  auto callg = call_op::create(cvg, {cvf, st[0]});

  h->finalize(callg);
  graph->add_export(h->output(), {ptrtype(h->type()), "h"});

  /* extract nodes */

  this->lambda_g = g;
  this->lambda_h = h;

  this->delta_f = dfNode;

  this->call_g = jive::node_output::node(callg[0]);
  this->constantFive = jive::node_output::node(five);

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

  /* global d1 */
  auto d1Node = delta::node::create(
    graph->root(),
    ptrtype(jive::bit32),
    "d1",
    linkage::external_linkage,
    false);
  auto d1 = d1Node->finalize(jive::create_bitconstant(d1Node->subregion(), 32, 0));

  /* global d2 */
  auto d2Node = delta::node::create(
    graph->root(),
    ptrtype(jive::bit32),
    "d2",
    linkage::external_linkage,
    false);
  auto d2 = d2Node->finalize(jive::create_bitconstant(d2Node->subregion(), 32, 0));

  /* function f1 */
  auto f1 = lambda::node::create(graph->root(), ft, "f1", linkage::external_linkage);
  auto cvd1 = f1->add_ctxvar(d1);
  auto b2 = jive::create_bitconstant(f1->subregion(), 32, 2);
  auto st = store_op::create(cvd1, b2, {f1->fctargument(0)}, 4);
  f1->finalize(st);

  /* function f2 */
  auto f2 = lambda::node::create(graph->root(), ft, "f2", linkage::external_linkage);
  cvd1 = f2->add_ctxvar(d1);
  auto cvd2 = f2->add_ctxvar(d2);
  auto cvf1 = f2->add_ctxvar(f1->output());
  auto b5 = jive::create_bitconstant(f2->subregion(), 32, 5);
  auto b42 = jive::create_bitconstant(f2->subregion(), 32, 42);
  st = store_op::create(cvd1, b5, {f2->fctargument(0)}, 4);
  auto callf1 = call_op::create(cvf1, st);
  st = store_op::create(cvd2, b42, callf1, 4);

  f2->finalize(st);
  graph->add_export(f2->output(), {ptrtype(f2->type()), "f2"});

  /* extract nodes */

  this->lambda_f1 = f1;
  this->lambda_f2 = f2;

  this->delta_d1 = d1Node;
  this->delta_d2 = d2Node;

  this->call_f1 = jive::node_output::node(callf1[0]);

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

  /* global d1 */
  auto d1 = graph->add_import(impport(ptrtype(jive::bit32), "d1", linkage::external_linkage));

  /* global d2 */
  auto d2 = graph->add_import(impport(ptrtype(jive::bit32), "d2", linkage::external_linkage));

  /* function f1 */
  auto f1 = lambda::node::create(graph->root(), ft, "f1", linkage::external_linkage);
  auto cvd1 = f1->add_ctxvar(d1);
  auto b5 = jive::create_bitconstant(f1->subregion(), 32, 5);
  auto st = store_op::create(cvd1, b5, {f1->fctargument(0)}, 4);
  f1->finalize(st);

  /* function f2 */
  auto f2 = lambda::node::create(graph->root(), ft, "f2", linkage::external_linkage);
  cvd1 = f2->add_ctxvar(d1);
  auto cvd2 = f2->add_ctxvar(d2);
  auto cvf1 = f2->add_ctxvar(f1->output());
  auto b2 = jive::create_bitconstant(f2->subregion(), 32, 2);
  auto b21 = jive::create_bitconstant(f2->subregion(), 32, 21);
  st = store_op::create(cvd1, b2, {f2->fctargument(0)}, 4);
  auto callf1 = call_op::create(cvf1, st);
  st = store_op::create(cvd2, b21, callf1, 4);

  f2->finalize(st);
  graph->add_export(f2->output(), {ptrtype(f2->type()), "f2"});

  /* extract nodes */

  this->lambda_f1 = f1;
  this->lambda_f2 = f2;

  this->call_f1 = jive::node_output::node(callf1[0]);

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

  /* fib function */
  jive::phi::builder pb;
  pb.begin(graph->root());
  auto fibrv = pb.add_recvar(pfibfcttype);

  auto fibfct = lambda::node::create(pb.subregion(), fibfcttype, "fib",
                                     linkage::external_linkage);
  auto fibcv = fibfct->add_ctxvar(fibrv->argument());

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
  auto callfibm1Results = call_op::create(
    fibev->argument(0),
    {nm1, resultev->argument(0),
     stateev->argument(0)});

  two = jive::create_bitconstant(gammaNode->subregion(0), 64, 2);
  auto nm2 = jive::bitsub_op::create(64, nev->argument(0), two);
  auto callfibm2Results = call_op::create(
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

  auto fib = fibfct->finalize({store[0]});

  fibrv->result()->divert_to(fib);
  auto phiNode = pb.end();

  /* test function */
  auto testfct = lambda::node::create(graph->root(), testfcttype, "test",
                                      linkage::external_linkage);
  fibcv = testfct->add_ctxvar(phiNode->output(0));

  auto ten = jive::create_bitconstant(testfct->subregion(), 64, 10);
  auto allocaResults = alloca_op::create(at, ten, 16);
  auto state = MemStateMergeOperator::Create({allocaResults[1], testfct->fctargument(0)});

  auto zero = jive::create_bitconstant(testfct->subregion(), 64, 0);
  auto gep = getelementptr_op::create(allocaResults[0], {zero, zero}, pbit64);

  auto call = call_op::create(fibcv, {ten, gep, state});

  testfct->finalize({call[0]});
  graph->add_export(testfct->output(), {ptrtype(testfcttype), "test"});

  /* extract nodes */

  this->lambda_fib = fibfct;
  this->lambda_test = testfct;

  this->gamma = gammaNode;
  this->phi = phiNode;

  this->callfibm1 = jive::node_output::node(callfibm1Results[0]);
  this->callfibm2 = jive::node_output::node(callfibm2Results[0]);

  this->callfib = jive::node_output::node(call[0]);

  this->alloca = jive::node_output::node(allocaResults[0]);

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