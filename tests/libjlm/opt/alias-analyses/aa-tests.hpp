/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/theta.hpp>

#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/operators.hpp>

/**
* FIXME: write some documentation
*/
class aatest {
public:
	aatest(std::unique_ptr<jlm::rvsdg_module> module)
	: module_(std::move(module))
	{}

	jlm::rvsdg_module &
	module() const noexcept
	{
		return *module_;
	}

	const jive::graph &
	graph() const noexcept
	{
		return *module_->graph();
	}

private:
	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() = 0;

	std::unique_ptr<jlm::rvsdg_module> module_;
};

/** FIXME: update documentation
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f()
*   {
*     uint32_t d;
*     uint32_t * c;
*     uint32_t ** b;
*     uint32_t *** a;
*
*     a = &b;
*     b = &c;
*     c = &d;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class store_test1 final : public aatest {
public:
	store_test1()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		auto ppt = ptrtype::create(*pt);
		auto pppt = ptrtype::create(*ppt);
		jive::fcttype fcttype(
		  {&jive::memtype::instance()}
		, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto size = jive::create_bitconstant(fct->subregion(), 32, 4);

		auto d = alloca_op::create(jive::bit32, size, 4);
		auto c = alloca_op::create(*pt, size, 4);
		auto b = alloca_op::create(*ppt, size, 4);
		auto a = alloca_op::create(*pppt, size, 4);

		auto mux_d = memstatemux_op::create_merge({d[1], fct->fctargument(0)});
		auto mux_c = memstatemux_op::create_merge(std::vector<jive::output*>({c[1], mux_d}));
		auto mux_b = memstatemux_op::create_merge(std::vector<jive::output*>({b[1], mux_c}));
		auto mux_a = memstatemux_op::create_merge(std::vector<jive::output*>({a[1], mux_b}));

		auto a_amp_b = store_op::create(a[0], b[0], {mux_a}, 4);
		auto b_amp_c = store_op::create(b[0], c[0], {a_amp_b[0]}, 4);
		auto c_amp_d = store_op::create(c[0], d[0], {b_amp_c[0]}, 4);

		fct->finalize({c_amp_d[0]});

		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */

		this->lambda = fct;

		this->size = jive::node_output::node(size);

		this->alloca_a = jive::node_output::node(a[0]);
		this->alloca_b = jive::node_output::node(b[0]);
		this->alloca_c = jive::node_output::node(c[0]);
		this->alloca_d = jive::node_output::node(d[0]);

		this->mux_a = jive::node_output::node(mux_a);
		this->mux_b = jive::node_output::node(mux_b);
		this->mux_c = jive::node_output::node(mux_c);
		this->mux_d = jive::node_output::node(mux_d);

		this->store_b = jive::node_output::node(a_amp_b[0]);
		this->store_c = jive::node_output::node(b_amp_c[0]);
		this->store_d = jive::node_output::node(c_amp_d[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_c;
	jive::node * alloca_d;

	jive::node * mux_a;
	jive::node * mux_b;
	jive::node * mux_c;
	jive::node * mux_d;

	jive::node * store_b;
	jive::node * store_c;
	jive::node * store_d;
};


/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f()
*   {
*     uint32_t a, b;
*     uint32_t * x, * y;
*     uint32_t ** p;
*
*     x = &a;
*     y = &b;
*     p = &x;
*     p = &y;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class store_test2 final : public aatest {
public:
	store_test2()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		auto ppt = ptrtype::create(*pt);
		jive::fcttype fcttype(
		  {&jive::memtype::instance()}
		, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto size = jive::create_bitconstant(fct->subregion(), 32, 4);

		auto a = alloca_op::create(jive::bit32, size, 4);
		auto b = alloca_op::create(jive::bit32, size, 4);
		auto x = alloca_op::create(*pt, size, 4);
		auto y = alloca_op::create(*pt, size, 4);
		auto p = alloca_op::create(*ppt, size, 4);

		auto mux_a = memstatemux_op::create_merge({a[1], fct->fctargument(0)});
		auto mux_b = memstatemux_op::create_merge(std::vector<jive::output*>({b[1], mux_a}));
		auto mux_x = memstatemux_op::create_merge(std::vector<jive::output*>({x[1], mux_b}));
		auto mux_y = memstatemux_op::create_merge(std::vector<jive::output*>({y[1], mux_x}));
		auto mux_p = memstatemux_op::create_merge(std::vector<jive::output*>({p[1], mux_y}));

		auto x_amp_a = store_op::create(x[0], a[0], {mux_p}, 4);
		auto y_amp_b = store_op::create(y[0], b[0], {x_amp_a[0]}, 4);
		auto p_amp_x = store_op::create(p[0], x[0], {y_amp_b[0]}, 4);
		auto p_amp_y = store_op::create(p[0], y[0], {p_amp_x[0]}, 4);

		fct->finalize({p_amp_y[0]});

		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */

		this->lambda = fct;

		this->size = jive::node_output::node(size);

		this->alloca_a = jive::node_output::node(a[0]);
		this->alloca_b = jive::node_output::node(b[0]);
		this->alloca_x = jive::node_output::node(x[0]);
		this->alloca_y = jive::node_output::node(y[0]);
		this->alloca_p = jive::node_output::node(p[0]);

		this->mux_a = jive::node_output::node(mux_a);
		this->mux_b = jive::node_output::node(mux_b);
		this->mux_x = jive::node_output::node(mux_x);
		this->mux_y = jive::node_output::node(mux_y);
		this->mux_p = jive::node_output::node(mux_p);

		this->store_a = jive::node_output::node(x_amp_a[0]);
		this->store_b = jive::node_output::node(y_amp_b[0]);
		this->store_x = jive::node_output::node(p_amp_x[0]);
		this->store_y = jive::node_output::node(p_amp_y[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_p;

	jive::node * mux_a;
	jive::node * mux_b;
	jive::node * mux_x;
	jive::node * mux_y;
	jive::node * mux_p;

	jive::node * store_a;
	jive::node * store_b;
	jive::node * store_x;
	jive::node * store_y;
};


/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f()
*   {
*     uint32_t a, b;
*     uint32_t * c, * d;
*     uint32_t ** x, ** y;
*
*     c = &a;
*     d = &b;
*     x = &c;
*     y = &d;
*     x = y;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
static inline std::unique_ptr<jlm::rvsdg_module>
setup_assignment_test()
{
	using namespace jlm;

	auto pt = ptrtype::create(jive::bit32);
	auto ppt = ptrtype::create(*pt);
	jive::fcttype fcttype(
	  {&jive::memtype::instance()}
	, {&jive::memtype::instance()});

	auto module = rvsdg_module::create(filepath(""), "", "");
	auto graph = module->graph();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

	auto size = jive::create_bitconstant(fct->subregion(), 32, 4);

	auto a = alloca_op::create(jive::bit32, size, 4);
	auto b = alloca_op::create(jive::bit32, size, 4);
	auto c = alloca_op::create(*pt, size, 4);
	auto d = alloca_op::create(*pt, size, 4);
	auto x = alloca_op::create(*ppt, size, 4);
	auto y = alloca_op::create(*ppt, size, 4);

	auto mux1 = memstatemux_op::create_merge(std::vector<jive::output*>({a[1], fct->fctargument(0)}));
	auto mux2 = memstatemux_op::create_merge(std::vector<jive::output*>({b[1], mux1}));
	auto mux3 = memstatemux_op::create_merge(std::vector<jive::output*>({c[1], mux2}));
	auto mux4 = memstatemux_op::create_merge(std::vector<jive::output*>({d[1], mux3}));
	auto mux5 = memstatemux_op::create_merge(std::vector<jive::output*>({x[1], mux4}));
	auto mux6 = memstatemux_op::create_merge(std::vector<jive::output*>({y[1], mux5}));

	auto c_amp_a = store_op::create(c[0], a[0], {mux6}, 4);
	auto d_amp_b = store_op::create(d[0], b[0], {c_amp_a[0]}, 4);
	auto x_amp_c = store_op::create(x[0], c[0], {d_amp_b[0]}, 4);
	auto y_amp_d = store_op::create(y[0], d[0], {x_amp_c[0]}, 4);

	auto x_eq_y = store_op::create(x[0], d[0], {y_amp_d[0]}, 4);

	fct->finalize({x_eq_y[0]});

	graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

	return module;
}

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   uint32_t f(uint32_t ** p)
*   {
*     uint32_t * x = *p;
*     uint32_t a = *x;
*     return a;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class load_test1 final : public aatest {
public:
	load_test1()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		auto ppt = ptrtype::create(*pt);
		jive::fcttype fcttype(
		  {ppt.get(), &jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

public:
	jlm::lambda::node * lambda;

	jive::node * load_p;
	jive::node * load_x;
};


/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f()
*   {
*     uint32_t a, b;
*     uint32_t * x, * y;
*     uint32_t ** p;
*
*     x = &a;
*     y = &b;
*     p = &x;
*     y = *p;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class load_test2 final : public aatest {
public:
	load_test2()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		auto ppt = ptrtype::create(*pt);
		jive::fcttype fcttype(
		  {&jive::memtype::instance()}
		, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto size = jive::create_bitconstant(fct->subregion(), 32, 4);

		auto a = alloca_op::create(jive::bit32, size, 4);
		auto b = alloca_op::create(jive::bit32, size, 4);
		auto x = alloca_op::create(*pt, size, 4);
		auto y = alloca_op::create(*pt, size, 4);
		auto p = alloca_op::create(*ppt, size, 4);

		auto mux_a = memstatemux_op::create_merge({a[1], fct->fctargument(0)});
		auto mux_b = memstatemux_op::create_merge(std::vector<jive::output*>({b[1], mux_a}));
		auto mux_x = memstatemux_op::create_merge(std::vector<jive::output*>({x[1], mux_b}));
		auto mux_y = memstatemux_op::create_merge(std::vector<jive::output*>({y[1], mux_x}));
		auto mux_p = memstatemux_op::create_merge(std::vector<jive::output*>({p[1], mux_y}));

		auto x_amp_a = store_op::create(x[0], a[0], {mux_p}, 4);
		auto y_amp_b = store_op::create(y[0], b[0], x_amp_a, 4);
		auto p_amp_x = store_op::create(p[0], x[0], y_amp_b, 4);

		auto ld1 = load_op::create(p[0], p_amp_x, 4);
		auto ld2 = load_op::create(ld1[0], {ld1[1]}, 4);
		auto y_star_p = store_op::create(y[0], ld2[0], {ld2[1]}, 4);

		fct->finalize({y_star_p[0]});

		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */

		this->lambda = fct;

		this->size = jive::node_output::node(size);

		this->alloca_a = jive::node_output::node(a[0]);
		this->alloca_b = jive::node_output::node(b[0]);
		this->alloca_x = jive::node_output::node(x[0]);
		this->alloca_y = jive::node_output::node(y[0]);
		this->alloca_p = jive::node_output::node(p[0]);

		this->mux_a = jive::node_output::node(mux_a);
		this->mux_b = jive::node_output::node(mux_b);
		this->mux_x = jive::node_output::node(mux_x);
		this->mux_y = jive::node_output::node(mux_y);
		this->mux_p = jive::node_output::node(mux_p);

		this->store_ax = jive::node_output::node(x_amp_a[0]);
		this->store_b = jive::node_output::node(y_amp_b[0]);
		this->store_x = jive::node_output::node(p_amp_x[0]);

		this->load_x = jive::node_output::node(ld1[0]);
		this->load_a = jive::node_output::node(ld2[0]);

		this->store_ay = jive::node_output::node(y_star_p[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_p;

	jive::node * mux_a;
	jive::node * mux_b;
	jive::node * mux_x;
	jive::node * mux_y;
	jive::node * mux_p;

	jive::node * store_ax;
	jive::node * store_b;
	jive::node * store_x;

	jive::node * load_x;
	jive::node * load_a;

	jive::node * store_ay;
};


/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   struct point {
*     uint32_t x;
*     uint32_t y;
*   };
*
*   uint32_t f(const struct point * p)
*   {
*     return p->x + p->y;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class GetElementPtrTest final : public aatest {
public:
	GetElementPtrTest()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto dcl = jive::rcddeclaration::create({&jive::bit32, &jive::bit32});
		jive::rcdtype rt(dcl.get());

		auto pt = ptrtype::create(rt);
		auto pbt = ptrtype::create(jive::bit32);
		jive::fcttype fcttype(
		  {pt.get(), &jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

public:
	jlm::lambda::node * lambda;

	jive::node * getElementPtrX;
	jive::node * getElementPtrY;
};

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   uint16_t * f(uint32_t * p)
*   {
*     return (uint16_t*)p;
*   }
* \endcode
*/
class BitCastTest final : public aatest {
public:
	BitCastTest()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pbt16 = ptrtype::create(jive::bit16);
		auto pbt32 = ptrtype::create(jive::bit32);
		jive::fcttype fcttype({pbt32.get()}, {pbt16.get()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

public:
	jlm::lambda::node * lambda;

	jive::node * bitCast;
};

/**
* This function sets up an RVSDG representing the following code snippet:
*
* \code{.c}
*   static void*
*   bits2ptr(ptrdiff_t i)
*   {
*     return (void*)i;
*   }
*
*   void
*   test(ptrdiff_t i)
*   {
*     bit2ptr(i);
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory operations.
*/
class bits2ptr_test final : public aatest {
public:
	bits2ptr_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		jive::memtype mt;
		auto pt = ptrtype::create(jive::bit8);
		jive::fcttype fctbits2ptrtype({&jive::bit64, &mt}, {pt.get(), &mt});
		jive::fcttype fcttesttype({&jive::bit64, &mt}, {&mt});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		/* bit2ptr function */
		auto bits2ptrfct = lambda::node::create(graph->root(), fctbits2ptrtype, "bit2ptr",
			linkage::external_linkage);
		auto cast = bits2ptr_op::create(bits2ptrfct->fctargument(0), *pt);
		auto bits2ptr = bits2ptrfct->finalize({cast, bits2ptrfct->fctargument(1)});

		/* test function */
		auto testfct = lambda::node::create(graph->root(), fcttesttype, "test",
			linkage::external_linkage);
		auto cvbits2ptr = testfct->add_ctxvar(bits2ptr);

		auto call = call_op::create(cvbits2ptr, {testfct->fctargument(0), testfct->fctargument(1)});

		testfct->finalize({call[1]});
		graph->add_export(testfct->output(), {ptrtype(testfct->type()), "testfct"});

		/* extract nodes */

		this->lambda_bits2ptr = bits2ptrfct;
		this->lambda_test = testfct;

		this->bits2ptr = jive::node_output::node(cast);

		this->call = jive::node_output::node(call[0]);

		return module;
	}

	jlm::lambda::node * lambda_bits2ptr;
	jlm::lambda::node * lambda_test;

	jive::node * bits2ptr;

	jive::node * call;
};

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f(uint32_t ** i)
*   {
*	    *i = NULL;
*   }
* \endcode
*/
class ConstantPointerNullTest final : public aatest {
public:
	ConstantPointerNullTest()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		auto ppt = ptrtype::create(*pt);
		jive::fcttype fcttype(
		  {ppt.get(), &jive::memtype::instance()}
		, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto null = ptr_constant_null_op::create(fct->subregion(), *pt);
		auto st = store_op::create(fct->fctargument(0), null, {fct->fctargument(1)}, 4);

		fct->finalize({st[0]});

		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */
		this->lambda = fct;
		this->null = jive::node_output::node(null);

		return module;
	}

public:
	jlm::lambda::node * lambda;

	jive::node * null;
};

/**
* FIXME: update documentation
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*	  static uint32_t
*	  f(uint32_t * x, uint32_t * y)
*	  {
*	    return *x + *y;
*	  }
*
*	  static uint32_t
*	  g(uint32_t * x, uint32_t * y)
*	  {
*	    return *x - *y;
*	  }
*
*	  uint32_t
*	  h()
*	  {
*	    uint32_t x = 5, y = 6, z = 7;
*	    return f(&x, &y) + g(&z, &z);
*	  }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations within each function.
*/
class call_test1 final : public aatest {
public:
	call_test1()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		jive::fcttype ft1(
			{pt.get(), pt.get(), &jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		jive::fcttype ft2(
		  {&jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

		auto mx = memstatemux_op::create_merge(std::vector<jive::output*>({x[1], h->fctargument(0)}));
		auto my = memstatemux_op::create_merge(std::vector<jive::output*>({y[1], mx}));
		auto mz = memstatemux_op::create_merge(std::vector<jive::output*>({z[1], my}));

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

		return module;
	}

public:
	jlm::lambda::node * lambda_f;
	jlm::lambda::node * lambda_g;
	jlm::lambda::node * lambda_h;

	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_z;
};


/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*	  static uint32_t *
*	  create(size_t n)
*	  {
*	    return (uint32_t*)malloc(n * sizeof(uint32_t));
*	  }
*
*	  static void
*	  destroy(uint32_t * p)
*	  {
*	    free(p);
*	  }
*
*	  void
*	  test()
*	  {
*		  uint32_t * p1 = create(6);
*		  uint32_t * p2 = create(7);
*
*	    destroy(p1);
*		  destroy(p2);
*	  }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations within each function.
*/
class call_test2 final : public aatest {
public:
	call_test2()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
    using namespace jlm;

		iostatetype iot;
		jive::memtype mt;
		auto pbit8 = ptrtype::create(jive::bit8);
		auto pbit32 = ptrtype::create(jive::bit32);

		jive::fcttype create_type({&jive::bit32, &mt, &iot}, {pbit32.get(), &mt, &iot});
		jive::fcttype destroy_type({pbit32.get(), &mt, &iot}, {&mt, &iot});
		jive::fcttype test_type({&mt, &iot}, {&mt, &iot});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		/* function create */
		auto create = lambda::node::create(graph->root(), create_type, "create",
			linkage::external_linkage);

		auto four = jive::create_bitconstant(create->subregion(), 32, 4);
		auto prod = jive::bitmul_op::create(32, create->fctargument(0), four);

		auto alloc = malloc_op::create(prod);
		auto cast = bitcast_op::create(alloc[0], *pbit32);
		auto mx = memstatemux_op::create_merge(std::vector<jive::output*>(
			{alloc[1], create->fctargument(1)}));

		create->finalize({cast, mx, create->fctargument(2)});

		/* function destroy */
		auto destroy = lambda::node::create(graph->root(), destroy_type, "destroy",
			linkage::external_linkage);

		cast = bitcast_op::create(destroy->fctargument(0), *pbit8);
		auto free = free_op::create(cast, {destroy->fctargument(1)}, destroy->fctargument(2));
		destroy->finalize(free);

		/* function test */
		auto test = lambda::node::create(graph->root(), test_type, "test", linkage::external_linkage);
		auto create_cv = test->add_ctxvar(create->output());
		auto destroy_cv = test->add_ctxvar(destroy->output());

		auto six = jive::create_bitconstant(test->subregion(), 32, 6);
		auto seven = jive::create_bitconstant(test->subregion(), 32, 7);

		auto call_create1 = call_op::create(create_cv,
			{six, test->fctargument(0), test->fctargument(1)});
		auto call_create2 = call_op::create(create_cv, {seven, call_create1[1], call_create1[2]});

		auto call_destroy1 = call_op::create(destroy_cv,
			{call_create1[0], call_create2[1], call_create2[2]});
		auto call_destroy2 = call_op::create(destroy_cv,
			{call_create2[0], call_destroy1[0], call_destroy1[1]});

		test->finalize(call_destroy2);
		graph->add_export(test->output(), {ptrtype(test->type()), "test"});

		/* extract nodes */

		this->lambda_create = create;
		this->lambda_destroy = destroy;
		this->lambda_test = test;

		this->malloc = jive::node_output::node(alloc[0]);
		this->free = jive::node_output::node(free[0]);

		this->call_create1 = jive::node_output::node(call_create1[0]);
		this->call_create2 = jive::node_output::node(call_create2[0]);

		this->call_destroy1 = jive::node_output::node(call_destroy1[0]);
		this->call_destroy2 = jive::node_output::node(call_destroy2[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda_create;
	jlm::lambda::node * lambda_destroy;
	jlm::lambda::node * lambda_test;

	jive::node * malloc;
	jive::node * free;

	jive::node * call_create1;
	jive::node * call_create2;

	jive::node * call_destroy1;
	jive::node * call_destroy2;
};

/**
* This function sets up an RVSDG representing the following function:
*
*	\code{.c}
*	  static uint32_t
*	  four()
*	  {
*	    return 4;
*	  }
*
*	  static uint32_t
*	  three()
*	  {
*	    return 3;
*	  }
*
*	  static uint32_t
*	  indcall(uint32_t (*f)())
*	  {
*	    return (*f)();
*	  }
*
*	  uint32_t
*	  test()
*	  {
*	    return call(&four) + call(&three);
*	  }
*	\endcode
*
*	It uses a single memory state to sequentialize the respective memory
* operations within each function.
*/
class indirect_call_test final : public aatest {
public:
	indirect_call_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		iostatetype iot;
		jive::memtype mt;

		jive::fcttype four_type({&mt, &iot}, {&jive::bit32, &mt, &iot});

		auto pfct = ptrtype::create(four_type);
		jive::fcttype indcall_type({pfct.get(), &mt, &iot}, {&jive::bit32, &mt, &iot});

		jive::fcttype test_type({&mt, &iot}, {&jive::bit32, &mt, &iot});


		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

public:
	jlm::lambda::node * lambda_three;
	jlm::lambda::node * lambda_four;
	jlm::lambda::node * lambda_indcall;
	jlm::lambda::node * lambda_test;

	jive::node * call_fctindcall;
	jive::node * call_fctthree;
	jive::node * call_fctfour;
};

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   uint32_t f(uint32_t c, uint32_t * p1, uint32_t * p2, uint32_t * p3, uint32_t * p4)
*   {
*		  uint32_t * tmp1, * tmp2;
*     if (c == 0) {
*		    tmp1 = p1;
*       tmp2 = p2;
*     } else {
*		    tmp1 = p3;
*       tmp2 = p4;
*     }
*		  return *tmp1 + *tmp2;
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class gamma_test final : public aatest {
public:
	gamma_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		jive::fcttype fcttype(
			{&jive::bit32, pt.get(), pt.get(), pt.get(), pt.get(), &jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto zero = jive::create_bitconstant(fct->subregion(), 32, 0);
		auto biteq = jive::biteq_op::create(32, fct->fctargument(0), zero);
		auto predicate = jive::match(1, {{0, 1}}, 0, 2, biteq);

		auto gamma = jive::gamma_node::create(predicate, 2);
		auto p1ev = gamma->add_entryvar(fct->fctargument(1));
		auto p2ev = gamma->add_entryvar(fct->fctargument(2));
		auto p3ev = gamma->add_entryvar(fct->fctargument(3));
		auto p4ev = gamma->add_entryvar(fct->fctargument(4));

		auto tmp1 = gamma->add_exitvar({p1ev->argument(0), p3ev->argument(1)});
		auto tmp2 = gamma->add_exitvar({p2ev->argument(0), p4ev->argument(1)});

		auto ld1 = load_op::create(tmp1, {fct->fctargument(5)}, 4);
		auto ld2 = load_op::create(tmp2, {ld1[1]}, 4);
		auto sum = jive::bitadd_op::create(32, ld1[0], ld2[0]);

		fct->finalize({sum, ld2[1]});

		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */

		this->lambda = fct;
		this->gamma = gamma;

		return module;
	}

public:
	jlm::lambda::node * lambda;

	jive::gamma_node * gamma;
};

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   void f(uint32_t l, uint32_t  a[], uint32_t c)
*   {
*		  uint32_t n = 0;
*		  do {
*		    a[n++] = c;
*		  } while (n < l);
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class theta_test final : public aatest {
public:
	theta_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		jive::fcttype fcttype(
			{&jive::bit32, pt.get(), &jive::bit32, &jive::memtype::instance()}
		, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

		auto zero = jive::create_bitconstant(fct->subregion(), 32, 0);

		auto theta = jive::theta_node::create(fct->subregion());

		auto n = theta->add_loopvar(zero);
		auto l = theta->add_loopvar(fct->fctargument(0));
		auto a = theta->add_loopvar(fct->fctargument(1));
		auto c = theta->add_loopvar(fct->fctargument(2));
		auto s = theta->add_loopvar(fct->fctargument(3));

		auto gep = getelementptr_op::create(a->argument(), {n->argument()}, *pt);
		auto store = store_op::create(gep, c->argument(), {s->argument()}, 4);

		auto one = jive::create_bitconstant(theta->subregion(), 32, 1);
		auto sum = jive::bitadd_op::create(32, n->argument(), one);
		auto cmp = jive::bitult_op::create(32, sum, l->argument());
		auto predicate = jive::match(1, {{1, 1}}, 0, 2, cmp);

		n->result()->divert_to(sum);
		s->result()->divert_to(store[0]);
		theta->set_predicate(predicate);

		fct->finalize({s});
		graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

		/* extract nodes */

		this->lambda = fct;
		this->theta = theta;
		this->gep = jive::node_output::node(gep);

		return module;
	}

	jlm::lambda::node * lambda;
	jive::theta_node * theta;
	jive::node * gep;
};

/**
* This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   static uint32_t f;
*
*   static uint32_t
*   g(uint32_t * v)
*   {
*     return *v;
*   }
*
*   uint32_t
*   h()
*   {
*     f = 5;
*     return g(&f);
*   }
* \endcode
*
* It uses a single memory state to sequentialize the respective memory
* operations.
*/
class delta_test1 final : public aatest {
public:
	delta_test1()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		auto pt = ptrtype::create(jive::bit32);
		jive::fcttype fctgtype(
			{pt.get(), &jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});
		jive::fcttype fcthtype(
			{&jive::memtype::instance()}
		, {&jive::bit32, &jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		/* global f */
		auto delta_f = delta::node::create(
			graph->root(),
			ptrtype(jive::bit32),
			"f",
			linkage::external_linkage,
			false);
		auto f = delta_f->finalize(jive::create_bitconstant(delta_f->subregion(), 32, 0));

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

		this->delta_f = delta_f;

		this->call_g = jive::node_output::node(callg[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda_g;
	jlm::lambda::node * lambda_h;

	jlm::delta::node * delta_f;

	jive::node * call_g;
};

/**
*	This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   static uint32_t d1 = 0;
*   static uint32_t d2 = 0;
*
*   static void
*   f1()
*   {
*     d1 = 2;
*   }
*
*   void
*   f2()
*   {
*     d1 = 5;
*     f1();
*     d2 = 42;
*   }
* \endcode
*
* It uses a signle memory state to sequentialize the respective memory
* operations.
*/
class delta_test2 final : public aatest {
public:
	delta_test2()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		jive::fcttype ft({&jive::memtype::instance()}, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

		auto nf = graph->node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		/* global d1 */
		auto delta_d1 = delta::node::create(
			graph->root(),
			ptrtype(jive::bit32),
			"d1",
			linkage::external_linkage,
			false);
		auto d1 = delta_d1->finalize(jive::create_bitconstant(delta_d1->subregion(), 32, 0));

		/* global d2 */
		auto delta_d2 = delta::node::create(
			graph->root(),
			ptrtype(jive::bit32),
			"d2",
			linkage::external_linkage,
			false);
		auto d2 = delta_d2->finalize(jive::create_bitconstant(delta_d2->subregion(), 32, 0));

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

		this->delta_d1 = delta_d1;
		this->delta_d2 = delta_d2;

		this->call_f1 = jive::node_output::node(callf1[0]);

		return module;
	}

public:
	jlm::lambda::node * lambda_f1;
	jlm::lambda::node * lambda_f2;

	jlm::delta::node * delta_d1;
	jlm::delta::node * delta_d2;

	jive::node * call_f1;
};

/**
*	This function sets up an RVSDG representing the following function:
*
* \code{.c}
*   extern uint32_t d1 = 0;
*   extern uint32_t d2 = 0;
*
*   static void
*   f1()
*   {
*     d1 = 5;
*   }
*
*   void
*   f2()
*   {
*     d1 = 2;
*     f1();
*     d2 = 21;
*   }
* \endcode
*
* It uses a signle memory state to sequentialize the respective memory
* operations.
*/
class import_test final : public aatest {
public:
	import_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		jive::fcttype ft({&jive::memtype::instance()}, {&jive::memtype::instance()});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

public:
	jlm::lambda::node * lambda_f1;
	jlm::lambda::node * lambda_f2;

	jive::node * call_f1;

	jive::argument * import_d1;
	jive::argument * import_d2;
};

/**
*	This function sets up an RVSDG representing the following code snippet:
*
* \code{.c}
*	  void
*	  fib(uint64_t n, uint64_t result[])
*	  {
*	    if (n < 2) {
*	      result[n] = n;
*	      return;
*	    }
*
*	    fib(n-1, result);
*	    fib(n-2, result);
*	    result[n] = result[n-1] + result[n-2];
*	  }
*
*	  void
*	  test()
*	  {
*	    uint64_t n = 10;
*	    uint64_t results[n];
*
*	    fib(n, results);
*	  }
* \endcode
*
* It uses a signle memory state to sequentialize the respective memory
* operations.
*/
class phi_test final : public aatest {
public:
	phi_test()
	: aatest(setup())
	{}

	virtual std::unique_ptr<jlm::rvsdg_module>
	setup() override
	{
		using namespace jlm;

		jive::memtype mt;

		arraytype at(jive::bit64, 10);
		ptrtype pat(at);

		ptrtype pbit64(jive::bit64);

		jive::fcttype fibfcttype(
			{&jive::bit64, &pbit64, &mt},
			{&mt});
		ptrtype pfibfcttype(fibfcttype);

		jive::fcttype testfcttype({&mt}, {&mt});

		auto module = rvsdg_module::create(filepath(""), "", "");
		auto graph = module->graph();

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

		auto gamma = jive::gamma_node::create(predicate, 2);
		auto nev = gamma->add_entryvar(fibfct->fctargument(0));
		auto resultev = gamma->add_entryvar(fibfct->fctargument(1));
		auto fibev = gamma->add_entryvar(fibfct->cvargument(0));
		auto stateev = gamma->add_entryvar(fibfct->fctargument(2));

		/* gamma subregion 0 */
		auto one = jive::create_bitconstant(gamma->subregion(0), 64, 1);
		auto nm1 = jive::bitsub_op::create(64, nev->argument(0), one);
		auto callfibm1 = call_op::create(fibev->argument(0), {nm1, resultev->argument(0),
			stateev->argument(0)});

		two = jive::create_bitconstant(gamma->subregion(0), 64, 2);
		auto nm2 = jive::bitsub_op::create(64, nev->argument(0), two);
		auto callfibm2 = call_op::create(fibev->argument(0), {nm2, resultev->argument(0),
			callfibm1[0]});

		auto gepnm1 = getelementptr_op::create(resultev->argument(0), {nm1}, pbit64);
		auto ldnm1 = load_op::create(gepnm1, {callfibm2[0]}, 8);

		auto gepnm2 = getelementptr_op::create(resultev->argument(0), {nm2}, pbit64);
		auto ldnm2 = load_op::create(gepnm2, {ldnm1[1]}, 8);

		auto sum = jive::bitadd_op::create(64, ldnm1[0], ldnm2[0]);

		/* gamma subregion 1 */
		/* Nothing needs to be done */

		auto sumex = gamma->add_exitvar({sum, nev->argument(1)});
		auto stateex = gamma->add_exitvar({ldnm2[1], stateev->argument(1)});

		auto gepn = getelementptr_op::create(fibfct->fctargument(1), {fibfct->fctargument(0)}, pbit64);
		auto store = store_op::create(gepn, sumex, {stateex}, 8);

		auto fib = fibfct->finalize({store[0]});

		fibrv->result()->divert_to(fib);
		auto phi = pb.end();

		/* test function */
		auto testfct = lambda::node::create(graph->root(), testfcttype, "test",
			linkage::external_linkage);
		fibcv = testfct->add_ctxvar(phi->output(0));

		auto ten = jive::create_bitconstant(testfct->subregion(), 64, 10);
		auto alloca = alloca_op::create(at, ten, 16);
		auto state = memstatemux_op::create_merge({alloca[1], testfct->fctargument(0)});

		auto zero = jive::create_bitconstant(testfct->subregion(), 64, 0);
		auto gep = getelementptr_op::create(alloca[0], {zero, zero}, pbit64);

		auto call = call_op::create(fibcv, {ten, gep, state});

		testfct->finalize({call[0]});
		graph->add_export(testfct->output(), {ptrtype(testfcttype), "test"});

		/* extract nodes */

		this->lambda_fib = fibfct;
		this->lambda_test = testfct;

		this->gamma = gamma;
		this->phi = phi;

		this->callfibm1 = jive::node_output::node(callfibm1[0]);
		this->callfibm2 = jive::node_output::node(callfibm2[0]);

		this->callfib = jive::node_output::node(call[0]);

		this->alloca = jive::node_output::node(alloca[0]);

		return module;
	}

	jlm::lambda::node * lambda_fib;
	jlm::lambda::node * lambda_test;

	jive::gamma_node * gamma;

	jive::phi::node * phi;

	jive::node * callfibm1;
	jive::node * callfibm2;

	jive::node * callfib;

	jive::node * alloca;
};
