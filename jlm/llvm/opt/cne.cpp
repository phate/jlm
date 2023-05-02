/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class cnestat final : public Statistics {
public:
	virtual
	~cnestat()
	{}

	cnestat()
	: Statistics(Statistics::Id::CommonNodeElimination)
  , nnodes_before_(0), nnodes_after_(0)
	, ninputs_before_(0), ninputs_after_(0)
	{}

	void
	start_mark_stat(const jive::graph & graph) noexcept
	{
		nnodes_before_ = jive::nnodes(graph.root());
		ninputs_before_ = jive::ninputs(graph.root());
		marktimer_.start();
	}

	void
	end_mark_stat() noexcept
	{
		marktimer_.stop();
	}

	void
	start_divert_stat() noexcept
	{
		diverttimer_.start();
	}

	void
	end_divert_stat(const jive::graph & graph) noexcept
	{
		nnodes_after_ = jive::nnodes(graph.root());
		ninputs_after_ = jive::ninputs(graph.root());
		diverttimer_.stop();
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("CNE ",
			nnodes_before_, " ", nnodes_after_, " ",
			ninputs_before_, " ", ninputs_after_, " ",
			marktimer_.ns(), " ", diverttimer_.ns()
		);
	}

  static std::unique_ptr<cnestat>
  Create()
  {
    return std::make_unique<cnestat>();
  }

private:
	size_t nnodes_before_, nnodes_after_;
	size_t ninputs_before_, ninputs_after_;
	jlm::timer marktimer_, diverttimer_;
};


typedef std::unordered_set<jive::output*> congruence_set;

class cnectx {
public:
	inline void
	mark(jive::output * o1, jive::output * o2)
	{
		auto s1 = set(o1);
		auto s2 = set(o2);

		if (s1 == s2)
			return;

		if (s2->size() < s1->size()) {
			s1 = outputs_[o2];
			s2 = outputs_[o1];
		}

		for (auto & o : *s1) {
			s2->insert(o);
			outputs_[o] = s2;
		}
	}

	inline void
	mark(const jive::node * n1, const jive::node * n2)
	{
		JLM_ASSERT(n1->noutputs() == n2->noutputs());

		for (size_t n = 0; n < n1->noutputs(); n++)
			mark(n1->output(n), n2->output(n));
	}

	inline bool
	congruent(jive::output * o1, jive::output * o2) const noexcept
	{
		if (o1 == o2)
			return true;

		auto it = outputs_.find(o1);
		if (it == outputs_.end())
			return false;

		return it->second->find(o2) != it->second->end();
	}

	inline bool
	congruent(const jive::input * i1, const jive::input * i2) const noexcept
	{
		return congruent(i1->origin(), i2->origin());
	}

	congruence_set *
	set(jive::output * output) noexcept
	{
		if (outputs_.find(output) == outputs_.end()) {
			std::unique_ptr<congruence_set> set(new congruence_set({output}));
			outputs_[output] = set.get();
			sets_.insert(std::move(set));
		}

		return outputs_[output];
	}

private:
	std::unordered_set<std::unique_ptr<congruence_set>> sets_;
	std::unordered_map<const jive::output*, congruence_set*> outputs_;
};

class vset {
public:
	void
	insert(const jive::output * o1, const jive::output * o2)
	{
		auto it = sets_.find(o1);
		if (it != sets_.end())
			sets_[o1].insert(o2);
		else
			sets_[o1] = {o2};

		it = sets_.find(o2);
		if (it != sets_.end())
			sets_[o2].insert(o1);
		else
			sets_[o2] = {o1};
	}

	bool
	visited(const jive::output * o1, const jive::output * o2) const
	{
		auto it = sets_.find(o1);
		if (it == sets_.end())
			return false;

		return it->second.find(o2) != it->second.end();
	}

private:
	std::unordered_map<
		const jive::output*,
		std::unordered_set<const jive::output*>
	> sets_;
};

/* mark phase */

static bool
congruent(
	jive::output * o1,
	jive::output * o2,
	vset & vs,
	cnectx & ctx)
{
	if (ctx.congruent(o1, o2) || vs.visited(o1, o2))
		return true;

	if (o1->type() != o2->type())
		return false;

	if (is_theta_argument(o1) && is_theta_argument(o2)) {
		JLM_ASSERT(o1->region()->node() == o2->region()->node());
		auto a1 = static_cast<jive::argument*>(o1);
		auto a2 = static_cast<jive::argument*>(o2);
		vs.insert(a1, a2);
		auto i1 = a1->input(), i2 = a2->input();
		if (!congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx))
			return false;

		auto output1 = o1->region()->node()->output(i1->index());
		auto output2 = o2->region()->node()->output(i2->index());
		return congruent(output1, output2, vs, ctx);
	}

	auto n1 = jive::node_output::node(o1);
	auto n2 = jive::node_output::node(o2);
	if (jive::is<jive::theta_op>(n1) && jive::is<jive::theta_op>(n2) && n1 == n2) {
		auto so1 = static_cast<jive::structural_output*>(o1);
		auto so2 = static_cast<jive::structural_output*>(o2);
		vs.insert(o1, o2);
		auto r1 = so1->results.first();
		auto r2 = so2->results.first();
		return congruent(r1->origin(), r2->origin(), vs, ctx);
	}

	if (jive::is<jive::gamma_op>(n1) && n1 == n2) {
		auto so1 = static_cast<jive::structural_output*>(o1);
		auto so2 = static_cast<jive::structural_output*>(o2);
		auto r1 = so1->results.begin();
		auto r2 = so2->results.begin();
		for (; r1 != so1->results.end(); r1++, r2++) {
			JLM_ASSERT(r1->region() == r2->region());
			if (!congruent(r1->origin(), r2->origin(), vs, ctx))
				return false;
		}
		return true;
	}

	if (is_gamma_argument(o1) && is_gamma_argument(o2)) {
		JLM_ASSERT(o1->region()->node() == o2->region()->node());
		auto a1 = static_cast<jive::argument*>(o1);
		auto a2 = static_cast<jive::argument*>(o2);
		return congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx);
	}

	if (jive::is<jive::simple_op>(n1)
	&& jive::is<jive::simple_op>(n2)
	&& n1->operation() == n2->operation()
	&& n1->ninputs() == n2->ninputs()
	&& o1->index() == o2->index()) {
		for (size_t n = 0; n < n1->ninputs(); n++) {
			auto origin1 = n1->input(n)->origin();
			auto origin2 = n2->input(n)->origin();
			if (!congruent(origin1, origin2, vs, ctx))
				return false;
		}
		return true;
	}

	return false;
}

static bool
congruent(jive::output * o1, jive::output * o2, cnectx & ctx)
{
	vset vs;
	return congruent(o1, o2, vs, ctx);
}

static void
mark_arguments(
	jive::structural_input * i1,
	jive::structural_input * i2,
	cnectx & ctx)
{
	JLM_ASSERT(i1->node() && i1->node() == i2->node());
	JLM_ASSERT(i1->arguments.size() == i2->arguments.size());

	auto a1 = i1->arguments.begin();
	auto a2 = i2->arguments.begin();
	for (; a1 != i1->arguments.end(); a1++, a2++) {
		JLM_ASSERT(a1->region() == a2->region());
		if (congruent(a1.ptr(), a2.ptr(), ctx))
			ctx.mark(a1.ptr(), a2.ptr());
	}
}

static void
mark(jive::region*, cnectx&);

static void
mark_gamma(const jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<jive::gamma_op>(node->operation()));

	/* mark entry variables */
	for (size_t i1 = 1; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++)
			mark_arguments(node->input(i1), node->input(i2), ctx);
	}

	for (size_t n = 0; n < node->nsubregions(); n++)
		mark(node->subregion(n), ctx);

	/* mark exit variables */
	for (size_t o1 = 0; o1 < node->noutputs(); o1++) {
		for (size_t o2 = o1+1; o2 < node->noutputs(); o2++) {
			if (congruent(node->output(o1), node->output(o2), ctx))
				ctx.mark(node->output(o1), node->output(o2));
		}
	}
}

static void
mark_theta(const jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<jive::theta_op>(node));
	auto theta = static_cast<const jive::theta_node*>(node);

	/* mark loop variables */
	for (size_t i1 = 0; i1 < theta->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < theta->ninputs(); i2++) {
			auto input1 = theta->input(i1);
			auto input2 = theta->input(i2);
			if (congruent(input1->argument(), input2->argument(), ctx)) {
				ctx.mark(input1->argument(), input2->argument());
				ctx.mark(input1->output(), input2->output());
			}
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_lambda(const jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<lambda::operation>(node));

	/* mark dependencies */
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			auto input1 = node->input(i1);
			auto input2 = node->input(i2);
			if (ctx.congruent(input1, input2))
				ctx.mark(input1->arguments.first(), input2->arguments.first());
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_phi(const jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(is<phi::operation>(node));

	/* mark dependencies */
	for (size_t i1 = 0; i1 < node->ninputs(); i1++) {
		for (size_t i2 = i1+1; i2 < node->ninputs(); i2++) {
			auto input1 = node->input(i1);
			auto input2 = node->input(i2);
			if (ctx.congruent(input1, input2))
				ctx.mark(input1->arguments.first(), input2->arguments.first());
		}
	}

	mark(node->subregion(0), ctx);
}

static void
mark_delta(const jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<delta::operation>(node));
}

static void
mark(const jive::structural_node * node, cnectx & ctx)
{
	static std::unordered_map<
		std::type_index
	, void(*)(const jive::structural_node*, cnectx&)
	> map({
	  {std::type_index(typeid(jive::gamma_op)), mark_gamma}
	, {std::type_index(typeid(jive::theta_op)), mark_theta}
	, {typeid(lambda::operation), mark_lambda}
	, {typeid(phi::operation), mark_phi}
	, {typeid(delta::operation), mark_delta}
	});

	auto & op = node->operation();
	JLM_ASSERT(map.find(typeid(op)) != map.end());
	map[typeid(op)](node, ctx);
}

static void
mark(const jive::simple_node * node, cnectx & ctx)
{
	if (node->ninputs() == 0) {
		for (const auto & other : node->region()->top_nodes) {
			if (&other != node && node->operation() == other.operation()) {
				ctx.mark(node, &other);
				break;
			}
		}
		return;
	}

	auto set = ctx.set(node->input(0)->origin());
	for (const auto & origin : *set) {
		for (const auto & user : *origin) {
			auto ni = dynamic_cast<const jive::node_input*>(user);
			auto other = ni ? ni->node() : nullptr;
			if (!other
			|| other == node
			|| other->operation() != node->operation()
			|| other->ninputs() != node->ninputs())
				continue;

			size_t n;
			for (n = 0; n < node->ninputs(); n++) {
				if (!ctx.congruent(node->input(n), other->input(n)))
					break;
			}
			if (n == node->ninputs())
				ctx.mark(node, other);
		}
	}
}

static void
mark(jive::region * region, cnectx & ctx)
{
	for (const auto & node : jive::topdown_traverser(region)) {
		if (auto simple = dynamic_cast<const jive::simple_node*>(node))
			mark(simple, ctx);
		else
			mark(static_cast<const jive::structural_node*>(node), ctx);
	}
}

/* divert phase */

static void
divert_users(jive::output * output, cnectx & ctx)
{
	auto set = ctx.set(output);
	for (auto & other : *set)
		other->divert_users(output);
	set->clear();
}

static void
divert_outputs(jive::node * node, cnectx & ctx)
{
	for (size_t n = 0; n < node->noutputs(); n++)
		divert_users(node->output(n), ctx);
}

static void
divert_arguments(jive::region * region, cnectx & ctx)
{
	for (size_t n = 0; n < region->narguments(); n++)
		divert_users(region->argument(n), ctx);
}

static void
divert(jive::region*, cnectx&);

static void
divert_gamma(jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<jive::gamma_op>(node));
	auto gamma = static_cast<jive::gamma_node*>(node);

	for (auto ev = gamma->begin_entryvar(); ev != gamma->end_entryvar(); ev++) {
		for (size_t n = 0; n < ev->narguments(); n++)
			divert_users(ev->argument(n), ctx);
	}

	for (size_t r = 0; r < node->nsubregions(); r++)
		divert(node->subregion(r), ctx);

	divert_outputs(node, ctx);
}

static void
divert_theta(jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<jive::theta_op>(node));
	auto theta = static_cast<jive::theta_node*>(node);
	auto subregion = node->subregion(0);

	for (const auto & lv : *theta) {
		JLM_ASSERT(ctx.set(lv->argument())->size() == ctx.set(lv)->size());
		divert_users(lv->argument(), ctx);
		divert_users(lv, ctx);
	}

	divert(subregion, ctx);
}

static void
divert_lambda(jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<lambda::operation>(node));

	divert_arguments(node->subregion(0), ctx);
	divert(node->subregion(0), ctx);
}

static void
divert_phi(jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(is<phi::operation>(node));

	divert_arguments(node->subregion(0), ctx);
	divert(node->subregion(0), ctx);
}

static void
divert_delta(jive::structural_node * node, cnectx & ctx)
{
	JLM_ASSERT(jive::is<delta::operation>(node));
}

static void
divert(jive::structural_node * node, cnectx & ctx)
{
	static std::unordered_map<
		std::type_index,
		void(*)(jive::structural_node*, cnectx&)
	> map({
	  {std::type_index(typeid(jive::gamma_op)), divert_gamma}
	, {std::type_index(typeid(jive::theta_op)), divert_theta}
	, {typeid(lambda::operation), divert_lambda}
	, {typeid(phi::operation),divert_phi}
	, {typeid(delta::operation), divert_delta}
	});

	auto & op = node->operation();
	JLM_ASSERT(map.find(typeid(op)) != map.end());
	map[typeid(op)](node, ctx);
}

static void
divert(jive::region * region, cnectx & ctx)
{
	for (const auto & node : jive::topdown_traverser(region)) {
		if (auto simple = dynamic_cast<jive::simple_node*>(node))
			divert_outputs(simple, ctx);
		else
			divert(static_cast<jive::structural_node*>(node), ctx);
	}
}

static void
cne(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto & graph = rm.Rvsdg();

	cnectx ctx;
	auto statistics = cnestat::Create();

	statistics->start_mark_stat(graph);
	mark(graph.root(), ctx);
	statistics->end_mark_stat();

	statistics->start_divert_stat();
	divert(graph.root(), ctx);
	statistics->end_divert_stat(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* cne class */

cne::~cne()
{}

void
cne::run(
  RvsdgModule & module,
  StatisticsCollector & statisticsCollector)
{
	jlm::cne(module, statisticsCollector);
}

}
