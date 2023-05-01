/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/rvsdg2jlm/context.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <deque>

namespace jlm {

class rvsdg_destruction_stat final : public Statistics {
public:
	virtual
	~rvsdg_destruction_stat()
	{}

	rvsdg_destruction_stat(const jlm::filepath & filename)
	: Statistics(Statistics::Id::RvsdgDestruction)
  , ntacs_(0)
	, nnodes_(0)
	, filename_(filename)
	{}

	void
	start(const jive::graph & graph) noexcept
	{
		nnodes_ = jive::nnodes(graph.root());
		timer_.start();
	}

	void
	end(const ipgraph_module & im)
	{
		ntacs_ = jlm::ntacs(im);
		timer_.stop();
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("RVSDGDESTRUCTION ",
			filename_.to_str(), " ",
			nnodes_, " ", ntacs_, " ",
			timer_.ns()
		);
	}

  static std::unique_ptr<rvsdg_destruction_stat>
  Create(const jlm::filepath & sourceFile)
  {
    return std::make_unique<rvsdg_destruction_stat>(sourceFile);
  }

private:
	size_t ntacs_;
	size_t nnodes_;
	jlm::timer timer_;
	jlm::filepath filename_;
};

namespace rvsdg2jlm {

static const FunctionType *
is_function_import(const jive::argument * argument)
{
    JLM_ASSERT(argument->region()->graph()->root() == argument->region());

    if (auto rvsdgImport = dynamic_cast<const impport*>(&argument->port()))
    {
        return dynamic_cast<const FunctionType*>(&rvsdgImport->GetValueType());
    }

    return nullptr;
}

static std::unique_ptr<data_node_init>
create_initialization(const delta::node * delta, context & ctx)
{
	auto subregion = delta->subregion();

	/* add delta dependencies to context */
	for (size_t n = 0; n < delta->ninputs(); n++) {
		auto v = ctx.variable(delta->input(n)->origin());
		ctx.insert(delta->input(n)->arguments.first(), v);
	}


	if (subregion->nnodes() == 0) {
		auto value = ctx.variable(subregion->result(0)->origin());
		return std::make_unique<data_node_init>(value);
	}


	tacsvector_t tacs;
	for (const auto & node : jive::topdown_traverser(delta->subregion())) {
		JLM_ASSERT(node->noutputs() == 1);
		auto output = node->output(0);

		/* collect operand variables */
		std::vector<const variable*> operands;
		for (size_t n = 0; n < node->ninputs(); n++)
			operands.push_back(ctx.variable(node->input(n)->origin()));

		/* convert node to tac */
		auto & op = *static_cast<const jive::simple_op*>(&node->operation());
		tacs.push_back(tac::create(op, operands));
		ctx.insert(output, tacs.back()->result(0));
	}

	return std::make_unique<data_node_init>(std::move(tacs));
}

static void
convert_node(const jive::node & node, context & ctx);

static inline void
convert_region(jive::region & region, context & ctx)
{
	auto entry = basic_block::create(*ctx.cfg());
	ctx.lpbb()->add_outedge(entry);
	ctx.set_lpbb(entry);

	for (const auto & node : jive::topdown_traverser(&region))
		convert_node(*node, ctx);

	auto exit = basic_block::create(*ctx.cfg());
	ctx.lpbb()->add_outedge(exit);
	ctx.set_lpbb(exit);
}

static inline std::unique_ptr<jlm::cfg>
create_cfg(const lambda::node & lambda, context & ctx)
{
	JLM_ASSERT(ctx.lpbb() == nullptr);
	std::unique_ptr<jlm::cfg> cfg(new jlm::cfg(ctx.module()));
	auto entry = basic_block::create(*cfg);
	cfg->exit()->divert_inedges(entry);
	ctx.set_lpbb(entry);
	ctx.set_cfg(cfg.get());

	/* add arguments */
	for (auto & fctarg : lambda.fctarguments()) {
		auto argument = jlm::argument::create("", fctarg.type(), fctarg.attributes());
		auto v = cfg->entry()->append_argument(std::move(argument));
		ctx.insert(&fctarg, v);
	}

	/* add context variables */
	for (auto & cv : lambda.ctxvars()) {
		auto v = ctx.variable(cv.origin());
		ctx.insert(cv.argument(), v);
	}

	convert_region(*lambda.subregion(), ctx);

	/* add results */
	for (auto & result : lambda.fctresults())
		cfg->exit()->append_result(ctx.variable(result.origin()));

	ctx.lpbb()->add_outedge(cfg->exit());
	ctx.set_lpbb(nullptr);
	ctx.set_cfg(nullptr);

	straighten(*cfg);
	JLM_ASSERT(is_closed(*cfg));
	return cfg;
}

static inline void
convert_simple_node(const jive::node & node, context & ctx)
{
	JLM_ASSERT(dynamic_cast<const jive::simple_op*>(&node.operation()));

	std::vector<const variable*> operands;
	for (size_t n = 0; n < node.ninputs(); n++)
		operands.push_back(ctx.variable(node.input(n)->origin()));

	auto & op = *static_cast<const jive::simple_op*>(&node.operation());
	ctx.lpbb()->append_last(tac::create(op, operands));

	for (size_t n = 0; n < node.noutputs(); n++)
		ctx.insert(node.output(n), ctx.lpbb()->last()->result(n));
}

static void
convert_empty_gamma_node(const jive::gamma_node * gamma, context & ctx)
{
	JLM_ASSERT(gamma->nsubregions() == 2);
	JLM_ASSERT(gamma->subregion(0)->nnodes() == 0 && gamma->subregion(1)->nnodes() == 0);

	/* both regions are empty, create only select instructions */

	auto predicate = gamma->predicate()->origin();
	auto cfg = ctx.cfg();

	auto bb = basic_block::create(*cfg);
	ctx.lpbb()->add_outedge(bb);

	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = gamma->output(n);

		auto a0 = static_cast<const jive::argument*>(gamma->subregion(0)->result(n)->origin());
		auto a1 = static_cast<const jive::argument*>(gamma->subregion(1)->result(n)->origin());
		auto o0 = a0->input()->origin();
		auto o1 = a1->input()->origin();

		/* both operands are the same, no select is necessary */
		if (o0 == o1) {
			ctx.insert(output, ctx.variable(o0));
			continue;
		}

		auto matchnode = jive::node_output::node(predicate);
		if (is<jive::match_op>(matchnode)) {
			auto matchop = static_cast<const jive::match_op*>(&matchnode->operation());
			auto d = matchop->default_alternative();
			auto c = ctx.variable(matchnode->input(0)->origin());
			auto t = d == 0 ? ctx.variable(o1) : ctx.variable(o0);
			auto f = d == 0 ? ctx.variable(o0) : ctx.variable(o1);
			bb->append_last(select_op::create(c, t, f));
		} else {
			auto vo0 = ctx.variable(o0);
			auto vo1 = ctx.variable(o1);
			bb->append_last(ctl2bits_op::create(ctx.variable(predicate), jive::bittype(1)));
			bb->append_last(select_op::create(bb->last()->result(0), vo0, vo1));
		}

		ctx.insert(output, bb->last()->result(0));
	}

	ctx.set_lpbb(bb);
}

static inline void
convert_gamma_node(const jive::node & node, context & ctx)
{
	JLM_ASSERT(is<jive::gamma_op>(&node));
	auto gamma = static_cast<const jive::gamma_node*>(&node);
	auto nalternatives = gamma->nsubregions();
	auto predicate = gamma->predicate()->origin();
	auto cfg = ctx.cfg();

	if (gamma->nsubregions() == 2
	&& gamma->subregion(0)->nnodes() == 0
	&& gamma->subregion(1)->nnodes() == 0)
		return convert_empty_gamma_node(gamma, ctx);

	auto entry = basic_block::create(*cfg);
	auto exit = basic_block::create(*cfg);
	ctx.lpbb()->add_outedge(entry);

	/* convert gamma regions */
	std::vector<cfg_node*> phi_nodes;
	entry->append_last(branch_op::create(nalternatives, ctx.variable(predicate)));
	for (size_t n = 0; n < gamma->nsubregions(); n++) {
		auto subregion = gamma->subregion(n);

		/* add arguments to context */
		for (size_t i = 0; i < subregion->narguments(); i++) {
			auto argument = subregion->argument(i);
			ctx.insert(argument, ctx.variable(argument->input()->origin()));
		}

		if (subregion->nnodes() == 0 && nalternatives == 2) {
			/* subregin is empty */
			phi_nodes.push_back(entry);
			entry->add_outedge(exit);
		} else {
			/* convert subregion */
			auto region_entry = basic_block::create(*cfg);
			entry->add_outedge(region_entry);
			ctx.set_lpbb(region_entry);
			convert_region(*subregion, ctx);

			phi_nodes.push_back(ctx.lpbb());
			ctx.lpbb()->add_outedge(exit);
		}
	}

	/* add phi instructions */
	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = gamma->output(n);

		bool invariant = true;
		auto matchnode = jive::node_output::node(predicate);
		bool select = (gamma->nsubregions() == 2) && is<jive::match_op>(matchnode);
		std::vector<std::pair<const variable*, cfg_node*>> arguments;
		for (size_t r = 0; r < gamma->nsubregions(); r++) {
			auto origin = gamma->subregion(r)->result(n)->origin();

			auto v = ctx.variable(origin);
			arguments.push_back(std::make_pair(v, phi_nodes[r]));
			invariant &= (v == ctx.variable(gamma->subregion(0)->result(n)->origin()));
			auto tmp = jive::node_output::node(origin);
			select &= (tmp == nullptr && origin->region()->node() == &node);
		}

		if (invariant) {
			/* all operands are the same */
			ctx.insert(output, arguments[0].first);
			continue;
		}

		if (select) {
			/* use select instead of phi */
			auto matchnode = jive::node_output::node(predicate);
			auto matchop = static_cast<const jive::match_op*>(&matchnode->operation());
			auto d = matchop->default_alternative();
			auto c = ctx.variable(matchnode->input(0)->origin());
			auto t = d == 0 ? arguments[1].first : arguments[0].first;
			auto f = d == 0 ? arguments[0].first : arguments[1].first;
			entry->append_first(select_op::create(c, t, f));
			ctx.insert(output, entry->first()->result(0));
			continue;
		}

		/* create phi instruction */
		exit->append_last(phi_op::create(arguments, output->type()));
		ctx.insert(output, exit->last()->result(0));
	}

	ctx.set_lpbb(exit);
}

static inline bool
phi_needed(const jive::input * i, const jlm::variable * v)
{
	auto node = input_node(i);
	JLM_ASSERT(is<jive::theta_op>(node));
	auto theta = static_cast<const jive::structural_node*>(node);
	auto input = static_cast<const jive::structural_input*>(i);
	auto output = theta->output(input->index());

	/* FIXME: solely decide on the input instead of using the variable */
	if (is<gblvariable>(v))
		return false;

	if (output->results.first()->origin() == input->arguments.first())
		return false;

	if (input->arguments.first()->nusers() == 0)
		return false;

	return true;
}

static inline void
convert_theta_node(const jive::node & node, context & ctx)
{
	JLM_ASSERT(is<jive::theta_op>(&node));
	auto subregion = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto predicate = subregion->result(0)->origin();

	auto pre_entry = ctx.lpbb();
	auto entry = basic_block::create(*ctx.cfg());
	pre_entry->add_outedge(entry);
	ctx.set_lpbb(entry);

	/* create phi nodes and add arguments to context */
	std::deque<jlm::tac*> phis;
	for (size_t n = 0; n < subregion->narguments(); n++) {
		auto argument = subregion->argument(n);
		auto v = ctx.variable(argument->input()->origin());
		if (phi_needed(argument->input(), v)) {
			auto phi = entry->append_last(phi_op::create({}, argument->type()));
			phis.push_back(phi);
			v = phi->result(0);
		}
		ctx.insert(argument, v);
	}

	convert_region(*subregion, ctx);

	/* add phi operands and results to context */
	for (size_t n = 1; n < subregion->nresults(); n++) {
		auto result = subregion->result(n);
		auto ve = ctx.variable(node.input(n-1)->origin());
		if (!phi_needed(node.input(n-1), ve)) {
			ctx.insert(result->output(), ctx.variable(result->origin()));
			continue;
		}

		auto vr = ctx.variable(result->origin());
		auto phi = phis.front();
		phis.pop_front();
		phi->replace(phi_op({pre_entry, ctx.lpbb()}, vr->type()), {ve, vr});
		ctx.insert(result->output(), vr);
	}
	JLM_ASSERT(phis.empty());

	ctx.lpbb()->append_last(branch_op::create(2, ctx.variable(predicate)));
	auto exit = basic_block::create(*ctx.cfg());
	ctx.lpbb()->add_outedge(exit);
	ctx.lpbb()->add_outedge(entry);
	ctx.set_lpbb(exit);
}

static inline void
convert_lambda_node(const jive::node & node, context & ctx)
{
	JLM_ASSERT(is<lambda::operation>(&node));
	auto lambda = static_cast<const lambda::node*>(&node);
	auto & module = ctx.module();
	auto & clg = module.ipgraph();

	auto f = function_node::create(clg, lambda->name(), lambda->type(), lambda->linkage(),
		lambda->attributes());
	auto v = module.create_variable(f);

	f->add_cfg(create_cfg(*lambda, ctx));
	ctx.insert(node.output(0), v);
}

static inline void
convert_phi_node(const jive::node & node, context & ctx)
{
	JLM_ASSERT(jive::is<phi::operation>(&node));
	auto phi = static_cast<const jive::structural_node*>(&node);
	auto subregion = phi->subregion(0);
	auto & module = ctx.module();
	auto & ipg = module.ipgraph();

	/* add dependencies to context */
	for (size_t n = 0; n < phi->ninputs(); n++) {
		auto v = ctx.variable(phi->input(n)->origin());
		ctx.insert(phi->input(n)->arguments.first(), v);
	}

	/* forward declare all functions and globals */
	for (size_t n = 0; n < subregion->nresults(); n++) {
		JLM_ASSERT(subregion->argument(n)->input() == nullptr);
		auto node = jive::node_output::node(subregion->result(n)->origin());

		if (auto lambda = dynamic_cast<const lambda::node*>(node)) {
			auto f = function_node::create(ipg, lambda->name(), lambda->type(), lambda->linkage(),
				lambda->attributes());
			ctx.insert(subregion->argument(n), module.create_variable(f));
		} else {
			JLM_ASSERT(is<delta::operation>(node));
			auto d = static_cast<const delta::node*>(node);
			auto data = data_node::Create(
        ipg,
        d->name(),
        d->type(),
        d->linkage(),
        d->Section(),
				d->constant());
			ctx.insert(subregion->argument(n), module.create_global_value(data));
		}
	}

	/* convert function bodies and global initializations */
	for (size_t n = 0; n < subregion->nresults(); n++) {
		JLM_ASSERT(subregion->argument(n)->input() == nullptr);
		auto result = subregion->result(n);
		auto node = jive::node_output::node(result->origin());

		if (auto lambda = dynamic_cast<const lambda::node*>(node)) {
			auto v = static_cast<const fctvariable*>(ctx.variable(subregion->argument(n)));
			v->function()->add_cfg(create_cfg(*lambda, ctx));
			ctx.insert(node->output(0), v);
		} else {
			JLM_ASSERT(is<delta::operation>(node));
			auto delta = static_cast<const delta::node*>(node);
			auto v = static_cast<const gblvalue*>(ctx.variable(subregion->argument(n)));

			v->node()->set_initialization(create_initialization(delta, ctx));
			ctx.insert(node->output(0), v);
		}
	}

	/* add functions and globals to context */
	JLM_ASSERT(node.noutputs() == subregion->nresults());
	for (size_t n = 0; n < node.noutputs(); n++)
		ctx.insert(node.output(n), ctx.variable(subregion->result(n)->origin()));
}

static inline void
convert_delta_node(const jive::node & node, context & ctx)
{
  JLM_ASSERT(is<delta::operation>(&node));
  auto delta = static_cast<const delta::node*>(&node);
  auto & m = ctx.module();

  auto dnode = data_node::Create(
    m.ipgraph(),
    delta->name(),
    delta->type(),
    delta->linkage(),
    delta->Section(),
    delta->constant());
  dnode->set_initialization(create_initialization(delta, ctx));
  auto v = m.create_global_value(dnode);
  ctx.insert(delta->output(), v);
}

static inline void
convert_node(const jive::node & node, context & ctx)
{
	static std::unordered_map<
	  std::type_index
	, std::function<void(const jive::node & node, context & ctx)
	>> map({
	  {typeid(lambda::operation), convert_lambda_node}
	, {std::type_index(typeid(jive::gamma_op)), convert_gamma_node}
	, {std::type_index(typeid(jive::theta_op)), convert_theta_node}
	, {typeid(phi::operation), convert_phi_node}
	, {typeid(delta::operation), convert_delta_node}
	});

	if (dynamic_cast<const jive::simple_op*>(&node.operation())) {
		convert_simple_node(node, ctx);
		return;
	}

	auto & op = node.operation();
	JLM_ASSERT(map.find(typeid(op)) != map.end());
	map[typeid(op)](node, ctx);
}

static void
convert_nodes(const jive::graph & graph, context & ctx)
{
	for (const auto & node : jive::topdown_traverser(graph.root()))
		convert_node(*node, ctx);
}

static void
convert_imports(const jive::graph & graph, ipgraph_module & im, context & ctx)
{
	auto & ipg = im.ipgraph();

	for (size_t n = 0; n < graph.root()->narguments(); n++) {
		auto argument = graph.root()->argument(n);
		auto import = static_cast<const jlm::impport*>(&argument->port());
		if (auto ftype = is_function_import(argument)) {
			auto f = function_node::create(ipg, import->name(), *ftype, import->linkage());
			auto v = im.create_variable(f);
			ctx.insert(argument, v);
		} else {
            auto dnode = data_node::Create(
                    ipg,
                    import->name(),
                    import->GetValueType(),
                    import->linkage(),
                    "",
                    false);
            auto v = im.create_global_value(dnode);
            ctx.insert(argument, v);
		}
	}
}

static std::unique_ptr<ipgraph_module>
convert_rvsdg(const RvsdgModule & rm)
{
	auto im = ipgraph_module::create(rm.SourceFileName(), rm.TargetTriple(), rm.DataLayout());

	context ctx(*im);
	convert_imports(rm.Rvsdg(), *im, ctx);
	convert_nodes(rm.Rvsdg(), ctx);

	return im;
}

std::unique_ptr<ipgraph_module>
rvsdg2jlm(
  const RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto statistics = rvsdg_destruction_stat::Create(rm.SourceFileName());

	statistics->start(rm.Rvsdg());
	auto im = convert_rvsdg(rm);
	statistics->end(*im);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

	return im;
}

}}
