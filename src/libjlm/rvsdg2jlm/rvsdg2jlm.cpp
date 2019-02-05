/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/phi.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#include <jlm/common.hpp>
#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators.hpp>
#include <jlm/jlm/ir/rvsdg.hpp>
#include <jlm/jlm/ir/tac.hpp>
#include <jlm/jlm/rvsdg2jlm/context.hpp>
#include <jlm/jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <deque>

namespace jlm {
namespace rvsdg2jlm {

static const jive::fcttype *
is_function_import(const jive::argument * argument)
{
	JLM_DEBUG_ASSERT(argument->region()->graph()->root() == argument->region());
	auto at = dynamic_cast<const ptrtype*>(&argument->type());
	JLM_DEBUG_ASSERT(at != nullptr);

	return dynamic_cast<const jive::fcttype*>(&at->pointee_type());
}

static inline const jive::output *
root_port(const jive::output * port)
{
	auto root = port->region()->graph()->root();
	if (port->region() == root)
		return port;

	auto node = port->node();
	JLM_DEBUG_ASSERT(node->noutputs() == 1);
	JLM_DEBUG_ASSERT(node->output(0)->nusers() == 1);
	auto user = *node->output(0)->begin();
	node = port->region()->node();
	JLM_DEBUG_ASSERT(node && dynamic_cast<const jive::phi_op*>(&node->operation()));
	port = node->output(user->index());

	JLM_DEBUG_ASSERT(port->region() == root);
	return port;
}

static inline bool
is_exported(const jive::output * port)
{
	port = root_port(port);
	for (const auto & user : *port) {
		if (dynamic_cast<const jive::result*>(user))
			return true;
	}

	return false;
}

static inline std::string
get_name(const jive::output * port)
{
	port = root_port(port);
	for (const auto & user : *port) {
		if (auto ep = dynamic_cast<const jive::expport*>(&user->port())) {
			return ep->name();
		}
	}

	static size_t c = 0;
	return strfmt("f", c++);
}

static inline const jlm::tac *
create_assignment_lpbb(const jlm::variable * argument, const jlm::variable * result, context & ctx)
{
	return ctx.lpbb()->append_last(create_assignment(argument->type(), argument, result));
}

static std::unique_ptr<data_node_init>
create_initialization(const delta_node * delta, context & ctx)
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
		JLM_DEBUG_ASSERT(node->noutputs() == 1);
		auto output = node->output(0);

		/* collect operand variables */
		std::vector<const variable*> operands;
		for (size_t n = 0; n < node->ninputs(); n++)
			operands.push_back(ctx.variable(node->input(n)->origin()));

		/* convert node to tac */
		auto v = ctx.module().create_tacvariable(output->type());
		auto & op = *static_cast<const jive::simple_op*>(&node->operation());
		tacs.push_back(tac::create(op, operands, {v}));
		ctx.insert(output, v);
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
create_cfg(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is<lambda_op>(&node));
	auto region = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto & module = ctx.module();

	JLM_DEBUG_ASSERT(ctx.lpbb() == nullptr);
	std::unique_ptr<jlm::cfg> cfg(new jlm::cfg(ctx.module()));
	auto entry = basic_block::create(*cfg);
	cfg->exit()->divert_inedges(entry);
	ctx.set_lpbb(entry);
	ctx.set_cfg(cfg.get());

	/* add arguments and dependencies */
	for (size_t n = 0; n < region->narguments(); n++) {
		const variable * v = nullptr;
		auto argument = region->argument(n);
		if (argument->input()) {
			v = ctx.variable(argument->input()->origin());
		} else {
			v = module.create_variable(argument->type(), "");
			cfg->entry()->append_argument(v);
		}

		ctx.insert(argument, v);
	}

	convert_region(*region, ctx);

	/* add results */
	for (size_t n = 0; n < region->nresults(); n++)
		cfg->exit()->append_result(ctx.variable(region->result(n)->origin()));

	ctx.lpbb()->add_outedge(cfg->exit());
	ctx.set_lpbb(nullptr);
	ctx.set_cfg(nullptr);

	straighten(*cfg);
	JLM_DEBUG_ASSERT(is_closed(*cfg));
	return cfg;
}

static inline void
convert_simple_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::simple_op*>(&node.operation()));

	std::vector<const variable*> operands;
	for (size_t n = 0; n < node.ninputs(); n++)
		operands.push_back(ctx.variable(node.input(n)->origin()));

	std::vector<tacvariable*> tvs;
	std::vector<const variable*> results;
	for (size_t n = 0; n < node.noutputs(); n++) {
		auto v = ctx.module().create_tacvariable(node.output(n)->type());
		ctx.insert(node.output(n), v);
		results.push_back(v);
		tvs.push_back(v);
	}

	auto & op = *static_cast<const jive::simple_op*>(&node.operation());
	ctx.lpbb()->append_last(tac::create(op, operands, results));
	/* FIXME: remove again once tacvariables owner's are tacs */
	for (const auto & tv : tvs)
		tv->set_tac(static_cast<const basic_block*>(ctx.lpbb())->tacs().last());
}

static void
convert_empty_gamma_node(const jive::gamma_node * gamma, context & ctx)
{
	JLM_DEBUG_ASSERT(gamma->nsubregions() == 2);
	JLM_DEBUG_ASSERT(gamma->subregion(0)->nnodes() == 0 && gamma->subregion(1)->nnodes() == 0);

	/* both regions are empty, create only select instructions */

	auto predicate = gamma->predicate()->origin();
	auto & module = ctx.module();
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

		auto v = module.create_variable(output->type());
		if (is<jive::match_op>(predicate->node())) {
			auto matchop = static_cast<const jive::match_op*>(&predicate->node()->operation());
			auto d = matchop->default_alternative();
			auto c = ctx.variable(predicate->node()->input(0)->origin());
			auto t = d == 0 ? ctx.variable(o1) : ctx.variable(o0);
			auto f = d == 0 ? ctx.variable(o0) : ctx.variable(o1);
			bb->append_last(create_select_tac(c, t, f, v));
		} else {
			auto c = module.create_variable(jive::bittype(1));
			bb->append_last(create_ctl2bits_tac(ctx.variable(predicate), c));
			bb->append_last(create_select_tac(c, ctx.variable(o1), ctx.variable(o0), v));
		}

		ctx.insert(output, v);
	}

	ctx.set_lpbb(bb);
}

static inline void
convert_gamma_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is<jive::gamma_op>(&node));
	auto gamma = static_cast<const jive::gamma_node*>(&node);
	auto nalternatives = gamma->nsubregions();
	auto predicate = gamma->predicate()->origin();
	auto & module = ctx.module();
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
	entry->append_last(create_branch_tac(nalternatives, ctx.variable(predicate)));
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
		bool select = (gamma->nsubregions() == 2) && is<jive::match_op>(predicate->node());
		std::vector<std::pair<const variable*, cfg_node*>> arguments;
		for (size_t r = 0; r < gamma->nsubregions(); r++) {
			auto origin = gamma->subregion(r)->result(n)->origin();

			auto v = ctx.variable(origin);
			arguments.push_back(std::make_pair(v, phi_nodes[r]));
			invariant &= (v == ctx.variable(gamma->subregion(0)->result(n)->origin()));
			select &= (origin->node() == nullptr && origin->region()->node() == &node);
		}

		if (invariant) {
			/* all operands are the same */
			ctx.insert(output, arguments[0].first);
			continue;
		}

		if (select) {
			/* use select instead of phi */
			auto matchop = static_cast<const jive::match_op*>(&predicate->node()->operation());
			auto d = matchop->default_alternative();
			auto v = module.create_variable(output->type());
			auto c = ctx.variable(predicate->node()->input(0)->origin());
			auto t = d == 0 ? arguments[1].first : arguments[0].first;
			auto f = d == 0 ? arguments[0].first : arguments[1].first;
			entry->append_first(create_select_tac(c, t, f, v));
			ctx.insert(output, v);
			continue;
		}

		/* create phi instruction */
		auto v = module.create_variable(output->type());
		exit->append_last(create_phi_tac(arguments, v));
		ctx.insert(output, v);
	}

	ctx.set_lpbb(exit);
}

static inline bool
phi_needed(const jive::input * i, const jlm::variable * v)
{
	JLM_DEBUG_ASSERT(is<jive::theta_op>(i->node()));
	auto theta = static_cast<const jive::structural_node*>(i->node());
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
	JLM_DEBUG_ASSERT(is<jive::theta_op>(&node));
	auto subregion = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto predicate = subregion->result(0)->origin();

	auto pre_entry = ctx.lpbb();
	auto entry = basic_block::create(*ctx.cfg());
	pre_entry->add_outedge(entry);
	ctx.set_lpbb(entry);

	/* create loop variables and add arguments to context */
	std::deque<jlm::variable*> lvs;
	for (size_t n = 0; n < subregion->narguments(); n++) {
		auto argument = subregion->argument(n);
		auto v = ctx.variable(argument->input()->origin());
		if (phi_needed(argument->input(), v)) {
			lvs.push_back(ctx.module().create_variable(argument->type()));
			v = lvs.back();
		}
		ctx.insert(argument, v);
	}

	convert_region(*subregion, ctx);

	/* add phi instructions and results to context */
	for (size_t n = 1; n < subregion->nresults(); n++) {
		auto result = subregion->result(n);
		auto ve = ctx.variable(node.input(n-1)->origin());
		if (!phi_needed(node.input(n-1), ve)) {
			ctx.insert(result->output(), ctx.variable(result->origin()));
			continue;
		}

		auto vr = ctx.variable(result->origin());
		auto v = lvs.front();
		lvs.pop_front();
		entry->append_last(create_phi_tac({{ve, pre_entry}, {vr, ctx.lpbb()}}, v));
		ctx.insert(result->output(), vr);
	}
	JLM_DEBUG_ASSERT(lvs.empty());

	ctx.lpbb()->append_last(create_branch_tac(2, ctx.variable(predicate)));
	auto exit = basic_block::create(*ctx.cfg());
	ctx.lpbb()->add_outedge(exit);
	ctx.lpbb()->add_outedge(entry);
	ctx.set_lpbb(exit);
}

static inline void
convert_lambda_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is<lambda_op>(&node));
	auto lambda = static_cast<const lambda_node*>(&node);
	auto & module = ctx.module();
	auto & clg = module.ipgraph();

	auto f = function_node::create(clg, lambda->name(), lambda->fcttype(), lambda->linkage());
	auto v = module.create_variable(f);

	f->add_cfg(create_cfg(node, ctx));
	ctx.insert(node.output(0), v);
}

static inline void
convert_phi_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is<jive::phi_op>(&node));
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
		JLM_DEBUG_ASSERT(subregion->argument(n)->input() == nullptr);
		auto node = subregion->result(n)->origin()->node();

		if (auto lambda = dynamic_cast<const lambda_node*>(node)) {
			auto f = function_node::create(ipg, lambda->name(), lambda->fcttype(), lambda->linkage());
			ctx.insert(subregion->argument(n), module.create_variable(f));
		} else {
			JLM_DEBUG_ASSERT(is<delta_op>(node));
			auto d = static_cast<const delta_node*>(node);
			auto data = data_node::create(ipg, get_name(d->output(0)), d->type(), d->linkage(),
				d->constant());
			ctx.insert(subregion->argument(n), module.create_global_value(data));
		}
	}

	/* convert function bodies and global initializations */
	for (size_t n = 0; n < subregion->nresults(); n++) {
		JLM_DEBUG_ASSERT(subregion->argument(n)->input() == nullptr);
		auto result = subregion->result(n);
		auto node = result->origin()->node();

		if (is<lambda_op>(node)) {
			auto v = static_cast<const fctvariable*>(ctx.variable(subregion->argument(n)));
			v->function()->add_cfg(create_cfg(*node, ctx));
			ctx.insert(node->output(0), v);
		} else {
			JLM_DEBUG_ASSERT(is<delta_op>(node));
			auto delta = static_cast<const delta_node*>(node);
			auto v = static_cast<const gblvalue*>(ctx.variable(subregion->argument(n)));

			v->node()->set_initialization(create_initialization(delta, ctx));
			ctx.insert(node->output(0), v);
		}
	}

	/* add functions and globals to context */
	JLM_DEBUG_ASSERT(node.noutputs() == subregion->nresults());
	for (size_t n = 0; n < node.noutputs(); n++)
		ctx.insert(node.output(n), ctx.variable(subregion->result(n)->origin()));
}

static inline void
convert_delta_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is<delta_op>(&node));
	auto delta = static_cast<const delta_node*>(&node);
	const auto & op = *static_cast<const jlm::delta_op*>(&node.operation());
	auto & module = ctx.module();

	JLM_DEBUG_ASSERT(delta->subregion()->nresults() == 1);
	auto result = delta->subregion()->result(0);

	auto name = get_name(result->output());
	auto dnode = data_node::create(module.ipgraph(), name, op.type(), op.linkage(), op.constant());
	dnode->set_initialization(create_initialization(delta, ctx));
	auto v = module.create_global_value(dnode);
	ctx.insert(result->output(), v);
}

static inline void
convert_node(const jive::node & node, context & ctx)
{
	static std::unordered_map<
	  std::type_index
	, std::function<void(const jive::node & node, context & ctx)
	>> map({
	  {typeid(lambda_op), convert_lambda_node}
	, {std::type_index(typeid(jive::gamma_op)), convert_gamma_node}
	, {std::type_index(typeid(jive::theta_op)), convert_theta_node}
	, {std::type_index(typeid(jive::phi_op)), convert_phi_node}
	, {typeid(jlm::delta_op), convert_delta_node}
	});

	if (dynamic_cast<const jive::simple_op*>(&node.operation())) {
		convert_simple_node(node, ctx);
		return;
	}

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node.operation()))) != map.end());
	map[std::type_index(typeid(node.operation()))](node, ctx);
}

std::unique_ptr<jlm::module>
rvsdg2jlm(const jlm::rvsdg & rvsdg)
{
	auto & dl = rvsdg.data_layout();
	auto & tt = rvsdg.target_triple();
	std::unique_ptr<jlm::module> module(new jlm::module(tt, dl));
	auto graph = rvsdg.graph();
	auto & clg = module->ipgraph();

	context ctx(*module);

	/* Add all imports to context */
	for (size_t n = 0; n < graph->root()->narguments(); n++) {
		auto argument = graph->root()->argument(n);
		auto import = static_cast<const jive::impport*>(&argument->port());
		if (auto ftype = is_function_import(argument)) {
			auto f = function_node::create(clg, import->name(), *ftype, linkage::external_linkage);
			auto v = module->create_variable(f);
			ctx.insert(argument, v);
		} else {
			JLM_DEBUG_ASSERT(dynamic_cast<const ptrtype*>(&argument->type()));
			auto & type = *static_cast<const ptrtype*>(&argument->type());
			const auto & name = import->name();
			auto dnode = data_node::create(clg, name, type, jlm::linkage::external_linkage, false);
			auto v = module->create_global_value(dnode);
			ctx.insert(argument, v);
		}
	}

	for (const auto & node : jive::topdown_traverser(graph->root()))
		convert_node(*node, ctx);

	return module;
}

}}
