/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/phi.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

#include <jlm/common.hpp>
#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/data.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/rvsdg2jlm/context.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

namespace jlm {
namespace rvsdg2jlm {

static inline const jive::oport *
root_port(const jive::oport * port)
{
	auto root = port->region()->graph()->root();
	if (port->region() == root)
		return port;

	auto node = port->node();
	JLM_DEBUG_ASSERT(node && dynamic_cast<const jive::fct::lambda_op*>(&node->operation()));
	JLM_DEBUG_ASSERT(node->output(0)->nusers() == 1);
	auto user = *node->output(0)->begin();
	node = port->region()->node();
	JLM_DEBUG_ASSERT(node && dynamic_cast<const jive::phi_op*>(&node->operation()));
	port = node->output(user->index());

	JLM_DEBUG_ASSERT(port->region() == root);
	return port;
}

static inline bool
is_exported(const jive::oport * port)
{
	port = root_port(port);
	for (const auto & user : *port) {
		if (dynamic_cast<const jive::result*>(user))
			return true;
	}

	return false;
}

static inline std::string
get_name(const jive::oport * port)
{
	port = root_port(port);
	for (const auto & user : *port) {
		if (auto result = dynamic_cast<const jive::result*>(user)) {
			JLM_DEBUG_ASSERT(result->gate());
			return result->gate()->name();
		}
	}

	return "";
}

static inline const jlm::tac *
append_tac(jlm::cfg_node * node, std::unique_ptr<jlm::tac> tac)
{
	JLM_DEBUG_ASSERT(is_basic_block(node));
	auto & bb = *static_cast<jlm::basic_block*>(&node->attribute());
	bb.append(std::move(tac));
	return bb.last();
}

static inline const jlm::tac *
create_assignment_lpbb(const jlm::variable * argument, const jlm::variable * result, context & ctx)
{
	return append_tac(ctx.lpbb(), create_assignment(argument->type(), argument, result));
}

static inline std::unique_ptr<const expr>
convert_port(const jive::oport * port)
{
	auto node = port->node();
	std::vector<std::unique_ptr<const expr>> operands;
	for (size_t n = 0; n < node->ninputs(); n++)
		operands.push_back(convert_port(node->input(n)->origin()));

	return std::make_unique<const jlm::expr>(node->operation(), std::move(operands));
}

static void
convert_node(const jive::node & node, context & ctx);

static inline void
convert_region(jive::region & region, context & ctx)
{
	auto entry = create_basic_block_node(ctx.cfg());
	ctx.lpbb()->add_outedge(entry, 0);
	ctx.set_lpbb(entry);

	for (const auto & node : jive::topdown_traverser(&region))
		convert_node(*node, ctx);

	auto exit = create_basic_block_node(ctx.cfg());
	ctx.lpbb()->add_outedge(exit, 0);
	ctx.set_lpbb(exit);
}

static inline std::unique_ptr<jlm::cfg>
create_cfg(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::lambda_op*>(&node.operation()));
	auto region = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto & module = ctx.module();

	JLM_DEBUG_ASSERT(ctx.lpbb() == nullptr);
	std::unique_ptr<jlm::cfg> cfg(new jlm::cfg(ctx.module()));
	auto entry = create_basic_block_node(cfg.get());
	cfg->exit_node()->divert_inedges(entry);
	ctx.set_lpbb(entry);
	ctx.set_cfg(cfg.get());

	/* add arguments and dependencies */
	for (size_t n = 0; n < region->narguments(); n++) {
		const variable * v = nullptr;
		auto argument = region->argument(n);
		if (argument->input()) {
			v = ctx.variable(argument->input()->origin());
		} else {
			v = module.create_variable(argument->type(), "", false);
			cfg->entry().append_argument(v);
		}

		ctx.insert(argument, v);
	}

	convert_region(*region, ctx);

	/* add results */
	for (size_t n = 0; n < region->nresults(); n++)
		cfg->exit().append_result(ctx.variable(region->result(n)->origin()));

	ctx.lpbb()->add_outedge(cfg->exit_node(),  0);
	ctx.set_lpbb(nullptr);
	ctx.set_cfg(nullptr);

	return cfg;
}

static inline void
convert_simple_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::simple_op*>(&node.operation()));

	std::vector<const variable*> operands;
	for (size_t n = 0; n < node.ninputs(); n++)
		operands.push_back(ctx.variable(node.input(n)->origin()));

	std::vector<const variable*> results;
	for (size_t n = 0; n < node.noutputs(); n++) {
		auto v = ctx.module().create_variable(node.output(n)->type(), false);
		ctx.insert(node.output(n), v);
		results.push_back(v);
	}

	append_tac(ctx.lpbb(), create_tac(node.operation(), operands, results));
}

static inline void
convert_gamma_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::gamma_op*>(&node.operation()));
	auto nalternatives = static_cast<const jive::gamma_op*>(&node.operation())->nalternatives();
	auto & snode = *static_cast<const jive::structural_node*>(&node);
	auto predicate = node.input(0)->origin();
	auto & module = ctx.module();
	auto cfg = ctx.cfg();

	auto entry = create_basic_block_node(cfg);
	auto exit = create_basic_block_node(cfg);
	append_tac(entry, create_branch_tac(nalternatives, ctx.variable(predicate)));
	ctx.lpbb()->add_outedge(entry, 0);

	/* convert gamma regions */
	for (size_t n = 0; n < snode.nsubregions(); n++) {
		auto subregion = snode.subregion(n);

		/* add arguments to context */
		for (size_t i = 0; i < subregion->narguments(); i++) {
			auto argument = subregion->argument(i);
			ctx.insert(argument, ctx.variable(argument->input()->origin()));
		}

		/* convert subregion */
		auto region_entry = create_basic_block_node(cfg);
		entry->add_outedge(region_entry, n);
		ctx.set_lpbb(region_entry);
		convert_region(*subregion, ctx);

		ctx.lpbb()->add_outedge(exit, 0);
	}

	/* add phi instructions */
	for (size_t n = 0; n < snode.noutputs(); n++) {
		auto output = snode.output(n);
		std::vector<const variable*> arguments;
		for (size_t i = 0; i < snode.nsubregions(); i++)
			arguments.push_back(ctx.variable(snode.subregion(i)->result(n)->origin()));

		auto v = module.create_variable(output->type(), false);
		append_tac(exit, create_phi_tac(arguments, v));
		ctx.insert(output, v);
	}

	ctx.set_lpbb(exit);
}

static inline void
convert_theta_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::theta_op*>(&node.operation()));
	auto subregion = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto predicate = subregion->result(0)->origin();

	auto entry = create_basic_block_node(ctx.cfg());
	ctx.lpbb()->add_outedge(entry, 0);
	ctx.set_lpbb(entry);

	/* create loop variables and add arguments to context */
	std::vector<const variable*> lvs;
	for (size_t n = 0; n < subregion->narguments(); n++) {
		auto argument = subregion->argument(n);
		auto lv = ctx.module().create_variable(argument->type(), false);
		ctx.insert(argument, lv);
		lvs.push_back(lv);
	}

	convert_region(*subregion, ctx);

	/* add results to context and phi instructions */
	for (size_t n = 1; n < subregion->nresults(); n++) {
		auto result = subregion->result(n);
		ctx.insert(result->output(), lvs[n-1]);

		auto v1 = ctx.variable(node.input(n-1)->origin());
		auto v2 = ctx.variable(result->origin());
		append_tac(entry, create_phi_tac({v1, v2}, lvs[n-1]));
	}

	append_tac(ctx.lpbb(), create_branch_tac(2, ctx.variable(predicate)));
	auto exit = create_basic_block_node(ctx.cfg());
	ctx.lpbb()->add_outedge(exit, 0);
	ctx.lpbb()->add_outedge(entry, 1);
	ctx.set_lpbb(exit);
}

static inline void
convert_lambda_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::lambda_op*>(&node.operation()));
	auto & module = ctx.module();
	auto & clg = module.clg();

	const auto & ftype = *static_cast<const jive::fct::type*>(&node.output(0)->type());
	/* FIXME: create/get names for lambdas */
	auto name = get_name(node.output(0));
	auto exported = is_exported(node.output(0));
	auto f = clg.add_function(name.c_str(), ftype, exported);
	auto v = module.create_variable(f);

	f->add_cfg(create_cfg(node, ctx));
	ctx.insert(node.output(0), v);
}

static inline void
convert_phi_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::phi_op*>(&node.operation()));
	auto subregion = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto & module = ctx.module();
	auto & clg = module.clg();

	/* FIXME: handle phi node dependencies */
	JLM_DEBUG_ASSERT(subregion->narguments() == subregion->nresults());

	/* forward declare all functions */
	for (size_t n = 0; n < subregion->nresults(); n++) {
		auto result = subregion->result(n);
		auto lambda = result->origin()->node();
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::lambda_op*>(&lambda->operation()));

		auto name = get_name(lambda->output(0));
		auto exported = is_exported(lambda->output(0));
		auto & ftype = *static_cast<const jive::fct::type*>(&result->type());
		auto f = clg.add_function(name.c_str(), ftype, exported);
		ctx.insert(subregion->argument(n), module.create_variable(f));
	}

	/* convert function bodies */
	for (size_t n = 0; n < subregion->nresults(); n++) {
		auto lambda = subregion->result(n)->origin()->node();
		auto v = static_cast<const jlm::function_variable*>(ctx.variable(subregion->argument(n)));
		v->function()->add_cfg(create_cfg(*lambda, ctx));
		ctx.insert(lambda->output(0), v);
	}

	/* add functions to context */
	JLM_DEBUG_ASSERT(node.noutputs() == subregion->nresults());
	for (size_t n = 0; n < node.noutputs(); n++)
		ctx.insert(node.output(0), ctx.variable(subregion->result(n)->origin()));
}

static inline void
convert_data_node(const jive::node & node, context & ctx)
{
	JLM_DEBUG_ASSERT(is_data_op(node.operation()));
	auto subregion = static_cast<const jive::structural_node*>(&node)->subregion(0);
	auto & module = ctx.module();

	JLM_DEBUG_ASSERT(subregion->nresults() == 1);
	auto result = subregion->result(0);

	auto name = get_name(result->output());
	auto exported = is_exported(result->output());
	auto expression = convert_port(result->origin());

	auto v = module.create_variable(result->type(), name, exported);
	module.add_global_variable(v, std::move(expression));
	ctx.insert(result->output(), v);
}

static inline void
convert_node(const jive::node & node, context & ctx)
{
	static std::unordered_map<
	  std::type_index
	, std::function<void(const jive::node & node, context & ctx)
	>> map({
	  {std::type_index(typeid(jive::fct::lambda_op)), convert_lambda_node}
	, {std::type_index(typeid(jive::gamma_op)), convert_gamma_node}
	, {std::type_index(typeid(jive::theta_op)), convert_theta_node}
	, {std::type_index(typeid(jive::phi_op)), convert_phi_node}
	, {std::type_index(typeid(jlm::data_op)), convert_data_node}
	});

	if (dynamic_cast<const jive::simple_op*>(&node.operation())) {
		convert_simple_node(node, ctx);
		return;
	}

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node.operation()))) != map.end());
	map[std::type_index(typeid(node.operation()))](node, ctx);
}

std::unique_ptr<jlm::module>
rvsdg2jlm(const jive::graph & graph)
{
	std::unique_ptr<jlm::module> module(new jlm::module());
	auto & clg = module->clg();

	context ctx(*module);
	for (size_t n = 0; n < graph.root()->narguments(); n++) {
		auto argument = graph.root()->argument(n);
		if (auto ftype = dynamic_cast<const jive::fct::type*>(&argument->type())) {
			auto f = clg.add_function(argument->gate()->name().c_str(), *ftype, false);
			auto v = module->create_variable(f);
			ctx.insert(argument, v);
		} else {
			JLM_DEBUG_ASSERT(0);
		}
	}

	for (const auto & node : jive::topdown_traverser(graph.root()))
		convert_node(*node, ctx);

	return module;
}

}}
