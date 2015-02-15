/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/destruction.hpp>
#include <jlm/destruction/restructuring.hpp>

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/tac/assignment.hpp>
#include <jlm/IR/tac/tac.hpp>

#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>

#include <stack>

typedef std::unordered_map<const jlm::frontend::variable*, jive::output*> variable_map;
typedef std::unordered_map<const jlm::frontend::cfg_node*, variable_map> node_map;
typedef std::unordered_map<const jlm::frontend::cfg_edge*, std::stack<jive::output*>> predicate_map;

namespace jlm {

static jive::output *
create_undefined_value(const jive::base::type & type, struct jive_graph * graph)
{
	if (auto t = dynamic_cast<const jive::bits::type*>(&type))
		return jive_bitconstant_undefined(graph, t->nbits());

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static bool
is_join(const jlm::frontend::basic_block * bb)
{
	return bb->ninedges() > 1;
}

static bool
visit_join(const jlm::frontend::basic_block * bb, predicate_map & pmap)
{
	JLM_DEBUG_ASSERT(is_join(bb));

	for (auto edge : bb->inedges()) {
		if (pmap.find(edge) == pmap.end())
			return false;
	}

	return true;
}

static void
convert_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	node_map & nmap,
	predicate_map & pmap)
{
	/* only visit a join if all incoming edges have already been visited */
	if (is_join(bb) && !visit_join(bb, pmap))
		return;

	variable_map vmap;
	std::stack<jive::output*> pstack;

	/* handle incoming edges */
	if (bb->ninedges() == 1) {
		vmap = nmap[bb->inedges().front()->source()];
		pstack = pmap[bb->inedges().front()];
	} else if (is_join(bb)) {
		pstack = pmap[bb->inedges().front()];
		JLM_DEBUG_ASSERT(pstack.size() != 0);
		jive::output * predicate = pstack.top();
		pstack.pop();

		for (auto edge : bb->inedges())
			JLM_DEBUG_ASSERT(predicate == pmap[edge].top());

		for (auto edge : bb->inedges())
			vmap.insert(nmap[edge->source()].begin(), nmap[edge->source()].end());

		std::vector<const jive::base::type*> types;
		for (auto vpair : vmap)
			types.push_back(&vpair.first->type());

		std::vector<std::vector<jive::output*>> alternatives;
		for (auto edge : bb->inedges()) {
			std::vector<jive::output*> values;
			for (auto vpair : vmap) {
				const jlm::frontend::variable * variable = vpair.first;
				if (nmap[edge->source()].find(variable) != nmap[edge->source()].end())
					values.push_back(nmap[edge->source()][variable]);
				else
					values.push_back(create_undefined_value(variable->type(), graph));
			}
			alternatives.push_back(values);
		}

		size_t n = 0;
		std::vector<jive::output*> results = jive_gamma(predicate, types, alternatives);
		JLM_DEBUG_ASSERT(results.size() == vmap.size());
		for (auto variable : vmap)
			vmap[variable.first] = results[n++];
	}

	/* handle TACs */
	std::list<const jlm::frontend::tac*> tacs = bb->tacs();
	for (auto tac : tacs) {
		if (dynamic_cast<const jlm::frontend::assignment_op*>(&tac->operation())) {
			vmap[tac->outputs()[0]->variable()] = vmap[tac->inputs()[0]->variable()];
			continue;
		}

		std::vector<jive::output*> operands;
		for (size_t n = 0; n < tac->inputs().size(); n++) {
			JLM_DEBUG_ASSERT(vmap.find(tac->inputs()[n]->variable()));
			operands.push_back(vmap[tac->inputs()[n]->variable()]);
		}

		std::vector<jive::output *> results;
		results = jive_node_create_normalized(graph, tac->operation(), operands);

		for (size_t n = 0; n < tac->outputs().size(); n++)
			vmap[tac->outputs()[n]->variable()] = results[n];
	}

	/* handle outgoing edges */
	if (bb->is_branch())
		pstack.push(vmap[bb->tacs().back()->outputs()[0]->variable()]);

	nmap[bb] = vmap;
	for (auto e : bb->outedges()) {
		pmap[e] = pstack;
		if (auto bb = dynamic_cast<jlm::frontend::basic_block *>(e->sink()))
			convert_basic_block(bb, graph, nmap, pmap);
	}
}

static jive::output * 
convert_cfg(
	jlm::frontend::cfg * cfg,
	struct jive_graph * graph)
{
	jive_cfg_view(*cfg);
	cfg->destruct_ssa();
	jive_cfg_view(*cfg);
	restructure(cfg);
	jive_cfg_view(*cfg);


	std::vector<const jlm::frontend::variable*> variables;
	std::vector<const char*> argument_names;
	std::vector<const jive::base::type*> argument_types;
	for (size_t n = 0; n < cfg->narguments(); n++) {
		variables.push_back(cfg->argument(n)->variable());
		argument_names.push_back(cfg->argument(n)->variable()->name().c_str());
		argument_types.push_back(&cfg->argument(n)->variable()->type());
	}

	struct jive_lambda * lambda = jive_lambda_begin(graph, variables.size(),
		&argument_types[0], &argument_names[0]);

	variable_map vmap;
	JLM_DEBUG_ASSERT(variables.size() == lambda->narguments);
	for (size_t n = 0; n < variables.size(); n++)
		vmap[variables[n]] = lambda->arguments[n];

	node_map nmap;
	nmap[cfg->enter()] = vmap;
	JLM_DEBUG_ASSERT(cfg->enter()->noutedges() == 1);
	const jlm::frontend::basic_block * bb;
	bb = static_cast<const jlm::frontend::basic_block*>(cfg->enter()->outedges()[0]->sink());

	predicate_map pmap;
	pmap[cfg->enter()->outedges()[0]] = std::stack<jive::output*>();

	convert_basic_block(bb, graph, nmap, pmap);

	JLM_DEBUG_ASSERT(cfg->exit()->ninedges() == 1);
	jlm::frontend::cfg_node * predecessor = cfg->exit()->inedges().front()->source();

	std::vector<jive::output*> results;
	std::vector<const jive::base::type*> result_types;
	for (size_t n = 0; n< cfg->nresults(); n++) {
		results.push_back(nmap[predecessor][cfg->result(n)]);
		result_types.push_back(&cfg->result(n)->type());
	}

	return jive_lambda_end(lambda, cfg->nresults(), &result_types[0], &results[0]);
}

static jive::output *
construct_lambda(struct jive_graph * graph, const jlm::frontend::clg_node * clg_node)
{
	//FIXME: check whether cfg_node has a CFG

	return convert_cfg(clg_node->cfg(), graph);
}


static void
handle_scc(struct jive_graph * graph, std::unordered_set<const jlm::frontend::clg_node*> & scc)
{
	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		construct_lambda(graph, *scc.begin());
	} else {
		JIVE_DEBUG_ASSERT(0);
		/* create phi */
	}
}

struct jive_graph *
construct_rvsdg(const jlm::frontend::clg & clg)
{
	struct ::jive_graph * graph = jive_graph_create();	

	std::vector<std::unordered_set<const jlm::frontend::clg_node*>> sccs = clg.find_sccs();
	for (auto scc : sccs)
		handle_scc(graph, scc);

	return graph;
}

}
