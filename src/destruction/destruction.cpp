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
#include <jive/vsdg/control.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/theta.h>

//to be removed
#include <jive/view.h>

#include <stack>

namespace jlm {

class theta {
public:
	jive_theta theta;
	std::unordered_map<const jlm::frontend::variable*, jive_theta_loopvar> loopvars;
};

}

typedef std::unordered_map<const jlm::frontend::variable*, jive::output*> variable_map;
typedef std::unordered_map<const jlm::frontend::cfg_node*, variable_map> node_map;
typedef std::unordered_map<const jlm::frontend::cfg_edge*, std::stack<jive::output*>> predicate_map;
typedef std::unordered_map<const jlm::frontend::cfg_edge*, std::stack<jlm::theta>> theta_map;

namespace jlm {

static jive::output *
create_undefined_value(const jive::base::type & type, struct jive_graph * graph)
{
	if (auto t = dynamic_cast<const jive::bits::type*>(&type))
		return jive_bitconstant_undefined(graph, t->nbits());

	/* FIXME: temporary solutation */
	if (dynamic_cast<const jive::ctl::type*>(&type))
		return jive_control_constant(graph, 2, 0);

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static bool
is_branch_join(
	const jlm::frontend::basic_block * bb,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (bb->ninedges() <= 1)
		return false;

	for (auto edge : bb->inedges()) {
		if (back_edges.find(edge) != back_edges.end())
			return false;
	}

	return true;
}

static bool
visit_branch_join(const jlm::frontend::basic_block * bb, predicate_map & pmap)
{
	for (auto edge : bb->inedges()) {
		if (pmap.find(edge) == pmap.end())
			return false;
	}

	return true;
}

static bool
is_loop_entry(
	const jlm::frontend::basic_block * bb,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (bb->inedges().size() != 2)
		return false;

	jlm::frontend::cfg_edge * edge1 = bb->inedges().front();
	jlm::frontend::cfg_edge * edge2 = *std::next(bb->inedges().begin());
	return back_edges.find(edge1) != back_edges.end() || back_edges.find(edge2) != back_edges.end();
}

static bool
is_loop_exit(
	const jlm::frontend::basic_block * bb,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	if (bb->outedges().size() != 2)
		return false;

	jlm::frontend::cfg_edge * edge1 = bb->outedges()[0];
	jlm::frontend::cfg_edge * edge2 = bb->outedges()[1];
	return back_edges.find(edge1) != back_edges.end() || back_edges.find(edge2) != back_edges.end();
}

static void
convert_basic_block(
	const jlm::frontend::basic_block * bb,
	struct jive_graph * graph,
	node_map & nmap,
	predicate_map & pmap,
	theta_map & tmap,
	const std::unordered_set<jlm::frontend::cfg_edge*> & back_edges)
{
	/* only visit a join if all incoming edges have already been visited */
	if (is_branch_join(bb, back_edges) && !visit_branch_join(bb, pmap))
		return;

	variable_map vmap;
	std::stack<jive::output*> pstack;
	std::stack<theta> tstack;

	/* handle incoming edges */
	if (bb->ninedges() == 1) {
		vmap = nmap[bb->inedges().front()->source()];
		pstack = pmap[bb->inedges().front()];
		tstack = tmap[bb->inedges().front()];
	} else if (is_branch_join(bb, back_edges)) {
		pstack = pmap[bb->inedges().front()];
		tstack = tmap[bb->inedges().front()];

		JLM_DEBUG_ASSERT(pstack.size() != 0);
		jive::output * predicate = pstack.top();
		for (auto edge : bb->inedges())
			JLM_DEBUG_ASSERT(predicate == pmap[edge].top());
		pstack.pop();

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
				if (nmap[edge->source()].find(variable) != nmap[edge->source()].end()) {
					jive::output * value = nmap[edge->source()][variable];
					JLM_DEBUG_ASSERT(value != nullptr);
					values.push_back(value);
				} else {
					jive::output * undef = create_undefined_value(variable->type(), graph);
					JLM_DEBUG_ASSERT(undef != nullptr);
					values.push_back(undef);
				}
			}
			alternatives.push_back(values);
		}

		size_t n = 0;
		std::vector<jive::output*> results = jive_gamma(predicate, types, alternatives);
		JLM_DEBUG_ASSERT(results.size() == vmap.size());
		for (auto variable : vmap)
			vmap[variable.first] = results[n++];
	} else if (is_loop_entry(bb, back_edges)) {
		jlm::frontend::cfg_edge * entry_edge = bb->inedges().front();
		jlm::frontend::cfg_edge * back_edge = *std::next(bb->inedges().begin());
		if (back_edges.find(entry_edge) != back_edges.end()) {
			jlm::frontend::cfg_edge * tmp = entry_edge;
			entry_edge = back_edge;
			back_edge = tmp;
		}

		tstack = tmap[entry_edge];
		pstack = pmap[entry_edge];
		vmap.insert(nmap[entry_edge->source()].begin(), nmap[entry_edge->source()].end());

		jlm::theta theta;
		theta.theta = jive_theta_begin(graph);
		for (auto vpair : vmap) {
			jive_theta_loopvar loopvar = jive_theta_loopvar_enter(theta.theta, vpair.second);
			JLM_DEBUG_ASSERT(loopvar.value);
			vmap[vpair.first] = loopvar.value;
			theta.loopvars[vpair.first] = loopvar;
		}
		tstack.push(theta);
	} else
		JLM_DEBUG_ASSERT(0);


	/* handle TACs */
	std::list<const jlm::frontend::tac*> tacs = bb->tacs();
	for (auto tac : tacs) {
		if (dynamic_cast<const jlm::frontend::assignment_op*>(&tac->operation())) {
			JLM_DEBUG_ASSERT(vmap.find(tac->input(0)) != vmap.end());
			vmap[tac->output(0)] = vmap[tac->input(0)];
			continue;
		}

		std::vector<jive::output*> operands;
		for (size_t n = 0; n < tac->ninputs(); n++) {
			JLM_DEBUG_ASSERT(vmap.find(tac->input(n)) != vmap.end());
			operands.push_back(vmap[tac->input(n)]);
		}

		std::vector<jive::output *> results;
		results = jive_node_create_normalized(graph, tac->operation(), operands);

		JLM_DEBUG_ASSERT(results.size() == tac->noutputs());
		for (size_t n = 0; n < tac->noutputs(); n++) {
			JLM_DEBUG_ASSERT(results[n] != nullptr);
			vmap[tac->output(n)] = results[n];
		}
	}

	/* handle outgoing edges */
	if (is_loop_exit(bb, back_edges)) {
		JLM_DEBUG_ASSERT(tstack.size() != 0);
		jlm::theta theta = tstack.top();
		tstack.pop();

		std::vector<jive_theta_loopvar> tmp;
		for (auto lvpair : theta.loopvars) {
			jive_theta_loopvar_leave(theta.theta, lvpair.second.gate, vmap[lvpair.first]);
			tmp.push_back(lvpair.second);
		}
		jive::output * predicate = vmap[bb->tacs().back()->output(0)];
		jive_theta_end(theta.theta, predicate, tmp.size(), &tmp[0]);
		size_t n = 0;
		for (auto it = theta.loopvars.begin(); it != theta.loopvars.end(); it++) {
			JLM_DEBUG_ASSERT(tmp[n].value != nullptr);
			vmap[it->first] = tmp[n++].value;
		}
	} else if (bb->is_branch()) {
		JLM_DEBUG_ASSERT(vmap[bb->tacs().back()->output(0)]);
		pstack.push(vmap[bb->tacs().back()->output(0)]);
	} else
		JLM_DEBUG_ASSERT(bb->outedges().size() == 1);

	nmap[bb] = vmap;
	for (auto e : bb->outedges()) {
		if (back_edges.find(e) != back_edges.end())
			continue;

		pmap[e] = pstack;
		tmap[e] = tstack;
		if (auto bb = dynamic_cast<jlm::frontend::basic_block *>(e->sink())) {
			convert_basic_block(bb, graph, nmap, pmap, tmap, back_edges);
		}
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
	std::unordered_set<jlm::frontend::cfg_edge*> back_edges = restructure(cfg);
	jive_cfg_view(*cfg);


	std::vector<const jlm::frontend::variable*> variables;
	std::vector<const char*> argument_names;
	std::vector<const jive::base::type*> argument_types;
	for (size_t n = 0; n < cfg->narguments(); n++) {
		variables.push_back(cfg->argument(n));
		argument_names.push_back(cfg->argument(n)->name().c_str());
		argument_types.push_back(&cfg->argument(n)->type());
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

	theta_map tmap;
	predicate_map pmap;
	pmap[cfg->enter()->outedges()[0]] = std::stack<jive::output*>();
	tmap[cfg->enter()->outedges()[0]] = std::stack<jlm::theta>();

	convert_basic_block(bb, graph, nmap, pmap, tmap, back_edges);

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
		JLM_DEBUG_ASSERT(0);
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
