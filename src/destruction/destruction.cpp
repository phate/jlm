/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/destruction/destruction.hpp>
#include <jlm/IR/clg.hpp>

#include <jive/types/function/fctlambda.h>
#include <jive/vsdg/basetype.h>
#include <jive/vsdg/graph.h>

namespace jlm {

static jive::output * 
construct_lambda(struct jive_graph * graph, const jlm::frontend::clg_node * clg_node)
{
	const jive::fct::type & fcttype = clg_node->type();

	std::vector<const jive::base::type*> argument_types;
	std::vector<const char *> argument_names;
	for (size_t n = 0; n < fcttype.narguments(); n++) {
		argument_types.push_back(fcttype.argument_type(n));
		argument_names.push_back("arg");
	}

	struct jive_lambda * lambda = jive_lambda_begin(graph, fcttype.narguments(),
		&argument_types[0], &argument_names[0]);

	JIVE_DEBUG_ASSERT(lambda->narguments == fcttype.nreturns());

	std::vector<const jive::base::type*> result_types;
	std::vector<jive::output*> results;
	for (size_t n = 0; n < fcttype.nreturns(); n++) {
		result_types.push_back(fcttype.return_type(n));
		results.push_back(lambda->arguments[n]);
	}

	return jive_lambda_end(lambda, fcttype.nreturns(), &result_types[0], &results[0]);
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
