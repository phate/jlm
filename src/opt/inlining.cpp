/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/lambda.hpp>
#include <jlm/ir/operators/call.hpp>
#include <jlm/opt/inlining.hpp>

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/substitution.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#ifdef ILNTIME
#include <chrono>
#include <iostream>
#endif

namespace jlm {

static bool
is_exported(jive::output * output)
{
	auto graph = output->region()->graph();

	for (const auto & user : *output) {
		if (dynamic_cast<const jive::result*>(user) && user->region() == graph->root())
			return true;
	}

	return false;
}

static std::vector<jive::simple_node*>
find_consumers(const jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(is_lambda_op(node->operation()));

	std::vector<jive::simple_node*> consumers;
	std::unordered_set<jive::output*> worklist({node->output(0)});
	while (!worklist.empty()) {
		auto output = *worklist.begin();
		worklist.erase(output);

		for (const auto & user : *output) {
			if (auto result = dynamic_cast<const jive::result*>(user)) {
				JLM_DEBUG_ASSERT(result->output() != nullptr);
				worklist.insert(result->output());
				continue;
			}

			if (auto simple = dynamic_cast<jive::simple_node*>(user->node())) {
				consumers.push_back(simple);
				continue;
			}

			auto sinput = dynamic_cast<jive::structural_input*>(user);
			for (auto & argument : sinput->arguments)
				worklist.insert(&argument);
		}
	}

	return consumers;
}

static jive::output *
find_producer(jive::input * input)
{
	auto graph = input->region()->graph();

	auto argument = dynamic_cast<jive::argument*>(input->origin());
	if (argument == nullptr)
		return input->origin();

	if (argument->region() == graph->root())
		return argument;

	JLM_DEBUG_ASSERT(argument->input() != nullptr);
	return find_producer(argument->input());
}

static jive::output *
route_to_region(jive::output * output, jive::region * region)
{
	JLM_DEBUG_ASSERT(region != nullptr);

	if (region == output->region())
		return output;

	output = route_to_region(output, region->node()->region());

	if (auto gamma = dynamic_cast<jive::gamma_node*>(region->node())) {
		gamma->add_entryvar(output);
		output = region->argument(region->narguments()-1);
	}	else if (auto theta = dynamic_cast<jive::theta_node*>(region->node())) {
		output = theta->add_loopvar(output)->argument();
	} else if (auto lambda = dynamic_cast<lambda_node*>(region->node())) {
		output = lambda->add_dependency(output);
	} else {
		JLM_DEBUG_ASSERT(0);
	}

	return output;
}

static std::vector<jive::output*>
route_dependencies(const jive::structural_node * lambda, const jive::simple_node * apply)
{
	JLM_DEBUG_ASSERT(is_lambda_op(lambda->operation()));
	JLM_DEBUG_ASSERT(dynamic_cast<const call_op*>(&apply->operation()));

	/* collect origins of dependencies */
	std::vector<jive::output*> deps;
	for (size_t n = 0; n < lambda->ninputs(); n++)
		deps.push_back(find_producer(lambda->input(n)));

	/* route dependencies to apply region */
	for (size_t n = 0; n < deps.size(); n++)
		deps[n] = route_to_region(deps[n], apply->region());

	return deps;
}

static void
inline_apply(const jive::structural_node * lambda, jive::simple_node * apply)
{
	JLM_DEBUG_ASSERT(is_lambda_op(lambda->operation()));
	JLM_DEBUG_ASSERT(dynamic_cast<const call_op*>(&apply->operation()));

	auto deps = route_dependencies(lambda, apply);

	jive::substitution_map smap;
	for (size_t n = 1; n < apply->ninputs(); n++) {
		auto argument = lambda->subregion(0)->argument(n-1);
		JLM_DEBUG_ASSERT(argument->input() == nullptr);
		smap.insert(argument, apply->input(n)->origin());
	}
	for (size_t n = 0; n < lambda->ninputs(); n++) {
		auto argument = lambda->input(n)->arguments.first();
		JLM_DEBUG_ASSERT(argument != nullptr);
		smap.insert(argument, deps[n]);
	}

	lambda->subregion(0)->copy(apply->region(), smap, false, false);

	for (size_t n = 0; n < apply->noutputs(); n++) {
		auto output = lambda->subregion(0)->result(n)->origin();
		JLM_DEBUG_ASSERT(smap.lookup(output));
		apply->output(n)->replace(smap.lookup(output));
	}
	remove(apply);
}

void
inlining(jive::graph & graph)
{
	auto root = graph.root();

	#ifdef ILNTIME
		auto nnodes = jive::nnodes(root);
		auto start = std::chrono::high_resolution_clock::now();
	#endif

	for (auto node : jive::topdown_traverser(root)) {
		if (!is_lambda_op(node->operation()))
			continue;

		if (is_exported(node->output(0)))
			continue;

		auto snode = static_cast<const jive::structural_node*>(node);
		auto consumers = find_consumers(snode);
		if (consumers.size() == 1
		&& dynamic_cast<const call_op*>(&consumers[0]->operation()))
			inline_apply(snode, static_cast<jive::simple_node*>(consumers[0]));
	}

	#ifdef ILNTIME
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "ILNTIME: "
		          << nnodes
		          << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()
		          << "\n";
	#endif
}

}
