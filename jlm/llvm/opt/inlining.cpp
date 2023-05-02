/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class ilnstat final : public Statistics {
public:
	virtual
	~ilnstat()
	{}

	ilnstat()
	: Statistics(Statistics::Id::FunctionInlining)
  , nnodes_before_(0)
  , nnodes_after_(0)
	{}

	void
	start(const jive::graph & graph)
	{
		nnodes_before_ = jive::nnodes(graph.root());
		timer_.start();
	}

	void
	stop(const jive::graph & graph)
	{
		nnodes_after_ = jive::nnodes(graph.root());
		timer_.stop();
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("ILN ", nnodes_before_, " ", nnodes_after_, " ", timer_.ns());
	}

  static std::unique_ptr<ilnstat>
  Create()
  {
    return std::make_unique<ilnstat>();
  }

private:
	size_t nnodes_before_, nnodes_after_;
	jlm::timer timer_;
};

jive::output *
find_producer(jive::input * input)
{
	auto graph = input->region()->graph();

	auto argument = dynamic_cast<jive::argument*>(input->origin());
	if (argument == nullptr)
		return input->origin();

	if (argument->region() == graph->root())
		return argument;

	JLM_ASSERT(argument->input() != nullptr);
	return find_producer(argument->input());
}

static jive::output *
route_to_region(jive::output * output, jive::region * region)
{
	JLM_ASSERT(region != nullptr);

	if (region == output->region())
		return output;

	output = route_to_region(output, region->node()->region());

	if (auto gamma = dynamic_cast<jive::gamma_node*>(region->node())) {
		gamma->add_entryvar(output);
		output = region->argument(region->narguments()-1);
	} else if (auto theta = dynamic_cast<jive::theta_node*>(region->node())) {
		output = theta->add_loopvar(output)->argument();
	} else if (auto lambda = dynamic_cast<lambda::node*>(region->node())) {
		output = lambda->add_ctxvar(output);
	} else if (auto phi = dynamic_cast<phi::node*>(region->node())) {
		output = phi->add_ctxvar(output);
	} else {
		JLM_UNREACHABLE("This should have never happened!");
	}

	return output;
}

static std::vector<jive::output*>
route_dependencies(const lambda::node * lambda, const jive::simple_node * apply)
{
	JLM_ASSERT(is<CallOperation>(apply));

	/* collect origins of dependencies */
	std::vector<jive::output*> deps;
	for (size_t n = 0; n < lambda->ninputs(); n++)
		deps.push_back(find_producer(lambda->input(n)));

	/* route dependencies to apply region */
	for (size_t n = 0; n < deps.size(); n++)
		deps[n] = route_to_region(deps[n], apply->region());

	return deps;
}

void
inlineCall(jive::simple_node * call, const lambda::node * lambda)
{
	JLM_ASSERT(is<CallOperation>(call));

	auto deps = route_dependencies(lambda, call);
	JLM_ASSERT(lambda->ncvarguments() == deps.size());

	jive::substitution_map smap;
	for (size_t n = 1; n < call->ninputs(); n++) {
		auto argument = lambda->fctargument(n-1);
		smap.insert(argument, call->input(n)->origin());
	}
	for (size_t n = 0; n < lambda->ncvarguments(); n++)
		smap.insert(lambda->cvargument(n), deps[n]);

	lambda->subregion()->copy(call->region(), smap, false, false);

	for (size_t n = 0; n < call->noutputs(); n++) {
		auto output = lambda->subregion()->result(n)->origin();
		JLM_ASSERT(smap.lookup(output));
		call->output(n)->divert_users(smap.lookup(output));
	}
	remove(call);
}

static void
inlining(jive::graph & graph)
{
	auto root = graph.root();

	for (auto node : jive::topdown_traverser(root)) {
		if (!is<lambda::operation>(node))
			continue;

		auto lambda = static_cast<const lambda::node*>(node);

		std::vector<jive::simple_node*> calls;
		bool onlyDirectCalls = lambda->direct_calls(&calls);

		if (onlyDirectCalls && calls.size() == 1)
			inlineCall(calls[0], lambda);
	}
}

static void
inlining(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto & graph = rm.Rvsdg();
	auto statistics = ilnstat::Create();

	statistics->start(graph);
	inlining(graph);
	statistics->stop(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* fctinline class */

fctinline::~fctinline()
{}

void
fctinline::run(
  RvsdgModule & module,
  StatisticsCollector & statisticsCollector)
{
	inlining(module, statisticsCollector);
}

}
