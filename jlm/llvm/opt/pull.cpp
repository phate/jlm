/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class pullstat final : public Statistics {
public:
	virtual
	~pullstat()
	{}

	pullstat()
	: Statistics(Statistics::Id::PullNodes)
  , ninputs_before_(0), ninputs_after_(0)
	{}

	void
	start(const jive::graph & graph) noexcept
	{
		ninputs_before_ = jive::ninputs(graph.root());
		timer_.start();
	}

	void
	end(const jive::graph & graph) noexcept
	{
		ninputs_after_ = jive::ninputs(graph.root());
		timer_.stop();
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("PULL ",
			ninputs_before_, " ", ninputs_after_, " ",
			timer_.ns()
		);
	}

  static std::unique_ptr<pullstat>
  Create()
  {
    return std::make_unique<pullstat>();
  }

private:
	size_t ninputs_before_, ninputs_after_;
	jlm::timer timer_;
};

static bool
empty(const jive::gamma_node * gamma)
{
	for (size_t n = 0; n < gamma->nsubregions(); n++) {
		if (gamma->subregion(n)->nnodes() != 0)
			return false;
	}

	return true;
}

static bool
single_successor(const jive::node * node)
{
	std::unordered_set<jive::node*> successors;
	for (size_t n = 0; n < node->noutputs(); n++) {
		for (const auto & user : *node->output(n))
			successors.insert(input_node(user));
	}

	return successors.size() == 1;
}

static void
remove(jive::gamma_input * input)
{
	auto gamma = input->node();

	for (size_t n = 0; n < gamma->nsubregions(); n++)
		gamma->subregion(n)->remove_argument(input->index()-1);
	gamma->remove_input(input->index());
}

static void
pullin_node(jive::gamma_node * gamma, jive::node * node)
{
	/* collect operands */
	std::vector<std::vector<jive::output*>> operands(gamma->nsubregions());
	for (size_t i = 0; i < node->ninputs(); i++) {
		auto ev = gamma->add_entryvar(node->input(i)->origin());
		for (size_t a = 0; a < ev->narguments(); a++)
			operands[a].push_back(ev->argument(a));
	}

	/* copy node into subregions */
	for (size_t r = 0; r < gamma->nsubregions(); r++) {
		auto copy = node->copy(gamma->subregion(r), operands[r]);

		/* redirect outputs */
		for (size_t o = 0; o < node->noutputs(); o++) {
			for (const auto & user : *node->output(o)) {
				JLM_ASSERT(dynamic_cast<jive::structural_input*>(user));
				auto sinput = static_cast<jive::structural_input*>(user);
				auto argument = gamma->subregion(r)->argument(sinput->index()-1);
				argument->divert_users(copy->output(o));
			}
		}
	}
}

static void
cleanup(jive::gamma_node * gamma, jive::node * node)
{
	JLM_ASSERT(single_successor(node));

	/* remove entry variables and node */
	for (size_t n = 0; n < node->noutputs(); n++) {
		while (node->output(n)->nusers() != 0)
			remove(static_cast<jive::gamma_input*>(*node->output(n)->begin()));
	}
	remove(node);
}

void
pullin_top(jive::gamma_node * gamma)
{
	/* FIXME: This is inefficient. We can do better. */
	auto ev = gamma->begin_entryvar();
	while (ev != gamma->end_entryvar()) {
		auto node = jive::node_output::node(ev->origin());
		auto tmp = jive::node_output::node(gamma->predicate()->origin());
		if (node && tmp != node && single_successor(node)) {
			pullin_node(gamma, node);

			cleanup(gamma, node);

			ev = gamma->begin_entryvar();
		} else {
			ev++;
		}
	}
}

void
pullin_bottom(jive::gamma_node * gamma)
{
	/* collect immediate successors of the gamma node */
	std::unordered_set<jive::node*> workset;
	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = gamma->output(n);
		for (const auto & user : *output) {
			auto node = input_node(user);
			if (node && node->depth() == gamma->depth()+1)
				workset.insert(node);
		}
	}

	while (!workset.empty()) {
		auto node = *workset.begin();
		workset.erase(node);

		/* copy node into subregions */
		std::vector<std::vector<jive::output*>> outputs(node->noutputs());
		for (size_t r = 0; r < gamma->nsubregions(); r++) {
			/* collect operands */
			std::vector<jive::output*> operands;
			for (size_t i = 0; i < node->ninputs(); i++) {
				auto input = node->input(i);
				if (jive::node_output::node(input->origin()) == gamma) {
					auto output = static_cast<jive::structural_output*>(input->origin());
					operands.push_back(gamma->subregion(r)->result(output->index())->origin());
				} else {
					auto ev = gamma->add_entryvar(input->origin());
					operands.push_back(ev->argument(r));
				}
			}

			auto copy = node->copy(gamma->subregion(r), operands);
			for (size_t o = 0; o < copy->noutputs(); o++)
				outputs[o].push_back(copy->output(o));
		}

		/* adjust outputs and update workset */
		for (size_t n = 0; n < node->noutputs(); n++) {
			auto output = node->output(n);
			for (const auto & user : *output) {
				auto tmp = input_node(user);
				if (tmp && tmp->depth() == node->depth()+1)
					workset.insert(tmp);
			}

			auto xv = gamma->add_exitvar(outputs[n]);
			output->divert_users(xv);
		}
	}
}

static size_t
is_used_in_nsubregions(const jive::gamma_node * gamma, const jive::node * node)
{
	JLM_ASSERT(single_successor(node));

	/* collect all gamma inputs */
	std::unordered_set<const jive::gamma_input*> inputs;
	for (size_t n = 0; n < node->noutputs(); n++) {
		for (const auto & user : *(node->output(n))) {
			JLM_ASSERT(is_gamma_input(user));
			inputs.insert(static_cast<const jive::gamma_input*>(user));
		}
	}

	/* collect subregions where node is used */
	std::unordered_set<jive::region*> subregions;
	for (const auto & input : inputs) {
		for (const auto & argument : *input) {
			if (argument.nusers() != 0)
				subregions.insert(argument.region());
		}
	}

	return subregions.size();
}

void
pull(jive::gamma_node * gamma)
{
	/*
		We don't want to pull anything into empty gammas with two subregions,
		as they are translated to select instructions in the r2j phase.
	*/
	if (gamma->nsubregions() == 2 && empty(gamma))
		return;

	auto prednode = jive::node_output::node(gamma->predicate()->origin());

	/* FIXME: This is inefficient. We can do better. */
	auto ev = gamma->begin_entryvar();
	while (ev != gamma->end_entryvar()) {
		auto node = jive::node_output::node(ev->origin());
		if (!node || prednode == node || !single_successor(node)) {
			ev++; continue;
		}

		if (is_used_in_nsubregions(gamma, node) == 1) {
			/*
				FIXME: This function pulls in the node to ALL subregions and
				not just the one we care about.
			*/
			pullin_node(gamma, node);
			cleanup(gamma, node);
			ev = gamma->begin_entryvar();
		} else {
			ev++;
		}
	}
}

void
pull(jive::region * region)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			if (auto gamma = dynamic_cast<jive::gamma_node*>(node))
				pull(gamma);

			for (size_t n = 0; n < structnode->nsubregions(); n++)
				pull(structnode->subregion(n));
		}
	}
}

static void
pull(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto statistics = pullstat::Create();

	statistics->start(rm.Rvsdg());
	pull(rm.Rvsdg().root());
	statistics->end(rm.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* pullin class */

pullin::~pullin()
{}

void
pullin::run(
  RvsdgModule & module,
  StatisticsCollector & statisticsCollector)
{
	pull(module, statisticsCollector);
}

}
