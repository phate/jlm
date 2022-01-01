/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/DeadNodeElimination.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/phi.hpp>
#include <jive/rvsdg/theta.hpp>
#include <jive/rvsdg/traverser.hpp>

namespace jlm {

static bool
is_phi_argument(const jive::output * output)
{
  auto argument = dynamic_cast<const jive::argument*>(output);
  return argument
         && argument->region()->node()
         && is<jive::phi::operation>(argument->region()->node());
}

/** \brief Dead Node Elimination statistics class
 *
 */
class DeadNodeElimination::Statistics final : public jlm::Statistics {
public:
	~Statistics() override = default;

	Statistics()
	: numNodesBefore_(0), numNodesAfter_(0)
	, numInputsBefore_(0), numInputsAfter_(0)
	{}

	void
	StartMarkStatistics(const jive::graph & graph) noexcept
	{
    numNodesBefore_ = jive::nnodes(graph.root());
    numInputsBefore_ = jive::ninputs(graph.root());
		markTimer_.start();
	}

	void
	StopMarkStatistics() noexcept
	{
		markTimer_.stop();
	}

	void
	StartSweepStatistics() noexcept
	{
		sweepTimer_.start();
	}

	void
	StopSweepStatistics(const jive::graph & graph) noexcept
	{
    numNodesAfter_ = jive::nnodes(graph.root());
    numInputsAfter_ = jive::ninputs(graph.root());
		sweepTimer_.stop();
	}

	std::string
	ToString() const override
	{
		return strfmt("DNE ",
                  numNodesBefore_, " ", numNodesAfter_, " ",
                  numInputsBefore_, " ", numInputsAfter_, " ",
                  markTimer_.ns(), " ", sweepTimer_.ns()
		);
	}

private:
	size_t numNodesBefore_;
  size_t numNodesAfter_;
	size_t numInputsBefore_;
  size_t numInputsAfter_;
	jlm::timer markTimer_;
  jlm::timer sweepTimer_;
};

DeadNodeElimination::~DeadNodeElimination()
= default;

void
DeadNodeElimination::run(jive::region & region)
{
  ResetState();
	Mark(region);
	Sweep(region);
}

void
DeadNodeElimination::run(
  rvsdg_module & module,
  const StatisticsDescriptor & sd)
{
  auto & graph = *module.graph();

  ResetState();

  Statistics statistics;
  statistics.StartMarkStatistics(graph);
  Mark(*graph.root());
  statistics.StopMarkStatistics();

  statistics.StartSweepStatistics();
  Sweep(graph);
  statistics.StopSweepStatistics(graph);

  if (sd.IsPrintable(StatisticsDescriptor::StatisticsId::DeadNodeElimination))
    sd.print_stat(statistics);
}

void
DeadNodeElimination::ResetState()
{
  context_.Clear();
}

void
DeadNodeElimination::Mark(const jive::region & region)
{
  for (size_t n = 0; n < region.nresults(); n++)
    Mark(*region.result(n)->origin());
}

void
DeadNodeElimination::Mark(const jive::output & output)
{
  if (context_.IsAlive(output))
    return;

  context_.Mark(output);

  if (is_import(&output))
    return;

  if (auto gammaOutput = is_gamma_output(&output)) {
    Mark(*gammaOutput->node()->predicate()->origin());
    for (const auto & result : gammaOutput->results)
      Mark(*result.origin());
    return;
  }

  if (auto argument = is_gamma_argument(&output)) {
    Mark(*argument->input()->origin());
    return;
  }

  if (auto thetaOutput = is_theta_output(&output)) {
    Mark(*thetaOutput->node()->predicate()->origin());
    Mark(*thetaOutput->result()->origin());
    Mark(*thetaOutput->input()->origin());
    return;
  }

  if (auto thetaArgument = is_theta_argument(&output)) {
    auto thetaInput = static_cast<const jive::theta_input*>(thetaArgument->input());
    Mark(*thetaInput->output());
    Mark(*thetaInput->origin());
    return;
  }

  if (auto o = dynamic_cast<const lambda::output*>(&output)) {
    for (auto & result : o->node()->fctresults())
      Mark(*result.origin());
    return;
  }

  if (is<lambda::fctargument>(&output))
    return;

  if (auto cv = dynamic_cast<const lambda::cvargument*>(&output)) {
    Mark(*cv->input()->origin());
    return;
  }

  if (is_phi_output(&output)) {
    auto soutput = static_cast<const jive::structural_output*>(&output);
    Mark(*soutput->results.first()->origin());
    return;
  }

  if (is_phi_argument(&output)) {
    auto argument = static_cast<const jive::argument*>(&output);
    if (argument->input()) Mark(*argument->input()->origin());
    else Mark(*argument->region()->result(argument->index())->origin());
    return;
  }

  if (auto deltaOutput = dynamic_cast<const delta::output*>(&output)) {
    auto deltaNode = deltaOutput->node();
    for (size_t n = 0; n < deltaNode->ninputs(); n++)
      Mark(*deltaNode->input(n)->origin());
    return;
  }

  if (auto simpleOutput = dynamic_cast<const jive::simple_output*>(&output)) {
    auto node = simpleOutput->node();

    /**
     * A single output of this simple node is alive, which
     * in turn means that the entire node is alive. Thus,
     * we proactively mark all other outputs of this node
     * as alive. This is not necessary for correctness, but
     * rather for performance. By marking all outputs alive,
     * we avoid that we reiterate through all inputs of this node
     * again in the future. The following example illustrates the
     * issue:
     *
     * o1 ... oN = Node2 i1 ... iN
     * p1 ... pN = Node1 o1 ... oN
     *
     * When we mark o1 as alive, we also mark with the following
     * loop o2 ... oN proactively alive. This means that when we
     * try to mark o2 alive in the future, we can immediately stop
     * marking instead of reiterating through i1 ... iN again. Thus,
     * proactively marking reduces the runtime for marking Node2 from
     * O(oN x iN) to O(oN + iN).
     */
    for (size_t n = 0; n < node->noutputs(); n++)
      context_.Mark(*node->output(n));

    for (size_t n = 0; n < node->ninputs(); n++)
      Mark(*node->input(n)->origin());
    return;
  }

  JLM_UNREACHABLE("We should have never reached this statement.");
}

void
DeadNodeElimination::Sweep(jive::graph & graph) const
{
  Sweep(*graph.root());
  for (ssize_t n = graph.root()->narguments()-1; n >= 0; n--) {
    if (!context_.IsAlive(*graph.root()->argument(n)))
      graph.root()->remove_argument(n);
  }
}

void
DeadNodeElimination::Sweep(jive::region & region) const
{
  for (const auto & node : jive::bottomup_traverser(&region)) {
    if (auto simple = dynamic_cast<jive::simple_node*>(node))
      Sweep(*simple);
    else
      Sweep(*static_cast<jive::structural_node*>(node));
  }

  JLM_ASSERT(region.bottom_nodes.empty());
}

void
DeadNodeElimination::Sweep(jive::simple_node & node) const
{
  if (!context_.IsAlive(node))
    remove(&node);
}

void
DeadNodeElimination::Sweep(jive::structural_node & node) const
{
  static std::unordered_map<
    std::type_index,
    std::function<void(const DeadNodeElimination&, jive::structural_node&)>
  > map({
    {typeid(jive::gamma_op),       [](auto & dne, auto & node){ dne.SweepGamma(node); }},
    {typeid(jive::theta_op),       [](auto & dne, auto & node){ dne.SweepTheta(node); }},
    {typeid(lambda::operation),    [](auto & dne, auto & node){ dne.SweepLambda(node); }},
    {typeid(jive::phi::operation), [](auto & dne, auto & node){ dne.SweepPhi(node); }},
    {typeid(delta::operation),     [](auto & dne, auto & node){ dne.SweepDelta(node); }}
  });

  auto & op = node.operation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](*this, node);
}

void
DeadNodeElimination::SweepGamma(jive::structural_node & node) const
{
  JLM_ASSERT(jive::is<jive::gamma_op>(&node));

  if (!context_.IsAlive(node)) {
    remove(&node);
    return;
  }

  /* remove outputs and results */
  for (ssize_t n = node.noutputs()-1; n >= 0; n--) {
    if (context_.IsAlive(*node.output(n)))
      continue;

    for (size_t r = 0; r < node.nsubregions(); r++)
      node.subregion(r)->remove_result(n);
    node.remove_output(n);
  }

  for (size_t r = 0; r < node.nsubregions(); r++)
    Sweep(*node.subregion(r));

  /* remove arguments and inputs */
  for (ssize_t n = node.ninputs()-1; n >=  1; n--) {
    auto input = node.input(n);

    bool alive = false;
    for (const auto & argument : input->arguments) {
      if (context_.IsAlive(argument)) {
        alive = true;
        break;
      }
    }
    if (!alive) {
      for (size_t r = 0; r < node.nsubregions(); r++)
        node.subregion(r)->remove_argument(n-1);
      node.remove_input(n);
    }
  }
}

void
DeadNodeElimination::SweepTheta(jive::structural_node & node) const
{
  JLM_ASSERT(jive::is<jive::theta_op>(&node));
  auto subregion = node.subregion(0);

  if (!context_.IsAlive(node)) {
    remove(&node);
    return;
  }

  /* remove results */
  for (ssize_t n = subregion->nresults()-1; n >= 1; n--) {
    if (!context_.IsAlive(*subregion->argument(n - 1))
    && !context_.IsAlive(*node.output(n - 1)))
      subregion->remove_result(n);
  }

  Sweep(*subregion);

  /* remove outputs, inputs, and arguments */
  for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
    if (!context_.IsAlive(*subregion->argument(n))
    && !context_.IsAlive(*node.output(n))) {
      JLM_ASSERT(node.output(n)->results.first() == nullptr);
      subregion->remove_argument(n);
      node.remove_input(n);
      node.remove_output(n);
    }
  }

  JLM_ASSERT(node.ninputs() == node.noutputs());
  JLM_ASSERT(subregion->narguments() == subregion->nresults()-1);
}

void
DeadNodeElimination::SweepLambda(jive::structural_node & node) const
{
  JLM_ASSERT(is<lambda::operation>(&node));
  auto subregion = node.subregion(0);

  if (!context_.IsAlive(node)) {
    remove(&node);
    return;
  }

  Sweep(*subregion);

  /* remove inputs and arguments */
  for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
    auto argument = subregion->argument(n);
    if (argument->input() == nullptr)
      continue;

    if (!context_.IsAlive(*argument)) {
      size_t index = argument->input()->index();
      subregion->remove_argument(n);
      node.remove_input(index);
    }
  }
}

void
DeadNodeElimination::SweepPhi(jive::structural_node & node) const
{
  JLM_ASSERT(is<jive::phi::operation>(&node));
  auto subregion = node.subregion(0);

  if (!context_.IsAlive(node)) {
    remove(&node);
    return;
  }

  /* remove outputs and results */
  for (ssize_t n = subregion->nresults()-1; n >= 1; n--) {
    auto result = subregion->result(n);
    if (!context_.IsAlive(*result->output())
        && !context_.IsAlive(*subregion->argument(result->index()))) {
      subregion->remove_result(n);
      node.remove_output(n);
    }
  }

  Sweep(*subregion);

  /* remove dead arguments and dependencies */
  for (ssize_t n = subregion->narguments()-1; n >= 0; n--) {
    auto argument = subregion->argument(n);
    auto input = argument->input();
    if (!context_.IsAlive(*argument)) {
      subregion->remove_argument(n);
      if (input) node.remove_input(input->index());
    }
  }
}

void
DeadNodeElimination::SweepDelta(jive::structural_node & node) const
{
  JLM_ASSERT(is<delta::operation>(&node));
  JLM_ASSERT(node.noutputs() == 1);

  if (!context_.IsAlive(node)) {
    remove(&node);
    return;
  }
}

}
