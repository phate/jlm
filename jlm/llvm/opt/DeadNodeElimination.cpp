/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

static bool
is_phi_argument(const jlm::rvsdg::output * output)
{
  auto argument = dynamic_cast<const jlm::rvsdg::argument*>(output);
  return argument
         && argument->region()->node()
         && is<phi::operation>(argument->region()->node());
}

/** \brief Dead Node Elimination statistics class
 *
 */
class DeadNodeElimination::Statistics final : public util::Statistics {
public:
	~Statistics() override = default;

	Statistics()
	: util::Statistics(Statistics::Id::DeadNodeElimination)
  , numNodesBefore_(0), numNodesAfter_(0)
	, numInputsBefore_(0), numInputsAfter_(0)
	{}

	void
	StartMarkStatistics(const jlm::rvsdg::graph & graph) noexcept
	{
    numNodesBefore_ = jlm::rvsdg::nnodes(graph.root());
    numInputsBefore_ = jlm::rvsdg::ninputs(graph.root());
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
	StopSweepStatistics(const jlm::rvsdg::graph & graph) noexcept
	{
    sweepTimer_.stop();
    numNodesAfter_ = jlm::rvsdg::nnodes(graph.root());
    numInputsAfter_ = jlm::rvsdg::ninputs(graph.root());
	}

	[[nodiscard]] std::string
	ToString() const override
	{
		return util::strfmt("DeadNodeElimination ",
                  "#RvsdgNodesBeforeDNE:", numNodesBefore_, " ",
                  "#RvsdgNodesAfterDNE:", numNodesAfter_, " ",
                  "#RvsdgInputsBeforeDNE:", numInputsBefore_, " ",
                  "#RvsdgInputsAfterDNE:", numInputsAfter_, " ",
                  "MarkTime[ns]:", markTimer_.ns(), " ",
                  "SweepTime[ns]:", sweepTimer_.ns()
		);
	}

  static std::unique_ptr<Statistics>
  Create()
  {
    return std::make_unique<Statistics>();
  }

private:
	size_t numNodesBefore_;
  size_t numNodesAfter_;
	size_t numInputsBefore_;
  size_t numInputsAfter_;
	util::timer markTimer_;
  util::timer sweepTimer_;
};

DeadNodeElimination::~DeadNodeElimination()
= default;

void
DeadNodeElimination::run(jlm::rvsdg::region & region)
{
  Context_ = Context::Create();

	Mark(region);
	Sweep(region);

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::run(
  RvsdgModule & module,
  jlm::util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create();

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create();
  statistics->StartMarkStatistics(rvsdg);
  Mark(*rvsdg.root());
  statistics->StopMarkStatistics();

  statistics->StartSweepStatistics();
  Sweep(rvsdg);
  statistics->StopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::Mark(const jlm::rvsdg::region & region)
{
  for (size_t n = 0; n < region.nresults(); n++)
    Mark(*region.result(n)->origin());
}

void
DeadNodeElimination::Mark(const jlm::rvsdg::output & output)
{
  if (Context_->IsAlive(output))
    return;

  Context_->MarkAlive(output);

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
    auto thetaInput = static_cast<const jlm::rvsdg::theta_input*>(thetaArgument->input());
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
    auto soutput = static_cast<const jlm::rvsdg::structural_output*>(&output);
    Mark(*soutput->results.first()->origin());
    return;
  }

  if (is_phi_argument(&output)) {
    auto argument = static_cast<const jlm::rvsdg::argument*>(&output);
    if (argument->input()) Mark(*argument->input()->origin());
    else Mark(*argument->region()->result(argument->index())->origin());
    return;
  }

  if (auto deltaOutput = dynamic_cast<const delta::output*>(&output)) {
    Mark(*deltaOutput->node()->subregion()->result(0)->origin());
    return;
  }

  if (auto deltaCvArgument = dynamic_cast<const delta::cvargument*>(&output)) {
    Mark(*deltaCvArgument->input()->origin());
    return;
  }

  if (auto simpleOutput = dynamic_cast<const jlm::rvsdg::simple_output*>(&output)) {
    auto node = simpleOutput->node();
    for (size_t n = 0; n < node->ninputs(); n++)
      Mark(*node->input(n)->origin());
    return;
  }

  JLM_UNREACHABLE("We should have never reached this statement.");
}

void
DeadNodeElimination::Sweep(jlm::rvsdg::graph & graph) const
{
  Sweep(*graph.root());

  /**
   * Remove dead imports
   */
  for (size_t n = graph.root()->narguments()-1; n != static_cast<size_t>(-1); n--) {
    if (!Context_->IsAlive(*graph.root()->argument(n)))
      graph.root()->remove_argument(n);
  }
}

void
DeadNodeElimination::Sweep(jlm::rvsdg::region & region) const
{
  region.prune(false);

  std::vector<std::vector<jlm::rvsdg::node*>> nodesTopDown(region.nnodes());
  for (auto & node : region.nodes)
    nodesTopDown[node.depth()].push_back(&node);

  for (auto it = nodesTopDown.rbegin(); it != nodesTopDown.rend(); it++) {
    for (auto node : *it) {
      if (!Context_->IsAlive(*node)) {
        remove(node);
        continue;
      }

      if (auto structuralNode = dynamic_cast<jlm::rvsdg::structural_node*>(node))
        Sweep(*structuralNode);
    }
  }

  JLM_ASSERT(region.bottom_nodes.empty());
}

void
DeadNodeElimination::Sweep(jlm::rvsdg::structural_node & node) const
{
  static std::unordered_map<
    std::type_index,
    std::function<void(const DeadNodeElimination&, jlm::rvsdg::structural_node&)>
  > map({
    {typeid(jlm::rvsdg::gamma_op),       [](auto & d, auto & n){ d.SweepGamma(*static_cast<jlm::rvsdg::gamma_node*>(&n)); }},
    {typeid(jlm::rvsdg::theta_op),       [](auto & d, auto & n){ d.SweepTheta(*static_cast<jlm::rvsdg::theta_node*>(&n)); }},
    {typeid(lambda::operation),    [](auto & d, auto & n){ d.SweepLambda(*static_cast<lambda::node*>(&n));    }},
    {typeid(phi::operation),       [](auto & d, auto & n){ d.SweepPhi(*static_cast<phi::node*>(&n));          }},
    {typeid(delta::operation),     [](auto & d, auto & n){ d.SweepDelta(*static_cast<delta::node*>(&n));      }}
  });

  auto & op = node.operation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](*this, node);
}

void
DeadNodeElimination::SweepGamma(jlm::rvsdg::gamma_node & gammaNode) const
{
  /**
   * Remove dead outputs and results
   */
  for (size_t n = gammaNode.noutputs()-1; n != static_cast<size_t>(-1); n--) {
    if (Context_->IsAlive(*gammaNode.output(n)))
      continue;

    for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      gammaNode.subregion(r)->remove_result(n);
    gammaNode.remove_output(n);
  }

  /**
   * Sweep Gamma subregions
   */
  for (size_t r = 0; r < gammaNode.nsubregions(); r++)
    Sweep(*gammaNode.subregion(r));

  /**
   * Remove dead arguments and inputs
   */
  for (size_t n = gammaNode.ninputs()-1; n >= 1; n--) {
    auto input = gammaNode.input(n);

    bool alive = false;
    for (auto & argument : input->arguments) {
      if (Context_->IsAlive(argument)) {
        alive = true;
        break;
      }
    }
    if (!alive) {
      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
        gammaNode.subregion(r)->remove_argument(n-1);
      gammaNode.remove_input(n);
    }
  }
}

void
DeadNodeElimination::SweepTheta(jlm::rvsdg::theta_node & thetaNode) const
{
  auto subregion = thetaNode.subregion();

  /**
   * Remove dead results
   */
  for (size_t n = thetaNode.noutputs()-1; n != static_cast<size_t>(-1); n--) {
    auto & thetaOutput = *thetaNode.output(n);
    auto & thetaArgument = *thetaOutput.argument();
    auto & thetaResult = *thetaOutput.result();

    if (!Context_->IsAlive(thetaArgument) && !Context_->IsAlive(thetaOutput))
      subregion->remove_result(thetaResult.index());
  }

  Sweep(*subregion);

  /**
   * Remove dead outputs, inputs, and arguments
   */
  for (size_t n = thetaNode.ninputs()-1; n != static_cast<size_t>(-1); n--) {
    auto & thetaInput = *thetaNode.input(n);
    auto & thetaArgument = *thetaInput.argument();
    auto & thetaOutput = *thetaInput.output();

    if (!Context_->IsAlive(thetaArgument) && !Context_->IsAlive(thetaOutput)) {
      JLM_ASSERT(thetaOutput.results.empty());
      subregion->remove_argument(thetaArgument.index());
      thetaNode.remove_input(thetaInput.index());
      thetaNode.remove_output(thetaOutput.index());
    }
  }

  JLM_ASSERT(thetaNode.ninputs() == thetaNode.noutputs());
  JLM_ASSERT(subregion->narguments() == subregion->nresults()-1);
}

void
DeadNodeElimination::SweepLambda(lambda::node & lambdaNode) const
{
  Sweep(*lambdaNode.subregion());

  /**
   * Remove dead arguments and inputs
   */
  for (size_t n = lambdaNode.ninputs()-1; n != static_cast<size_t>(-1); n--) {
    auto input = lambdaNode.input(n);

    if (!Context_->IsAlive(*input->argument())) {
      lambdaNode.subregion()->remove_argument(input->argument()->index());
      lambdaNode.remove_input(n);
    }
  }
}

void
DeadNodeElimination::SweepPhi(phi::node & phiNode) const
{
  auto subregion = phiNode.subregion();

  /**
   * Remove dead outputs and results
   */
  for (size_t n = subregion->nresults()-1; n != static_cast<size_t>(-1); n--) {
    auto result = subregion->result(n);
    if (!Context_->IsAlive(*result->output())
        && !Context_->IsAlive(*subregion->argument(result->index()))) {
      subregion->remove_result(n);
      phiNode.remove_output(n);
    }
  }

  Sweep(*subregion);

  /**
   * Remove dead arguments and inputs
   */
  for (size_t n = subregion->narguments()-1; n != static_cast<size_t>(-1); n--) {
    auto argument = subregion->argument(n);
    auto input = argument->input();
    if (!Context_->IsAlive(*argument)) {
      subregion->remove_argument(n);
      if (input) {
        phiNode.remove_input(input->index());
      }
    }
  }
}

void
DeadNodeElimination::SweepDelta(delta::node & deltaNode) const
{
  /**
   * A delta subregion can only contain simple nodes. Thus, a simple prune is sufficient.
   */
  deltaNode.subregion()->prune(false);

  /**
   * Remove dead arguments and inputs.
   */
  for (size_t n = deltaNode.ninputs()-1; n != static_cast<size_t>(-1); n--) {
    auto input = deltaNode.input(n);
    if (!Context_->IsAlive(*input->argument())) {
      deltaNode.subregion()->remove_argument(input->argument()->index());
      deltaNode.remove_input(input->index());
    }
  }
}

}
