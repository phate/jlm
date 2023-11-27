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
  auto argument = dynamic_cast<const jlm::rvsdg::argument *>(output);
  return argument && argument->region()->node() && is<phi::operation>(argument->region()->node());
}

/** \brief Dead Node Elimination context class
 *
 * This class keeps track of all the nodes and outputs that are alive. In contrast to all other
 * nodes, a simple node is considered alive if already a single of its outputs is alive. For this
 * reason, this class keeps separately track of simple nodes and therefore avoids to store all its
 * outputs (and instead stores the node itself). By marking the entire node as alive, we also avoid
 * that we reiterate through all inputs of this node again in the future. The following example
 * illustrates the issue:
 *
 * o1 ... oN = Node2 i1 ... iN
 * p1 ... pN = Node1 o1 ... oN
 *
 * When we mark o1 as alive, we actually mark the entire Node2 as alive. This means that when we try
 * to mark o2 alive in the future, we can immediately stop marking instead of reiterating through i1
 * ... iN again. Thus, by marking the entire simple node instead of just its outputs, we reduce the
 * runtime for marking Node2 from O(oN x iN) to O(oN + iN).
 */
class DeadNodeElimination::Context final
{
public:
  void
  MarkAlive(const jlm::rvsdg::output & output)
  {
    if (auto simpleOutput = dynamic_cast<const jlm::rvsdg::simple_output *>(&output))
    {
      SimpleNodes_.Insert(simpleOutput->node());
      return;
    }

    Outputs_.Insert(&output);
  }

  bool
  IsAlive(const jlm::rvsdg::output & output) const noexcept
  {
    if (auto simpleOutput = dynamic_cast<const jlm::rvsdg::simple_output *>(&output))
    {
      return SimpleNodes_.Contains(simpleOutput->node());
    }

    return Outputs_.Contains(&output);
  }

  bool
  IsAlive(const jlm::rvsdg::node & node) const noexcept
  {
    if (auto simpleNode = dynamic_cast<const jlm::rvsdg::simple_node *>(&node))
    {
      return SimpleNodes_.Contains(simpleNode);
    }

    for (size_t n = 0; n < node.noutputs(); n++)
    {
      if (IsAlive(*node.output(n)))
      {
        return true;
      }
    }

    return false;
  }

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

private:
  util::HashSet<const jlm::rvsdg::simple_node *> SimpleNodes_;
  util::HashSet<const jlm::rvsdg::output *> Outputs_;
};

/** \brief Dead Node Elimination statistics class
 *
 */
class DeadNodeElimination::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(util::filepath sourceFile)
      : util::Statistics(Statistics::Id::DeadNodeElimination),
        SourceFile_(std::move(sourceFile)),
        NumRvsdgNodesBefore_(0),
        NumRvsdgNodesAfter_(0),
        NumInputsBefore_(0),
        NumInputsAfter_(0)
  {}

  void
  StartMarkStatistics(const jlm::rvsdg::graph & graph) noexcept
  {
    NumRvsdgNodesBefore_ = jlm::rvsdg::nnodes(graph.root());
    NumInputsBefore_ = jlm::rvsdg::ninputs(graph.root());
    MarkTimer_.start();
  }

  void
  StopMarkStatistics() noexcept
  {
    MarkTimer_.stop();
  }

  void
  StartSweepStatistics() noexcept
  {
    SweepTimer_.start();
  }

  void
  StopSweepStatistics(const jlm::rvsdg::graph & graph) noexcept
  {
    SweepTimer_.stop();
    NumRvsdgNodesAfter_ = jlm::rvsdg::nnodes(graph.root());
    NumInputsAfter_ = jlm::rvsdg::ninputs(graph.root());
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return util::strfmt(
        "DeadNodeElimination ",
        SourceFile_.to_str(),
        " ",
        "#RvsdgNodesBeforeDNE:",
        NumRvsdgNodesBefore_,
        " ",
        "#RvsdgNodesAfterDNE:",
        NumRvsdgNodesAfter_,
        " ",
        "#RvsdgInputsBeforeDNE:",
        NumInputsBefore_,
        " ",
        "#RvsdgInputsAfterDNE:",
        NumInputsAfter_,
        " ",
        "MarkTime[ns]:",
        MarkTimer_.ns(),
        " ",
        "SweepTime[ns]:",
        SweepTimer_.ns());
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

private:
  util::filepath SourceFile_;

  size_t NumRvsdgNodesBefore_;
  size_t NumRvsdgNodesAfter_;
  size_t NumInputsBefore_;
  size_t NumInputsAfter_;

  util::timer MarkTimer_;
  util::timer SweepTimer_;
};

DeadNodeElimination::~DeadNodeElimination() noexcept = default;

DeadNodeElimination::DeadNodeElimination() = default;

void
DeadNodeElimination::run(jlm::rvsdg::region & region)
{
  Context_ = Context::Create();

  MarkRegion(region);
  SweepRegion(region);

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::run(RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create();

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFileName());
  statistics->StartMarkStatistics(rvsdg);
  MarkRegion(*rvsdg.root());
  statistics->StopMarkStatistics();

  statistics->StartSweepStatistics();
  SweepRvsdg(rvsdg);
  statistics->StopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::MarkRegion(const jlm::rvsdg::region & region)
{
  for (size_t n = 0; n < region.nresults(); n++)
  {
    MarkOutput(*region.result(n)->origin());
  }
}

void
DeadNodeElimination::MarkOutput(const jlm::rvsdg::output & output)
{
  if (Context_->IsAlive(output))
  {
    return;
  }

  Context_->MarkAlive(output);

  if (is_import(&output))
  {
    return;
  }

  if (auto gammaOutput = is_gamma_output(&output))
  {
    MarkOutput(*gammaOutput->node()->predicate()->origin());
    for (const auto & result : gammaOutput->results)
    {
      MarkOutput(*result.origin());
    }
    return;
  }

  if (auto argument = is_gamma_argument(&output))
  {
    MarkOutput(*argument->input()->origin());
    return;
  }

  if (auto thetaOutput = is_theta_output(&output))
  {
    MarkOutput(*thetaOutput->node()->predicate()->origin());
    MarkOutput(*thetaOutput->result()->origin());
    MarkOutput(*thetaOutput->input()->origin());
    return;
  }

  if (auto thetaArgument = is_theta_argument(&output))
  {
    auto thetaInput = util::AssertedCast<const jlm::rvsdg::theta_input>(thetaArgument->input());
    MarkOutput(*thetaInput->output());
    MarkOutput(*thetaInput->origin());
    return;
  }

  if (auto o = dynamic_cast<const lambda::output *>(&output))
  {
    for (auto & result : o->node()->fctresults())
    {
      MarkOutput(*result.origin());
    }
    return;
  }

  if (is<lambda::fctargument>(&output))
  {
    return;
  }

  if (auto cv = dynamic_cast<const lambda::cvargument *>(&output))
  {
    MarkOutput(*cv->input()->origin());
    return;
  }

  if (is_phi_output(&output))
  {
    auto structuralOutput = util::AssertedCast<const jlm::rvsdg::structural_output>(&output);
    MarkOutput(*structuralOutput->results.first()->origin());
    return;
  }

  if (is_phi_argument(&output))
  {
    auto argument = util::AssertedCast<const jlm::rvsdg::argument>(&output);
    if (argument->input())
    {
      MarkOutput(*argument->input()->origin());
    }
    else
    {
      MarkOutput(*argument->region()->result(argument->index())->origin());
    }
    return;
  }

  if (auto deltaOutput = dynamic_cast<const delta::output *>(&output))
  {
    MarkOutput(*deltaOutput->node()->subregion()->result(0)->origin());
    return;
  }

  if (auto deltaCvArgument = dynamic_cast<const delta::cvargument *>(&output))
  {
    MarkOutput(*deltaCvArgument->input()->origin());
    return;
  }

  if (auto simpleOutput = dynamic_cast<const jlm::rvsdg::simple_output *>(&output))
  {
    auto node = simpleOutput->node();
    for (size_t n = 0; n < node->ninputs(); n++)
    {
      MarkOutput(*node->input(n)->origin());
    }
    return;
  }

  JLM_UNREACHABLE("We should have never reached this statement.");
}

void
DeadNodeElimination::SweepRvsdg(jlm::rvsdg::graph & rvsdg) const
{
  SweepRegion(*rvsdg.root());

  // Remove dead imports
  for (size_t n = rvsdg.root()->narguments() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (!Context_->IsAlive(*rvsdg.root()->argument(n)))
    {
      rvsdg.root()->RemoveArgument(n);
    }
  }
}

void
DeadNodeElimination::SweepRegion(jlm::rvsdg::region & region) const
{
  region.prune(false);

  std::vector<std::vector<jlm::rvsdg::node *>> nodesTopDown(region.nnodes());
  for (auto & node : region.nodes)
  {
    nodesTopDown[node.depth()].push_back(&node);
  }

  for (auto it = nodesTopDown.rbegin(); it != nodesTopDown.rend(); it++)
  {
    for (auto node : *it)
    {
      if (!Context_->IsAlive(*node))
      {
        remove(node);
        continue;
      }

      if (auto structuralNode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
      {
        SweepStructuralNode(*structuralNode);
      }
    }
  }

  JLM_ASSERT(region.bottom_nodes.empty());
}

void
DeadNodeElimination::SweepStructuralNode(jlm::rvsdg::structural_node & node) const
{
  auto sweepGamma = [](auto & d, auto & n)
  {
    d.SweepGamma(*util::AssertedCast<jlm::rvsdg::gamma_node>(&n));
  };
  auto sweepTheta = [](auto & d, auto & n)
  {
    d.SweepTheta(*util::AssertedCast<jlm::rvsdg::theta_node>(&n));
  };
  auto sweepLambda = [](auto & d, auto & n)
  {
    d.SweepLambda(*util::AssertedCast<lambda::node>(&n));
  };
  auto sweepPhi = [](auto & d, auto & n)
  {
    d.SweepPhi(*util::AssertedCast<phi::node>(&n));
  };
  auto sweepDelta = [](auto & d, auto & n)
  {
    d.SweepDelta(*util::AssertedCast<delta::node>(&n));
  };

  static std::unordered_map<
      std::type_index,
      std::function<void(const DeadNodeElimination &, jlm::rvsdg::structural_node &)>>
      map({ { typeid(jlm::rvsdg::gamma_op), sweepGamma },
            { typeid(jlm::rvsdg::theta_op), sweepTheta },
            { typeid(lambda::operation), sweepLambda },
            { typeid(phi::operation), sweepPhi },
            { typeid(delta::operation), sweepDelta } });

  auto & op = node.operation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](*this, node);
}

void
DeadNodeElimination::SweepGamma(jlm::rvsdg::gamma_node & gammaNode) const
{
  // Remove dead outputs and results
  for (size_t n = gammaNode.noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (Context_->IsAlive(*gammaNode.output(n)))
    {
      continue;
    }

    for (size_t r = 0; r < gammaNode.nsubregions(); r++)
    {
      gammaNode.subregion(r)->RemoveResult(n);
    }
    gammaNode.RemoveOutput(n);
  }

  // Sweep gamma subregions
  for (size_t r = 0; r < gammaNode.nsubregions(); r++)
  {
    SweepRegion(*gammaNode.subregion(r));
  }

  // Remove dead arguments and inputs
  for (size_t n = gammaNode.ninputs() - 1; n >= 1; n--)
  {
    auto input = gammaNode.input(n);

    bool alive = false;
    for (auto & argument : input->arguments)
    {
      if (Context_->IsAlive(argument))
      {
        alive = true;
        break;
      }
    }
    if (!alive)
    {
      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      {
        gammaNode.subregion(r)->RemoveArgument(n - 1);
      }
      gammaNode.RemoveInput(n);
    }
  }
}

void
DeadNodeElimination::SweepTheta(jlm::rvsdg::theta_node & thetaNode) const
{
  auto & thetaSubregion = *thetaNode.subregion();

  auto matchOutput = [&](const rvsdg::theta_output & output)
  {
    auto & argument = *output.argument();
    return !Context_->IsAlive(argument) && !Context_->IsAlive(output);
  };
  auto deadInputs = thetaNode.RemoveThetaOutputsWhere(matchOutput);

  SweepRegion(thetaSubregion);

  auto matchInput = [&](const rvsdg::theta_input & input)
  {
    return deadInputs.Contains(&input);
  };
  thetaNode.RemoveThetaInputsWhere(matchInput);

  JLM_ASSERT(thetaNode.ninputs() == thetaNode.noutputs());
  JLM_ASSERT(thetaSubregion.narguments() == thetaSubregion.nresults() - 1);
}

void
DeadNodeElimination::SweepLambda(lambda::node & lambdaNode) const
{
  SweepRegion(*lambdaNode.subregion());
  lambdaNode.PruneLambdaInputs();
}

void
DeadNodeElimination::SweepPhi(phi::node & phiNode) const
{
  util::HashSet<const rvsdg::argument *> deadRecursionArguments;

  auto isDeadOutput = [&](const phi::rvoutput & output)
  {
    auto argument = output.argument();

    // A recursion variable is only dead iff its output AND argument are dead
    auto isDead = !Context_->IsAlive(output) && !Context_->IsAlive(*argument);
    if (isDead)
    {
      deadRecursionArguments.Insert(argument);
    }

    return isDead;
  };
  phiNode.RemovePhiOutputsWhere(isDeadOutput);

  SweepRegion(*phiNode.subregion());

  auto isDeadArgument = [&](const rvsdg::argument & argument)
  {
    if (argument.input())
    {
      // It is always safe to remove context variables if they are dead
      return argument.IsDead();
    }

    // Only remove the recursion argument if its output was removed in isDeadOutput()
    JLM_ASSERT(is<phi::rvargument>(&argument));
    return deadRecursionArguments.Contains(&argument);
  };
  phiNode.RemovePhiArgumentsWhere(isDeadArgument);
}

void
DeadNodeElimination::SweepDelta(delta::node & deltaNode)
{
  // A delta subregion can only contain simple nodes. Thus, a simple prune is sufficient.
  deltaNode.subregion()->prune(false);

  deltaNode.PruneDeltaInputs();
}

}
