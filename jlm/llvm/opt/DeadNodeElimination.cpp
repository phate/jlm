/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <typeindex>

namespace jlm::llvm
{

DNEStructuralNodeHandler::~DNEStructuralNodeHandler() = default;

DNEGammaNodeHandler::~DNEGammaNodeHandler() = default;

DNEGammaNodeHandler::DNEGammaNodeHandler() = default;

std::type_index
DNEGammaNodeHandler::GetTypeInfo() const
{
  return typeid(rvsdg::GammaNode);
}

std::optional<std::vector<rvsdg::output *>>
DNEGammaNodeHandler::ComputeMarkPhaseContinuations(const rvsdg::output & output) const
{
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    return std::vector{ gammaNode->MapBranchArgumentEntryVar(output).input->origin() };
  }

  if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    std::vector continuations({ gammaNode->predicate()->origin() });
    for (const auto & result : gammaNode->MapOutputExitVar(output).branchResult)
    {
      continuations.push_back(result->origin());
    }
    return continuations;
  }

  return std::nullopt;
}

void
DNEGammaNodeHandler::SweepNodeEntry(
    rvsdg::StructuralNode & structuralNode,
    const DNEContext & context) const
{
  auto & gammaNode = *util::AssertedCast<rvsdg::GammaNode>(&structuralNode);

  // Remove dead arguments and inputs
  for (size_t n = gammaNode.ninputs() - 1; n >= 1; n--)
  {
    auto input = gammaNode.input(n);

    bool alive = false;
    for (auto & argument : input->arguments)
    {
      if (context.IsAlive(argument))
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
DNEGammaNodeHandler::SweepNodeExit(
    rvsdg::StructuralNode & structuralNode,
    const DNEContext & context) const
{
  auto & gammaNode = *util::AssertedCast<rvsdg::GammaNode>(&structuralNode);

  // Remove dead outputs and results
  for (size_t n = gammaNode.noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (context.IsAlive(*gammaNode.output(n)))
    {
      continue;
    }

    for (size_t r = 0; r < gammaNode.nsubregions(); r++)
    {
      gammaNode.subregion(r)->RemoveResult(n);
    }
    gammaNode.RemoveOutput(n);
  }
}

DNEStructuralNodeHandler *
DNEGammaNodeHandler::GetInstance()
{
  static DNEGammaNodeHandler singleton;
  return &singleton;
}

/** \brief Dead Node Elimination statistics class
 *
 */
class DeadNodeElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::DeadNodeElimination, sourceFile)
  {}

  void
  StartMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  StopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  StartSweepStatistics() noexcept
  {
    AddTimer(SweepTimerLabel_).start();
  }

  void
  StopSweepStatistics(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(SweepTimerLabel_).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

DeadNodeElimination::~DeadNodeElimination() noexcept = default;

DeadNodeElimination::DeadNodeElimination(
    const std::vector<const DNEStructuralNodeHandler *> & handlers)
{
  for (const auto handler : handlers)
  {
    JLM_ASSERT(Handlers_.find(handler->GetTypeInfo()) == Handlers_.end());
    Handlers_[handler->GetTypeInfo()] = handler;
  }
}

void
DeadNodeElimination::run(rvsdg::Region & region)
{
  Context_ = DNEContext{};

  MarkRegion(region);
  SweepRegion(region);
}

void
DeadNodeElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto & rvsdg = module.Rvsdg();

  Context_ = DNEContext{};
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->StartMarkStatistics(rvsdg);
  MarkRegion(rvsdg.GetRootRegion());
  statistics->StopMarkStatistics();

  statistics->StartSweepStatistics();
  SweepRvsdg(rvsdg);
  statistics->StopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
DeadNodeElimination::MarkRegion(const rvsdg::Region & region)
{
  for (size_t n = 0; n < region.nresults(); n++)
  {
    MarkOutput(*region.result(n)->origin());
  }
}

void
DeadNodeElimination::MarkOutput(const jlm::rvsdg::output & output)
{
  if (Context_.IsAlive(output))
  {
    return;
  }

  Context_.MarkAlive(output);

  if (is<rvsdg::GraphImport>(&output))
  {
    return;
  }

  for (const auto [_, handler] : Handlers_)
  {
    if (const auto continuations = handler->ComputeMarkPhaseContinuations(output))
    {
      for (const auto & continuation : continuations.value())
      {
        MarkOutput(*continuation);
      }
      return;
    }
  }

  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    auto loopvar = theta->MapOutputLoopVar(output);
    MarkOutput(*theta->predicate()->origin());
    MarkOutput(*loopvar.post->origin());
    MarkOutput(*loopvar.input->origin());
    return;
  }

  if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    auto loopvar = theta->MapPreLoopVar(output);
    MarkOutput(*loopvar.output);
    MarkOutput(*loopvar.input->origin());
    return;
  }

  if (auto lambda = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(output))
  {
    for (auto & result : lambda->GetFunctionResults())
    {
      MarkOutput(*result->origin());
    }
    return;
  }

  if (auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    if (auto ctxvar = lambda->MapBinderContextVar(output))
    {
      // Bound context variable.
      MarkOutput(*ctxvar->input->origin());
      return;
    }
    else
    {
      // Function argument.
      return;
    }
  }

  if (const auto phiNode = rvsdg::TryGetOwnerNode<phi::node>(output))
  {
    const auto result = phiNode->subregion()->result(output.index());
    MarkOutput(*result->origin());
    return;
  }

  if (const auto phiNode = rvsdg::TryGetRegionParentNode<phi::node>(output))
  {
    const auto argument = util::AssertedCast<const rvsdg::RegionArgument>(&output);
    if (argument->input())
    {
      // Bound context variable
      MarkOutput(*argument->input()->origin());
      return;
    }

    // Recursion argument
    const auto result = phiNode->subregion()->result(argument->index());
    MarkOutput(*result->origin());
    return;
  }

  if (const auto deltaNode = rvsdg::TryGetOwnerNode<delta::node>(output))
  {
    const auto result = deltaNode->subregion()->result(0);
    MarkOutput(*result->origin());
    return;
  }

  if (rvsdg::TryGetRegionParentNode<delta::node>(output))
  {
    const auto argument = util::AssertedCast<const rvsdg::RegionArgument>(&output);
    MarkOutput(*argument->input()->origin());
    return;
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    for (size_t n = 0; n < simpleNode->ninputs(); n++)
    {
      MarkOutput(*simpleNode->input(n)->origin());
    }
    return;
  }

  JLM_UNREACHABLE("We should have never reached this statement.");
}

void
DeadNodeElimination::SweepRvsdg(rvsdg::Graph & rvsdg) const
{
  SweepRegion(rvsdg.GetRootRegion());

  // Remove dead imports
  for (size_t n = rvsdg.GetRootRegion().narguments() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (!Context_.IsAlive(*rvsdg.GetRootRegion().argument(n)))
    {
      rvsdg.GetRootRegion().RemoveArgument(n);
    }
  }
}

void
DeadNodeElimination::SweepRegion(rvsdg::Region & region) const
{
  region.prune(false);

  std::vector<std::vector<rvsdg::Node *>> nodesTopDown(region.nnodes());
  for (auto & node : region.Nodes())
  {
    nodesTopDown[node.depth()].push_back(&node);
  }

  for (auto it = nodesTopDown.rbegin(); it != nodesTopDown.rend(); it++)
  {
    for (auto node : *it)
    {
      if (!Context_.IsAlive(*node))
      {
        remove(node);
        continue;
      }

      if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
      {
        SweepStructuralNode(*structuralNode);
      }
    }
  }

  JLM_ASSERT(region.NumBottomNodes() == 0);
}

void
DeadNodeElimination::SweepStructuralNode(rvsdg::StructuralNode & node) const
{
  if (const auto it = Handlers_.find(typeid(node)); it != Handlers_.end())
  {
    const auto handler = it->second;
    handler->SweepNodeExit(node, Context_);

    for (size_t r = 0; r < node.nsubregions(); r++)
    {
      SweepRegion(*node.subregion(r));
    }

    handler->SweepNodeEntry(node, Context_);
    return;
  }

  auto sweepTheta = [](auto & d, auto & n)
  {
    d.SweepTheta(*util::AssertedCast<rvsdg::ThetaNode>(&n));
  };
  auto sweepLambda = [](auto & d, auto & n)
  {
    d.SweepLambda(*util::AssertedCast<rvsdg::LambdaNode>(&n));
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
      std::function<void(const DeadNodeElimination &, rvsdg::StructuralNode &)>>
      map({ { typeid(rvsdg::ThetaOperation), sweepTheta },
            { typeid(llvm::LlvmLambdaOperation), sweepLambda },
            { typeid(phi::operation), sweepPhi },
            { typeid(delta::operation), sweepDelta } });

  auto & op = node.GetOperation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](*this, node);
}

void
DeadNodeElimination::SweepTheta(rvsdg::ThetaNode & thetaNode) const
{
  // Determine loop variables to be removed.
  std::vector<rvsdg::ThetaNode::LoopVar> loopvars;
  for (const auto & loopvar : thetaNode.GetLoopVars())
  {
    if (!Context_.IsAlive(*loopvar.pre) && !Context_.IsAlive(*loopvar.output))
    {
      loopvar.post->divert_to(loopvar.pre);
      loopvars.push_back(loopvar);
    }
  }

  // Now that the loop variables to be eliminated only point to
  // their own pre-iteration values, any outputs within the subregion
  // that only contributed to computing the post-iteration values
  // of the variables are unlinked and can be removed as well.
  SweepRegion(*thetaNode.subregion());

  // There are now no other users of the pre-iteration values of the
  // variables to be removed left in the subregion anymore.
  // The variables have become "loop-invariant" and can simply
  // be eliminated from the theta node.
  thetaNode.RemoveLoopVars(std::move(loopvars));
}

void
DeadNodeElimination::SweepLambda(rvsdg::LambdaNode & lambdaNode) const
{
  SweepRegion(*lambdaNode.subregion());
  lambdaNode.PruneLambdaInputs();
}

void
DeadNodeElimination::SweepPhi(phi::node & phiNode) const
{
  util::HashSet<const rvsdg::RegionArgument *> deadRecursionArguments;

  auto isDeadOutput = [&](const phi::rvoutput & output)
  {
    auto argument = output.argument();

    // A recursion variable is only dead iff its output AND argument are dead
    auto isDead = !Context_.IsAlive(output) && !Context_.IsAlive(*argument);
    if (isDead)
    {
      deadRecursionArguments.Insert(argument);
    }

    return isDead;
  };
  phiNode.RemovePhiOutputsWhere(isDeadOutput);

  SweepRegion(*phiNode.subregion());

  auto isDeadArgument = [&](const rvsdg::RegionArgument & argument)
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
