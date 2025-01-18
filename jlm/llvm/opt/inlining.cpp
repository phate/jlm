/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class ilnstat final : public util::Statistics
{
public:
  ~ilnstat() override = default;

  explicit ilnstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::FunctionInlining, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph)
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  stop(const rvsdg::Graph & graph)
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<ilnstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<ilnstat>(sourceFile);
  }
};

jlm::rvsdg::output *
find_producer(jlm::rvsdg::input * input)
{
  auto graph = input->region()->graph();

  auto argument = dynamic_cast<rvsdg::RegionArgument *>(input->origin());
  if (argument == nullptr)
    return input->origin();

  if (argument->region() == &graph->GetRootRegion())
    return argument;

  JLM_ASSERT(argument->input() != nullptr);
  return find_producer(argument->input());
}

static jlm::rvsdg::output *
route_to_region(jlm::rvsdg::output * output, rvsdg::Region * region)
{
  JLM_ASSERT(region != nullptr);

  if (region == output->region())
    return output;

  output = route_to_region(output, region->node()->region());

  if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(region->node()))
  {
    gamma->AddEntryVar(output);
    output = region->argument(region->narguments() - 1);
  }
  else if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(region->node()))
  {
    output = theta->AddLoopVar(output).pre;
  }
  else if (auto lambda = dynamic_cast<lambda::node *>(region->node()))
  {
    output = lambda->AddContextVar(*output).inner;
  }
  else if (auto phi = dynamic_cast<phi::node *>(region->node()))
  {
    output = phi->add_ctxvar(output);
  }
  else
  {
    JLM_UNREACHABLE("This should have never happened!");
  }

  return output;
}

static std::vector<jlm::rvsdg::output *>
route_dependencies(const lambda::node * lambda, const jlm::rvsdg::SimpleNode * apply)
{
  JLM_ASSERT(is<CallOperation>(apply));

  /* collect origins of dependencies */
  std::vector<jlm::rvsdg::output *> deps;
  for (size_t n = 0; n < lambda->ninputs(); n++)
    deps.push_back(find_producer(lambda->input(n)));

  /* route dependencies to apply region */
  for (size_t n = 0; n < deps.size(); n++)
    deps[n] = route_to_region(deps[n], apply->region());

  return deps;
}

void
inlineCall(jlm::rvsdg::SimpleNode * call, const lambda::node * lambda)
{
  JLM_ASSERT(is<CallOperation>(call));

  auto deps = route_dependencies(lambda, call);
  auto ctxvars = lambda->GetContextVars();
  JLM_ASSERT(ctxvars.size() == deps.size());

  rvsdg::SubstitutionMap smap;
  auto args = lambda->GetFunctionArguments();
  for (size_t n = 1; n < call->ninputs(); n++)
  {
    auto argument = args[n - 1];
    smap.insert(argument, call->input(n)->origin());
  }
  for (size_t n = 0; n < ctxvars.size(); n++)
    smap.insert(ctxvars[n].inner, deps[n]);

  lambda->subregion()->copy(call->region(), smap, false, false);

  for (size_t n = 0; n < call->noutputs(); n++)
  {
    auto output = lambda->subregion()->result(n)->origin();
    JLM_ASSERT(smap.lookup(output));
    call->output(n)->divert_users(smap.lookup(output));
  }
  remove(call);
}

static void
inlining(rvsdg::Graph & rvsdg)
{
  for (auto node : rvsdg::topdown_traverser(&rvsdg.GetRootRegion()))
  {
    if (auto lambda = dynamic_cast<const lambda::node *>(node))
    {
      auto callSummary = jlm::llvm::ComputeCallSummary(*lambda);

      if (callSummary.HasOnlyDirectCalls() && callSummary.NumDirectCalls() == 1)
      {
        inlineCall(*callSummary.DirectCalls().begin(), lambda);
      }
    }
  }
}

static void
inlining(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto & graph = rvsdgModule.Rvsdg();
  auto statistics = ilnstat::Create(rvsdgModule.SourceFilePath().value());

  statistics->start(graph);
  inlining(graph);
  statistics->stop(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* fctinline class */

fctinline::~fctinline()
{}

void
fctinline::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  inlining(module, statisticsCollector);
}

}
