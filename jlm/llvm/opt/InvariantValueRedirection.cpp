/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/gamma.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class InvariantValueRedirectionStatistics final : public Statistics {
public:
  ~InvariantValueRedirectionStatistics() override
  = default;

  InvariantValueRedirectionStatistics()
    : Statistics(Statistics::Id::InvariantValueRedirection)
  {}

  void
  Start(const jive::graph & graph) noexcept
  {
    Timer_.start();
  }

  void
  Stop(const jive::graph & graph) noexcept
  {
    Timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("InvariantValueRedirection ",
                  "Time[ns]:", Timer_.ns()
    );
  }

  static std::unique_ptr<InvariantValueRedirectionStatistics>
  Create()
  {
    return std::make_unique<InvariantValueRedirectionStatistics>();
  }

private:
  jlm::timer Timer_;
};

InvariantValueRedirection::~InvariantValueRedirection()
= default;

void
InvariantValueRedirection::run(
  RvsdgModule & rvsdgModule,
  StatisticsCollector & statisticsCollector)
{
  auto & rvsdg = rvsdgModule.Rvsdg();
  auto statistics = InvariantValueRedirectionStatistics::Create();

  statistics->Start(rvsdg);
  RedirectInvariantValues(*rvsdg.root());
  statistics->Stop(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
InvariantValueRedirection::RedirectInvariantValues(jive::region & region)
{
  for (auto node : jive::topdown_traverser(&region)) {
    if (jive::is<jive::simple_op>(node))
      continue;

    auto & structuralNode = *AssertedCast<jive::structural_node>(node);
    for (size_t n = 0; n < structuralNode.nsubregions(); n++)
      RedirectInvariantValues(*structuralNode.subregion(n));

    if (auto gammaNode = dynamic_cast<jive::gamma_node*>(&structuralNode)) {
      RedirectInvariantGammaOutputs(*gammaNode);
      continue;
    }

    if (auto thetaNode = dynamic_cast<jive::theta_node*>(&structuralNode)) {
      RedirectInvariantThetaOutputs(*thetaNode);
      continue;
    }
  }
}

void
InvariantValueRedirection::RedirectInvariantGammaOutputs(jive::gamma_node & gammaNode)
{
  for (auto it = gammaNode.begin_exitvar(); it != gammaNode.end_exitvar(); it++) {
    auto & gammaOutput = *it;

    if (auto origin = is_invariant(&gammaOutput))
      it->divert_users(origin);
  }
}

void
InvariantValueRedirection::RedirectInvariantThetaOutputs(jive::theta_node & thetaNode)
{
  for (const auto & thetaOutput : thetaNode) {
    /* FIXME: In order to also redirect loop state type variables, we need to know whether a loop terminates.*/
    if (jive::is<loopstatetype>(thetaOutput->type()))
      continue;

    if (jive::is_invariant(thetaOutput))
      thetaOutput->divert_users(thetaOutput->input()->origin());
  }
}

}
