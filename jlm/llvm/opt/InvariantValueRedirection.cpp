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

namespace jlm::llvm
{

class InvariantValueRedirectionStatistics final : public util::Statistics
{
public:
  ~InvariantValueRedirectionStatistics() override = default;

  InvariantValueRedirectionStatistics()
      : Statistics(Statistics::Id::InvariantValueRedirection)
  {}

  void
  Start(const jlm::rvsdg::graph & graph) noexcept
  {
    Timer_.start();
  }

  void
  Stop(const jlm::rvsdg::graph & graph) noexcept
  {
    Timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return util::strfmt("InvariantValueRedirection ", "Time[ns]:", Timer_.ns());
  }

  static std::unique_ptr<InvariantValueRedirectionStatistics>
  Create()
  {
    return std::make_unique<InvariantValueRedirectionStatistics>();
  }

private:
  util::timer Timer_;
};

InvariantValueRedirection::~InvariantValueRedirection() = default;

void
InvariantValueRedirection::run(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto & rvsdg = rvsdgModule.Rvsdg();
  auto statistics = InvariantValueRedirectionStatistics::Create();

  statistics->Start(rvsdg);
  RedirectInvariantValues(*rvsdg.root());
  statistics->Stop(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
InvariantValueRedirection::RedirectInvariantValues(jlm::rvsdg::region & region)
{
  for (auto node : jlm::rvsdg::topdown_traverser(&region))
  {
    if (jlm::rvsdg::is<jlm::rvsdg::simple_op>(node))
      continue;

    auto & structuralNode = *util::AssertedCast<jlm::rvsdg::structural_node>(node);
    for (size_t n = 0; n < structuralNode.nsubregions(); n++)
      RedirectInvariantValues(*structuralNode.subregion(n));

    if (auto gammaNode = dynamic_cast<jlm::rvsdg::gamma_node *>(&structuralNode))
    {
      RedirectInvariantGammaOutputs(*gammaNode);
      continue;
    }

    if (auto thetaNode = dynamic_cast<jlm::rvsdg::theta_node *>(&structuralNode))
    {
      RedirectInvariantThetaOutputs(*thetaNode);
      continue;
    }
  }
}

void
InvariantValueRedirection::RedirectInvariantGammaOutputs(jlm::rvsdg::gamma_node & gammaNode)
{
  for (auto it = gammaNode.begin_exitvar(); it != gammaNode.end_exitvar(); it++)
  {
    auto & gammaOutput = *it;

    if (auto origin = is_invariant(&gammaOutput))
      it->divert_users(origin);
  }
}

void
InvariantValueRedirection::RedirectInvariantThetaOutputs(jlm::rvsdg::theta_node & thetaNode)
{
  for (const auto & thetaOutput : thetaNode)
  {
    /* FIXME: In order to also redirect loop state type variables, we need to know whether a loop
     * terminates.*/
    if (jlm::rvsdg::is<loopstatetype>(thetaOutput->type()))
      continue;

    if (jlm::rvsdg::is_invariant(thetaOutput))
      thetaOutput->divert_users(thetaOutput->input()->origin());
  }
}

}
