/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class redstat final : public util::Statistics
{
public:
  ~redstat() override = default;

  explicit redstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::ReduceNodes, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<redstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<redstat>(sourceFile);
  }
};

static void
enable_store_reductions(rvsdg::Graph & graph)
{
  auto nf = StoreNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(true);
  nf->set_store_mux_reducible(true);
  nf->set_store_store_reducible(true);
  nf->set_store_alloca_reducible(true);
  nf->set_multiple_origin_reducible(true);
}

static void
enable_load_reductions(rvsdg::Graph & graph)
{
  auto nf = LoadNonVolatileOperation::GetNormalForm(&graph);
  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  nf->set_load_store_reducible(true);
  nf->set_load_alloca_reducible(true);
  nf->set_multiple_origin_reducible(true);
  nf->set_load_store_state_reducible(true);
  nf->set_load_load_state_reducible(true);
}

static void
enable_gamma_reductions(rvsdg::Graph & graph)
{
  auto nf = rvsdg::GammaOperation::normal_form(&graph);
  nf->set_mutable(true);
  nf->set_predicate_reduction(true);
  // set_control_constante_reduction cause a PHI node input type error
  // github issue #303
  nf->set_control_constant_reduction(false);
}

static void
enable_unary_reductions(rvsdg::Graph & graph)
{
  auto nf = jlm::rvsdg::unary_op::normal_form(&graph);
  // set_mutable generates incorrect output for a number of
  // llvm suite tests when used in combination with other
  // optimizations than the set_reducible
  nf->set_mutable(false);
  // set_reducible generates incorrect output for 18 llvm suite tests
  // github issue #304
  nf->set_reducible(false);
}

static void
enable_binary_reductions(rvsdg::Graph & graph)
{
  auto nf = jlm::rvsdg::binary_op::normal_form(&graph);
  nf->set_mutable(true);
  nf->set_reducible(true);
}

static void
reduce(RvsdgModule & rm, util::StatisticsCollector & statisticsCollector)
{
  auto & graph = rm.Rvsdg();
  auto statistics = redstat::Create(rm.SourceFileName());

  statistics->start(graph);

  enable_store_reductions(graph);
  enable_load_reductions(graph);
  enable_gamma_reductions(graph);
  enable_unary_reductions(graph);
  enable_binary_reductions(graph);

  graph.Normalize();
  statistics->end(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* nodereduction class */

nodereduction::~nodereduction()
{}

void
nodereduction::run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  reduce(module, statisticsCollector);
}

}
