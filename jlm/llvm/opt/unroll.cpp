/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class LoopUnrolling::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::LoopUnrolling, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

/* helper functions */

static bool
is_eqcmp(const rvsdg::Operation & op)
{
  return dynamic_cast<const IntegerUgeOperation *>(&op)
      || dynamic_cast<const IntegerSgeOperation *>(&op)
      || dynamic_cast<const IntegerUleOperation *>(&op)
      || dynamic_cast<const IntegerSleOperation *>(&op);
}

static bool
is_theta_invariant(const jlm::rvsdg::Output * output)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(output->region()->node()));

  if (jlm::rvsdg::is<IntegerConstantOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*output)))
    return true;

  auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*output);
  if (!theta)
    return false;

  auto loopVar = theta->MapPreLoopVar(*output);
  return ThetaLoopVarIsInvariant(loopVar);
}

static rvsdg::Output *
push_from_theta(jlm::rvsdg::Output * output)
{
  auto argument = dynamic_cast<rvsdg::RegionArgument *>(output);
  if (argument)
    return argument;

  auto tmp = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*output);
  JLM_ASSERT(jlm::rvsdg::is<IntegerConstantOperation>(tmp));
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(tmp->region()->node()));
  auto theta = static_cast<rvsdg::ThetaNode *>(tmp->region()->node());

  auto node = tmp->copy(theta->region(), {});
  auto lv = theta->AddLoopVar(node->output(0));
  output->divert_users(lv.pre);

  return lv.pre;
}

static bool
is_idv(jlm::rvsdg::Input * input)
{
  using namespace jlm::rvsdg;

  auto node = TryGetOwnerNode<SimpleNode>(*input);
  JLM_ASSERT(is<IntegerAddOperation>(node) || is<IntegerSubOperation>(node));

  if (auto theta = rvsdg::TryGetRegionParentNode<ThetaNode>(*input->origin()))
  {
    auto loopvar = theta->MapPreLoopVar(*input->origin());
    return rvsdg::TryGetOwnerNode<Node>(*loopvar.post->origin()) == node;
  }

  return false;
}

std::unique_ptr<IntegerValueRepresentation>
LoopUnrollInfo::niterations() const noexcept
{
  if (!is_known() || *step_value() == 0)
    return nullptr;

  auto start = is_additive() ? *init_value() : *end_value();
  auto step = is_additive() ? *step_value() : step_value()->neg();
  auto end = is_additive() ? *end_value() : *init_value();

  if (is_eqcmp(cmpnode()->GetOperation()))
    end = end.add({ nbits(), 1 });

  auto range = end.sub(start);
  if (range.is_negative())
    return nullptr;

  if (range.umod(step) != 0)
    return nullptr;

  return std::make_unique<IntegerValueRepresentation>(range.udiv(step));
}

static bool
is_integer_comparison(const rvsdg::SimpleNode * node)
{
  using namespace jlm::rvsdg;
  return is<IntegerEqOperation>(node) || is<IntegerNeOperation>(node)
      || is<IntegerSgeOperation>(node) || is<IntegerSgtOperation>(node)
      || is<IntegerSleOperation>(node) || is<IntegerSltOperation>(node)
      || is<IntegerUgeOperation>(node) || is<IntegerUgtOperation>(node)
      || is<IntegerUleOperation>(node) || is<IntegerUltOperation>(node);
}

size_t
LoopUnrollInfo::nbits() const noexcept
{
  JLM_ASSERT(is_integer_comparison(cmpnode()));
  return static_cast<const IntegerBinaryOperation &>(cmpnode()->GetOperation()).Type().nbits();
}

std::unique_ptr<LoopUnrollInfo>
LoopUnrollInfo::create(rvsdg::ThetaNode * theta)
{
  using namespace jlm::rvsdg;

  const auto matchNode = rvsdg::TryGetOwnerNode<SimpleNode>(*theta->predicate()->origin());
  if (!is<MatchOperation>(matchNode))
    return nullptr;

  auto cmpnode = rvsdg::TryGetOwnerNode<SimpleNode>(*matchNode->input(0)->origin());
  if (!is_integer_comparison(cmpnode))
    return nullptr;

  auto o0 = cmpnode->input(0)->origin();
  auto o1 = cmpnode->input(1)->origin();
  auto end = is_theta_invariant(o0) ? o0 : (is_theta_invariant(o1) ? o1 : nullptr);
  if (!end)
    return nullptr;

  auto armnode = rvsdg::TryGetOwnerNode<SimpleNode>(*(end == o0 ? o1 : o0));
  if (!is<IntegerAddOperation>(armnode) && !is<IntegerSubOperation>(armnode))
    return nullptr;
  if (armnode->ninputs() != 2)
    return nullptr;

  auto i0 = armnode->input(0);
  auto i1 = armnode->input(1);
  if (!is_idv(i0) && !is_idv(i1))
    return nullptr;

  auto idv = static_cast<rvsdg::RegionArgument *>(is_idv(i0) ? i0->origin() : i1->origin());

  auto step = idv == i0->origin() ? i1->origin() : i0->origin();
  if (!is_theta_invariant(step))
    return nullptr;

  auto endarg = push_from_theta(end);
  auto steparg = push_from_theta(step);
  return std::unique_ptr<LoopUnrollInfo>(
      new LoopUnrollInfo(cmpnode, armnode, idv, steparg, endarg));
}

/* loop unrolling */

static void
unroll_body(
    const rvsdg::ThetaNode * theta,
    rvsdg::Region * target,
    rvsdg::SubstitutionMap & smap,
    size_t factor)
{
  for (size_t n = 0; n < factor - 1; n++)
  {
    theta->subregion()->copy(target, smap);
    rvsdg::SubstitutionMap tmap;
    for (const auto & olv : theta->GetLoopVars())
      tmap.insert(olv.pre, &smap.lookup(*olv.post->origin()));
    smap = tmap;
  }
  theta->subregion()->copy(target, smap);
}

/*
  Copy the body of the theta and unroll it factor number of times.
  The unrolled body has the same inputs and outputs as the theta.
  The theta itself is not deleted.
*/
static void
copy_body_and_unroll(const rvsdg::ThetaNode * theta, size_t factor)
{
  rvsdg::SubstitutionMap smap;
  for (const auto & olv : theta->GetLoopVars())
    smap.insert(olv.pre, olv.input->origin());

  unroll_body(theta, theta->region(), smap, factor);

  for (const auto & olv : theta->GetLoopVars())
    olv.output->divert_users(&smap.lookup(*olv.post->origin()));
}

/*
  Unroll theta node by given factor.
*/
static void
unroll_theta(const LoopUnrollInfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);
  auto unrolled_theta = rvsdg::ThetaNode::create(theta->region());

  auto oldLoopVars = theta->GetLoopVars();
  for (const auto & olv : oldLoopVars)
  {
    auto nlv = unrolled_theta->AddLoopVar(olv.input->origin());
    smap.insert(olv.pre, nlv.pre);
  }

  unroll_body(theta, unrolled_theta->subregion(), smap, factor);
  unrolled_theta->set_predicate(&smap.lookup(*theta->predicate()->origin()));

  auto newLoopVars = unrolled_theta->GetLoopVars();
  for (size_t i = 0; i < oldLoopVars.size(); ++i)
  {
    const auto & olv = oldLoopVars[i];
    const auto & nlv = newLoopVars[i];
    auto origin = &smap.lookup(*olv.post->origin());
    nlv.post->divert_to(origin);
    smap.insert(olv.output, nlv.output);
  }

  if (remainder != 0)
  {
    /*
      We have residual iterations. Adjust the end value of the unrolled loop
      to a multiple of the step value.
    */
    auto cmpnode = ui.cmpnode();
    auto cmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(smap.lookup(*cmpnode->output(0)));
    auto input = cmp->input(0)->origin() == &smap.lookup(*ui.end()) ? cmp->input(0) : cmp->input(1);
    JLM_ASSERT(input->origin() == &smap.lookup(*ui.end()));

    auto sv = ui.is_additive() ? *ui.step_value() : ui.step_value()->neg();
    auto end = remainder.mul(sv);
    auto ev = ui.is_additive() ? ui.end_value()->sub(end) : ui.end_value()->add(end);

    auto c = IntegerConstantOperation::Create(*unrolled_theta->subregion(), ev).output(0);
    input->divert_to(c);
  }
}

/*
  Adde the reminder for the lopp if any
*/
static void
add_remainder(const LoopUnrollInfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);

  if (remainder == 0)
  {
    /*
      We only need to redirect the users of the outputs of the old theta node
      to the outputs of the new theta node, as there are no residual iterations.
    */
    for (const auto & olv : theta->GetLoopVars())
      olv.output->divert_users(&smap.lookup(*olv.output));
    return remove(theta);
  }

  /*
    Add the old theta node as epilogue after the unrolled loop by simply
    redirecting the inputs of the old theta to the outputs of the unrolled
    theta.
  */
  for (const auto & olv : theta->GetLoopVars())
    olv.input->divert_to(&smap.lookup(*olv.output));

  if (remainder == 1)
  {
    /*
      There is only one loop iteration remaining.
      Simply copy the body of the theta to replace it.
    */
    copy_body_and_unroll(theta, 1);
    remove(theta);
  }
}

static void
unroll_known_theta(const LoopUnrollInfo & ui, size_t factor)
{
  JLM_ASSERT(ui.is_known() && ui.niterations());
  auto niterations = ui.niterations();
  auto original_theta = ui.theta();
  auto nbits = ui.nbits();

  // A theta always executes at least once; zero is not a valid trip count.
  if (niterations->to_uint() == 0)
    return;
  if (niterations->ule({ nbits, (int64_t)factor }) == '1')
  {
    /*
      Completely unroll the loop body and then remove the theta node,
      as the number of iterations is smaller than the unroll factor.
    */
    copy_body_and_unroll(original_theta, niterations->to_uint());
    return remove(original_theta);
  }

  JLM_ASSERT(niterations->ugt({ nbits, (int64_t)factor }) == '1');

  /*
    Unroll the theta
  */
  rvsdg::SubstitutionMap smap;
  unroll_theta(ui, smap, factor);

  /*
    Add code for any potential iterations that remains
  */
  add_remainder(ui, smap, factor);
}

static rvsdg::Output *
advance_idv(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output * value,
    rvsdg::Output * step)
{
  auto first = ui.armnode()->input(0)->origin() == ui.idv();
  return rvsdg::SimpleNode::Create(
             region,
             ui.armoperation().copy(),
             first ? std::vector<rvsdg::Output *>{ value, step }
                   : std::vector<rvsdg::Output *>{ step, value })
      .output(0);
}

static rvsdg::Output *
loop_comparison(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output * value,
    rvsdg::Output * end)
{
  auto first = ui.cmpnode()->input(0)->origin() == ui.end();
  return rvsdg::SimpleNode::Create(
             region,
             ui.cmpoperation().copy(),
             first ? std::vector<rvsdg::Output *>{ end, value }
                   : std::vector<rvsdg::Output *>{ value, end })
      .output(0);
}

static rvsdg::Output *
batch_guard(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output * value,
    rvsdg::Output * step,
    rvsdg::Output * end,
    size_t count)
{
  // Check each original comparison, preserving wrapping and operand order.
  JLM_ASSERT(count >= 1);
  rvsdg::Output * guard = nullptr;
  for (size_t i = 0; i < count; ++i)
  {
    auto cmp = loop_comparison(ui, region, value, end);
    guard = guard ? IntegerAndOperation::createNode(1, *guard, *cmp).output(0) : cmp;
    if (i + 1 < count)
      value = advance_idv(ui, region, value, step);
  }
  return guard;
}

static rvsdg::Output *
predicate(rvsdg::Output * condition)
{
  return rvsdg::MatchOperation::CreateNode(*condition, { { 1, 1 } }, 0, 2).output(0);
}

static rvsdg::Output *
first_batch_guard(const LoopUnrollInfo & ui, size_t factor)
{
  auto region = ui.theta()->region();
  auto step = ui.theta()->MapPreLoopVar(*ui.step()).input->origin();
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  // The first iteration is unconditional; only the remaining copies need checks.
  auto value = advance_idv(ui, *region, ui.init(), step);
  return batch_guard(ui, *region, value, step, end, factor - 1);
}

static rvsdg::Output *
next_batch_guard(const rvsdg::SubstitutionMap & smap, const LoopUnrollInfo & ui, size_t factor)
{
  auto cmp = &smap.lookup(*ui.cmpnode()->output(0));
  auto region = cmp->region();
  auto cmpnode = rvsdg::TryGetOwnerNode<rvsdg::Node>(*cmp);
  auto step = &smap.lookup(*ui.step());
  auto end = &smap.lookup(*ui.end());
  auto idv = cmpnode->input(cmpnode->input(0)->origin() == end ? 1 : 0)->origin();
  return batch_guard(ui, *region, idv, step, end, factor);
}

static rvsdg::Output *
residual_guard(
    const rvsdg::SubstitutionMap & smap,
    const LoopUnrollInfo & ui,
    rvsdg::Output * firstGuard)
{
  auto region = ui.theta()->region();
  auto idv = &smap.lookup(*ui.theta()->MapPreLoopVar(*ui.idv()).output);
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  auto cmp = loop_comparison(ui, *region, idv, end);
  // A declined first batch must still execute the original do-while loop.
  auto one = IntegerConstantOperation::Create(*region, 1, 1).output(0);
  auto declined = IntegerXorOperation::createNode(1, *firstGuard, *one).output(0);
  return IntegerOrOperation::createNode(1, *cmp, *declined).output(0);
}

static void
unroll_unknown_theta(const LoopUnrollInfo & ui, size_t factor)
{
  auto otheta = ui.theta();
  auto firstGuard = first_batch_guard(ui, factor);

  /* handle gamma with unrolled loop */
  rvsdg::SubstitutionMap smap;
  {
    auto pred = predicate(firstGuard);
    auto ngamma = rvsdg::GammaNode::create(pred, 2);
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto ev = ngamma->AddEntryVar(olv.input->origin());
      auto nlv = ntheta->AddLoopVar(ev.branchArgument[1]);
      rmap[0].insert(olv.output, ev.branchArgument[0]);
      rmap[1].insert(olv.pre, nlv.pre);
    }

    unroll_body(otheta, ntheta->subregion(), rmap[1], factor);
    pred = predicate(next_batch_guard(rmap[1], ui, factor));
    ntheta->set_predicate(pred);

    auto oldLoopVars = otheta->GetLoopVars();
    auto newLoopVars = ntheta->GetLoopVars();
    for (std::size_t n = 0; n < oldLoopVars.size(); ++n)
    {
      auto & olv = oldLoopVars[n];
      auto & nlv = newLoopVars[n];
      auto origin = &rmap[1].lookup(*olv.post->origin());
      nlv.post->divert_to(origin);
      rmap[1].insert(olv.output, nlv.output);
    }

    for (const auto & olv : oldLoopVars)
    {
      auto xv =
          ngamma->AddExitVar({ &rmap[0].lookup(*olv.output), &rmap[1].lookup(*olv.output) }).output;
      smap.insert(olv.output, xv);
    }
  }

  /* handle gamma for residual iterations */
  {
    auto pred = predicate(residual_guard(smap, ui, firstGuard));
    auto ngamma = rvsdg::GammaNode::create(pred, 2);
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    auto oldLoopVars = otheta->GetLoopVars();
    for (const auto & olv : oldLoopVars)
    {
      auto ev = ngamma->AddEntryVar(&smap.lookup(*olv.output));
      auto nlv = ntheta->AddLoopVar(ev.branchArgument[1]);
      rmap[0].insert(olv.output, ev.branchArgument[0]);
      rmap[1].insert(olv.pre, nlv.pre);
    }

    otheta->subregion()->copy(ntheta->subregion(), rmap[1]);
    ntheta->set_predicate(&rmap[1].lookup(*otheta->predicate()->origin()));

    auto newLoopVars = ntheta->GetLoopVars();

    for (std::size_t n = 0; n < oldLoopVars.size(); ++n)
    {
      auto & olv = oldLoopVars[n];
      auto & nlv = newLoopVars[n];
      auto origin = &rmap[1].lookup(*olv.post->origin());
      nlv.post->divert_to(origin);
      auto xv = ngamma->AddExitVar({ &rmap[0].lookup(*olv.output), nlv.output }).output;
      smap.insert(olv.output, xv);
    }
  }
  for (const auto & olv : otheta->GetLoopVars())
    olv.output->divert_users(&smap.lookup(*olv.output));
  remove(otheta);
}

void
unroll(rvsdg::ThetaNode * otheta, size_t factor)
{
  if (factor < 2)
    return;

  auto ui = LoopUnrollInfo::create(otheta);
  if (!ui)
    return;

  if (ui->is_known() && ui->niterations())
    unroll_known_theta(*ui, factor);
  else
    unroll_unknown_theta(*ui, factor);
}

static bool
unroll(rvsdg::Region * region, size_t factor)
{
  bool unrolled = false;
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        unrolled = unroll(structnode->subregion(n), factor);

      /* Try to unroll if an inner loop hasn't already been found */
      if (!unrolled)
      {
        if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(node))
        {
          unroll(theta, factor);
          unrolled = true;
        }
      }
    }
  }
  return unrolled;
}

LoopUnrolling::~LoopUnrolling() noexcept = default;

void
LoopUnrolling::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  if (factor_ < 2)
    return;

  auto & graph = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->start(module.Rvsdg());
  unroll(&graph.GetRootRegion(), factor_);
  statistics->end(module.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}
