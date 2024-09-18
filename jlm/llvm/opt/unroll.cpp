/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class unrollstat final : public util::Statistics
{
public:
  ~unrollstat() override = default;

  explicit unrollstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::LoopUnrolling, sourceFile)
  {}

  void
  start(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(graph.root()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(graph.root()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<unrollstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<unrollstat>(sourceFile);
  }
};

/* helper functions */

static bool
is_eqcmp(const jlm::rvsdg::operation & op)
{
  return dynamic_cast<const jlm::rvsdg::bituge_op *>(&op)
      || dynamic_cast<const jlm::rvsdg::bitsge_op *>(&op)
      || dynamic_cast<const jlm::rvsdg::bitule_op *>(&op)
      || dynamic_cast<const jlm::rvsdg::bitsle_op *>(&op);
}

/* unrollinfo methods */

static bool
is_theta_invariant(const jlm::rvsdg::output * output)
{
  JLM_ASSERT(is<rvsdg::ThetaOperation>(output->region()->node()));

  if (jlm::rvsdg::is<jlm::rvsdg::bitconstant_op>(jlm::rvsdg::node_output::node(output)))
    return true;

  auto argument = dynamic_cast<const rvsdg::RegionArgument *>(output);
  if (!argument)
    return false;

  return is_invariant(static_cast<const rvsdg::ThetaInput *>(argument->input()));
}

static rvsdg::RegionArgument *
push_from_theta(jlm::rvsdg::output * output)
{
  auto argument = dynamic_cast<rvsdg::RegionArgument *>(output);
  if (argument)
    return argument;

  auto tmp = jlm::rvsdg::node_output::node(output);
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::bitconstant_op>(tmp));
  JLM_ASSERT(is<rvsdg::ThetaOperation>(tmp->region()->node()));
  auto theta = static_cast<rvsdg::ThetaNode *>(tmp->region()->node());

  auto node = tmp->copy(theta->region(), {});
  auto lv = theta->add_loopvar(node->output(0));
  output->divert_users(lv->argument());

  return lv->argument();
}

static bool
is_idv(jlm::rvsdg::input * input)
{
  using namespace jlm::rvsdg;

  auto node = rvsdg::input::GetNode(*input);
  JLM_ASSERT(is<bitadd_op>(node) || is<bitsub_op>(node));

  auto a = dynamic_cast<rvsdg::RegionArgument *>(input->origin());
  if (!a)
    return false;

  auto tinput = static_cast<const ThetaInput *>(a->input());
  return jlm::rvsdg::node_output::node(tinput->result()->origin()) == node;
}

std::unique_ptr<jlm::rvsdg::bitvalue_repr>
unrollinfo::niterations() const noexcept
{
  if (!is_known() || step_value() == 0)
    return nullptr;

  auto start = is_additive() ? *init_value() : *end_value();
  auto step = is_additive() ? *step_value() : step_value()->neg();
  auto end = is_additive() ? *end_value() : *init_value();

  if (is_eqcmp(cmpnode()->operation()))
    end = end.add({ nbits(), 1 });

  auto range = end.sub(start);
  if (range.is_negative())
    return nullptr;

  if (range.umod(step) != 0)
    return nullptr;

  return std::make_unique<jlm::rvsdg::bitvalue_repr>(range.udiv(step));
}

std::unique_ptr<unrollinfo>
unrollinfo::create(rvsdg::ThetaNode * theta)
{
  using namespace jlm::rvsdg;

  auto matchnode = jlm::rvsdg::node_output::node(theta->predicate()->origin());
  if (!is<match_op>(matchnode))
    return nullptr;

  auto cmpnode = jlm::rvsdg::node_output::node(matchnode->input(0)->origin());
  if (!is<bitcompare_op>(cmpnode))
    return nullptr;

  auto o0 = cmpnode->input(0)->origin();
  auto o1 = cmpnode->input(1)->origin();
  auto end = is_theta_invariant(o0) ? o0 : (is_theta_invariant(o1) ? o1 : nullptr);
  if (!end)
    return nullptr;

  auto armnode = jlm::rvsdg::node_output::node((end == o0 ? o1 : o0));
  if (!is<bitadd_op>(armnode) && !is<bitsub_op>(armnode))
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
  return std::unique_ptr<unrollinfo>(new unrollinfo(cmpnode, armnode, idv, steparg, endarg));
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
    theta->subregion()->copy(target, smap, false, false);
    rvsdg::SubstitutionMap tmap;
    for (const auto & olv : *theta)
      tmap.insert(olv->argument(), smap.lookup(olv->result()->origin()));
    smap = tmap;
  }
  theta->subregion()->copy(target, smap, false, false);
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
  for (const auto & olv : *theta)
    smap.insert(olv->argument(), olv->input()->origin());

  unroll_body(theta, theta->region(), smap, factor);

  for (const auto & olv : *theta)
    olv->divert_users(smap.lookup(olv->result()->origin()));
}

/*
  Unroll theta node by given factor.
*/
static void
unroll_theta(const unrollinfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);
  auto unrolled_theta = rvsdg::ThetaNode::create(theta->region());

  for (const auto & olv : *theta)
  {
    auto nlv = unrolled_theta->add_loopvar(olv->input()->origin());
    smap.insert(olv->argument(), nlv->argument());
  }

  unroll_body(theta, unrolled_theta->subregion(), smap, factor);
  unrolled_theta->set_predicate(smap.lookup(theta->predicate()->origin()));

  for (auto olv = theta->begin(), nlv = unrolled_theta->begin(); olv != theta->end(); olv++, nlv++)
  {
    auto origin = smap.lookup((*olv)->result()->origin());
    (*nlv)->result()->divert_to(origin);
    smap.insert(*olv, *nlv);
  }

  if (remainder != 0)
  {
    /*
      We have residual iterations. Adjust the end value of the unrolled loop
      to a multiple of the step value.
    */
    auto cmpnode = ui.cmpnode();
    auto cmp = jlm::rvsdg::node_output::node(smap.lookup(cmpnode->output(0)));
    auto input = cmp->input(0)->origin() == smap.lookup(ui.end()) ? cmp->input(0) : cmp->input(1);
    JLM_ASSERT(input->origin() == smap.lookup(ui.end()));

    auto sv = ui.is_additive() ? *ui.step_value() : ui.step_value()->neg();
    auto end = remainder.mul(sv);
    auto ev = ui.is_additive() ? ui.end_value()->sub(end) : ui.end_value()->add(end);

    auto c = jlm::rvsdg::create_bitconstant(unrolled_theta->subregion(), ev);
    input->divert_to(c);
  }
}

/*
  Adde the reminder for the lopp if any
*/
static void
add_remainder(const unrollinfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);

  if (remainder == 0)
  {
    /*
      We only need to redirect the users of the outputs of the old theta node
      to the outputs of the new theta node, as there are no residual iterations.
    */
    for (const auto & olv : *theta)
      olv->divert_users(smap.lookup(olv));
    return remove(theta);
  }

  /*
    Add the old theta node as epilogue after the unrolled loop by simply
    redirecting the inputs of the old theta to the outputs of the unrolled
    theta.
  */
  for (const auto & olv : *theta)
    olv->input()->divert_to(smap.lookup(olv));

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
unroll_known_theta(const unrollinfo & ui, size_t factor)
{
  JLM_ASSERT(ui.is_known() && ui.niterations());
  auto niterations = ui.niterations();
  auto original_theta = ui.theta();
  auto nbits = ui.nbits();

  JLM_ASSERT(niterations != 0);
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

static jlm::rvsdg::output *
create_unrolled_gamma_predicate(const unrollinfo & ui, size_t factor)
{
  auto region = ui.theta()->region();
  auto nbits = ui.nbits();
  auto step = ui.step()->input()->origin();
  auto end = ui.end()->input()->origin();

  auto uf = jlm::rvsdg::create_bitconstant(region, nbits, factor);
  auto mul = jlm::rvsdg::bitmul_op::create(nbits, step, uf);
  auto arm =
      jlm::rvsdg::simple_node::create_normalized(region, ui.armoperation(), { ui.init(), mul })[0];
  /* FIXME: order of operands */
  auto cmp = jlm::rvsdg::simple_node::create_normalized(region, ui.cmpoperation(), { arm, end })[0];
  auto pred = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  return pred;
}

static jlm::rvsdg::output *
create_unrolled_theta_predicate(
    rvsdg::Region * target,
    const rvsdg::SubstitutionMap & smap,
    const unrollinfo & ui,
    size_t factor)
{
  using namespace jlm::rvsdg;

  auto region = smap.lookup(ui.cmpnode()->output(0))->region();
  auto cmpnode = jlm::rvsdg::node_output::node(smap.lookup(ui.cmpnode()->output(0)));
  auto step = smap.lookup(ui.step());
  auto end = smap.lookup(ui.end());
  auto nbits = ui.nbits();

  auto i0 = cmpnode->input(0);
  auto i1 = cmpnode->input(1);
  auto iend = i0->origin() == end ? i0 : i1;
  auto idv = i0->origin() == end ? i1 : i0;

  auto uf = create_bitconstant(region, nbits, factor);
  auto mul = bitmul_op::create(nbits, step, uf);
  auto arm = simple_node::create_normalized(region, ui.armoperation(), { idv->origin(), mul })[0];
  /* FIXME: order of operands */
  auto cmp = simple_node::create_normalized(region, ui.cmpoperation(), { arm, iend->origin() })[0];
  auto pred = match(1, { { 1, 1 } }, 0, 2, cmp);

  return pred;
}

static jlm::rvsdg::output *
create_residual_gamma_predicate(const rvsdg::SubstitutionMap & smap, const unrollinfo & ui)
{
  auto region = ui.theta()->region();
  auto idv = smap.lookup(ui.theta()->output(ui.idv()->input()->index()));
  auto end = ui.end()->input()->origin();

  /* FIXME: order of operands */
  auto cmp = jlm::rvsdg::simple_node::create_normalized(region, ui.cmpoperation(), { idv, end })[0];
  auto pred = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  return pred;
}

static void
unroll_unknown_theta(const unrollinfo & ui, size_t factor)
{
  auto otheta = ui.theta();

  /* handle gamma with unrolled loop */
  rvsdg::SubstitutionMap smap;
  {
    auto pred = create_unrolled_gamma_predicate(ui, factor);
    auto ngamma = rvsdg::GammaNode::create(pred, 2);
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    for (const auto & olv : *otheta)
    {
      auto ev = ngamma->add_entryvar(olv->input()->origin());
      auto nlv = ntheta->add_loopvar(ev->argument(1));
      rmap[0].insert(olv, ev->argument(0));
      rmap[1].insert(olv->argument(), nlv->argument());
    }

    unroll_body(otheta, ntheta->subregion(), rmap[1], factor);
    pred = create_unrolled_theta_predicate(ntheta->subregion(), rmap[1], ui, factor);
    ntheta->set_predicate(pred);

    for (auto olv = otheta->begin(), nlv = ntheta->begin(); olv != otheta->end(); olv++, nlv++)
    {
      auto origin = rmap[1].lookup((*olv)->result()->origin());
      (*nlv)->result()->divert_to(origin);
      rmap[1].insert(*olv, *nlv);
    }

    for (const auto & olv : *otheta)
    {
      auto xv = ngamma->add_exitvar({ rmap[0].lookup(olv), rmap[1].lookup(olv) });
      smap.insert(olv, xv);
    }
  }

  /* handle gamma for residual iterations */
  {
    auto pred = create_residual_gamma_predicate(smap, ui);
    auto ngamma = rvsdg::GammaNode::create(pred, 2);
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    for (const auto & olv : *otheta)
    {
      auto ev = ngamma->add_entryvar(smap.lookup(olv));
      auto nlv = ntheta->add_loopvar(ev->argument(1));
      rmap[0].insert(olv, ev->argument(0));
      rmap[1].insert(olv->argument(), nlv->argument());
    }

    otheta->subregion()->copy(ntheta->subregion(), rmap[1], false, false);
    ntheta->set_predicate(rmap[1].lookup(otheta->predicate()->origin()));

    for (auto olv = otheta->begin(), nlv = ntheta->begin(); olv != otheta->end(); olv++, nlv++)
    {
      auto origin = rmap[1].lookup((*olv)->result()->origin());
      (*nlv)->result()->divert_to(origin);
      auto xv = ngamma->add_exitvar({ rmap[0].lookup(*olv), *nlv });
      smap.insert(*olv, xv);
    }
  }
  for (const auto & olv : *otheta)
    olv->divert_users(smap.lookup(olv));
  remove(otheta);
}

void
unroll(rvsdg::ThetaNode * otheta, size_t factor)
{
  if (factor < 2)
    return;

  auto ui = unrollinfo::create(otheta);
  if (!ui)
    return;

  auto nf = otheta->graph()->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  if (ui->is_known() && ui->niterations())
    unroll_known_theta(*ui, factor);
  else
    unroll_unknown_theta(*ui, factor);

  nf->set_mutable(true);
}

static bool
unroll(rvsdg::Region * region, size_t factor)
{
  bool unrolled = false;
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
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

/* loopunroll class */

loopunroll::~loopunroll()
{}

void
loopunroll::run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  if (factor_ < 2)
    return;

  auto & graph = module.Rvsdg();
  auto statistics = unrollstat::Create(module.SourceFileName());

  statistics->start(module.Rvsdg());
  unroll(graph.root(), factor_);
  statistics->end(module.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}
