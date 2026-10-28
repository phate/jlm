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

/**
 * Returns true if the operation is a comparison whose loop condition is satisfied for the value
 * equal to the bound, i.e. a loop of the form \code do { } while (idv <= end) \endcode rather than
 * \code do { } while (idv < end) \endcode.
 */
static bool
is_eqcmp(const rvsdg::Operation & op)
{
  return dynamic_cast<const IntegerUgeOperation *>(&op)
      || dynamic_cast<const IntegerSgeOperation *>(&op)
      || dynamic_cast<const IntegerUleOperation *>(&op)
      || dynamic_cast<const IntegerSleOperation *>(&op);
}

/**
 * Returns true if the operation is an equality or inequality comparison.
 *
 * Such loops are not unrolled: they can exit in the middle of a batch, e.g. when the step does not
 * divide the remaining range, and no guard over a whole batch can express stopping there.
 */
static bool
is_equality_comparison(const rvsdg::Operation & op)
{
  return dynamic_cast<const IntegerEqOperation *>(&op)
      || dynamic_cast<const IntegerNeOperation *>(&op);
}

/**
 * Returns true if the output is constant and defined outside of the loop, i.e. it does not change
 * between iterations.
 */
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

/**
 * Makes the value available in the parent region of the loop, so that it can be referenced from
 * outside the loop body. Values that already live outside the loop are returned as is; constants
 * defined inside the loop body are copied out and turned into a loop variable.
 */
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

std::unique_ptr<jlm::llvm::IntegerValueRepresentation>
LoopUnrollInfo::niterations() const noexcept
{
  if (!is_known() || step_value() == 0)
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

  return std::make_unique<jlm::llvm::IntegerValueRepresentation>(range.udiv(step));
}

std::unique_ptr<LoopUnrollInfo>
LoopUnrollInfo::create(rvsdg::ThetaNode * theta)
{
  using namespace jlm::rvsdg;

  const auto matchNode = rvsdg::TryGetOwnerNode<SimpleNode>(*theta->predicate()->origin());
  if (!is<MatchOperation>(matchNode))
    return nullptr;

  auto cmpnode = rvsdg::TryGetOwnerNode<SimpleNode>(*matchNode->input(0)->origin());
  if (!is<IntegerBinaryOperation>(cmpnode))
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

/**
 * Creates a constant in the loop's operation family.
 */
static rvsdg::Output *
createConstant(rvsdg::Region & region, const IntegerValueRepresentation & value)
{
  return IntegerConstantOperation::Create(region, value).output(0);
}

static void
unroll_body(
    const rvsdg::ThetaNode * theta,
    rvsdg::Region * target,
    rvsdg::SubstitutionMap & smap,
    size_t factor)
{
  /*
    `factor` is the number of body copies, so it must be at least one: the loop bound `factor - 1`
    is evaluated on an unsigned value and would wrap around for zero.
  */
  JLM_ASSERT(factor != 0);
  if (factor == 0)
    return;

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

    auto c = createConstant(*unrolled_theta->subregion(), ev);
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

  /*
    A theta executes its body at least once, so the smallest representable loop runs exactly one
    iteration. A zero iteration count therefore does not describe a loop this transformation can
    build, and the theta is left unchanged.
  */
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

/**
 * Advances \p value by one iteration, using the loop's own arm operation and operand order.
 *
 * The result is bit-identical to the value the original loop would hold after that iteration,
 * because it is computed with the same wrapping arithmetic on the same operands.
 */
static rvsdg::Output *
advanceInductionVariable(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output & value,
    rvsdg::Output & step)
{
  auto armnode = ui.armnode();
  auto idvIsFirstOperand = armnode->input(0)->origin() == ui.idv();

  std::vector<rvsdg::Output *> operands = idvIsFirstOperand
                                            ? std::vector<rvsdg::Output *>{ &value, &step }
                                            : std::vector<rvsdg::Output *>{ &step, &value };

  return rvsdg::SimpleNode::Create(region, ui.armoperation().copy(), operands).output(0);
}

/**
 * Evaluates the loop's own comparison at \p value against the loop's bound, using the loop's
 * original comparison operation and operand order.
 */
static rvsdg::Output *
createLoopComparison(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output & value,
    rvsdg::Output & bound,
    bool boundIsFirstOperand)
{
  std::vector<rvsdg::Output *> operands = boundIsFirstOperand
                                            ? std::vector<rvsdg::Output *>{ &bound, &value }
                                            : std::vector<rvsdg::Output *>{ &value, &bound };

  return rvsdg::SimpleNode::Create(region, ui.cmpoperation().copy(), operands).output(0);
}

/**
 * Creates the guard that checks \p numComparisons consecutive iterations, starting with the
 * iteration decided by the loop predicate at \p firstComparisonValue.
 *
 * A theta evaluates its predicate after each body execution, and the predicate decides whether the
 * next iteration runs. Iteration \c k+1 therefore runs exactly when the loop's comparison holds at
 * the value the induction variable holds after iteration \c k. A guard for a run of iterations is
 * the conjunction of those comparisons:
 * \code cmp(i_a, end) && cmp(i_a + step, end) && ... \endcode
 * for \p numComparisons terms, where \c i_a is \p firstComparisonValue.
 *
 * Computing each term by repeatedly applying the loop's own arm operation guarantees that the
 * values are exactly the ones the original loop would have seen. This is why the guard is immune to
 * the overflow and underflow that plague the two tempting closed forms:
 *
 *  - \code cmp(i_0, end - factor*step) \endcode underflows whenever fewer than
 *    \code factor*step \endcode values remain, turning the bound into a huge unsigned value and
 *    making the guard admit a batch for a loop that is about to finish.
 *  - \code cmp(i_0 + factor*step, end) \endcode overflows for descending loops and for ascending
 *    loops close to the maximum value, again admitting a batch that the loop would not have run.
 *
 * Both make the unrolled body execute iterations the original loop was not supposed to run, which
 * turns bounded loops into effectively unbounded ones.
 *
 * The two guards built on this helper pass different starting values and counts, because the
 * unrolled body is entered at a different point in each case:
 *
 *  - The gamma enclosing the first batch runs iterations 1 to \c factor. Iteration 1 is
 *    unconditional, since a theta always executes its body at least once, so the first comparison
 *    is the one that decides iteration 2, i.e. at \c i_1. That leaves \c factor-1 comparisons.
 *  - The unrolled theta's predicate is evaluated after a batch, at \c i_factor, and decides whether
 *    iteration \c factor+1 runs. The first comparison is therefore at \c i_factor itself, and all
 *    \c factor copies of the next batch need a comparison.
 */
static rvsdg::Output *
createBatchGuardCondition(
    const LoopUnrollInfo & ui,
    rvsdg::Region & region,
    rvsdg::Output & firstComparisonValue,
    rvsdg::Output & step,
    rvsdg::Output & bound,
    bool boundIsFirstOperand,
    size_t numComparisons)
{
  /*
    A guard with no comparisons would be a constant `true`, which would admit every batch
    unconditionally. `unroll()` never asks for that, and neither guard may degenerate to it.
  */
  JLM_ASSERT(numComparisons >= 1);

  rvsdg::Output * value = &firstComparisonValue;
  rvsdg::Output * guard = nullptr;
  for (size_t comparison = 0; comparison < numComparisons; ++comparison)
  {
    auto holds = createLoopComparison(ui, region, *value, bound, boundIsFirstOperand);
    guard = guard ? IntegerAndOperation::createNode(1, *guard, *holds).output(0) : holds;

    value = advanceInductionVariable(ui, region, *value, step);
  }

  return guard;
}

/**
 * Turns a 1-bit condition into a gamma predicate, mapping a set bit to the gamma's then-branch.
 */
static rvsdg::Output *
createGammaPredicate(rvsdg::Output & condition)
{
  auto & predNode = rvsdg::MatchOperation::CreateNode(condition, { { 1, 1 } }, 0, 2);

  return predNode.output(0);
}

/**
 * Creates the guard that admits the first batch, i.e. the guard of the gamma that encloses the
 * unrolled loop.
 */
static jlm::rvsdg::Output *
create_unrolled_gamma_predicate(const LoopUnrollInfo & ui, size_t factor)
{
  JLM_ASSERT(factor >= 2);

  auto region = ui.theta()->region();
  auto step = ui.theta()->MapPreLoopVar(*ui.step()).input->origin();
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  auto boundIsFirstOperand = ui.cmpnode()->input(0)->origin() == ui.end();

  /*
    Iteration 1 of the batch is unconditional, since a theta always executes its body at least once.
    The first comparison therefore decides iteration 2 and is evaluated at `i_1`, leaving
    `factor - 1` comparisons.
  */
  auto firstComparisonValue = advanceInductionVariable(ui, *region, *ui.init(), *step);

  auto condition = createBatchGuardCondition(
      ui,
      *region,
      *firstComparisonValue,
      *step,
      *end,
      boundIsFirstOperand,
      factor - 1);

  return createGammaPredicate(*condition);
}

/**
 * Creates the predicate of the unrolled loop, i.e. the guard deciding whether another batch may run
 * after the one that just finished. It is the same batch guard, evaluated at the value the
 * induction variable holds after the batch.
 */
static jlm::rvsdg::Output *
create_unrolled_theta_predicate(
    rvsdg::Region *,
    const rvsdg::SubstitutionMap & smap,
    const LoopUnrollInfo & ui,
    size_t factor)
{
  using namespace jlm::rvsdg;

  auto region = smap.lookup(*ui.cmpnode()->output(0)).region();
  auto cmpnode = rvsdg::TryGetOwnerNode<Node>(smap.lookup(*ui.cmpnode()->output(0)));
  auto step = &smap.lookup(*ui.step());
  auto end = &smap.lookup(*ui.end());

  auto i0 = cmpnode->input(0);
  auto i1 = cmpnode->input(1);
  auto boundIsFirstOperand = i0->origin() == end;
  auto idv = boundIsFirstOperand ? i1 : i0;

  /*
    This predicate is evaluated after a batch, so all `factor` copies of the next batch are
    conditional and each needs its own comparison.
  */
  auto condition = createBatchGuardCondition(
      ui,
      *region,
      *idv->origin(),
      *step,
      *end,
      boundIsFirstOperand,
      factor);

  return createGammaPredicate(*condition);
}

/**
 * Creates the guard predicate for the residual loop, i.e. the loop that runs the leftover
 * iterations after the unrolled batches.
 *
 * The residual theta is a do-while, so entering it at a value \c v runs one body and then loops
 * while \code cmp(v) \endcode. That is exactly "the rest of the original loop from \c v", so the
 * residual loop must be entered whenever the loop is not already finished. Two situations reach it:
 *
 *  - The batch was declined at the entry value \c i_0. A theta runs its body unconditionally, so
 * the original loop executes one iteration here even when \code cmp(i_0) \endcode is false. This is
 *    the case that broke `GCC-C-execute-loop-4b`:
 *    \code
 *      int j = 1; long i = 0x60000000L;
 *      do { j <<= 1; i += 0x10000000L; } while (i < -0x60000000L);   // one iteration
 *    \endcode
 *    Guarding with \code cmp(i_0, end) \endcode skipped the residual loop here and the body never
 *    ran.
 *  - The unrolled loop exited at \c i_k. The loop is finished exactly when \code cmp(i_k) \endcode
 * is false, so the residual loop runs iff \code cmp(i_k) \endcode holds.
 *
 * The guard is therefore \code cmp(v) || (v == init) \endcode. Negating the batch guard instead
 * would be wrong in the second case, because the batch guard asks whether a further batch of copies
 * fits, not whether the loop is finished.
 */
static jlm::rvsdg::Output *
create_residual_gamma_predicate(const rvsdg::SubstitutionMap & smap, const LoopUnrollInfo & ui)
{
  auto region = ui.theta()->region();
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  auto boundIsFirstOperand = ui.cmpnode()->input(0)->origin() == ui.end();
  auto idv = &smap.lookup(*ui.theta()->MapPreLoopVar(*ui.idv()).output);

  /*
    The loop is not finished while the predicate still holds, which covers the state after the
    unrolled loop exits.
  */
  auto notFinished = createLoopComparison(ui, *region, *idv, *end, boundIsFirstOperand);

  /*
    At the entry value the body runs unconditionally, so the residual loop must be entered even when
    the predicate already fails. Detecting the entry value is enough, since the unrolled loop only
    ever exits after at least one batch.
  */
  auto atEntryValue = IntegerEqOperation::createNode(ui.nbits(), *idv, *ui.init()).output(0);

  auto condition = IntegerOrOperation::createNode(1, *notFinished, *atEntryValue).output(0);

  return createGammaPredicate(*condition);
}

static void
unroll_unknown_theta(const LoopUnrollInfo & ui, size_t factor)
{
  auto otheta = ui.theta();

  /* handle gamma with unrolled loop */
  rvsdg::SubstitutionMap smap;
  {
    auto pred = create_unrolled_gamma_predicate(ui, factor);
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
    pred = create_unrolled_theta_predicate(ntheta->subregion(), rmap[1], ui, factor);
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
    auto pred = create_residual_gamma_predicate(smap, ui);
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
  {
    /*
      The unknown-bound path unrolls behind a batch guard, which decides up front whether a whole
      batch may run. A loop bounded by `==` or `!=` can exit in the middle of a batch, e.g. when the
      step does not divide the remaining range, and no batch guard can express stopping there. Such
      loops are therefore left alone.
    */
    if (is_equality_comparison(ui->cmpoperation()))
      return;

    unroll_unknown_theta(*ui, factor);
  }
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
