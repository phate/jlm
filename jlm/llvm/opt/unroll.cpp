/*
 * Copyright 2017 Nico Reißmann <nico.reißmann@gmail.com>
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
isInclusiveComparison(const rvsdg::Operation & op)
{
  return dynamic_cast<const IntegerUgeOperation *>(&op)
      || dynamic_cast<const IntegerSgeOperation *>(&op)
      || dynamic_cast<const IntegerUleOperation *>(&op)
      || dynamic_cast<const IntegerSleOperation *>(&op);
}

/**
 * Returns true if the output is constant and defined outside of the loop, i.e. it does not change
 * between iterations.
 */
static bool
isThetaInvariant(const jlm::rvsdg::Output * output)
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
pushFromTheta(jlm::rvsdg::Output * output)
{
  auto argument = dynamic_cast<rvsdg::RegionArgument *>(output);
  if (argument)
    return argument;

  auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*output);
  JLM_ASSERT(jlm::rvsdg::is<IntegerConstantOperation>(node));
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node->region()->node()));
  auto theta = static_cast<rvsdg::ThetaNode *>(node->region()->node());

  auto constant = node->copy(theta->region(), {});
  auto loopVar = theta->AddLoopVar(constant->output(0));
  output->divert_users(loopVar.pre);

  return loopVar.pre;
}

static bool
isInductionVariable(jlm::rvsdg::Input * input)
{
  using namespace jlm::rvsdg;

  auto node = TryGetOwnerNode<SimpleNode>(*input);
  JLM_ASSERT(is<IntegerAddOperation>(node) || is<IntegerSubOperation>(node));

  if (auto theta = rvsdg::TryGetRegionParentNode<ThetaNode>(*input->origin()))
  {
    auto loopVar = theta->MapPreLoopVar(*input->origin());
    return rvsdg::TryGetOwnerNode<Node>(*loopVar.post->origin()) == node;
  }

  return false;
}

std::unique_ptr<jlm::llvm::IntegerValueRepresentation>
LoopUnrollInfo::niterations() const noexcept
{
  if (!isKnown() || stepValue()->isZero())
    return nullptr;

  auto start = isAdditive() ? *initValue() : *endValue();
  auto step = isAdditive() ? *stepValue() : stepValue()->neg();
  auto end = isAdditive() ? *endValue() : *initValue();

  if (isInclusiveComparison(cmpNode()->GetOperation()))
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

  auto cmpNode = rvsdg::TryGetOwnerNode<SimpleNode>(*matchNode->input(0)->origin());
  if (!is<IntegerBinaryOperation>(cmpNode))
    return nullptr;

  auto operand0 = cmpNode->input(0)->origin();
  auto operand1 = cmpNode->input(1)->origin();
  auto end =
      isThetaInvariant(operand0) ? operand0 : (isThetaInvariant(operand1) ? operand1 : nullptr);
  if (!end)
    return nullptr;

  auto armNode = rvsdg::TryGetOwnerNode<SimpleNode>(*(end == operand0 ? operand1 : operand0));
  if (!is<IntegerAddOperation>(armNode) && !is<IntegerSubOperation>(armNode))
    return nullptr;
  if (armNode->ninputs() != 2)
    return nullptr;

  auto input0 = armNode->input(0);
  auto input1 = armNode->input(1);
  if (!isInductionVariable(input0) && !isInductionVariable(input1))
    return nullptr;

  auto idv = static_cast<rvsdg::RegionArgument *>(
      isInductionVariable(input0) ? input0->origin() : input1->origin());

  auto step = idv == input0->origin() ? input1->origin() : input0->origin();
  if (!isThetaInvariant(step))
    return nullptr;

  auto endArgument = pushFromTheta(end);
  auto stepArgument = pushFromTheta(step);
  return std::unique_ptr<LoopUnrollInfo>(
      new LoopUnrollInfo(cmpNode, armNode, idv, stepArgument, endArgument));
}

/* loop unrolling */

static void
unrollBody(
    const rvsdg::ThetaNode * theta,
    rvsdg::Region * target,
    rvsdg::SubstitutionMap & smap,
    size_t factor)
{
  /*
    `factor` is the number of body copies, so it must be at least one: the loop bound `factor - 1`
    is evaluated on an unsigned value and would wrap around for zero.
  */
  if (factor == 0)
    return;

  for (size_t n = 0; n < factor - 1; n++)
  {
    theta->subregion()->copy(target, smap);
    rvsdg::SubstitutionMap tmap;
    for (const auto & oldLoopVar : theta->GetLoopVars())
      tmap.insert(oldLoopVar.pre, &smap.lookup(*oldLoopVar.post->origin()));
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
copyBodyAndUnroll(const rvsdg::ThetaNode * theta, size_t factor)
{
  rvsdg::SubstitutionMap smap;
  for (const auto & oldLoopVar : theta->GetLoopVars())
    smap.insert(oldLoopVar.pre, oldLoopVar.input->origin());

  unrollBody(theta, theta->region(), smap, factor);

  for (const auto & oldLoopVar : theta->GetLoopVars())
    oldLoopVar.output->divert_users(&smap.lookup(*oldLoopVar.post->origin()));
}

/*
  Unroll theta node by given factor.
*/
static void
unrollTheta(const LoopUnrollInfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);
  auto unrolledTheta = rvsdg::ThetaNode::create(theta->region());

  auto oldLoopVars = theta->GetLoopVars();
  for (const auto & oldLoopVar : oldLoopVars)
  {
    auto newLoopVar = unrolledTheta->AddLoopVar(oldLoopVar.input->origin());
    smap.insert(oldLoopVar.pre, newLoopVar.pre);
  }

  unrollBody(theta, unrolledTheta->subregion(), smap, factor);
  unrolledTheta->set_predicate(&smap.lookup(*theta->predicate()->origin()));

  auto newLoopVars = unrolledTheta->GetLoopVars();
  for (size_t i = 0; i < oldLoopVars.size(); ++i)
  {
    const auto & oldLoopVar = oldLoopVars[i];
    const auto & newLoopVar = newLoopVars[i];
    auto origin = &smap.lookup(*oldLoopVar.post->origin());
    newLoopVar.post->divert_to(origin);
    smap.insert(oldLoopVar.output, newLoopVar.output);
  }

  if (remainder != 0)
  {
    /*
      We have residual iterations. Adjust the end value of the unrolled loop
      to a multiple of the step value.
    */
    auto cmpNode = ui.cmpNode();
    auto cmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(smap.lookup(*cmpNode->output(0)));
    auto input = cmp->input(0)->origin() == &smap.lookup(*ui.end()) ? cmp->input(0) : cmp->input(1);
    JLM_ASSERT(input->origin() == &smap.lookup(*ui.end()));

    auto stepValue = ui.isAdditive() ? *ui.stepValue() : ui.stepValue()->neg();
    auto endAdjustment = remainder.mul(stepValue);
    auto adjustedEnd =
        ui.isAdditive() ? ui.endValue()->sub(endAdjustment) : ui.endValue()->add(endAdjustment);

    auto c = IntegerConstantOperation::Create(*unrolledTheta->subregion(), adjustedEnd).output(0);
    input->divert_to(c);
  }
}

/*
  Add the old theta node as an epilogue for the residual iterations, if any.
*/
static void
addRemainder(const LoopUnrollInfo & ui, rvsdg::SubstitutionMap & smap, size_t factor)
{
  auto theta = ui.theta();
  auto remainder = ui.remainder(factor);

  if (remainder == 0)
  {
    /*
      We only need to redirect the users of the outputs of the old theta node
      to the outputs of the new theta node, as there are no residual iterations.
    */
    for (const auto & oldLoopVar : theta->GetLoopVars())
      oldLoopVar.output->divert_users(&smap.lookup(*oldLoopVar.output));
    return remove(theta);
  }

  /*
    Add the old theta node as epilogue after the unrolled loop by simply
    redirecting the inputs of the old theta to the outputs of the unrolled
    theta.
  */
  for (const auto & oldLoopVar : theta->GetLoopVars())
    oldLoopVar.input->divert_to(&smap.lookup(*oldLoopVar.output));

  if (remainder == 1)
  {
    /*
      There is only one loop iteration remaining.
      Simply copy the body of the theta to replace it.
    */
    copyBodyAndUnroll(theta, 1);
    remove(theta);
  }
}

static void
unrollKnownTheta(const LoopUnrollInfo & ui, size_t factor)
{
  JLM_ASSERT(ui.isKnown() && ui.niterations());
  auto niterations = ui.niterations();
  auto originalTheta = ui.theta();
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
    copyBodyAndUnroll(originalTheta, niterations->to_uint());
    return remove(originalTheta);
  }

  JLM_ASSERT(niterations->ugt({ nbits, (int64_t)factor }) == '1');

  /*
    Unroll the theta
  */
  rvsdg::SubstitutionMap smap;
  unrollTheta(ui, smap, factor);

  /*
    Add code for any potential iterations that remains
  */
  addRemainder(ui, smap, factor);
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
  auto armNode = ui.armNode();
  auto idvIsFirstOperand = armNode->input(0)->origin() == ui.idv();

  std::vector<rvsdg::Output *> operands = idvIsFirstOperand
                                            ? std::vector<rvsdg::Output *>{ &value, &step }
                                            : std::vector<rvsdg::Output *>{ &step, &value };

  return rvsdg::SimpleNode::Create(region, ui.armOperation().copy(), operands).output(0);
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

  return rvsdg::SimpleNode::Create(region, ui.cmpOperation().copy(), operands).output(0);
}

/**
 * Creates the guard condition that checks \p numComparisons consecutive iterations, starting with
 * the iteration decided by the loop predicate at \p firstComparisonValue.
 *
 * A theta evaluates its predicate after each body execution, and the predicate decides whether the
 * next iteration runs. Iteration \c k+1 therefore runs exactly when the loop's comparison holds at
 * the value the induction variable holds after iteration \c k. A guard for a run of iterations is
 * the conjunction of those comparisons:
 * \code cmp(i_a, end) && cmp(i_a + step, end) && ... \endcode
 * for \p numComparisons terms, where \c i_a is \p firstComparisonValue.
 *
 * Computing each term by repeatedly applying the loop's own arm operation guarantees that the
 * values are exactly the ones the original loop would have seen. This is why the guard is immune
 * to the overflow and underflow that plague the two tempting closed forms:
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
 * Turns a 1-bit condition into a gamma or theta predicate, mapping a set bit to the then-branch or
 * to another loop iteration.
 */
static rvsdg::Output *
createGammaPredicate(rvsdg::Output & condition)
{
  auto & predNode = rvsdg::MatchOperation::CreateNode(condition, { { 1, 1 } }, 0, 2);

  return predNode.output(0);
}

/**
 * Creates the guard condition that admits the first batch, i.e. the condition of the gamma that
 * encloses the unrolled loop.
 */
static rvsdg::Output *
createFirstBatchGuard(const LoopUnrollInfo & ui, size_t factor)
{
  JLM_ASSERT(factor >= 2);

  auto region = ui.theta()->region();
  auto step = ui.theta()->MapPreLoopVar(*ui.step()).input->origin();
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  auto boundIsFirstOperand = ui.cmpNode()->input(0)->origin() == ui.end();

  /*
    Iteration 1 of the batch is unconditional, since a theta always executes its body at least once.
    The first comparison therefore decides iteration 2 and is evaluated at `i_1`, leaving
    `factor - 1` comparisons.
  */
  auto firstComparisonValue = advanceInductionVariable(ui, *region, *ui.init(), *step);

  return createBatchGuardCondition(
      ui,
      *region,
      *firstComparisonValue,
      *step,
      *end,
      boundIsFirstOperand,
      factor - 1);
}

/**
 * Creates the guard condition of the unrolled loop, i.e. the condition deciding whether another
 * batch may run after the one that just finished. It is the same batch guard, evaluated at the
 * value the induction variable holds after the batch.
 */
static rvsdg::Output *
createNextBatchGuard(const rvsdg::SubstitutionMap & smap, const LoopUnrollInfo & ui, size_t factor)
{
  using namespace jlm::rvsdg;

  auto cmpOutput = &smap.lookup(*ui.cmpNode()->output(0));
  auto region = cmpOutput->region();
  auto cmpNode = rvsdg::TryGetOwnerNode<Node>(*cmpOutput);
  auto step = &smap.lookup(*ui.step());
  auto end = &smap.lookup(*ui.end());

  auto input0 = cmpNode->input(0);
  auto input1 = cmpNode->input(1);
  auto boundIsFirstOperand = input0->origin() == end;
  auto idv = boundIsFirstOperand ? input1 : input0;

  /*
    This guard is evaluated after a batch, so all `factor` copies of the next batch are conditional
    and each needs its own comparison.
  */
  return createBatchGuardCondition(
      ui,
      *region,
      *idv->origin(),
      *step,
      *end,
      boundIsFirstOperand,
      factor);
}

/**
 * Creates the guard condition for the residual loop, i.e. the loop that runs the leftover
 * iterations after the unrolled batches.
 *
 * The residual theta is a do-while, so entering it at a value \c v runs one body and then loops
 * while \code cmp(v) \endcode. That is exactly "the rest of the original loop from \c v". Two
 * situations reach it:
 *
 *  - The first batch was declined. The original loop still executes its body at least once, and
 *    the residual loop reproduces the entire original loop from the entry value, so it must be
 *    entered even when the comparison already fails at the entry value:
 *    \code
 *      int j = 1; long i = 0x60000000L;
 *      do { j <<= 1; i += 0x10000000L; } while (i < -0x60000000L);   // one iteration
 *    \endcode
 *  - The unrolled loop exited at \c i_k. The remaining iterations are exactly the ones the original
 *    loop would run from \c i_k, and the first of them runs precisely when \code cmp(i_k) \endcode
 *    holds.
 *
 * The guard is therefore \code !batchGuard || cmp(idv) \endcode. The declined-batch decision must
 * be taken from \p batchGuard rather than from the induction variable's value: wrapping arithmetic
 * can bring the variable back to its initial value with the comparison already failing, so no
 * comparison against that value can identify the declined case, and such a guard would run the
 * entire loop a second time.
 */
static rvsdg::Output *
createResidualLoopGuard(
    const rvsdg::SubstitutionMap & smap,
    const LoopUnrollInfo & ui,
    rvsdg::Output & batchGuard)
{
  auto region = ui.theta()->region();
  auto end = ui.theta()->MapPreLoopVar(*ui.end()).input->origin();
  auto boundIsFirstOperand = ui.cmpNode()->input(0)->origin() == ui.end();
  auto idv = &smap.lookup(*ui.theta()->MapPreLoopVar(*ui.idv()).output);

  /*
    The loop is not finished while the predicate still holds, which covers the state after the
    unrolled loop exits.
  */
  auto notFinished = createLoopComparison(ui, *region, *idv, *end, boundIsFirstOperand);

  /*
    A set bit in \p batchGuard means the first batch was taken, so its negation records whether the
    residual loop must reproduce the entire original loop.
  */
  auto one = IntegerConstantOperation::Create(*region, 1, 1).output(0);
  auto batchDeclined = IntegerXorOperation::createNode(1, batchGuard, *one).output(0);

  return IntegerOrOperation::createNode(1, *notFinished, *batchDeclined).output(0);
}

static void
unrollUnknownTheta(const LoopUnrollInfo & ui, size_t factor)
{
  auto oldTheta = ui.theta();

  /* handle gamma with unrolled loop */
  rvsdg::SubstitutionMap smap;
  auto batchGuard = createFirstBatchGuard(ui, factor);
  {
    auto pred = createGammaPredicate(*batchGuard);
    auto newGamma = rvsdg::GammaNode::create(pred, 2);
    auto newTheta = rvsdg::ThetaNode::create(newGamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    for (const auto & oldLoopVar : oldTheta->GetLoopVars())
    {
      auto entryVar = newGamma->AddEntryVar(oldLoopVar.input->origin());
      auto newLoopVar = newTheta->AddLoopVar(entryVar.branchArgument[1]);
      rmap[0].insert(oldLoopVar.output, entryVar.branchArgument[0]);
      rmap[1].insert(oldLoopVar.pre, newLoopVar.pre);
    }

    unrollBody(oldTheta, newTheta->subregion(), rmap[1], factor);
    pred = createGammaPredicate(*createNextBatchGuard(rmap[1], ui, factor));
    newTheta->set_predicate(pred);

    auto oldLoopVars = oldTheta->GetLoopVars();
    auto newLoopVars = newTheta->GetLoopVars();
    for (std::size_t n = 0; n < oldLoopVars.size(); ++n)
    {
      auto & oldLoopVar = oldLoopVars[n];
      auto & newLoopVar = newLoopVars[n];
      auto origin = &rmap[1].lookup(*oldLoopVar.post->origin());
      newLoopVar.post->divert_to(origin);
      rmap[1].insert(oldLoopVar.output, newLoopVar.output);
    }

    for (const auto & oldLoopVar : oldLoopVars)
    {
      auto exitVar = newGamma
                         ->AddExitVar({ &rmap[0].lookup(*oldLoopVar.output),
                                        &rmap[1].lookup(*oldLoopVar.output) })
                         .output;
      smap.insert(oldLoopVar.output, exitVar);
    }
  }

  /* handle gamma for residual iterations */
  {
    auto pred = createGammaPredicate(*createResidualLoopGuard(smap, ui, *batchGuard));
    auto newGamma = rvsdg::GammaNode::create(pred, 2);
    auto newTheta = rvsdg::ThetaNode::create(newGamma->subregion(1));

    rvsdg::SubstitutionMap rmap[2];
    auto oldLoopVars = oldTheta->GetLoopVars();
    for (const auto & oldLoopVar : oldLoopVars)
    {
      auto entryVar = newGamma->AddEntryVar(&smap.lookup(*oldLoopVar.output));
      auto newLoopVar = newTheta->AddLoopVar(entryVar.branchArgument[1]);
      rmap[0].insert(oldLoopVar.output, entryVar.branchArgument[0]);
      rmap[1].insert(oldLoopVar.pre, newLoopVar.pre);
    }

    oldTheta->subregion()->copy(newTheta->subregion(), rmap[1]);
    newTheta->set_predicate(&rmap[1].lookup(*oldTheta->predicate()->origin()));

    auto newLoopVars = newTheta->GetLoopVars();

    for (std::size_t n = 0; n < oldLoopVars.size(); ++n)
    {
      auto & oldLoopVar = oldLoopVars[n];
      auto & newLoopVar = newLoopVars[n];
      auto origin = &rmap[1].lookup(*oldLoopVar.post->origin());
      newLoopVar.post->divert_to(origin);
      auto exitVar =
          newGamma->AddExitVar({ &rmap[0].lookup(*oldLoopVar.output), newLoopVar.output }).output;
      smap.insert(oldLoopVar.output, exitVar);
    }
  }
  for (const auto & oldLoopVar : oldTheta->GetLoopVars())
    oldLoopVar.output->divert_users(&smap.lookup(*oldLoopVar.output));
  remove(oldTheta);
}

void
unroll(rvsdg::ThetaNode * oldTheta, size_t factor)
{
  if (factor < 2)
    return;

  auto ui = LoopUnrollInfo::create(oldTheta);
  if (!ui)
    return;

  if (ui->isKnown() && ui->niterations())
    unrollKnownTheta(*ui, factor);
  else
    unrollUnknownTheta(*ui, factor);
}

static bool
unroll(rvsdg::Region * region, size_t factor)
{
  bool unrolled = false;
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structNode->nsubregions(); n++)
        unrolled = unroll(structNode->subregion(n), factor);

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
