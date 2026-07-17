/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Gamma.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::llvm
{

static bool
reduceStaticallyKnownPredicate(rvsdg::GammaNode & gammaNode)
{
  const auto & tracedPredicate = llvm::traceOutput(*gammaNode.predicate()->origin());
  auto [ctlConstantNode, ctlConstantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(tracedPredicate);
  if (!ctlConstantOperation)
  {
    return false;
  }

  const auto ctlAlternative = ctlConstantOperation->value().alternative();

  rvsdg::SubstitutionMap smap;
  for (const auto & [input, branchArgument] : gammaNode.GetEntryVars())
    smap.insert(branchArgument[ctlAlternative], input->origin());

  gammaNode.subregion(ctlAlternative)->copy(gammaNode.region(), smap);

  for (auto [branchResult, output] : gammaNode.GetExitVars())
    output->divert_users(&smap.lookup(*branchResult[ctlAlternative]->origin()));

  remove(&gammaNode);
  return true;
}

}
