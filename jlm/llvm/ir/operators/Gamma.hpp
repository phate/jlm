/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_GAMMA_HPP
#define JLM_LLVM_IR_OPERATORS_GAMMA_HPP

namespace jlm::rvsdg
{
class GammaNode;
}

namespace jlm::llvm
{

/**
 * Reduces a gamma node with a statically known predicate to the respective subregion determined
 * by the value of the predicate.
 *
 * c = gamma 0
 *   []
 *     x = 45
 *   [c <= x]
 *   []
 *     y = 37
 *   [c <= y]
 * ... = add c + 5
 * =>
 * c = 45
 * ... = add c + 5
 *
 * @param gammaNode A gamma node that is supposed to be reduced.
 * @return True, if transformation was successful, otherwise false.
 */
static bool
reduceStaticallyKnownPredicate(rvsdg::GammaNode & gammaNode);

}

#endif
