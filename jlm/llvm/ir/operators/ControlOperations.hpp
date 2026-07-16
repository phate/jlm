/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_CONTROLOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_CONTROLOPERATIONS_HPP

#include <jlm/rvsdg/control.hpp>

#include <optional>
#include <vector>

namespace jlm::llvm
{

std::optional<std::vector<rvsdg::Output *>>
foldMatchOperationWithConstant(
    const rvsdg::MatchOperation & matchOperation,
    const std::vector<rvsdg::Output *> & operands);

}

#endif
