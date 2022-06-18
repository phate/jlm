/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/operators/call.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>

namespace jlm::aa {

MemoryStateEncoder::~MemoryStateEncoder()
= default;

}