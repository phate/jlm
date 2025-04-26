/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_MLIRCONVERTERCOMMON_HPP
#define JLM_MLIR_MLIRCONVERTERCOMMON_HPP

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/util/BijectiveMap.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace jlm::mlir
{

/**
 * Get a bijective mapping between MLIR floating-point comparison predicates and JLM fpcmp values.
 *
 * @return A reference to the static mapping between MLIR CmpFPredicate and JLM fpcmp
 */
const util::BijectiveMap<::mlir::arith::CmpFPredicate, llvm::fpcmp> &
GetFpCmpPredicateMap();

} // namespace jlm::mlir

#endif // JLM_MLIR_MLIRCONVERTERCOMMON_HPP
