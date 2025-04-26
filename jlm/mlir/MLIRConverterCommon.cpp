/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/MLIRConverterCommon.hpp>

namespace jlm::mlir
{

// Mapping between each MLIR FP comparison predicate and JLM fpcmp value
const util::BijectiveMap<::mlir::arith::CmpFPredicate, llvm::fpcmp> &
GetFpCmpPredicateMap()
{
  static util::BijectiveMap<::mlir::arith::CmpFPredicate, llvm::fpcmp> mapping = {
    { ::mlir::arith::CmpFPredicate::AlwaysTrue, llvm::fpcmp::TRUE },
    { ::mlir::arith::CmpFPredicate::AlwaysFalse, llvm::fpcmp::FALSE },
    { ::mlir::arith::CmpFPredicate::OEQ, llvm::fpcmp::oeq },
    { ::mlir::arith::CmpFPredicate::OGT, llvm::fpcmp::ogt },
    { ::mlir::arith::CmpFPredicate::OGE, llvm::fpcmp::oge },
    { ::mlir::arith::CmpFPredicate::OLT, llvm::fpcmp::olt },
    { ::mlir::arith::CmpFPredicate::OLE, llvm::fpcmp::ole },
    { ::mlir::arith::CmpFPredicate::ONE, llvm::fpcmp::one },
    { ::mlir::arith::CmpFPredicate::ORD, llvm::fpcmp::ord },
    { ::mlir::arith::CmpFPredicate::UEQ, llvm::fpcmp::ueq },
    { ::mlir::arith::CmpFPredicate::UGT, llvm::fpcmp::ugt },
    { ::mlir::arith::CmpFPredicate::UGE, llvm::fpcmp::uge },
    { ::mlir::arith::CmpFPredicate::ULT, llvm::fpcmp::ult },
    { ::mlir::arith::CmpFPredicate::ULE, llvm::fpcmp::ule },
    { ::mlir::arith::CmpFPredicate::UNE, llvm::fpcmp::une },
    { ::mlir::arith::CmpFPredicate::UNO, llvm::fpcmp::uno }
  };
  return mapping;
}

} // namespace jlm::mlir
