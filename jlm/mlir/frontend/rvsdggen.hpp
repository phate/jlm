/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_FRONTEND_RVSDGGEN_HPP
#define JLM_MLIR_FRONTEND_RVSDGGEN_HPP

#ifdef MLIR_ENABLED

// #include <jlm/llvm/ir/operators/GetElementPtr.hpp>
// #include <jlm/llvm/ir/operators/load.hpp>
// #include <jlm/llvm/ir/operators/operators.hpp>
// #include <jlm/llvm/ir/operators/sext.hpp>
// #include <jlm/llvm/ir/operators/store.hpp>
// #include <jlm/rvsdg/bitstring/comparison.hpp>
// #include <jlm/rvsdg/bitstring/type.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

#include <JLM/JLMDialect.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace jlm::mlirrvsdg
{

class RVSDGGen final
{
public:
  RVSDGGen()
  {
    Context_ = std::make_unique<mlir::MLIRContext>();
    // Load the RVSDG dialect
    Context_->getOrLoadDialect<mlir::rvsdg::RVSDGDialect>();
    // Load the JLM dialect
    Context_->getOrLoadDialect<mlir::jlm::JLMDialect>();
    // Load the Arith dialect
    Context_->getOrLoadDialect<mlir::arith::ArithDialect>();
  }

  RVSDGGen(const RVSDGGen &) = delete;

  RVSDGGen(RVSDGGen &&) = delete;

  RVSDGGen &
  operator=(const RVSDGGen &) = delete;

  RVSDGGen &
  operator=(RVSDGGen &&) = delete;

  std::unique_ptr<mlir::Block>
  readRvsdgMlir(const util::filepath & filePath);

  std::unique_ptr<llvm::RvsdgModule>
  convertMlir(std::unique_ptr<mlir::Block> & block);

private:
  void
  convertBlock(mlir::Block & block, rvsdg::region & rvsdgRegion);

  rvsdg::node *
  convertOperation(mlir::Operation & operation, rvsdg::region & rvsdgRegion);

  void
  convertRegion(mlir::Region & region, rvsdg::region & rvsdgRegion);

  void
  convertOmega(mlir::Operation & omega, rvsdg::region & rvsdgRegion);

  rvsdg::node *
  convertLambda(mlir::Operation & lambda, rvsdg::region & rvsdgRegion);

  rvsdg::type *
  convertType(mlir::Type & type);

  std::unique_ptr<mlir::MLIRContext> Context_;
};

} // namespace jlm::mlirrvsdg

#endif // MLIR_ENABLED

#endif // JLM_MLIR_FRONTEND_RVSDGGEN_HPP
