/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_BACKEND_MLIRGEN_HPP
#define JLM_MLIR_BACKEND_MLIRGEN_HPP

#ifdef MLIR_ENABLED

// JLM
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

// MLIR RVSDG dialects
#include <JLM/JLMDialect.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

// MLIR generic dialects
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace jlm::rvsdgmlir
{

class MLIRGen final
{
public:
  MLIRGen()
  {
    Context_ = std::make_unique<mlir::MLIRContext>();
    // Load the RVSDG dialect
    Context_->getOrLoadDialect<mlir::rvsdg::RVSDGDialect>();
    // Load the JLM dialect
    Context_->getOrLoadDialect<mlir::jlm::JLMDialect>();
    // Load the Arith dialect
    Context_->getOrLoadDialect<mlir::arith::ArithDialect>();
    Builder_ = std::make_unique<mlir::OpBuilder>(Context_.get());
  }

  MLIRGen(const MLIRGen &) = delete;

  MLIRGen(MLIRGen &&) = delete;

  MLIRGen &
  operator=(const MLIRGen &) = delete;

  MLIRGen &
  operator=(MLIRGen &&) = delete;

  void
  print(std::unique_ptr<mlir::rvsdg::OmegaNode> & omega, const util::filepath & filePath);

  std::unique_ptr<mlir::rvsdg::OmegaNode>
  convertModule(const llvm::RvsdgModule & rvsdgModule);

private:
  std::unique_ptr<mlir::rvsdg::OmegaNode>
  convertOmega(const rvsdg::graph & graph);

  ::llvm::SmallVector<mlir::Value>
  convertSubregion(rvsdg::region & region, mlir::Block & block);

  mlir::Value
  convertNode(const rvsdg::node & node, mlir::Block & block);

  mlir::Value
  convertSimpleNode(const rvsdg::simple_node & node, mlir::Block & block);

  mlir::Value
  convertLambda(const llvm::lambda::node & node, mlir::Block & block);

  mlir::Type
  convertType(const rvsdg::type & type);

  std::unique_ptr<mlir::OpBuilder> Builder_;
  std::unique_ptr<mlir::MLIRContext> Context_;
};

} // namespace jlm::rvsdgmlir

#endif // MLIR_ENABLED

#endif // JLM_MLIR_BACKEND_MLIRGEN_HPP
