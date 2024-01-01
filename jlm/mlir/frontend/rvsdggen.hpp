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
  /**
   * Converts the MLIR region and all operations in it
   * MLIR uses blocks as the innermost "container" so this function gets the
   * block of the region and converts it.
   * \param region The MLIR region to the converted
   * \param rvsdgRegion The corresponding RVSDG region that will be populated with all the contents
   * of the MLIR region. \return The results of the region are returned as a std::vector
   */
  std::unique_ptr<std::vector<jlm::rvsdg::output *>>
  convertRegion(mlir::Region & region, rvsdg::region & rvsdgRegion);

  /**
   * Converts the MLIR block and all operations in it
   * \param block The MLIR block to the converted
   * \param rvsdgRegion The corresponding RVSDG region that will be populated with all the contents
   * of the MLIR region. \return The results of the region are returned as a std::vector
   */
  std::unique_ptr<std::vector<jlm::rvsdg::output *>>
  convertBlock(mlir::Block & block, rvsdg::region & rvsdgRegion);

  rvsdg::node *
  convertOperation(mlir::Operation & mlirOperation, rvsdg::region & rvsdgRegion, std::vector<const rvsdg::output *> & inputs);

  /**
   * Converts an MLIR omega operation and insterst it into an RVSDG region.
   * \param omega The MLIR omega opeation to the converted
   * \param rvsdgRegion The RVSDG region that the omega node will reside in.
   */
  void
  convertOmega(mlir::Operation & mlirOmega, rvsdg::region & rvsdgRegion);

  /**
   * Converts an MLIR lambda operation and inserts it into an RVSDG region.
   * \param mlirLambda The MLIR lambda opeation to the converted
   * \param rvsdgRegion The RVSDG region that the lambda node will reside in.
   */
  rvsdg::node *
  convertLambda(mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion);

  /**
   * Converts an MLIR type into an RVSDG type.
   * \param type The MLIR type to be converted.
   * \result The converted RVSDG type.
   */
  rvsdg::type *
  convertType(mlir::Type & type);

  std::unique_ptr<mlir::MLIRContext> Context_;
};

} // namespace jlm::mlirrvsdg

#endif // MLIR_ENABLED

#endif // JLM_MLIR_FRONTEND_RVSDGGEN_HPP
