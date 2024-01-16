/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_BACKEND_JLMTOMLIRCONVERTER_HPP
#define JLM_MLIR_BACKEND_JLMTOMLIRCONVERTER_HPP

// JLM
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

// MLIR RVSDG dialects
#include <JLM/JLMDialect.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

// MLIR generic dialects
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace jlm::mlir
{

class JlmToMlirConverter final
{
public:
  JlmToMlirConverter()
  {
    Context_ = std::make_unique<::mlir::MLIRContext>();
    // Load the RVSDG dialect
    Context_->getOrLoadDialect<::mlir::rvsdg::RVSDGDialect>();
    // Load the JLM dialect
    Context_->getOrLoadDialect<::mlir::jlm::JLMDialect>();
    // Load the Arith dialect
    Context_->getOrLoadDialect<::mlir::arith::ArithDialect>();
    Builder_ = std::make_unique<::mlir::OpBuilder>(Context_.get());
  }

  JlmToMlirConverter(const JlmToMlirConverter &) = delete;

  JlmToMlirConverter(JlmToMlirConverter &&) = delete;

  JlmToMlirConverter &
  operator=(const JlmToMlirConverter &) = delete;

  JlmToMlirConverter &
  operator=(JlmToMlirConverter &&) = delete;

  /**
   * Prints MLIR RVSDG to a file.
   * \param omega The MLIR RVSDG Omega node to be printed.
   * \param filePath The path to the file to print the MLIR to.
   */
  void
  Print(::mlir::rvsdg::OmegaNode & omega, const util::filepath & filePath);

  /**
   * Converts an RVSDG module to MLIR RVSDG.
   * \param rvsdgModule The RVSDG module to be converted.
   * \return An MLIR RVSDG OmegaNode containing the whole graph of the rvsdgModule. It is
   * the responsibility of the caller to call ->destroy() on the returned omega, once it is no
   * longer needed.
   */
  ::mlir::rvsdg::OmegaNode
  ConvertModule(const llvm::RvsdgModule & rvsdgModule);

private:
  /**
   * Converts an omega and all nodes in its (sub)region(s) to an MLIR RVSDG OmegaNode.
   * \param graph The root RVSDG graph.
   * \return An MLIR RVSDG OmegaNode.
   */
  ::mlir::rvsdg::OmegaNode
  ConvertOmega(const rvsdg::graph & graph);

  /**
   * Converts all nodes in an RVSDG region. Conversion of structural nodes cause their regions to
   * also be converted.
   * \param region The RVSDG region to be converted
   * \param block The MLIR RVSDG block that corresponds to this RVSDG region, and to which
   *              converted nodes are insterted.
   * \return A list of outputs of the converted region/block.
   */
  ::llvm::SmallVector<::mlir::Value>
  ConvertRegion(rvsdg::region & region, ::mlir::Block & block);

  /**
   * Converts an RVSDG node to an MLIR RVSDG operation.
   * \param node The RVSDG node to be converted
   * \param block The MLIR RVSDG block to insert the converted node.
   * \return The converted MLIR RVSDG operation.
   */
  ::mlir::Value
  ConvertNode(const rvsdg::node & node, ::mlir::Block & block);

  /**
   * Converts an RVSDG simple_node to an MLIR RVSDG operation.
   * \param node The RVSDG node to be converted
   * \param block The MLIR RVSDG block to insert the converted node.
   * \return The converted MLIR RVSDG operation.
   */
  ::mlir::Value
  ConvertSimpleNode(const rvsdg::simple_node & node, ::mlir::Block & block);

  /**
   * Converts an RVSDG lambda node to an MLIR RVSDG LambdaNode.
   * \param node The RVSDG lambda node to be converted
   * \param block The MLIR RVSDG block to insert the lambda node.
   * \return The converted MLIR RVSDG LambdaNode.
   */
  ::mlir::Value
  ConvertLambda(const llvm::lambda::node & node, ::mlir::Block & block);

  /**
   * Converts an RVSDG type to an MLIR RVSDG type.
   * \param type The RVSDG type to be converted.
   * \result The corresponding MLIR RVSDG type.
   */
  ::mlir::Type
  ConvertType(const rvsdg::type & type);

  std::unique_ptr<::mlir::OpBuilder> Builder_;
  std::unique_ptr<::mlir::MLIRContext> Context_;
};

} // namespace jlm::mlir

#endif // JLM_MLIR_BACKEND_JLMTOMLIRCONVERTER_HPP
