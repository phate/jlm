/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP
#define JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <JLM/JLMDialect.h>
#include <JLM/JLMOps.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace jlm::mlir
{

class MlirToJlmConverter final
{
public:
  MlirToJlmConverter()
      : Context_(std::make_unique<::mlir::MLIRContext>())
  {
    // Load the RVSDG dialect
    Context_->getOrLoadDialect<::mlir::rvsdg::RVSDGDialect>();
    // Load the JLM dialect
    Context_->getOrLoadDialect<::mlir::jlm::JLMDialect>();
    // Load the Arith dialect
    Context_->getOrLoadDialect<::mlir::arith::ArithDialect>();
  }

  MlirToJlmConverter(const MlirToJlmConverter &) = delete;

  MlirToJlmConverter(MlirToJlmConverter &&) = delete;

  MlirToJlmConverter &
  operator=(const MlirToJlmConverter &) = delete;

  MlirToJlmConverter &
  operator=(MlirToJlmConverter &&) = delete;

  /**
   * Reads RVSDG MLIR from a file and converts it,
   * \param filePath The path to the file containing RVSDG MLIR IR.
   * \return The converted RVSDG graph.
   */
  std::unique_ptr<llvm::RvsdgModule>
  ReadAndConvertMlir(const util::filepath & filePath);

  /**
   * Converts the MLIR block and all operations in it, including their respective regions.
   * \param block The RVSDG MLIR block to be converted.
   * \return The converted RVSDG graph.
   */
  std::unique_ptr<llvm::RvsdgModule>
  ConvertMlir(std::unique_ptr<::mlir::Block> & block);

  /**
   * Temporarily creates an MlirToJlmConverter that is used to convert an MLIR block to an RVSDG
   * graph.
   * \param block The RVSDG MLIR block to be converted.
   * \return The converted RVSDG graph.
   */
  static std::unique_ptr<llvm::RvsdgModule>
  CreateAndConvert(std::unique_ptr<::mlir::Block> & block)
  {
    jlm::mlir::MlirToJlmConverter converter;
    return converter.ConvertMlir(block);
  }

private:
  /**
   * Converts the MLIR region and all operations in it
   * MLIR uses blocks as the innermost "container" so this function gets the
   * block of the region and converts it.
   * \param region The MLIR region to the converted
   * \param rvsdgRegion The corresponding RVSDG region that will be populated with all the contents
   * of the MLIR region.
   * \return The results of the region are returned as a std::vector
   */
  ::llvm::SmallVector<jlm::rvsdg::output *>
  ConvertRegion(::mlir::Region & region, rvsdg::Region & rvsdgRegion);

  /**
   * Converts the MLIR block and all operations in it
   * \param block The MLIR block to the converted
   * \param rvsdgRegion The corresponding RVSDG region that will be populated with all the contents
   * of the MLIR region.
   * \return The results of the region are returned as a std::vector
   */
  ::llvm::SmallVector<jlm::rvsdg::output *>
  ConvertBlock(::mlir::Block & block, rvsdg::Region & rvsdgRegion);

  /**
   * Retreive the previously converted RVSDG ouputs from the map of operations
   * and return them in the inputs vector.
   * \param mlirOp The MLIR operation that the inputs are retrieved for.
   * \param operationsMap The map of operations that have been converted.
   * \param rvsdgRegion The RVSDG region that the inputs are retrieved from (if it's a region
   * argument). \return The vector that is populated with the inputs.
   */
  static ::llvm::SmallVector<jlm::rvsdg::output *>
  GetConvertedInputs(
      ::mlir::Operation & mlirOp,
      const std::unordered_map<::mlir::Operation *, rvsdg::Node *> & operationsMap,
      const rvsdg::Region & rvsdgRegion);

  /**
   * Converts an MLIR integer comparison operation into an RVSDG node.
   * \param CompOp The MLIR comparison operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \param nbits The number of bits in the comparison.
   * \result The converted RVSDG node.
   */
  rvsdg::Node *
  ConvertCmpIOp(
      ::mlir::arith::CmpIOp & CompOp,
      const ::llvm::SmallVector<rvsdg::output *> & inputs,
      size_t nbits);

  /**
   * Converts an MLIR fp binary operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node OR nullptr if the operation cannot be casted to an operation
   */
  rvsdg::node *
  ConvertFPBinaryNode(
      const ::mlir::Operation & mlirOperation,
      rvsdg::Region & rvsdgRegion,
      const ::llvm::SmallVector<rvsdg::output *> & inputs);

  /**
   * Converts an MLIR integer binary operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node OR nullptr if the operation cannot be casted to an operation
   */
  rvsdg::Node *
  ConvertBitBinaryNode(
      const ::mlir::Operation & mlirOperation,
      const ::llvm::SmallVector<rvsdg::output *> & inputs);

  /**
   * Converts an MLIR operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param rvsdgRegion The RVSDG region that the generated RVSDG node is inserted into.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node.
   */
  rvsdg::Node *
  ConvertOperation(
      ::mlir::Operation & mlirOperation,
      rvsdg::Region & rvsdgRegion,
      const ::llvm::SmallVector<rvsdg::output *> & inputs);

  /**
   * Converts a floating point size to jlm::llvm::fpsize.
   * \param size unsinged int representing the size.
   * \result The fpsize.
   */
  llvm::fpsize
  ConvertFPSize(unsigned int size);

  /**
   * Converts an MLIR omega operation and insterst it into an RVSDG region.
   * \param mlirOmega The MLIR omega opeation to the converted
   * \param rvsdgRegion The RVSDG region that the omega node will reside in.
   */
  void
  ConvertOmega(::mlir::Operation & mlirOmega, rvsdg::Region & rvsdgRegion);

  /**
   * Converts an MLIR lambda operation and inserts it into an RVSDG region.
   * \param mlirLambda The MLIR lambda opeation to the converted
   * \param rvsdgRegion The RVSDG region that the lambda node will reside in.
   * \result The converted Lambda node.
   */
  rvsdg::Node *
  ConvertLambda(::mlir::Operation & mlirLambda, rvsdg::Region & rvsdgRegion);

  /**
   * Converts an MLIR type into an RVSDG type.
   * \param type The MLIR type to be converted.
   * \result The converted RVSDG type.
   */
  static std::unique_ptr<rvsdg::Type>
  ConvertType(::mlir::Type & type);

  std::unique_ptr<::mlir::MLIRContext> Context_;
};

} // namespace jlm::mlir

#endif // JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP
