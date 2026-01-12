/*
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP
#define JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/BijectiveMap.hpp>

#include <JLM/JLMDialect.h>
#include <JLM/JLMOps.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

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
    // Load the LLVM dialect
    Context_->getOrLoadDialect<::mlir::LLVM::LLVMDialect>();
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
  std::unique_ptr<llvm::LlvmRvsdgModule>
  ReadAndConvertMlir(const util::FilePath & filePath);

  /**
   * Converts the MLIR block and all operations in it, including their respective regions.
   * \param block The RVSDG MLIR block to be converted.
   * \return The converted RVSDG graph.
   */
  std::unique_ptr<llvm::LlvmRvsdgModule>
  ConvertMlir(std::unique_ptr<::mlir::Block> & block);

  /**
   * Temporarily creates an MlirToJlmConverter that is used to convert an MLIR block to an RVSDG
   * graph.
   * \param block The RVSDG MLIR block to be converted.
   * \return The converted RVSDG graph.
   */
  static std::unique_ptr<llvm::LlvmRvsdgModule>
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
  ::llvm::SmallVector<jlm::rvsdg::Output *>
  ConvertRegion(::mlir::Region & region, rvsdg::Region & rvsdgRegion);

  /**
   * Converts the MLIR block and all operations in it
   * \param block The MLIR block to the converted
   * \param rvsdgRegion The corresponding RVSDG region that will be populated with all the contents
   * of the MLIR region.
   * \return The results of the region are returned as a std::vector
   */
  ::llvm::SmallVector<jlm::rvsdg::Output *>
  ConvertBlock(::mlir::Block & block, rvsdg::Region & rvsdgRegion);

  /**
   * Retreive the previously converted RVSDG ouputs from the map of operations
   * and return them in the inputs vector.
   * \param mlirOp The MLIR operation that the inputs are retrieved for.
   * \param outputMap The map of operations that have been converted.
   * argument). \return The vector that is populated with the inputs.
   */
  static ::llvm::SmallVector<jlm::rvsdg::Output *>
  GetConvertedInputs(
      ::mlir::Operation & mlirOp,
      const std::unordered_map<void *, rvsdg::Output *> & outputMap);

  /**
   * Converts an MLIR arith integer comparison operation into an RVSDG node.
   * \param CompOp The MLIR comparison operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \param nbits The number of bits in the comparison.
   * \result The converted RVSDG node.
   */
  rvsdg::Node *
  ConvertCmpIOp(
      ::mlir::arith::CmpIOp & CompOp,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs,
      size_t nbits);

  /**
   * Converts an MLIR LLVM integer comparison operation into an RVSDG node.
   * \param operation The MLIR comparison operation to be converted.
   * \param rvsdgRegion The RVSDG region that the generated RVSDG node is inserted into.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node.
   */
  rvsdg::Node *
  ConvertICmpOp(
      ::mlir::LLVM::ICmpOp & operation,
      rvsdg::Region & rvsdgRegion,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs);

  /**
   * Converts an MLIR floating point binary operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node OR nullptr if the operation cannot be casted to an operation
   */
  rvsdg::Node *
  ConvertFPBinaryNode(
      const ::mlir::Operation & mlirOperation,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs);

  /**
   * Converts a floating point compare predicate to jlm::llvm::fpcmp.
   * \param op the predicate.
   * \result The corresponding fpcmp.
   */
  jlm::llvm::fpcmp
  TryConvertFPCMP(const ::mlir::arith::CmpFPredicate & op);

  /**
   * Converts an MLIR integer binary operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted RVSDG node OR nullptr if the operation cannot be casted to an operation
   */
  rvsdg::Node *
  ConvertBitBinaryNode(
      ::mlir::Operation & mlirOperation,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs);

  /**
   * Converts an MLIR operation into an RVSDG node.
   * \param mlirOperation The MLIR operation to be converted.
   * \param rvsdgRegion The RVSDG region that the generated RVSDG node is inserted into.
   * \param inputs The inputs for the RVSDG node.
   * \result The outputs of the RVSDG node.
   */
  std::vector<jlm::rvsdg::Output *>
  ConvertOperation(
      ::mlir::Operation & mlirOperation,
      rvsdg::Region & rvsdgRegion,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs);

  /**
   * Converts a floating point size to jlm::llvm::fpsize.
   * \param size unsinged int representing the size.
   * \result The fpsize.
   */
  llvm::fpsize
  ConvertFPSize(unsigned int size);

  /**
   * Converts a string representing a linkage to jlm::llvm::linkage.
   * \param stringValue The string to be converted.
   * \result The linkage.
   */
  llvm::Linkage
  ConvertLinkage(std::string stringValue);

  /**
   * Converts an MLIR omega operation and insterst it into an RVSDG region.
   * \param omegaNode The MLIR omega opeation to the converted
   * \return The converted RVSDG graph.
   */
  std::unique_ptr<llvm::LlvmRvsdgModule>
  ConvertOmega(::mlir::rvsdg::OmegaNode & omegaNode);

  /**
   * Converts an MLIR lambda operation and inserts it into an RVSDG region.
   * \param mlirLambda The MLIR lambda opeation to the converted
   * \param rvsdgRegion The RVSDG region that the lambda node will reside in.
   * \param inputs The inputs for the RVSDG node.
   * \result The converted Lambda node.
   */
  rvsdg::Node *
  ConvertLambda(
      ::mlir::Operation & mlirLambda,
      rvsdg::Region & rvsdgRegion,
      const ::llvm::SmallVector<rvsdg::Output *> & inputs);

  /**
   * Converts an MLIR type into an RVSDG type.
   * \param type The MLIR type to be converted.
   * \result The converted RVSDG type.
   */
  std::shared_ptr<const rvsdg::Type>
  ConvertType(const ::mlir::Type & type);

  /**
   * This function should return an architecture dependent number of bits that is used to represent
   * the MLIR intex type.
   * \result The number of bits used to represent the index type (currently
   * hardcoded to 32 bits for x86)
   */
  static size_t
  GetIndexBitWidth()
  {
    // TODO
    // This should return an architeture dependent size
    // 32-bits are used for x86.

    return 32;
  }

  std::unique_ptr<::mlir::MLIRContext> Context_;
  util::BijectiveMap<::mlir::LLVM::LLVMStructType *, std::shared_ptr<const llvm::StructType>>
      StructTypeMap_;
};

} // namespace jlm::mlir

#endif // JLM_MLIR_FRONTEND_MLIRTOJLMCONVERTER_HPP
