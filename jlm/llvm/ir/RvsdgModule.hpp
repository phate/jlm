/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_RVSDGMODULE_HPP
#define JLM_LLVM_IR_RVSDGMODULE_HPP

#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/llvm/ir/Linkage.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/file.hpp>

namespace jlm::llvm
{

/**
 * Represents an import into the RVSDG of an external entity.
 * It is used to model LLVM module declarations.
 */
class LlvmGraphImport final : public rvsdg::GraphImport
{
  LlvmGraphImport(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::Type> valueType,
      std::shared_ptr<const rvsdg::Type> importedType,
      std::string name,
      Linkage linkage,
      CallingConvention callingConvention,
      const bool isConstant,
      const size_t alignment)
      : GraphImport(graph, importedType, std::move(name)),
        ValueType_(std::move(valueType)),
        ImportedType_(std::move(importedType)),
        Linkage_(std::move(linkage)),
        callingConvention_(callingConvention),
        isConstant_(isConstant),
        alignment_(alignment)
  {}

public:
  [[nodiscard]] size_t
  getAlignment() const noexcept
  {
    return alignment_;
  }

  [[nodiscard]] const Linkage &
  linkage() const noexcept
  {
    return Linkage_;
  }

  /**
   * The calling convention of the imported symbol.
   * Only applies to imported functions, not global variables.
   * @return the calling convention of the imported function.
   */
  [[nodiscard]] const CallingConvention &
  callingConvention() const noexcept
  {
    return callingConvention_;
  }

  [[nodiscard]] bool
  isConstant() const noexcept
  {
    return isConstant_;
  }

  /**
   * Symbols are imported as references to an underlying function or global variable.
   * The value type is the type of this underlying target.
   *
   * @return the underlying type of the imported symbol
   * @see ImportedType
   */
  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::Type> &
  ValueType() const noexcept
  {
    return ValueType_;
  }

  /**
   * Symbols are imported as references to an underlying function or global variable.
   * The imported type is the type of this reference, e.g., a pointer or function type.
   * The imported type is the actual type of the RVSDG region argument.
   *
   * @return the type used to reference the imported symbol.
   * @see ValueType
   */
  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::Type> &
  ImportedType() const noexcept
  {
    return ImportedType_;
  }

  /**
   * Makes a copy of this LlvmGraphImport in the given region.
   * @param region the region to create the copy in. Must be a root region.
   * @param input must be nullptr
   * @return the created copy
   */
  LlvmGraphImport &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const override;

  [[nodiscard]] static LlvmGraphImport &
  create(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::Type> valueType,
      std::shared_ptr<const rvsdg::Type> importedType,
      std::string name,
      Linkage linkage,
      CallingConvention callingConvention,
      const bool isConstant,
      const size_t alignment)
  {
    auto graphImport = new LlvmGraphImport(
        graph,
        std::move(valueType),
        std::move(importedType),
        std::move(name),
        std::move(linkage),
        callingConvention,
        isConstant,
        alignment);
    graph.GetRootRegion().addArgument(std::unique_ptr<RegionArgument>(graphImport));
    return *graphImport;
  }

  /**
   * Creates a new graph import representing a global variable.
   * @param graph the graph whose root region will contain the import as an argument
   * @param valueType the underlying type of the global variable
   * @param importedType the reference type used to access the global variable (e.g. PointerType)
   * @param name the name of the global variable symbol
   * @param linkage the linkage of the global variable symbol
   * @param isConstant true if the global variable is constant memory
   * @param alignment the alignment of the global variable
   */
  [[nodiscard]] static LlvmGraphImport &
  createGlobalImport(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::Type> valueType,
      std::shared_ptr<const rvsdg::Type> importedType,
      std::string name,
      Linkage linkage,
      const bool isConstant,
      const size_t alignment)
  {
    return create(
        graph,
        std::move(valueType),
        std::move(importedType),
        std::move(name),
        std::move(linkage),
        CallingConvention::Default, // Global variables do not have a calling convention
        isConstant,
        alignment);
  }

  /**
   * Creates a new graph import representing an imported function.
   * @param graph the graph whose root region will contain the import as an argument
   * @param functionType the type of the function, which is also the type of the argument
   * @param name the name of the function symbol
   * @param linkage the linkage of the function symbol
   * @param callingConvention the calling convention of the function
   */
  [[nodiscard]] static LlvmGraphImport &
  createFunctionImport(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::FunctionType> functionType,
      std::string name,
      Linkage linkage,
      CallingConvention callingConvention)
  {
    return create(
        graph,
        functionType,
        functionType,
        std::move(name),
        std::move(linkage),
        callingConvention,
        true, // Functions are always considered constants
        1);   // We are not responsible for function alignment
  }

private:
  std::shared_ptr<const rvsdg::Type> ValueType_;
  std::shared_ptr<const rvsdg::Type> ImportedType_;
  llvm::Linkage Linkage_;
  CallingConvention callingConvention_;
  bool isConstant_;
  size_t alignment_;
};

/**
 * An LLVM module utilizing the RVSDG representation.
 */
class LlvmRvsdgModule final : public rvsdg::RvsdgModule
{
public:
  ~LlvmRvsdgModule() noexcept override = default;

  LlvmRvsdgModule(util::FilePath sourceFileName, std::string targetTriple, std::string dataLayout)
      : RvsdgModule(std::move(sourceFileName)),
        DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple))
  {}

  LlvmRvsdgModule(
      util::FilePath sourceFileName,
      std::string targetTriple,
      std::string dataLayout,
      std::unique_ptr<rvsdg::Graph> rvsdg)
      : RvsdgModule(std::move(sourceFileName), std::move(rvsdg)),
        DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple))
  {}

  LlvmRvsdgModule(const LlvmRvsdgModule &) = delete;

  LlvmRvsdgModule(LlvmRvsdgModule &&) = delete;

  LlvmRvsdgModule &
  operator=(const LlvmRvsdgModule &) = delete;

  LlvmRvsdgModule &
  operator=(LlvmRvsdgModule &&) = delete;

  std::unique_ptr<RvsdgModule>
  copy() const override;

  [[nodiscard]] const util::FilePath &
  SourceFileName() const noexcept
  {
    return SourceFilePath().value();
  }

  [[nodiscard]] const std::string &
  TargetTriple() const noexcept
  {
    return TargetTriple_;
  }

  [[nodiscard]] const std::string &
  DataLayout() const noexcept
  {
    return DataLayout_;
  }

  static std::unique_ptr<LlvmRvsdgModule>
  Create(
      const util::FilePath & sourceFileName,
      const std::string & targetTriple,
      const std::string & dataLayout)
  {
    return std::make_unique<LlvmRvsdgModule>(sourceFileName, targetTriple, dataLayout);
  }

private:
  std::string DataLayout_;
  std::string TargetTriple_;
};

}

#endif
