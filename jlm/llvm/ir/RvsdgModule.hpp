/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_RVSDGMODULE_HPP
#define JLM_LLVM_IR_RVSDGMODULE_HPP

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
      bool isConstant)
      : rvsdg::GraphImport(graph, importedType, std::move(name)),
        ValueType_(std::move(valueType)),
        ImportedType_(std::move(importedType)),
        Linkage_(std::move(linkage)),
        isConstant_(isConstant)
  {}

public:
  [[nodiscard]] const Linkage &
  linkage() const noexcept
  {
    return Linkage_;
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

  LlvmGraphImport &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const override;

  static LlvmGraphImport &
  Create(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::Type> valueType,
      std::shared_ptr<const rvsdg::Type> importedType,
      std::string name,
      Linkage linkage,
      bool isConstant = false)
  {
    auto graphImport = new LlvmGraphImport(
        graph,
        std::move(valueType),
        std::move(importedType),
        std::move(name),
        std::move(linkage),
        isConstant);
    graph.GetRootRegion().addArgument(std::unique_ptr<RegionArgument>(graphImport));
    return *graphImport;
  }

private:
  std::shared_ptr<const rvsdg::Type> ValueType_;
  std::shared_ptr<const rvsdg::Type> ImportedType_;
  llvm::Linkage Linkage_;
  bool isConstant_;
};

/**
 * An LLVM module utilizing the RVSDG representation.
 */
class LlvmRvsdgModule final : public rvsdg::RvsdgModule
{
public:
  ~LlvmRvsdgModule() noexcept override = default;

  LlvmRvsdgModule(util::FilePath sourceFileName, std::string targetTriple, std::string dataLayout)
      : rvsdg::RvsdgModule(std::move(sourceFileName)),
        DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple))
  {}

  LlvmRvsdgModule(const LlvmRvsdgModule &) = delete;

  LlvmRvsdgModule(LlvmRvsdgModule &&) = delete;

  LlvmRvsdgModule &
  operator=(const LlvmRvsdgModule &) = delete;

  LlvmRvsdgModule &
  operator=(LlvmRvsdgModule &&) = delete;

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
