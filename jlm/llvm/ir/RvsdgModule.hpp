/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_RVSDGMODULE_HPP
#define JLM_LLVM_IR_RVSDGMODULE_HPP

#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/file.hpp>

namespace jlm::llvm
{

/**
 * Represents an import into the RVSDG of an external entity.
 * It is used to model LLVM module declarations.
 */
class GraphImport final : public rvsdg::GraphImport
{
private:
  GraphImport(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::ValueType> valueType,
      std::shared_ptr<const rvsdg::ValueType> importedType,
      std::string name,
      llvm::linkage linkage)
      : rvsdg::GraphImport(graph, importedType, std::move(name)),
        Linkage_(std::move(linkage)),
        ValueType_(std::move(valueType)),
        ImportedType_(std::move(importedType))
  {}

public:
  [[nodiscard]] const linkage &
  Linkage() const noexcept
  {
    return Linkage_;
  }

  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::ValueType> &
  ValueType() const noexcept
  {
    return ValueType_;
  }

  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::ValueType> &
  ImportedType() const noexcept
  {
    return ImportedType_;
  }

  GraphImport &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

  static GraphImport &
  Create(
      rvsdg::Graph & graph,
      std::shared_ptr<const rvsdg::ValueType> valueType,
      std::shared_ptr<const rvsdg::ValueType> importedType,
      std::string name,
      llvm::linkage linkage)
  {
    auto graphImport = new GraphImport(
        graph,
        std::move(valueType),
        std::move(importedType),
        std::move(name),
        std::move(linkage));
    graph.GetRootRegion().append_argument(graphImport);
    return *graphImport;
  }

private:
  llvm::linkage Linkage_;
  std::shared_ptr<const rvsdg::ValueType> ValueType_;
  std::shared_ptr<const rvsdg::ValueType> ImportedType_;
};

#if 0
/**
 * Represents an export from the RVSDG of an internal entity.
 * It is used to model externally visible entities from LLVM modules.
 */
class GraphExport final : public rvsdg::GraphExport
{
private:
  GraphExport(rvsdg::Output & origin, std::string name)
      : rvsdg::GraphExport(origin, std::move(name))
  {}

public:
  GraphExport &
  Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) override;

  static GraphExport &
  Create(rvsdg::Output & origin, std::string name)
  {
    auto graphExport = new GraphExport(origin, std::move(name));
    origin.region()->graph()->GetRootRegion().append_result(graphExport);
    return *graphExport;
  }
};
#endif

/**
 * An LLVM module utilizing the RVSDG representation.
 */
class RvsdgModule final : public rvsdg::RvsdgModule
{
public:
  ~RvsdgModule() noexcept override = default;

  RvsdgModule(jlm::util::FilePath sourceFileName, std::string targetTriple, std::string dataLayout)
      : RvsdgModule(std::move(sourceFileName), std::move(targetTriple), std::move(dataLayout), {})
  {}

  RvsdgModule(
      util::FilePath sourceFileName,
      std::string targetTriple,
      std::string dataLayout,
      std::vector<std::unique_ptr<StructType::Declaration>> declarations)
      : rvsdg::RvsdgModule(std::move(sourceFileName)),
        DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple)),
        StructTypeDeclarations_(std::move(declarations))
  {}

  RvsdgModule(const RvsdgModule &) = delete;

  RvsdgModule(RvsdgModule &&) = delete;

  RvsdgModule &
  operator=(const RvsdgModule &) = delete;

  RvsdgModule &
  operator=(RvsdgModule &&) = delete;

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

  /**
   * Adds a struct type declaration to the module. The module becomes the owner of the declaration.
   *
   * @param declaration A declaration that is added to the module.
   * @return A reference to the added documentation.
   */
  const StructType::Declaration &
  AddStructTypeDeclaration(std::unique_ptr<StructType::Declaration> declaration)
  {
    StructTypeDeclarations_.emplace_back(std::move(declaration));
    return *StructTypeDeclarations_.back();
  }

  /**
   * Releases all struct type declarations from the module to the caller. The caller is the new
   * owner of the declarations.
   *
   * @return A vector of declarations.
   */
  std::vector<std::unique_ptr<StructType::Declaration>> &&
  ReleaseStructTypeDeclarations()
  {
    return std::move(StructTypeDeclarations_);
  }

  static std::unique_ptr<RvsdgModule>
  Create(
      const jlm::util::FilePath & sourceFileName,
      const std::string & targetTriple,
      const std::string & dataLayout)
  {
    return Create(sourceFileName, targetTriple, dataLayout, {});
  }

  static std::unique_ptr<RvsdgModule>
  Create(
      const jlm::util::FilePath & sourceFileName,
      const std::string & targetTriple,
      const std::string & dataLayout,
      std::vector<std::unique_ptr<StructType::Declaration>> declarations)
  {
    return std::make_unique<RvsdgModule>(
        sourceFileName,
        targetTriple,
        dataLayout,
        std::move(declarations));
  }

private:
  std::string DataLayout_;
  std::string TargetTriple_;
  std::vector<std::unique_ptr<StructType::Declaration>> StructTypeDeclarations_;
};

}

#endif
