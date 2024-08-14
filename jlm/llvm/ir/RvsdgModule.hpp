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

// FIXME: add documentation
class GraphImport final : public rvsdg::GraphImport
{
private:
  GraphImport(
      rvsdg::graph & graph,
      std::shared_ptr<const rvsdg::valuetype> valueType,
      std::string name,
      llvm::linkage linkage)
      : rvsdg::GraphImport(graph, PointerType::Create(), std::move(name)),
        Linkage_(std::move(linkage)),
        ValueType_(std::move(valueType))
  {}

public:
  [[nodiscard]] const linkage &
  Linkage() const noexcept
  {
    return Linkage_;
  }

  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::valuetype> &
  ValueType() const noexcept
  {
    return ValueType_;
  }

  static GraphImport &
  Create(
      rvsdg::graph & graph,
      std::shared_ptr<const rvsdg::valuetype> valueType,
      std::string name,
      llvm::linkage linkage)
  {
    auto graphImport =
        new GraphImport(graph, std::move(valueType), std::move(name), std::move(linkage));
    graph.root()->append_argument(graphImport);
    return *graphImport;
  }

private:
  llvm::linkage Linkage_;
  std::shared_ptr<const rvsdg::valuetype> ValueType_;
};

static inline bool
is_export(const jlm::rvsdg::input * input)
{
  auto graph = input->region()->graph();

  auto result = dynamic_cast<const jlm::rvsdg::result *>(input);
  return result && result->region() == graph->root();
}

/**
 * An LLVM module utilizing the RVSDG representation.
 */
class RvsdgModule final : public rvsdg::RvsdgModule
{
public:
  ~RvsdgModule() noexcept override = default;

  RvsdgModule(jlm::util::filepath sourceFileName, std::string targetTriple, std::string dataLayout)
      : RvsdgModule(std::move(sourceFileName), std::move(targetTriple), std::move(dataLayout), {})
  {}

  RvsdgModule(
      jlm::util::filepath sourceFileName,
      std::string targetTriple,
      std::string dataLayout,
      std::vector<std::unique_ptr<StructType::Declaration>> declarations)
      : DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple)),
        SourceFileName_(std::move(sourceFileName)),
        StructTypeDeclarations_(std::move(declarations))
  {}

  RvsdgModule(const RvsdgModule &) = delete;

  RvsdgModule(RvsdgModule &&) = delete;

  RvsdgModule &
  operator=(const RvsdgModule &) = delete;

  RvsdgModule &
  operator=(RvsdgModule &&) = delete;

  [[nodiscard]] const jlm::util::filepath &
  SourceFileName() const noexcept
  {
    return SourceFileName_;
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
      const jlm::util::filepath & sourceFileName,
      const std::string & targetTriple,
      const std::string & dataLayout)
  {
    return Create(sourceFileName, targetTriple, dataLayout, {});
  }

  static std::unique_ptr<RvsdgModule>
  Create(
      const jlm::util::filepath & sourceFileName,
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
  const jlm::util::filepath SourceFileName_;
  std::vector<std::unique_ptr<StructType::Declaration>> StructTypeDeclarations_;
};

}

#endif
