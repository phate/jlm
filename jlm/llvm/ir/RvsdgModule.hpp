/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_RVSDGMODULE_HPP
#define JLM_LLVM_IR_RVSDGMODULE_HPP

#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/util/file.hpp>

namespace jlm::llvm
{

/* impport class */

class impport final : public jlm::rvsdg::impport
{
public:
  virtual ~impport();

  impport(const jlm::rvsdg::valuetype & valueType, const std::string & name, const linkage & lnk)
      : jlm::rvsdg::impport(PointerType(), name),
        linkage_(lnk),
        ValueType_(valueType.copy())
  {}

  impport(const impport & other)
      : jlm::rvsdg::impport(other),
        linkage_(other.linkage_),
        ValueType_(other.ValueType_->copy())
  {}

  impport(impport && other)
      : jlm::rvsdg::impport(other),
        linkage_(std::move(other.linkage_)),
        ValueType_(std::move(other.ValueType_))
  {}

  impport &
  operator=(const impport &) = delete;

  impport &
  operator=(impport &&) = delete;

  const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] const jlm::rvsdg::valuetype &
  GetValueType() const noexcept
  {
    return *jlm::util::AssertedCast<jlm::rvsdg::valuetype>(ValueType_.get());
  }

  virtual bool
  operator==(const port &) const noexcept override;

  virtual std::unique_ptr<port>
  copy() const override;

private:
  jlm::llvm::linkage linkage_;
  std::unique_ptr<jlm::rvsdg::type> ValueType_;
};

static inline bool
is_import(const jlm::rvsdg::output * output)
{
  auto graph = output->region()->graph();

  auto argument = dynamic_cast<const jlm::rvsdg::argument *>(output);
  return argument && argument->region() == graph->root();
}

static inline bool
is_export(const jlm::rvsdg::input * input)
{
  auto graph = input->region()->graph();

  auto result = dynamic_cast<const jlm::rvsdg::result *>(input);
  return result && result->region() == graph->root();
}

/** \brief RVSDG module class
 *
 */
class RvsdgModule final
{
public:
  RvsdgModule(jlm::util::filepath sourceFileName, std::string targetTriple, std::string dataLayout)
      : DataLayout_(std::move(dataLayout)),
        TargetTriple_(std::move(targetTriple)),
        SourceFileName_(std::move(sourceFileName))
  {}

  RvsdgModule(const RvsdgModule &) = delete;

  RvsdgModule(RvsdgModule &&) = delete;

  RvsdgModule &
  operator=(const RvsdgModule &) = delete;

  RvsdgModule &
  operator=(RvsdgModule &&) = delete;

  jlm::rvsdg::graph &
  Rvsdg() noexcept
  {
    return Rvsdg_;
  }

  const jlm::rvsdg::graph &
  Rvsdg() const noexcept
  {
    return Rvsdg_;
  }

  const jlm::util::filepath &
  SourceFileName() const noexcept
  {
    return SourceFileName_;
  }

  const std::string &
  TargetTriple() const noexcept
  {
    return TargetTriple_;
  }

  const std::string &
  DataLayout() const noexcept
  {
    return DataLayout_;
  }

  static std::unique_ptr<RvsdgModule>
  Create(
      const jlm::util::filepath & sourceFileName,
      const std::string & targetTriple,
      const std::string & dataLayout)
  {
    return std::make_unique<RvsdgModule>(sourceFileName, targetTriple, dataLayout);
  }

private:
  jlm::rvsdg::graph Rvsdg_;
  std::string DataLayout_;
  std::string TargetTriple_;
  const jlm::util::filepath SourceFileName_;
};

}

#endif
