/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_RVSDGMODULE_HPP
#define JLM_RVSDG_RVSDGMODULE_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/util/file.hpp>
#include <optional>

namespace jlm::rvsdg
{

/**
 * Top-level class for a module with an RVSDG.
 */
class RvsdgModule
{
public:
  virtual ~RvsdgModule() noexcept;

  RvsdgModule() = default;

  explicit RvsdgModule(util::FilePath sourceFilePath)
      : rvsdg_(new Graph()),
        SourceFilePath_(std::move(sourceFilePath))
  {}

  RvsdgModule(util::FilePath sourceFilePath, std::unique_ptr<Graph> rvsdg)
      : rvsdg_(std::move(rvsdg)),
        SourceFilePath_(std::move(sourceFilePath))
  {}

  RvsdgModule(const RvsdgModule &) = delete;

  RvsdgModule(RvsdgModule &&) = delete;

  RvsdgModule &
  operator=(const RvsdgModule &) = delete;

  RvsdgModule &
  operator=(RvsdgModule &&) = delete;

  /**
   * @return A copy of the instance.
   */
  [[nodiscard]] virtual std::unique_ptr<RvsdgModule>
  copy() const;

  /**
   *
   * @return The RVSDG of the module.
   */
  Graph &
  Rvsdg() noexcept
  {
    return *rvsdg_;
  }

  /**
   *
   * @return The RVSDG of the module.
   */
  [[nodiscard]] const Graph &
  Rvsdg() const noexcept
  {
    return *rvsdg_;
  }

  [[nodiscard]] const std::optional<util::FilePath> &
  SourceFilePath() const noexcept
  {
    return SourceFilePath_;
  }

private:
  std::unique_ptr<Graph> rvsdg_{};
  std::optional<util::FilePath> SourceFilePath_{};
};

}

#endif // JLM_RVSDG_RVSDGMODULE_HPP
