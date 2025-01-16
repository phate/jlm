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
  virtual ~RvsdgModule() noexcept = default;

  RvsdgModule() = default;

  explicit RvsdgModule(util::filepath sourceFilePath)
      : SourceFilePath_(std::move(sourceFilePath))
  {}

  RvsdgModule(const RvsdgModule &) = delete;

  RvsdgModule(RvsdgModule &&) = delete;

  RvsdgModule &
  operator=(const RvsdgModule &) = delete;

  RvsdgModule &
  operator=(RvsdgModule &&) = delete;

  /**
   *
   * @return The RVSDG of the module.
   */
  jlm::rvsdg::Graph &
  Rvsdg() noexcept
  {
    return Rvsdg_;
  }

  /**
   *
   * @return The RVSDG of the module.
   */
  [[nodiscard]] const Graph &
  Rvsdg() const noexcept
  {
    return Rvsdg_;
  }

  [[nodiscard]] const std::optional<util::filepath> &
  SourceFilePath() const noexcept
  {
    return SourceFilePath_;
  }

private:
  Graph Rvsdg_;
  std::optional<util::filepath> SourceFilePath_{};
};

}

#endif // JLM_RVSDG_RVSDGMODULE_HPP
