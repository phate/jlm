/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP
#define JLM_LLVM_OPT_RVSDGTREEPRINTER_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/util/file.hpp>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

class RvsdgModule;

/** \brief RVSDG tree printer debug pass
 *
 * Prints an RVSDG tree to a file.
 */
class RvsdgTreePrinter final : public optimization
{
  class Statistics;

public:
  /**
   * Configuration for the \ref RvsdgTreePrinter.
   */
  class Configuration final
  {
  public:
    explicit Configuration(const util::filepath & outputDirectory)
        : OutputDirectory_(std::move(outputDirectory))
    {
      JLM_ASSERT(outputDirectory.IsDirectory());
      JLM_ASSERT(outputDirectory.Exists());
    }

    /**
     * The output directory for the RVSDG tree files.
     */
    [[nodiscard]] const util::filepath &
    OutputDirectory() const noexcept
    {
      return OutputDirectory_;
    }

  private:
    util::filepath OutputDirectory_;
  };

  ~RvsdgTreePrinter() noexcept override;

  explicit RvsdgTreePrinter(Configuration configuration)
      : Configuration_(std::move(configuration))
  {}

  RvsdgTreePrinter(const RvsdgTreePrinter &) = delete;

  RvsdgTreePrinter(RvsdgTreePrinter &&) = delete;

  RvsdgTreePrinter &
  operator=(const RvsdgTreePrinter &) = delete;

  RvsdgTreePrinter &
  operator=(RvsdgTreePrinter &&) = delete;

  void
  run(RvsdgModule & rvsdgModule, jlm::util::StatisticsCollector & statisticsCollector) override;

  void
  run(RvsdgModule & rvsdgModule);

private:
  void
  WriteTreeToFile(const RvsdgModule & rvsdgModule, const std::string & tree) const;

  [[nodiscard]] util::file
  CreateOutputFile(const RvsdgModule & rvsdgModule) const;

  static uint64_t
  GetOutputFileNameCounter(const RvsdgModule & rvsdgModule);

  Configuration Configuration_;
};

}

#endif
