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

// FIXME: add documentation
class RvsdgTreePrinter final : public optimization
{
  class Statistics;

public:
  // FIXME: add documentation
  class Configuration final
  {
  public:
    explicit Configuration(const util::filepath & outputDirectory)
        : OutputDirectory_(std::move(outputDirectory))
    {
      JLM_ASSERT(outputDirectory.IsDirectory());
      JLM_ASSERT(outputDirectory.Exists());
    }

    // FIXME: add documentation
    [[nodiscard]] const util::filepath &
    OutputDirectory() const noexcept
    {
      return OutputDirectory_;
    }

    // FIXME: add documentation
    static Configuration
    DefaultConfiguration()
    {
      util::filepath outputDirectory("/tmp");
      return Configuration(std::move(outputDirectory));
    }

  private:
    util::filepath OutputDirectory_;
  };

  ~RvsdgTreePrinter() noexcept override;

  RvsdgTreePrinter() = default;

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

  // FIXME: add documentation
  void
  SetConfiguration(Configuration configuration);

private:
  void
  WriteTreeToFile(const RvsdgModule & rvsdgModule, const std::string & tree) const;

  [[nodiscard]] util::file
  CreateOutputFile(const RvsdgModule & rvsdgModule) const;

  static uint64_t
  GetOutputFileNameCounter(const RvsdgModule & rvsdgModule);

  Configuration Configuration_ = Configuration::DefaultConfiguration();
};

}

#endif
