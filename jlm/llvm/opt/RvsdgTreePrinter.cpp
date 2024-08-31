/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/util/Statistics.hpp>

#include <fstream>

namespace jlm::llvm
{

class RvsdgTreePrinter::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(util::Statistics::Id::RvsdgTreePrinter, sourceFile)
  {}

  void
  Start() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  Stop() noexcept
  {
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

RvsdgTreePrinter::~RvsdgTreePrinter() noexcept = default;

void
RvsdgTreePrinter::run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
  statistics->Start();

  auto tree = rvsdg::region::ToTree(*rvsdgModule.Rvsdg().root());
  WriteTreeToFile(rvsdgModule, tree);

  statistics->Stop();
}

void
RvsdgTreePrinter::run(RvsdgModule & rvsdgModule)
{
  util::StatisticsCollector collector;
  run(rvsdgModule, collector);
}

void
RvsdgTreePrinter::WriteTreeToFile(const RvsdgModule & rvsdgModule, const std::string & tree) const
{
  auto outputFile = CreateOutputFile(rvsdgModule);

  outputFile.open("w");
  fprintf(outputFile.fd(), "%s\n", tree.c_str());
  outputFile.close();
}

util::file
RvsdgTreePrinter::CreateOutputFile(const RvsdgModule & rvsdgModule) const
{
  auto fileName = util::strfmt(
      Configuration_.OutputDirectory().to_str(),
      "/",
      rvsdgModule.SourceFileName().base().c_str(),
      "-rvsdgTree-",
      GetOutputFileNameCounter(rvsdgModule));
  return util::filepath(fileName);
}

uint64_t
RvsdgTreePrinter::GetOutputFileNameCounter(const RvsdgModule & rvsdgModule)
{
  static std::unordered_map<const RvsdgModule *, uint64_t> RvsdgModuleCounterMap_;

  return RvsdgModuleCounterMap_[&rvsdgModule]++;
}

}
