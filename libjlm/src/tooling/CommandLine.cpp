/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandLine.hpp>

#include <unordered_map>

namespace jlm
{

CommandLineOptions::~CommandLineOptions()
= default;

std::string
JlcCommandLineOptions::ToString(const OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<OptimizationLevel, const char*>
    map({
          {OptimizationLevel::O0, "O0"},
          {OptimizationLevel::O1, "O1"},
          {OptimizationLevel::O2, "O2"},
          {OptimizationLevel::O3, "O3"},
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::string
JlcCommandLineOptions::ToString(const LanguageStandard & languageStandard)
{
  static std::unordered_map<LanguageStandard, const char*>
    map({
          {LanguageStandard::None, ""},
          {LanguageStandard::Gnu89, "gnu89"},
          {LanguageStandard::Gnu99, "gnu99"},
          {LanguageStandard::C89, "c89"},
          {LanguageStandard::C99, "c99"},
          {LanguageStandard::C11, "c11"},
          {LanguageStandard::Cpp98, "c++98"},
          {LanguageStandard::Cpp03, "c++03"},
          {LanguageStandard::Cpp11, "c++11"},
          {LanguageStandard::Cpp14, "c++14"}
        });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

void
JlcCommandLineOptions::Reset() noexcept
{
  OnlyPrintCommands_ = false;
  GenerateDebugInformation_ = false;
  Verbose_ = false;
  Rdynamic_ = false;
  Suppress_ = false;
  UsePthreads_ = false;

  Md_ = false;

  OptimizationLevel_ = OptimizationLevel::O0;
  LanguageStandard_ = LanguageStandard::None;

  OutputFile_ = filepath("a.out");
  Libraries_.clear();
  MacroDefinitions_.clear();
  LibraryPaths_.clear();
  Warnings_.clear();
  IncludePaths_.clear();
  Flags_.clear();
  JlmOptOptimizations_.clear();

  Compilations_.clear();
}

}