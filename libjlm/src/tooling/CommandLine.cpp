/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandLine.hpp>

#include <llvm/Support/CommandLine.h>

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

CommandLineParser::~CommandLineParser() noexcept
= default;

JlcCommandLineParser::~JlcCommandLineParser() noexcept
= default;

const JlcCommandLineOptions &
JlcCommandLineParser::ParseCommandLineArguments(int argc, char **argv)
{
  CommandLineOptions_.Reset();

  using namespace llvm;

  /*
    FIXME: The command line parser setup is currently redone
    for every invocation of parse_cmdline. We should be able
    to do it only once and then reset the parser on every
    invocation of parse_cmdline.
  */

  cl::TopLevelSubCommand->reset();

  cl::opt<bool> onlyPrintCommands(
    "###",
    cl::ValueDisallowed,
    cl::desc("Print (but do not run) the commands for this compilation."));

  cl::list<std::string> inputFiles(
    cl::Positional,
    cl::desc("<inputs>"));

  cl::list<std::string> includePaths(
    "I",
    cl::Prefix,
    cl::desc("Add directory <dir> to include search paths."),
    cl::value_desc("dir"));

  cl::list<std::string> libraryPaths(
    "L",
    cl::Prefix,
    cl::desc("Add directory <dir> to library search paths."),
    cl::value_desc("dir"));

  cl::list<std::string> libraries(
    "l",
    cl::Prefix,
    cl::desc("Search the library <lib> when linking."),
    cl::value_desc("lib"));

  cl::opt<std::string> outputFile(
    "o",
    cl::desc("Write output to <file>."),
    cl::value_desc("file"));

  cl::opt<bool> generateDebugInformation(
    "g",
    cl::ValueDisallowed,
    cl::desc("Generate source-level debug information."));

  cl::opt<bool> noLinking(
    "c",
    cl::ValueDisallowed,
    cl::desc("Only run preprocess, compile, and assemble steps."));

  cl::opt<std::string> optimizationLevel(
    "O",
    cl::Prefix,
    cl::ValueOptional,
    cl::desc("Optimization level. [O0, O1, O2, O3]"),
    cl::value_desc("#"));

  cl::list<std::string> macroDefinitions(
    "D",
    cl::Prefix,
    cl::desc("Add <macro> to preprocessor macros."),
    cl::value_desc("macro"));

  cl::list<std::string> warnings(
    "W",
    cl::Prefix,
    cl::desc("Enable specified warning."),
    cl::value_desc("warning"));

  cl::opt<std::string> languageStandard(
    "std",
    cl::desc("Language standard."),
    cl::value_desc("standard"));

  cl::list<std::string> flags(
    "f",
    cl::Prefix,
    cl::desc("Specify flags."),
    cl::value_desc("flag"));

  cl::list<std::string> jlmOptimizations(
    "J",
    cl::Prefix,
    cl::desc("jlm-opt optimization. Run 'jlm-opt -help' for viable options."),
    cl::value_desc("jlmopt"));

  cl::opt<bool> verbose(
    "v",
    cl::ValueDisallowed,
    cl::desc("Show commands to run and use verbose output. (Affects only clang for now)"));

  cl::opt<bool> rDynamic(
    "rdynamic",
    cl::ValueDisallowed,
    cl::desc("rDynamic option passed to clang"));

  cl::opt<bool> suppress(
    "w",
    cl::ValueDisallowed,
    cl::desc("Suppress all warnings"));

  cl::opt<bool> usePthreads(
    "pthread",
    cl::ValueDisallowed,
    cl::desc("Support POSIX threads in generated code"));

  cl::opt<bool> mD(
    "MD",
    cl::ValueDisallowed,
    cl::desc("Write a depfile containing user and system headers"));

  cl::opt<std::string> mF(
    "MF",
    cl::desc("Write depfile output from -mD to <file>."),
    cl::value_desc("file"));

  cl::opt<std::string> mT(
    "MT",
    cl::desc("Specify name of main file output in depfile."),
    cl::value_desc("value"));

  cl::ParseCommandLineOptions(argc, argv);

  /* Process parsed options */

  static std::unordered_map<std::string, jlm::JlcCommandLineOptions::OptimizationLevel>
    optimizationLevelMap({
              {"0", JlcCommandLineOptions::OptimizationLevel::O0},
              {"1", JlcCommandLineOptions::OptimizationLevel::O1},
              {"2", JlcCommandLineOptions::OptimizationLevel::O2},
              {"3", JlcCommandLineOptions::OptimizationLevel::O3}
            });

  static std::unordered_map<std::string, jlm::JlcCommandLineOptions::LanguageStandard>
    languageStandardMap({
             {"gnu89", jlm::JlcCommandLineOptions::LanguageStandard::Gnu89},
             {"gnu99", jlm::JlcCommandLineOptions::LanguageStandard::Gnu99},
             {"c89",   jlm::JlcCommandLineOptions::LanguageStandard::C89},
             {"c90",   jlm::JlcCommandLineOptions::LanguageStandard::C99},
             {"c99",   jlm::JlcCommandLineOptions::LanguageStandard::C99},
             {"c11",   jlm::JlcCommandLineOptions::LanguageStandard::C11},
             {"c++98", jlm::JlcCommandLineOptions::LanguageStandard::Cpp98},
             {"c++03", jlm::JlcCommandLineOptions::LanguageStandard::Cpp03},
             {"c++11", jlm::JlcCommandLineOptions::LanguageStandard::Cpp11},
             {"c++14", jlm::JlcCommandLineOptions::LanguageStandard::Cpp14}
           });

  if (!optimizationLevel.empty()) {
    auto iterator = optimizationLevelMap.find(optimizationLevel);
    if (iterator == optimizationLevelMap.end()) {
      std::cerr << "Unknown optimization level.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.OptimizationLevel_ = iterator->second;
  }

  if (!languageStandard.empty()) {
    auto iterator = languageStandardMap.find(languageStandard);
    if (iterator == languageStandardMap.end()) {
      std::cerr << "Unknown language standard.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.LanguageStandard_ = iterator->second;
  }

  if (inputFiles.empty()) {
    std::cerr << "jlc: no input files.\n";
    exit(EXIT_FAILURE);
  }

  if (inputFiles.size() > 1 && noLinking && !outputFile.empty()) {
    std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
    exit(EXIT_FAILURE);
  }

  CommandLineOptions_.Libraries_ = libraries;
  CommandLineOptions_.MacroDefinitions_ = macroDefinitions;
  CommandLineOptions_.LibraryPaths_ = libraryPaths;
  CommandLineOptions_.Warnings_ = warnings;
  CommandLineOptions_.IncludePaths_ = includePaths;
  CommandLineOptions_.OnlyPrintCommands_ = onlyPrintCommands;
  CommandLineOptions_.GenerateDebugInformation_ = generateDebugInformation;
  CommandLineOptions_.Flags_ = flags;
  CommandLineOptions_.JlmOptOptimizations_ = jlmOptimizations;
  CommandLineOptions_.Verbose_ = verbose;
  CommandLineOptions_.Rdynamic_ = rDynamic;
  CommandLineOptions_.Suppress_ = suppress;
  CommandLineOptions_.UsePthreads_ = usePthreads;
  CommandLineOptions_.Md_ = mD;

  for (auto & inputFile : inputFiles) {
    if (IsObjectFile(inputFile)) {
      /* FIXME: print a warning like clang if noLinking is true */
      CommandLineOptions_.Compilations_.push_back({
                                                    inputFile,
                                                    jlm::filepath(""),
                                                    inputFile,
                                                    "",
                                                    false,
                                                    false,
                                                    false,
                                                    true});

      continue;
    }

    CommandLineOptions_.Compilations_.push_back({
                                                  inputFile,
                                                  mF.empty() ? ToDependencyFile(inputFile) : jlm::filepath(mF),
                                                  ToObjectFile(inputFile),
                                                  mT.empty() ? ToObjectFile(inputFile).name() : mT,
                                                  true,
                                                  true,
                                                  true,
                                                  !noLinking});
  }

  if (!outputFile.empty()) {
    if (noLinking) {
      JLM_ASSERT(CommandLineOptions_.Compilations_.size() == 1);
      CommandLineOptions_.Compilations_[0].SetOutputFile(outputFile);
    } else {
      CommandLineOptions_.OutputFile_ = filepath(outputFile);
    }
  }

  return CommandLineOptions_;
}

}