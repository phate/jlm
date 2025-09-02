/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <llvm/Support/CommandLine.h>

#include <unordered_map>

namespace jlm::tooling
{

CommandLineOptions::~CommandLineOptions() = default;

std::string
JlcCommandLineOptions::ToString(const OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<OptimizationLevel, const char *> map({
      { OptimizationLevel::O0, "O0" },
      { OptimizationLevel::O1, "O1" },
      { OptimizationLevel::O2, "O2" },
      { OptimizationLevel::O3, "O3" },
  });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::string
JlcCommandLineOptions::ToString(const LanguageStandard & languageStandard)
{
  static std::unordered_map<LanguageStandard, const char *> map(
      { { LanguageStandard::None, "" },
        { LanguageStandard::Gnu89, "gnu89" },
        { LanguageStandard::Gnu99, "gnu99" },
        { LanguageStandard::C89, "c89" },
        { LanguageStandard::C99, "c99" },
        { LanguageStandard::C11, "c11" },
        { LanguageStandard::Cpp98, "c++98" },
        { LanguageStandard::Cpp03, "c++03" },
        { LanguageStandard::Cpp11, "c++11" },
        { LanguageStandard::Cpp14, "c++14" } });

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

  OutputFile_ = util::FilePath("a.out");
  Libraries_.clear();
  MacroDefinitions_.clear();
  LibraryPaths_.clear();
  Warnings_.clear();
  IncludePaths_.clear();
  Flags_.clear();
  JlmOptOptimizations_.clear();

  Compilations_.clear();
}

void
JlmOptCommandLineOptions::Reset() noexcept
{
  InputFile_ = util::FilePath("");
  OutputFile_ = util::FilePath("");
  OutputFormat_ = OutputFormat::Llvm;
  StatisticsCollectorSettings_ = util::StatisticsCollectorSettings();
  OptimizationIds_.clear();
}

JlmOptCommandLineOptions::OptimizationId
JlmOptCommandLineOptions::FromCommandLineArgumentToOptimizationId(
    const std::string & commandLineArgument)
{
  static std::unordered_map<std::string, OptimizationId> map(
      { { OptimizationCommandLineArgument::AaAndersenAgnostic_,
          OptimizationId::AAAndersenAgnostic },
        { OptimizationCommandLineArgument::AaAndersenRegionAware_,
          OptimizationId::AAAndersenRegionAware },
        { OptimizationCommandLineArgument::AaAndersenTopDownLifetimeAware_,
          OptimizationId::AAAndersenTopDownLifetimeAware },
        { OptimizationCommandLineArgument::AaSteensgaardAgnostic_,
          OptimizationId::AASteensgaardAgnostic },
        { OptimizationCommandLineArgument::AaSteensgaardRegionAware_,
          OptimizationId::AASteensgaardRegionAware },
        { OptimizationCommandLineArgument::CommonNodeElimination_,
          OptimizationId::CommonNodeElimination },
        { OptimizationCommandLineArgument::DeadNodeElimination_,
          OptimizationId::DeadNodeElimination },
        { OptimizationCommandLineArgument::FunctionInlining_, OptimizationId::FunctionInlining },
        { OptimizationCommandLineArgument::IfConversion_, OptimizationId::IfConversion },
        { OptimizationCommandLineArgument::InvariantValueRedirection_,
          OptimizationId::InvariantValueRedirection },
        { OptimizationCommandLineArgument::NodePushOut_, OptimizationId::NodePushOut },
        { OptimizationCommandLineArgument::NodePullIn_, OptimizationId::NodePullIn },
        { OptimizationCommandLineArgument::NodeReduction_, OptimizationId::NodeReduction },
        { OptimizationCommandLineArgument::RvsdgTreePrinter_, OptimizationId::RvsdgTreePrinter },
        { OptimizationCommandLineArgument::ThetaGammaInversion_,
          OptimizationId::ThetaGammaInversion },
        { OptimizationCommandLineArgument::LoopUnrolling_, OptimizationId::LoopUnrolling } });

  if (map.find(commandLineArgument) != map.end())
    return map[commandLineArgument];

  throw util::Error("Unknown command line argument: " + commandLineArgument);
}

const char *
JlmOptCommandLineOptions::ToCommandLineArgument(OptimizationId optimizationId)
{
  static std::unordered_map<OptimizationId, const char *> map(
      { { OptimizationId::AAAndersenAgnostic,
          OptimizationCommandLineArgument::AaAndersenAgnostic_ },
        { OptimizationId::AAAndersenRegionAware,
          OptimizationCommandLineArgument::AaAndersenRegionAware_ },
        { OptimizationId::AAAndersenTopDownLifetimeAware,
          OptimizationCommandLineArgument::AaAndersenTopDownLifetimeAware_ },
        { OptimizationId::AASteensgaardAgnostic,
          OptimizationCommandLineArgument::AaSteensgaardAgnostic_ },
        { OptimizationId::AASteensgaardRegionAware,
          OptimizationCommandLineArgument::AaSteensgaardRegionAware_ },
        { OptimizationId::CommonNodeElimination,
          OptimizationCommandLineArgument::CommonNodeElimination_ },
        { OptimizationId::DeadNodeElimination,
          OptimizationCommandLineArgument::DeadNodeElimination_ },
        { OptimizationId::FunctionInlining, OptimizationCommandLineArgument::FunctionInlining_ },
        { OptimizationId::IfConversion, OptimizationCommandLineArgument::IfConversion_ },
        { OptimizationId::InvariantValueRedirection,
          OptimizationCommandLineArgument::InvariantValueRedirection_ },
        { OptimizationId::LoopUnrolling, OptimizationCommandLineArgument::LoopUnrolling_ },
        { OptimizationId::NodePullIn, OptimizationCommandLineArgument::NodePullIn_ },
        { OptimizationId::NodePushOut, OptimizationCommandLineArgument::NodePushOut_ },
        { OptimizationId::NodeReduction, OptimizationCommandLineArgument::NodeReduction_ },
        { OptimizationId::RvsdgTreePrinter, OptimizationCommandLineArgument::RvsdgTreePrinter_ },
        { OptimizationId::ThetaGammaInversion,
          OptimizationCommandLineArgument::ThetaGammaInversion_ } });

  if (map.find(optimizationId) != map.end())
    return map[optimizationId];

  throw util::Error("Unknown optimization identifier");
}

util::Statistics::Id
JlmOptCommandLineOptions::FromCommandLineArgumentToStatisticsId(
    const std::string & commandLineArgument)
{
  try
  {
    return GetStatisticsIdCommandLineArguments().LookupValue(commandLineArgument);
  }
  catch (...)
  {
    throw util::Error("Unknown command line argument: " + commandLineArgument);
  }
}

const char *
JlmOptCommandLineOptions::ToCommandLineArgument(util::Statistics::Id statisticsId)
{
  try
  {
    return GetStatisticsIdCommandLineArguments().LookupKey(statisticsId).data();
  }
  catch (...)
  {
    throw util::Error("Unknown statistics identifier");
  }
}

const char *
JlmOptCommandLineOptions::ToCommandLineArgument(InputFormat inputFormat)
{
  static std::unordered_map<InputFormat, const char *> map(
      { { InputFormat::Llvm, "llvm" }, { InputFormat::Mlir, "mlir" } });

  if (map.find(inputFormat) != map.end())
    return map[inputFormat];

  throw util::Error("Unknown input format");
}

const char *
JlmOptCommandLineOptions::ToCommandLineArgument(OutputFormat outputFormat)
{
  auto & mapping = GetOutputFormatCommandLineArguments();
  return mapping.at(outputFormat).data();
}

const util::BijectiveMap<util::Statistics::Id, std::string_view> &
JlmOptCommandLineOptions::GetStatisticsIdCommandLineArguments()
{
  static util::BijectiveMap<util::Statistics::Id, std::string_view> mapping = {
    { util::Statistics::Id::AliasAnalysisPrecisionEvaluation, "print-aa-precision-evaluation" },
    { util::Statistics::Id::Aggregation, "print-aggregation-time" },
    { util::Statistics::Id::AgnosticModRefSummarizer, "print-agnostic-mod-ref-summarization" },
    { util::Statistics::Id::AndersenAnalysis, "print-andersen-analysis" },
    { util::Statistics::Id::Annotation, "print-annotation-time" },
    { util::Statistics::Id::CommonNodeElimination, "print-cne-stat" },
    { util::Statistics::Id::ControlFlowRecovery, "print-cfr-time" },
    { util::Statistics::Id::DataNodeToDelta, "printDataNodeToDelta" },
    { util::Statistics::Id::DeadNodeElimination, "print-dne-stat" },
    { util::Statistics::Id::FunctionInlining, "print-iln-stat" },
    { util::Statistics::Id::IfConversion, "print-if-conversion" },
    { util::Statistics::Id::InvariantValueRedirection, "printInvariantValueRedirection" },
    { util::Statistics::Id::JlmToRvsdgConversion, "print-jlm-rvsdg-conversion" },
    { util::Statistics::Id::LoopUnrolling, "print-unroll-stat" },
    { util::Statistics::Id::MemoryStateEncoder, "print-basicencoder-encoding" },
    { util::Statistics::Id::PullNodes, "print-pull-stat" },
    { util::Statistics::Id::PushNodes, "print-push-stat" },
    { util::Statistics::Id::ReduceNodes, "print-reduction-stat" },
    { util::Statistics::Id::RegionAwareModRefSummarizer, "print-mod-ref-summarization" },
    { util::Statistics::Id::RvsdgConstruction, "print-rvsdg-construction" },
    { util::Statistics::Id::RvsdgDestruction, "print-rvsdg-destruction" },
    { util::Statistics::Id::RvsdgOptimization, "print-rvsdg-optimization" },
    { util::Statistics::Id::RvsdgTreePrinter, "print-rvsdg-tree" },
    { util::Statistics::Id::SteensgaardAnalysis, "print-steensgaard-analysis" },
    { util::Statistics::Id::ThetaGammaInversion, "print-ivt-stat" },
    { util::Statistics::Id::TopDownMemoryNodeEliminator, "TopDownMemoryNodeEliminator" }
  };

  auto firstIndex = static_cast<size_t>(util::Statistics::Id::FirstEnumValue);
  auto lastIndex = static_cast<size_t>(util::Statistics::Id::LastEnumValue);
  JLM_ASSERT(mapping.Size() == lastIndex - firstIndex - 1);
  return mapping;
}

const std::unordered_map<JlmOptCommandLineOptions::OutputFormat, std::string_view> &
JlmOptCommandLineOptions::GetOutputFormatCommandLineArguments()
{
  static std::unordered_map<OutputFormat, std::string_view> mapping = {
    { OutputFormat::Ascii, "ascii" }, { OutputFormat::Dot, "dot" },
    { OutputFormat::Llvm, "llvm" },   { OutputFormat::Mlir, "mlir" },
    { OutputFormat::Tree, "tree" },   { OutputFormat::Xml, "xml" }
  };

  auto firstIndex = static_cast<size_t>(OutputFormat::FirstEnumValue);
  auto lastIndex = static_cast<size_t>(OutputFormat::LastEnumValue);
  JLM_ASSERT(mapping.size() == lastIndex - firstIndex - 1);
  return mapping;
}

void
JlmHlsCommandLineOptions::Reset() noexcept
{
  InputFile_ = util::FilePath("");
  OutputFiles_ = util::FilePath("");
  OutputFormat_ = OutputFormat::Firrtl;
  HlsFunction_ = "";
  ExtractHlsFunction_ = false;
  MemoryLatency_ = 10;
}

void
JhlsCommandLineOptions::Reset() noexcept
{
  *this = JhlsCommandLineOptions();
}

static ::llvm::cl::OptionEnumValue
CreateStatisticsOption(util::Statistics::Id statisticsId, const char * description)
{
  return ::clEnumValN(
      statisticsId,
      JlmOptCommandLineOptions::ToCommandLineArgument(statisticsId),
      description);
}

static ::llvm::cl::OptionEnumValue
CreateOutputFormatOption(
    JlmOptCommandLineOptions::OutputFormat outputFormat,
    const char * description)
{
  return ::clEnumValN(
      outputFormat,
      JlmOptCommandLineOptions::ToCommandLineArgument(outputFormat),
      description);
}

CommandLineParser::~CommandLineParser() noexcept = default;

CommandLineParser::Exception::~Exception() noexcept = default;

JlcCommandLineParser::~JlcCommandLineParser() noexcept = default;

const JlcCommandLineOptions &
JlcCommandLineParser::ParseCommandLineArguments(int argc, const char * const * argv)
{
  auto checkAndConvertJlmOptOptimizations =
      [](const ::llvm::cl::list<std::string> & optimizations,
         JlcCommandLineOptions::OptimizationLevel optimizationLevel)
  {
    if (optimizations.empty() && optimizationLevel == JlcCommandLineOptions::OptimizationLevel::O3)
    {
      return std::vector({
          JlmOptCommandLineOptions::OptimizationId::AAAndersenRegionAware,
          JlmOptCommandLineOptions::OptimizationId::FunctionInlining,
          JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection,
          JlmOptCommandLineOptions::OptimizationId::NodeReduction,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::ThetaGammaInversion,
          JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::NodePushOut,
          JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::NodeReduction,
          JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::NodePullIn,
          JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::LoopUnrolling,
          JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection,
          JlmOptCommandLineOptions::OptimizationId::IfConversion,
          JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination,
          JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination,
      });
    }

    std::vector<JlmOptCommandLineOptions::OptimizationId> optimizationIds;
    for (auto & optimization : optimizations)
    {
      auto optimizationId = JlmOptCommandLineOptions::OptimizationId::FirstEnumValue;
      try
      {
        optimizationId =
            JlmOptCommandLineOptions::FromCommandLineArgumentToOptimizationId(optimization);
      }
      catch (util::Error &)
      {
        throw CommandLineParser::Exception("Unknown jlm-opt optimization: " + optimization);
      }

      optimizationIds.emplace_back(optimizationId);
    }

    return optimizationIds;
  };

  CommandLineOptions_.Reset();

  using namespace ::llvm;

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

  cl::list<std::string> inputFiles(cl::Positional, cl::desc("<inputs>"));

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

  cl::opt<std::string> outputFile("o", cl::desc("Write output to <file>."), cl::value_desc("file"));

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

  cl::list<std::string> flags("f", cl::Prefix, cl::desc("Specify flags."), cl::value_desc("flag"));

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

  cl::opt<bool> suppress("w", cl::ValueDisallowed, cl::desc("Suppress all warnings"));

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

  cl::list<util::Statistics::Id> jlmOptPassStatistics(
      "JlmOptPassStatistics",
      cl::values(
          CreateStatisticsOption(
              util::Statistics::Id::Aggregation,
              "Collect control flow graph aggregation pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::AgnosticModRefSummarizer,
              "Collect agnostic mod/ref summarizer pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::AndersenAnalysis,
              "Collect Andersen alias analysis pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::Annotation,
              "Collect aggregation tree annotation pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::MemoryStateEncoder,
              "Collect memory state encoding pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::CommonNodeElimination,
              "Collect common node elimination pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::ControlFlowRecovery,
              "Collect control flow recovery pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::DataNodeToDelta,
              "Collect data node to delta node conversion pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::DeadNodeElimination,
              "Collect dead node elimination pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::FunctionInlining,
              "Collect function inlining pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::InvariantValueRedirection,
              "Collect invariant value redirection pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::JlmToRvsdgConversion,
              "Collect Jlm to RVSDG conversion pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::LoopUnrolling,
              "Collect loop unrolling pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::PullNodes,
              "Collect node pull pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::PushNodes,
              "Collect node push pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::ReduceNodes,
              "Collect node reduction pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RegionAwareModRefSummarizer,
              "Collect region-aware mod/ref summarizer pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgConstruction,
              "Collect RVSDG construction pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgDestruction,
              "Collect RVSDG destruction pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgOptimization,
              "Collect RVSDG optimization pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgTreePrinter,
              "Collect RVSDG tree printer pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::SteensgaardAnalysis,
              "Collect Steensgaard alias analysis pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::ThetaGammaInversion,
              "Collect theta-gamma inversion pass statistics.")),
      cl::desc("Collect jlm-opt pass statistics"));

  cl::ParseCommandLineOptions(argc, argv);

  /* Process parsed options */

  static std::unordered_map<std::string, JlcCommandLineOptions::OptimizationLevel>
      optimizationLevelMap({ { "0", JlcCommandLineOptions::OptimizationLevel::O0 },
                             { "1", JlcCommandLineOptions::OptimizationLevel::O1 },
                             { "2", JlcCommandLineOptions::OptimizationLevel::O2 },
                             { "3", JlcCommandLineOptions::OptimizationLevel::O3 } });

  static std::unordered_map<std::string, JlcCommandLineOptions::LanguageStandard>
      languageStandardMap({ { "gnu89", JlcCommandLineOptions::LanguageStandard::Gnu89 },
                            { "gnu99", JlcCommandLineOptions::LanguageStandard::Gnu99 },
                            { "c89", JlcCommandLineOptions::LanguageStandard::C89 },
                            { "c90", JlcCommandLineOptions::LanguageStandard::C99 },
                            { "c99", JlcCommandLineOptions::LanguageStandard::C99 },
                            { "c11", JlcCommandLineOptions::LanguageStandard::C11 },
                            { "c++98", JlcCommandLineOptions::LanguageStandard::Cpp98 },
                            { "c++03", JlcCommandLineOptions::LanguageStandard::Cpp03 },
                            { "c++11", JlcCommandLineOptions::LanguageStandard::Cpp11 },
                            { "c++14", JlcCommandLineOptions::LanguageStandard::Cpp14 } });

  if (!optimizationLevel.empty())
  {
    auto iterator = optimizationLevelMap.find(optimizationLevel);
    if (iterator == optimizationLevelMap.end())
    {
      std::cerr << "Unknown optimization level.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.OptimizationLevel_ = iterator->second;
  }

  if (!languageStandard.empty())
  {
    auto iterator = languageStandardMap.find(languageStandard);
    if (iterator == languageStandardMap.end())
    {
      std::cerr << "Unknown language standard.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.LanguageStandard_ = iterator->second;
  }

  if (inputFiles.empty())
  {
    std::cerr << "jlc: no input files.\n";
    exit(EXIT_FAILURE);
  }

  if (inputFiles.size() > 1 && noLinking && !outputFile.empty())
  {
    std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
    exit(EXIT_FAILURE);
  }

  auto jlmOptOptimizations =
      checkAndConvertJlmOptOptimizations(jlmOptimizations, CommandLineOptions_.OptimizationLevel_);

  CommandLineOptions_.Libraries_ = libraries;
  CommandLineOptions_.MacroDefinitions_ = macroDefinitions;
  CommandLineOptions_.LibraryPaths_ = libraryPaths;
  CommandLineOptions_.Warnings_ = warnings;
  CommandLineOptions_.IncludePaths_ = includePaths;
  CommandLineOptions_.OnlyPrintCommands_ = onlyPrintCommands;
  CommandLineOptions_.GenerateDebugInformation_ = generateDebugInformation;
  CommandLineOptions_.Flags_ = flags;
  CommandLineOptions_.JlmOptOptimizations_ = jlmOptOptimizations;
  CommandLineOptions_.JlmOptPassStatistics_ = util::HashSet<util::Statistics::Id>(
      { jlmOptPassStatistics.begin(), jlmOptPassStatistics.end() });
  CommandLineOptions_.Verbose_ = verbose;
  CommandLineOptions_.Rdynamic_ = rDynamic;
  CommandLineOptions_.Suppress_ = suppress;
  CommandLineOptions_.UsePthreads_ = usePthreads;
  CommandLineOptions_.Md_ = mD;

  for (auto & inputFile : inputFiles)
  {
    util::FilePath inputFilePath(inputFile);
    if (IsObjectFile(inputFilePath))
    {
      /* FIXME: print a warning like clang if noLinking is true */
      CommandLineOptions_.Compilations_.push_back(
          { inputFilePath, util::FilePath(""), inputFilePath, "", false, false, false, true });

      continue;
    }

    CommandLineOptions_.Compilations_.push_back(
        { inputFilePath,
          mF.empty() ? ToDependencyFile(inputFilePath) : util::FilePath(mF),
          ToObjectFile(inputFilePath),
          mT.empty() ? ToObjectFile(inputFilePath).name() : mT,
          true,
          true,
          true,
          !noLinking });
  }

  if (!outputFile.empty())
  {
    util::FilePath outputFilePath(outputFile);
    if (noLinking)
    {
      JLM_ASSERT(CommandLineOptions_.Compilations_.size() == 1);
      CommandLineOptions_.Compilations_[0].SetOutputFile(outputFilePath);
    }
    else
    {
      CommandLineOptions_.OutputFile_ = outputFilePath;
    }
  }

  return CommandLineOptions_;
}

JlmOptCommandLineParser::~JlmOptCommandLineParser() noexcept = default;

const JlmOptCommandLineOptions &
JlmOptCommandLineParser::ParseCommandLineArguments(int argc, const char * const * argv)
{
  using namespace ::llvm;

  /*
    FIXME: The command line parser setup is currently redone
    for every invocation of parse_cmdline. We should be able
    to do it only once and then reset the parser on every
    invocation of parse_cmdline.
  */

  cl::TopLevelSubCommand->reset();

  cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input>"));

  cl::opt<std::string> outputFile(
      "o",
      cl::init(""),
      cl::desc("Write output to <file>"),
      cl::value_desc("file"));

  const auto statisticsDirectoryDefault = util::FilePath::TempDirectoryPath().Join("jlm");
  const auto statisticDirectoryDescription =
      "Write statistics and debug output to files in <dir>. Default is "
      + statisticsDirectoryDefault.to_str() + ".";
  cl::opt<std::string> statisticDirectory(
      "s",
      cl::init(statisticsDirectoryDefault.to_str()),
      cl::desc(statisticDirectoryDescription),
      cl::value_desc("dir"));

  cl::list<util::Statistics::Id> printStatistics(
      cl::values(
          CreateStatisticsOption(
              util::Statistics::Id::AliasAnalysisPrecisionEvaluation,
              "Evaluate alias analysis precision and store to file"),
          CreateStatisticsOption(
              util::Statistics::Id::Aggregation,
              "Write aggregation statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::AgnosticModRefSummarizer,
              "Collect agnostic mod/ref summarization pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::AndersenAnalysis,
              "Collect Andersen alias analysis pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::Annotation,
              "Write annotation statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::MemoryStateEncoder,
              "Write encoding statistics of basic encoder to file."),
          CreateStatisticsOption(
              util::Statistics::Id::CommonNodeElimination,
              "Write common node elimination statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::ControlFlowRecovery,
              "Write control flow recovery statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::DataNodeToDelta,
              "Write data node to delta node conversion statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::DeadNodeElimination,
              "Write dead node elimination statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::FunctionInlining,
              "Write function inlining statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::IfConversion,
              "Collect if-conversion transformation statistics"),
          CreateStatisticsOption(
              util::Statistics::Id::InvariantValueRedirection,
              "Write invariant value redirection statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::JlmToRvsdgConversion,
              "Write Jlm to RVSDG conversion statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::LoopUnrolling,
              "Write loop unrolling statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::PullNodes,
              "Write node pull statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::PushNodes,
              "Write node push statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::ReduceNodes,
              "Write node reduction statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::RegionAwareModRefSummarizer,
              "Collect region-aware mod/ref summarization statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgConstruction,
              "Write RVSDG construction statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgDestruction,
              "Write RVSDG destruction statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgOptimization,
              "Write RVSDG optimization statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::RvsdgTreePrinter,
              "Write RVSDG tree printer pass statistics."),
          CreateStatisticsOption(
              util::Statistics::Id::SteensgaardAnalysis,
              "Write Steensgaard analysis statistics to file."),
          CreateStatisticsOption(
              util::Statistics::Id::ThetaGammaInversion,
              "Write theta-gamma inversion statistics to file.")),
      cl::desc("Write statistics"));

#ifdef ENABLE_MLIR
  auto llvmInputFormat = JlmOptCommandLineOptions::InputFormat::Llvm;
  auto mlirInputFormat = JlmOptCommandLineOptions::InputFormat::Mlir;

  cl::opt<JlmOptCommandLineOptions::InputFormat> inputFormat(
      "input-format",
      cl::desc("Select input format:"),
      cl::values(
          ::clEnumValN(
              llvmInputFormat,
              JlmOptCommandLineOptions::ToCommandLineArgument(llvmInputFormat),
              "Input LLVM IR [default]"),
          ::clEnumValN(
              mlirInputFormat,
              JlmOptCommandLineOptions::ToCommandLineArgument(mlirInputFormat),
              "Input MLIR")),
      cl::init(llvmInputFormat));
#else
  auto inputFormat = JlmOptCommandLineOptions::InputFormat::Llvm;
#endif

  cl::opt<JlmOptCommandLineOptions::OutputFormat> outputFormat(
      "output-format",
      cl::desc("Select output format:"),
      cl::values(
          CreateOutputFormatOption(JlmOptCommandLineOptions::OutputFormat::Ascii, "Output Ascii"),
          CreateOutputFormatOption(JlmOptCommandLineOptions::OutputFormat::Dot, "Output Dot"),
          CreateOutputFormatOption(
              JlmOptCommandLineOptions::OutputFormat::Llvm,
              "Output LLVM IR [default]"),
#ifdef ENABLE_MLIR
          CreateOutputFormatOption(JlmOptCommandLineOptions::OutputFormat::Mlir, "Output MLIR"),
#endif
          CreateOutputFormatOption(
              JlmOptCommandLineOptions::OutputFormat::Tree,
              "Output Rvsdg Tree"),
          CreateOutputFormatOption(JlmOptCommandLineOptions::OutputFormat::Xml, "Output XML")),
      cl::init(JlmOptCommandLineOptions::OutputFormat::Llvm));

  auto aAAndersenAgnostic = JlmOptCommandLineOptions::OptimizationId::AAAndersenAgnostic;
  auto aAAndersenRegionAware = JlmOptCommandLineOptions::OptimizationId::AAAndersenRegionAware;
  auto aAAndersenTopDownLifetimeAware =
      JlmOptCommandLineOptions::OptimizationId::AAAndersenTopDownLifetimeAware;
  auto aASteensgaardAgnostic = JlmOptCommandLineOptions::OptimizationId::AASteensgaardAgnostic;
  auto aASteensgaardRegionAware =
      JlmOptCommandLineOptions::OptimizationId::AASteensgaardRegionAware;
  auto commonNodeElimination = JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination;
  auto deadNodeElimination = JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination;
  auto functionInlining = JlmOptCommandLineOptions::OptimizationId::FunctionInlining;
  auto ifConversion = JlmOptCommandLineOptions::OptimizationId::IfConversion;
  auto invariantValueRedirection =
      JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection;
  auto nodePushOut = JlmOptCommandLineOptions::OptimizationId::NodePushOut;
  auto nodePullIn = JlmOptCommandLineOptions::OptimizationId::NodePullIn;
  auto nodeReduction = JlmOptCommandLineOptions::OptimizationId::NodeReduction;
  auto rvsdgTreePrinter = JlmOptCommandLineOptions::OptimizationId::RvsdgTreePrinter;
  auto thetaGammaInversion = JlmOptCommandLineOptions::OptimizationId::ThetaGammaInversion;
  auto loopUnrolling = JlmOptCommandLineOptions::OptimizationId::LoopUnrolling;

  cl::list<JlmOptCommandLineOptions::OptimizationId> optimizationIds(
      cl::values(
          ::clEnumValN(
              aAAndersenAgnostic,
              JlmOptCommandLineOptions::ToCommandLineArgument(aAAndersenAgnostic),
              "Andersen alias analysis with agnostic memory state encoding"),
          ::clEnumValN(
              aAAndersenRegionAware,
              JlmOptCommandLineOptions::ToCommandLineArgument(aAAndersenRegionAware),
              "Andersen alias analysis with region-aware memory state encoding"),
          ::clEnumValN(
              aAAndersenTopDownLifetimeAware,
              JlmOptCommandLineOptions::ToCommandLineArgument(aAAndersenTopDownLifetimeAware),
              "Andersen alias analysis with top-down lifetime-aware memory node elimination"),
          ::clEnumValN(
              aASteensgaardAgnostic,
              JlmOptCommandLineOptions::ToCommandLineArgument(aASteensgaardAgnostic),
              "Steensgaard alias analysis with agnostic memory state encoding"),
          ::clEnumValN(
              aASteensgaardRegionAware,
              JlmOptCommandLineOptions::ToCommandLineArgument(aASteensgaardRegionAware),
              "Steensgaard alias analysis with region-aware memory state encoding"),
          ::clEnumValN(
              commonNodeElimination,
              JlmOptCommandLineOptions::ToCommandLineArgument(commonNodeElimination),
              "Common Node Elimination"),
          ::clEnumValN(
              deadNodeElimination,
              JlmOptCommandLineOptions::ToCommandLineArgument(deadNodeElimination),
              "Dead Node Elimination"),
          ::clEnumValN(
              functionInlining,
              JlmOptCommandLineOptions::ToCommandLineArgument(functionInlining),
              "Function Inlining"),
          ::clEnumValN(
              ifConversion,
              JlmOptCommandLineOptions::ToCommandLineArgument(ifConversion),
              "Convert pass-through values of gamma nodes to select operations"),
          ::clEnumValN(
              invariantValueRedirection,
              JlmOptCommandLineOptions::ToCommandLineArgument(invariantValueRedirection),
              "Invariant Value Redirection"),
          ::clEnumValN(
              nodePushOut,
              JlmOptCommandLineOptions::ToCommandLineArgument(nodePushOut),
              "Node Push Out"),
          ::clEnumValN(
              nodePullIn,
              JlmOptCommandLineOptions::ToCommandLineArgument(nodePullIn),
              "Node Pull In"),
          ::clEnumValN(
              nodeReduction,
              JlmOptCommandLineOptions::ToCommandLineArgument(nodeReduction),
              "Node Reduction"),
          ::clEnumValN(
              rvsdgTreePrinter,
              JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgTreePrinter),
              "Rvsdg Tree Printer"),
          ::clEnumValN(
              thetaGammaInversion,
              JlmOptCommandLineOptions::ToCommandLineArgument(thetaGammaInversion),
              "Theta-Gamma Inversion"),
          ::clEnumValN(
              loopUnrolling,
              JlmOptCommandLineOptions::ToCommandLineArgument(loopUnrolling),
              "Loop Unrolling")),
      cl::desc("Perform optimization"));

  cl::list<llvm::RvsdgTreePrinter::Configuration::Annotation> rvsdgTreePrinterAnnotations(
      "annotations",
      cl::values(::clEnumValN(
          llvm::RvsdgTreePrinter::Configuration::Annotation::NumAllocaNodes,
          "NumAllocaNodes",
          "Annotate number of AllocaOperation nodes")),
      cl::values(::clEnumValN(
          llvm::RvsdgTreePrinter::Configuration::Annotation::NumLoadNodes,
          "NumLoadNodes",
          "Annotate number of LoadOperation nodes")),
      cl::values(::clEnumValN(
          llvm::RvsdgTreePrinter::Configuration::Annotation::NumMemoryStateInputsOutputs,
          "NumMemoryStateInputsOutputs",
          "Annotate number of inputs/outputs with memory state type")),
      cl::values(::clEnumValN(
          llvm::RvsdgTreePrinter::Configuration::Annotation::NumRvsdgNodes,
          "NumRvsdgNodes",
          "Annotate number of RVSDG nodes")),
      cl::values(::clEnumValN(
          llvm::RvsdgTreePrinter::Configuration::Annotation::NumStoreNodes,
          "NumStoreNodes",
          "Annotate number of StoreOperation nodes")),
      cl::CommaSeparated,
      cl::desc("Comma separated list of RVSDG tree printer annotations"));

  cl::ParseCommandLineOptions(argc, argv);

  jlm::util::FilePath statisticsDirectoryFilePath(statisticDirectory);
  jlm::util::FilePath inputFilePath(inputFile);

  util::HashSet<util::Statistics::Id> demandedStatistics(
      { printStatistics.begin(), printStatistics.end() });

  util::StatisticsCollectorSettings statisticsCollectorSettings(
      std::move(demandedStatistics),
      statisticsDirectoryFilePath,
      inputFilePath.base());

  util::HashSet<llvm::RvsdgTreePrinter::Configuration::Annotation> demandedAnnotations(
      { rvsdgTreePrinterAnnotations.begin(), rvsdgTreePrinterAnnotations.end() });

  llvm::RvsdgTreePrinter::Configuration treePrinterConfiguration(std::move(demandedAnnotations));

  CommandLineOptions_ = JlmOptCommandLineOptions::Create(
      std::move(inputFilePath),
      inputFormat,
      util::FilePath(outputFile),
      outputFormat,
      std::move(statisticsCollectorSettings),
      std::move(treePrinterConfiguration),
      std::move(optimizationIds));

  return *CommandLineOptions_;
}

const JlmOptCommandLineOptions &
JlmOptCommandLineParser::Parse(int argc, const char * const * argv)
{
  static JlmOptCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

JlmHlsCommandLineParser::~JlmHlsCommandLineParser() noexcept = default;

const JlmHlsCommandLineOptions &
JlmHlsCommandLineParser::ParseCommandLineArguments(int argc, const char * const * argv)
{
  CommandLineOptions_.Reset();

  using namespace ::llvm;

  /*
    FIXME: The command line parser setup is currently redone
    for every invocation of parse_cmdline. We should be able
    to do it only once and then reset the parser on every
    invocation of parse_cmdline.
  */

  cl::TopLevelSubCommand->reset();

  cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input>"));

  cl::opt<std::string> outputFolder(
      "o",
      cl::desc("Write output to <folder>"),
      cl::value_desc("folder"));

  cl::opt<std::string> hlsFunction(
      "hls-function",
      cl::Prefix,
      cl::desc("Function that should be accelerated"),
      cl::value_desc("hls-function"));

  cl::opt<int> latency(
      "latency",
      cl::Prefix,
      cl::init(CommandLineOptions_.MemoryLatency_),
      cl::desc("Memory latency"),
      cl::value_desc("latency"));

  cl::opt<bool> extractHlsFunction(
      "extract",
      cl::Prefix,
      cl::desc("Extracts function specified by hls-function"));

  cl::opt<JlmHlsCommandLineOptions::OutputFormat> format(
      cl::values(
          ::clEnumValN(
              JlmHlsCommandLineOptions::OutputFormat::Firrtl,
              "fir",
              "Output FIRRTL [default]"),
          ::clEnumValN(JlmHlsCommandLineOptions::OutputFormat::Dot, "dot", "Output DOT graph")),
      cl::desc("Select output format"));

  cl::ParseCommandLineOptions(argc, argv);

  if (outputFolder.empty())
    throw util::Error("jlm-hls no output directory provided, i.e, -o.\n");

  if (extractHlsFunction && hlsFunction.empty())
    throw util::Error(
        "jlm-hls: --hls-function is not specified.\n         which is required for --extract\n");

  CommandLineOptions_.InputFile_ = util::FilePath(inputFile);
  CommandLineOptions_.HlsFunction_ = std::move(hlsFunction);
  CommandLineOptions_.OutputFiles_ = util::FilePath(outputFolder);
  CommandLineOptions_.ExtractHlsFunction_ = extractHlsFunction;
  CommandLineOptions_.OutputFormat_ = format;

  if (latency < 1)
  {
    throw util::Error("The --latency must be set to a number larger than zero.");
  }
  CommandLineOptions_.MemoryLatency_ = latency;

  return CommandLineOptions_;
}

const JlmHlsCommandLineOptions &
JlmHlsCommandLineParser::Parse(int argc, const char * const * argv)
{
  static JlmHlsCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

JhlsCommandLineParser::~JhlsCommandLineParser() noexcept = default;

const JhlsCommandLineOptions &
JhlsCommandLineParser::ParseCommandLineArguments(int argc, const char * const * argv)
{
  CommandLineOptions_.Reset();

  using namespace ::llvm;

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

  cl::list<std::string> inputFiles(cl::Positional, cl::desc("<inputs>"));

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

  cl::opt<std::string> outputFile("o", cl::desc("Write output to <file>."), cl::value_desc("file"));

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

  cl::list<std::string> flags("f", cl::Prefix, cl::desc("Specify flags."), cl::value_desc("flag"));

  cl::list<std::string> jlmHlsOptimizations(
      "J",
      cl::Prefix,
      cl::desc("jlm-hls optimization. Run 'jlm-hls -help' for viable options."),
      cl::value_desc("jlmhls"));

  cl::opt<bool> verbose(
      "v",
      cl::ValueDisallowed,
      cl::desc("Show commands to run and use verbose output. (Affects only clang for now)"));

  cl::opt<bool> rdynamic(
      "rdynamic",
      cl::ValueDisallowed,
      cl::desc("rdynamic option passed to clang"));

  cl::opt<bool> suppress("w", cl::ValueDisallowed, cl::desc("Suppress all warnings"));

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
      cl::desc("Write depfile output from -MD to <file>."),
      cl::value_desc("file"));

  cl::opt<std::string> mT(
      "MT",
      cl::desc("Specify name of main file output in depfile."),
      cl::value_desc("value"));

  cl::list<std::string> hlsFunction(
      "hls-function",
      cl::Prefix,
      cl::desc("function that should be accelerated"),
      cl::value_desc("regex"));

  cl::opt<bool> generateFirrtl("firrtl", cl::ValueDisallowed, cl::desc("Generate firrtl"));

  cl::opt<bool> useCirct(
      "circt",
      cl::Prefix,
      cl::desc("DEPRACATED - CIRCT is always used to generate FIRRTL"));

  cl::ParseCommandLineOptions(argc, argv);

  /* Process parsed options */

  static std::unordered_map<std::string, JhlsCommandLineOptions::OptimizationLevel> Olvlmap(
      { { "0", JhlsCommandLineOptions::OptimizationLevel::O0 },
        { "1", JhlsCommandLineOptions::OptimizationLevel::O1 },
        { "2", JhlsCommandLineOptions::OptimizationLevel::O2 },
        { "3", JhlsCommandLineOptions::OptimizationLevel::O3 } });

  static std::unordered_map<std::string, JhlsCommandLineOptions::LanguageStandard> stdmap(
      { { "Gnu89", JhlsCommandLineOptions::LanguageStandard::Gnu89 },
        { "Gnu99", JhlsCommandLineOptions::LanguageStandard::Gnu99 },
        { "C89", JhlsCommandLineOptions::LanguageStandard::C89 },
        { "C99", JhlsCommandLineOptions::LanguageStandard::C99 },
        { "C11", JhlsCommandLineOptions::LanguageStandard::C11 },
        { "C++98", JhlsCommandLineOptions::LanguageStandard::Cpp98 },
        { "C++03", JhlsCommandLineOptions::LanguageStandard::Cpp03 },
        { "C++11", JhlsCommandLineOptions::LanguageStandard::Cpp11 },
        { "C++14", JhlsCommandLineOptions::LanguageStandard::Cpp14 } });

  if (!optimizationLevel.empty())
  {
    auto it = Olvlmap.find(optimizationLevel);
    if (it == Olvlmap.end())
    {
      std::cerr << "Unknown optimization level.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.OptimizationLevel_ = it->second;
  }

  if (!languageStandard.empty())
  {
    auto it = stdmap.find(languageStandard);
    if (it == stdmap.end())
    {
      std::cerr << "Unknown language standard.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.LanguageStandard_ = it->second;
  }

  if (inputFiles.empty())
  {
    std::cerr << "jlc: no input files.\n";
    exit(EXIT_FAILURE);
  }

  if (inputFiles.size() > 1 && noLinking && !outputFile.empty())
  {
    std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
    exit(EXIT_FAILURE);
  }

  if (!hlsFunction.empty())
  {
    CommandLineOptions_.Hls_ = true;
    CommandLineOptions_.HlsFunctionRegex_ = hlsFunction.front();
  }

  if (hlsFunction.size() > 1)
  {
    std::cerr << "jlc-hls: more than one function regex specified\n";
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
  CommandLineOptions_.JlmHls_ = jlmHlsOptimizations;
  CommandLineOptions_.Verbose_ = verbose;
  CommandLineOptions_.Rdynamic_ = rdynamic;
  CommandLineOptions_.Suppress_ = suppress;
  CommandLineOptions_.UsePthreads_ = usePthreads;
  CommandLineOptions_.Md_ = mD;
  CommandLineOptions_.GenerateFirrtl_ = generateFirrtl;

  for (auto & inputFile : inputFiles)
  {
    util::FilePath inputFilePath(inputFile);
    if (IsObjectFile(inputFilePath))
    {
      /* FIXME: print a warning like clang if noLinking is true */
      CommandLineOptions_.Compilations_.push_back(
          { inputFilePath, util::FilePath(""), inputFilePath, "", false, false, false, true });

      continue;
    }

    CommandLineOptions_.Compilations_.push_back(
        { inputFilePath,
          mF.empty() ? CreateDependencyFileFromFile(inputFilePath) : util::FilePath(mF),
          CreateObjectFileFromFile(inputFilePath),
          mT.empty() ? CreateObjectFileFromFile(inputFilePath).name() : mT,
          true,
          true,
          true,
          !noLinking });
  }

  if (!outputFile.empty())
  {
    util::FilePath outputFilePath(outputFile);
    if (noLinking)
    {
      JLM_ASSERT(CommandLineOptions_.Compilations_.size() == 1);
      CommandLineOptions_.Compilations_[0].SetOutputFile(outputFilePath);
    }
    else
    {
      CommandLineOptions_.OutputFile_ = outputFilePath;
    }
  }

  return CommandLineOptions_;
}

bool
JhlsCommandLineParser::IsObjectFile(const util::FilePath & file)
{
  return file.suffix() == "o";
}

util::FilePath
JhlsCommandLineParser::CreateObjectFileFromFile(const util::FilePath & f)
{
  return f.Dirname().Join(f.base() + ".o");
}

util::FilePath
JhlsCommandLineParser::CreateDependencyFileFromFile(const util::FilePath & f)
{
  return f.Dirname().Join(f.base() + ".d");
}

const JhlsCommandLineOptions &
JhlsCommandLineParser::Parse(int argc, const char * const * argv)
{
  static JhlsCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

}
