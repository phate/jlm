/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <llvm/Support/CommandLine.h>

#include <unordered_map>

namespace jlm::tooling
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

  OutputFile_ = util::filepath("a.out");
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
  InputFile_ = util::filepath("");
  OutputFile_ = util::filepath("");
  OutputFormat_ = OutputFormat::Llvm;
  StatisticsCollectorSettings_ = util::StatisticsCollectorSettings();
  OptimizationIds_.clear();
}

std::vector<llvm::optimization*>
JlmOptCommandLineOptions::GetOptimizations() const noexcept
{
  std::vector<llvm::optimization*> optimizations;
  optimizations.reserve(OptimizationIds_.size());

  for (auto & optimizationId: OptimizationIds_)
  {
    optimizations.emplace_back(GetOptimization(optimizationId));
  }

  return optimizations;
}

JlmOptCommandLineOptions::OptimizationId
JlmOptCommandLineOptions::FromCommandLineArgumentToOptimizationId(const std::string& commandLineArgument)
{
  static std::unordered_map<std::string, OptimizationId> map(
    {
      {OptimizationCommandLineArgument::AaSteensgaardAgnostic_,           OptimizationId::AASteensgaardAgnostic},
      {OptimizationCommandLineArgument::AaSteensgaardRegionAware_,        OptimizationId::AASteensgaardRegionAware},
      {OptimizationCommandLineArgument::CommonNodeElimination_,           OptimizationId::CommonNodeElimination},
      {OptimizationCommandLineArgument::CommonNodeEliminationDeprecated_, OptimizationId::cne},
      {OptimizationCommandLineArgument::DeadNodeElimination_,             OptimizationId::DeadNodeElimination},
      {OptimizationCommandLineArgument::DeadNodeEliminationDeprecated_,   OptimizationId::dne},
      {OptimizationCommandLineArgument::FunctionInlining_,                OptimizationId::FunctionInlining},
      {OptimizationCommandLineArgument::FunctionInliningDeprecated_,      OptimizationId::iln},
      {OptimizationCommandLineArgument::InvariantValueRedirection_,       OptimizationId::InvariantValueRedirection},
      {OptimizationCommandLineArgument::NodePushOut_,                     OptimizationId::NodePushOut},
      {OptimizationCommandLineArgument::NodePushOutDeprecated_,           OptimizationId::psh},
      {OptimizationCommandLineArgument::NodePullIn_,                      OptimizationId::NodePullIn},
      {OptimizationCommandLineArgument::NodePullInDeprecated_,            OptimizationId::pll},
      {OptimizationCommandLineArgument::NodeReduction_,                   OptimizationId::NodeReduction},
      {OptimizationCommandLineArgument::NodeReductionDeprecated_,         OptimizationId::red},
      {OptimizationCommandLineArgument::ThetaGammaInversion_,             OptimizationId::ThetaGammaInversion},
      {OptimizationCommandLineArgument::ThetaGammaInversionDeprecated_,   OptimizationId::ivt},
      {OptimizationCommandLineArgument::LoopUnrolling_,                   OptimizationId::LoopUnrolling},
      {OptimizationCommandLineArgument::LoopUnrollingDeprecated_,         OptimizationId::url}
    });

  if (map.find(commandLineArgument) != map.end())
    return map[commandLineArgument];

  throw util::error("Unknown command line argument: " + commandLineArgument);
}

const char*
JlmOptCommandLineOptions::ToCommandLineArgument(OptimizationId optimizationId)
{
  static std::unordered_map<OptimizationId, const char*> map(
    {
      {OptimizationId::AASteensgaardAgnostic,     OptimizationCommandLineArgument::AaSteensgaardAgnostic_},
      {OptimizationId::AASteensgaardRegionAware,  OptimizationCommandLineArgument::AaSteensgaardRegionAware_},
      {OptimizationId::cne,                       OptimizationCommandLineArgument::CommonNodeEliminationDeprecated_},
      {OptimizationId::CommonNodeElimination,     OptimizationCommandLineArgument::CommonNodeElimination_},
      {OptimizationId::DeadNodeElimination,       OptimizationCommandLineArgument::DeadNodeElimination_},
      {OptimizationId::dne,                       OptimizationCommandLineArgument::DeadNodeEliminationDeprecated_},
      {OptimizationId::FunctionInlining,          OptimizationCommandLineArgument::FunctionInlining_},
      {OptimizationId::iln,                       OptimizationCommandLineArgument::FunctionInliningDeprecated_},
      {OptimizationId::InvariantValueRedirection, OptimizationCommandLineArgument::InvariantValueRedirection_},
      {OptimizationId::LoopUnrolling,             OptimizationCommandLineArgument::LoopUnrolling_},
      {OptimizationId::NodePullIn,                OptimizationCommandLineArgument::NodePullIn_},
      {OptimizationId::NodePushOut,               OptimizationCommandLineArgument::NodePushOut_},
      {OptimizationId::NodeReduction,             OptimizationCommandLineArgument::NodeReduction_},
      {OptimizationId::psh,                       OptimizationCommandLineArgument::NodePushOutDeprecated_},
      {OptimizationId::pll,                       OptimizationCommandLineArgument::NodePullInDeprecated_},
      {OptimizationId::red,                       OptimizationCommandLineArgument::NodeReductionDeprecated_},
      {OptimizationId::ivt,                       OptimizationCommandLineArgument::ThetaGammaInversionDeprecated_},
      {OptimizationId::url,                       OptimizationCommandLineArgument::LoopUnrollingDeprecated_},
      {OptimizationId::ThetaGammaInversion,       OptimizationCommandLineArgument::ThetaGammaInversion_}
    });

  if (map.find(optimizationId) != map.end())
    return map[optimizationId];

  throw util::error("Unknown optimization identifier");
}

util::Statistics::Id
JlmOptCommandLineOptions::FromCommandLineArgumentToStatisticsId(const std::string& commandLineArgument)
{
  static std::unordered_map<std::string, util::Statistics::Id> map(
    {
      {StatisticsCommandLineArgument::Aggregation_,               util::Statistics::Id::Aggregation},
      {StatisticsCommandLineArgument::BasicEncoderEncoding_,      util::Statistics::Id::BasicEncoderEncoding},
      {StatisticsCommandLineArgument::Annotation_,                util::Statistics::Id::Annotation},
      {StatisticsCommandLineArgument::CommonNodeElimination_,     util::Statistics::Id::CommonNodeElimination},
      {StatisticsCommandLineArgument::ControlFlowRecovery_,       util::Statistics::Id::ControlFlowRecovery},
      {StatisticsCommandLineArgument::DataNodeToDelta_,           util::Statistics::Id::DataNodeToDelta},
      {StatisticsCommandLineArgument::DeadNodeElimination_,       util::Statistics::Id::DeadNodeElimination},
      {StatisticsCommandLineArgument::FunctionInlining_,          util::Statistics::Id::FunctionInlining},
      {StatisticsCommandLineArgument::InvariantValueRedirection_, util::Statistics::Id::InvariantValueRedirection},
      {StatisticsCommandLineArgument::JlmToRvsdgConversion_,      util::Statistics::Id::JlmToRvsdgConversion},
      {StatisticsCommandLineArgument::LoopUnrolling_,             util::Statistics::Id::LoopUnrolling},
      {StatisticsCommandLineArgument::MemoryNodeProvisioning_,    util::Statistics::Id::MemoryNodeProvisioning},
      {StatisticsCommandLineArgument::PullNodes_,                 util::Statistics::Id::PullNodes},
      {StatisticsCommandLineArgument::PushNodes_,                 util::Statistics::Id::PushNodes},
      {StatisticsCommandLineArgument::ReduceNodes_,               util::Statistics::Id::ReduceNodes},
      {StatisticsCommandLineArgument::RvsdgConstruction_,         util::Statistics::Id::RvsdgConstruction},
      {StatisticsCommandLineArgument::RvsdgDestruction_,          util::Statistics::Id::RvsdgDestruction},
      {StatisticsCommandLineArgument::RvsdgOptimization_,         util::Statistics::Id::RvsdgOptimization},
      {StatisticsCommandLineArgument::SteensgaardAnalysis_,       util::Statistics::Id::SteensgaardAnalysis},
      {StatisticsCommandLineArgument::ThetaGammaInversion_,       util::Statistics::Id::ThetaGammaInversion}
    });

  if (map.find(commandLineArgument) != map.end())
    return map[commandLineArgument];

  throw util::error("Unknown command line argument: " + commandLineArgument);
}

const char*
JlmOptCommandLineOptions::ToCommandLineArgument(jlm::util::Statistics::Id statisticsId)
{
  static std::unordered_map<util::Statistics::Id, const char*> map(
    {
      {util::Statistics::Id::Aggregation,               StatisticsCommandLineArgument::Aggregation_},
      {util::Statistics::Id::BasicEncoderEncoding,      StatisticsCommandLineArgument::BasicEncoderEncoding_},
      {util::Statistics::Id::Annotation,                StatisticsCommandLineArgument::Annotation_},
      {util::Statistics::Id::CommonNodeElimination,     StatisticsCommandLineArgument::CommonNodeElimination_},
      {util::Statistics::Id::ControlFlowRecovery,       StatisticsCommandLineArgument::ControlFlowRecovery_},
      {util::Statistics::Id::DataNodeToDelta,           StatisticsCommandLineArgument::DataNodeToDelta_},
      {util::Statistics::Id::DeadNodeElimination,       StatisticsCommandLineArgument::DeadNodeElimination_},
      {util::Statistics::Id::FunctionInlining,          StatisticsCommandLineArgument::FunctionInlining_},
      {util::Statistics::Id::InvariantValueRedirection, StatisticsCommandLineArgument::InvariantValueRedirection_},
      {util::Statistics::Id::JlmToRvsdgConversion,      StatisticsCommandLineArgument::JlmToRvsdgConversion_},
      {util::Statistics::Id::LoopUnrolling,             StatisticsCommandLineArgument::LoopUnrolling_},
      {util::Statistics::Id::MemoryNodeProvisioning,    StatisticsCommandLineArgument::MemoryNodeProvisioning_},
      {util::Statistics::Id::PullNodes,                 StatisticsCommandLineArgument::PullNodes_},
      {util::Statistics::Id::PushNodes,                 StatisticsCommandLineArgument::PushNodes_},
      {util::Statistics::Id::ReduceNodes,               StatisticsCommandLineArgument::ReduceNodes_},
      {util::Statistics::Id::RvsdgConstruction,         StatisticsCommandLineArgument::RvsdgConstruction_},
      {util::Statistics::Id::RvsdgDestruction,          StatisticsCommandLineArgument::RvsdgDestruction_},
      {util::Statistics::Id::RvsdgOptimization,         StatisticsCommandLineArgument::RvsdgOptimization_},
      {util::Statistics::Id::SteensgaardAnalysis,       StatisticsCommandLineArgument::SteensgaardAnalysis_},
      {util::Statistics::Id::ThetaGammaInversion,       StatisticsCommandLineArgument::ThetaGammaInversion_}
    });

  if (map.find(statisticsId) != map.end())
    return map[statisticsId];

  throw util::error("Unknown statistics identifier");
}

const char*
JlmOptCommandLineOptions::ToCommandLineArgument(OutputFormat outputFormat)
{
  static std::unordered_map<OutputFormat, const char*> map(
    {
      {OutputFormat::Llvm, "llvm"},
      {OutputFormat::Xml,  "xml"}
    });

  if (map.find(outputFormat) != map.end())
    return map[outputFormat];

  throw util::error("Unknown output format");
}

llvm::optimization *
JlmOptCommandLineOptions::GetOptimization(enum OptimizationId id)
{
  static llvm::aa::SteensgaardAgnostic steensgaardAgnostic;
  static llvm::aa::SteensgaardRegionAware steensgaardRegionAware;
  static llvm::cne commonNodeElimination;
  static llvm::DeadNodeElimination deadNodeElimination;
  static llvm::fctinline functionInlining;
  static llvm::InvariantValueRedirection invariantValueRedirection;
  static llvm::pullin nodePullIn;
  static llvm::pushout nodePushOut;
  static llvm::tginversion thetaGammaInversion;
  static llvm::loopunroll loopUnrolling(4);
  static llvm::nodereduction nodeReduction;

  static std::unordered_map<OptimizationId, llvm::optimization*> map(
    {
      {OptimizationId::AASteensgaardAgnostic,     &steensgaardAgnostic},
      {OptimizationId::AASteensgaardRegionAware,  &steensgaardRegionAware},
      {OptimizationId::cne,                       &commonNodeElimination},
      {OptimizationId::CommonNodeElimination,     &commonNodeElimination},
      {OptimizationId::DeadNodeElimination,       &deadNodeElimination},
      {OptimizationId::dne,                       &deadNodeElimination},
      {OptimizationId::FunctionInlining,          &functionInlining},
      {OptimizationId::iln,                       &functionInlining},
      {OptimizationId::InvariantValueRedirection, &invariantValueRedirection},
      {OptimizationId::LoopUnrolling,             &loopUnrolling},
      {OptimizationId::NodePullIn,                &nodePullIn},
      {OptimizationId::NodePushOut,               &nodePushOut},
      {OptimizationId::NodeReduction,             &nodeReduction},
      {OptimizationId::pll,                       &nodePullIn},
      {OptimizationId::psh,                       &nodePushOut},
      {OptimizationId::ivt,                       &thetaGammaInversion},
      {OptimizationId::url,                       &loopUnrolling},
      {OptimizationId::red,                       &nodeReduction},
      {OptimizationId::ThetaGammaInversion,       &thetaGammaInversion}
    });

  if (map.find(id) != map.end())
    return map[id];

  throw util::error("Unknown optimization identifier");
}

void
JlmHlsCommandLineOptions::Reset() noexcept
{
  InputFile_ = util::filepath("");
  OutputFolder_ = util::filepath("");
  OutputFormat_ = OutputFormat::Firrtl;
  HlsFunction_ = "";
  ExtractHlsFunction_ = false;
  UseCirct_ = false;
}

void
JhlsCommandLineOptions::Reset() noexcept
{
  *this = JhlsCommandLineOptions();
}

CommandLineParser::~CommandLineParser() noexcept
= default;

CommandLineParser::Exception::~Exception() noexcept
= default;

JlcCommandLineParser::~JlcCommandLineParser() noexcept
= default;

const JlcCommandLineOptions &
JlcCommandLineParser::ParseCommandLineArguments(int argc, char **argv)
{
  auto checkAndConvertJlmOptOptimizations = [](
    const ::llvm::cl::list<std::string>& optimizations,
    JlcCommandLineOptions::OptimizationLevel optimizationLevel)
  {
    if (optimizations.empty()
        && optimizationLevel == JlcCommandLineOptions::OptimizationLevel::O3)
    {
      return std::vector<JlmOptCommandLineOptions::OptimizationId>
        ({
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
           JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection
         });
    }

    std::vector<JlmOptCommandLineOptions::OptimizationId> optimizationIds;
    for (auto & optimization : optimizations)
    {
      JlmOptCommandLineOptions::OptimizationId optimizationId;
      try
      {
        optimizationId = JlmOptCommandLineOptions::FromCommandLineArgumentToOptimizationId(optimization);
      }
      catch (util::error&)
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

  auto aggregationStatisticsId = util::Statistics::Id::Aggregation;
  auto annotationStatisticsId = util::Statistics::Id::Annotation;
  auto basicEncoderEncodingStatisticsId = util::Statistics::Id::BasicEncoderEncoding;
  auto commonNodeEliminationStatisticsId = util::Statistics::Id::CommonNodeElimination;
  auto controlFlowRecoveryStatisticsId = util::Statistics::Id::ControlFlowRecovery;
  auto dataNodeToDeltaStatisticsId = util::Statistics::Id::DataNodeToDelta;
  auto deadNodeEliminationStatisticsId = util::Statistics::Id::DeadNodeElimination;
  auto functionInliningStatisticsId = util::Statistics::Id::FunctionInlining;
  auto invariantValueRedirectionStatisticsId = util::Statistics::Id::InvariantValueRedirection;
  auto jlmToRvsdgConversionStatisticsId = util::Statistics::Id::JlmToRvsdgConversion;
  auto loopUnrollingStatisticsId = util::Statistics::Id::LoopUnrolling;
  auto memoryNodeProvisioningStatisticsId = util::Statistics::Id::MemoryNodeProvisioning;
  auto pullNodesStatisticsId = util::Statistics::Id::PullNodes;
  auto pushNodesStatisticsId = util::Statistics::Id::PushNodes;
  auto reduceNodesStatisticsId = util::Statistics::Id::ReduceNodes;
  auto rvsdgConstructionStatisticsId = util::Statistics::Id::RvsdgConstruction;
  auto rvsdgDestructionStatisticsId = util::Statistics::Id::RvsdgDestruction;
  auto rvsdgOptimizationStatisticsId = util::Statistics::Id::RvsdgOptimization;
  auto steensgaardAnalysisStatisticsId = util::Statistics::Id::SteensgaardAnalysis;
  auto thetaGammaInversionStatisticsId = util::Statistics::Id::ThetaGammaInversion;

  cl::list<util::Statistics::Id> jlmOptPassStatistics(
    "JlmOptPassStatistics",
    cl::values(
      ::clEnumValN(
        aggregationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(aggregationStatisticsId),
        "Collect control flow graph aggregation pass statistics."),
      ::clEnumValN(
        annotationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(annotationStatisticsId),
        "Collect aggregation tree annotation pass statistics."),
      ::clEnumValN(
        basicEncoderEncodingStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(basicEncoderEncodingStatisticsId),
        "Collect memory state encoding pass statistics."),
      ::clEnumValN(
        commonNodeEliminationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(commonNodeEliminationStatisticsId),
        "Collect common node elimination pass statistics."),
      ::clEnumValN(
        controlFlowRecoveryStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(controlFlowRecoveryStatisticsId),
        "Collect control flow recovery pass statistics."),
      ::clEnumValN(
        dataNodeToDeltaStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(dataNodeToDeltaStatisticsId),
        "Collect data node to delta node conversion pass statistics."),
      ::clEnumValN(
        deadNodeEliminationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(deadNodeEliminationStatisticsId),
        "Collect dead node elimination pass statistics."),
      ::clEnumValN(
        functionInliningStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(functionInliningStatisticsId),
        "Collect function inlining pass statistics."),
      ::clEnumValN(
        invariantValueRedirectionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(invariantValueRedirectionStatisticsId),
        "Collect invariant value redirection pass statistics."),
      ::clEnumValN(
        jlmToRvsdgConversionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(jlmToRvsdgConversionStatisticsId),
        "Collect Jlm to RVSDG conversion pass statistics."),
      ::clEnumValN(
        loopUnrollingStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(loopUnrollingStatisticsId),
        "Collect loop unrolling pass statistics."),
      ::clEnumValN(
        memoryNodeProvisioningStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(memoryNodeProvisioningStatisticsId),
        "Collect memory node provisioning pass statistics."),
      ::clEnumValN(
        pullNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(pullNodesStatisticsId),
        "Collect node pull pass statistics."),
      ::clEnumValN(
        pushNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(pushNodesStatisticsId),
        "Collect node push pass statistics."),
      ::clEnumValN(
        reduceNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(reduceNodesStatisticsId),
        "Collect node reduction pass statistics."),
      ::clEnumValN(
        rvsdgConstructionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgConstructionStatisticsId),
        "Collect RVSDG construction pass statistics."),
      ::clEnumValN(
        rvsdgDestructionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgDestructionStatisticsId),
        "Collect RVSDG destruction pass statistics."),
      ::clEnumValN(
        rvsdgOptimizationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgOptimizationStatisticsId),
        "Collect RVSDG optimization pass statistics."),
      ::clEnumValN(
        steensgaardAnalysisStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(steensgaardAnalysisStatisticsId),
        "Collect Steensgaard alias analysis pass statistics."),
      ::clEnumValN(
        thetaGammaInversionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(thetaGammaInversionStatisticsId),
        "Collect theta-gamma inversion pass statistics.")),
    cl::desc("Collect jlm-opt pass statistics"));

  cl::ParseCommandLineOptions(argc, argv);

  /* Process parsed options */

  static std::unordered_map<std::string, JlcCommandLineOptions::OptimizationLevel> optimizationLevelMap(
    {
      {"0", JlcCommandLineOptions::OptimizationLevel::O0},
      {"1", JlcCommandLineOptions::OptimizationLevel::O1},
      {"2", JlcCommandLineOptions::OptimizationLevel::O2},
      {"3", JlcCommandLineOptions::OptimizationLevel::O3}
    });

  static std::unordered_map<std::string, JlcCommandLineOptions::LanguageStandard> languageStandardMap(
    {
      {"gnu89", JlcCommandLineOptions::LanguageStandard::Gnu89},
      {"gnu99", JlcCommandLineOptions::LanguageStandard::Gnu99},
      {"c89",   JlcCommandLineOptions::LanguageStandard::C89},
      {"c90",   JlcCommandLineOptions::LanguageStandard::C99},
      {"c99",   JlcCommandLineOptions::LanguageStandard::C99},
      {"c11",   JlcCommandLineOptions::LanguageStandard::C11},
      {"c++98", JlcCommandLineOptions::LanguageStandard::Cpp98},
      {"c++03", JlcCommandLineOptions::LanguageStandard::Cpp03},
      {"c++11", JlcCommandLineOptions::LanguageStandard::Cpp11},
      {"c++14", JlcCommandLineOptions::LanguageStandard::Cpp14}
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

  auto jlmOptOptimizations = checkAndConvertJlmOptOptimizations(
    jlmOptimizations,
    CommandLineOptions_.OptimizationLevel_);

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
    {
      jlmOptPassStatistics.begin(),
      jlmOptPassStatistics.end()
    });
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
                                                    util::filepath(""),
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
                                                  mF.empty() ? ToDependencyFile(inputFile) : util::filepath(mF),
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
      CommandLineOptions_.OutputFile_ = util::filepath(outputFile);
    }
  }

  return CommandLineOptions_;
}

JlmOptCommandLineParser::~JlmOptCommandLineParser() noexcept
= default;

const JlmOptCommandLineOptions &
JlmOptCommandLineParser::ParseCommandLineArguments(int argc, char **argv)
{
  using namespace ::llvm;

  /*
    FIXME: The command line parser setup is currently redone
    for every invocation of parse_cmdline. We should be able
    to do it only once and then reset the parser on every
    invocation of parse_cmdline.
  */

  cl::TopLevelSubCommand->reset();

  util::StatisticsCollectorSettings statisticsCollectorSettings;

  cl::opt<std::string> inputFile(
    cl::Positional,
    cl::desc("<input>"));

  cl::opt<std::string> outputFile(
    "o",
    cl::init(""),
    cl::desc("Write output to <file>"),
    cl::value_desc("file"));

  const auto statisticFileDesc = "Write stats to <file>. Default is "
                                 + statisticsCollectorSettings.GetFilePath().to_str()
                                 + ".";
  cl::opt<std::string> statisticFile(
    "s",
    cl::init(statisticsCollectorSettings.GetFilePath().to_str()),
    cl::desc(statisticFileDesc),
    cl::value_desc("file"));

  auto aggregationStatisticsId = util::Statistics::Id::Aggregation;
  auto annotationStatisticsId = util::Statistics::Id::Annotation;
  auto basicEncoderEncodingStatisticsId = util::Statistics::Id::BasicEncoderEncoding;
  auto commonNodeEliminationStatisticsId = util::Statistics::Id::CommonNodeElimination;
  auto controlFlowRecoveryStatisticsId = util::Statistics::Id::ControlFlowRecovery;
  auto dataNodeToDeltaStatisticsId = util::Statistics::Id::DataNodeToDelta;
  auto deadNodeEliminationStatisticsId = util::Statistics::Id::DeadNodeElimination;
  auto functionInliningStatisticsId = util::Statistics::Id::FunctionInlining;
  auto invariantValueRedirectionStatisticsId = util::Statistics::Id::InvariantValueRedirection;
  auto jlmToRvsdgConversionStatisticsId = util::Statistics::Id::JlmToRvsdgConversion;
  auto loopUnrollingStatisticsId = util::Statistics::Id::LoopUnrolling;
  auto memoryNodeProvisioningStatisticsId = util::Statistics::Id::MemoryNodeProvisioning;
  auto pullNodesStatisticsId = util::Statistics::Id::PullNodes;
  auto pushNodesStatisticsId = util::Statistics::Id::PushNodes;
  auto reduceNodesStatisticsId = util::Statistics::Id::ReduceNodes;
  auto rvsdgConstructionStatisticsId = util::Statistics::Id::RvsdgConstruction;
  auto rvsdgDestructionStatisticsId = util::Statistics::Id::RvsdgDestruction;
  auto rvsdgOptimizationStatisticsId = util::Statistics::Id::RvsdgOptimization;
  auto steensgaardAnalysisStatisticsId = util::Statistics::Id::SteensgaardAnalysis;
  auto thetaGammaInversionStatisticsId = util::Statistics::Id::ThetaGammaInversion;

  cl::list<util::Statistics::Id> printStatistics(
    cl::values(
      ::clEnumValN(
        aggregationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(aggregationStatisticsId),
        "Write aggregation statistics to file."),
      ::clEnumValN(
        annotationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(annotationStatisticsId),
        "Write annotation statistics to file."),
      ::clEnumValN(
        basicEncoderEncodingStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(basicEncoderEncodingStatisticsId),
        "Write encoding statistics of basic encoder to file."),
      ::clEnumValN(
        commonNodeEliminationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(commonNodeEliminationStatisticsId),
        "Write common node elimination statistics to file."),
      ::clEnumValN(
        controlFlowRecoveryStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(controlFlowRecoveryStatisticsId),
        "Write control flow recovery statistics to file."),
      ::clEnumValN(
        dataNodeToDeltaStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(dataNodeToDeltaStatisticsId),
        "Write data node to delta node conversion statistics to file."),
      ::clEnumValN(
        deadNodeEliminationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(deadNodeEliminationStatisticsId),
        "Write dead node elimination statistics to file."),
      ::clEnumValN(
        functionInliningStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(functionInliningStatisticsId),
        "Write function inlining statistics to file."),
      ::clEnumValN(
        invariantValueRedirectionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(invariantValueRedirectionStatisticsId),
        "Write invariant value redirection statistics to file."),
      ::clEnumValN(
        jlmToRvsdgConversionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(jlmToRvsdgConversionStatisticsId),
        "Write Jlm to RVSDG conversion statistics to file."),
      ::clEnumValN(
        loopUnrollingStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(loopUnrollingStatisticsId),
        "Write loop unrolling statistics to file."),
      ::clEnumValN(
        memoryNodeProvisioningStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(memoryNodeProvisioningStatisticsId),
        "Write memory node provisioning statistics to file."),
      ::clEnumValN(
        pullNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(pullNodesStatisticsId),
        "Write node pull statistics to file."),
      ::clEnumValN(
        pushNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(pushNodesStatisticsId),
        "Write node push statistics to file."),
      ::clEnumValN(
        reduceNodesStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(reduceNodesStatisticsId),
        "Write node reduction statistics to file."),
      ::clEnumValN(
        rvsdgConstructionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgConstructionStatisticsId),
        "Write RVSDG construction statistics to file."),
      ::clEnumValN(
        rvsdgDestructionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgDestructionStatisticsId),
        "Write RVSDG destruction statistics to file."),
      ::clEnumValN(
        rvsdgOptimizationStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(rvsdgOptimizationStatisticsId),
        "Write RVSDG optimization statistics to file."),
      ::clEnumValN(
        steensgaardAnalysisStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(steensgaardAnalysisStatisticsId),
        "Write Steensgaard analysis statistics to file."),
      ::clEnumValN(
        thetaGammaInversionStatisticsId,
        JlmOptCommandLineOptions::ToCommandLineArgument(thetaGammaInversionStatisticsId),
        "Write theta-gamma inversion statistics to file.")),
    cl::desc("Write statistics"));

  auto llvmOutputFormat = JlmOptCommandLineOptions::OutputFormat::Llvm;
  auto xmlOutputFormat = JlmOptCommandLineOptions::OutputFormat::Xml;

  cl::opt<JlmOptCommandLineOptions::OutputFormat> outputFormat(
    cl::values(
      ::clEnumValN(
        llvmOutputFormat,
        JlmOptCommandLineOptions::ToCommandLineArgument(llvmOutputFormat),
        "Output LLVM IR [default]"),
      ::clEnumValN(
        xmlOutputFormat,
        JlmOptCommandLineOptions::ToCommandLineArgument(xmlOutputFormat),
        "Output XML")),
    cl::init(llvmOutputFormat),
    cl::desc("Select output format"));

  auto aASteensgaardAgnostic = JlmOptCommandLineOptions::OptimizationId::AASteensgaardAgnostic;
  auto aASteensgaardRegionAware = JlmOptCommandLineOptions::OptimizationId::AASteensgaardRegionAware;
  auto commonNodeElimination = JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination;
  auto commonNodeEliminationDeprecated = JlmOptCommandLineOptions::OptimizationId::cne;
  auto deadNodeElimination = JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination;
  auto deadNodeEliminationDeprecated = JlmOptCommandLineOptions::OptimizationId::dne;
  auto functionInlining = JlmOptCommandLineOptions::OptimizationId::FunctionInlining;
  auto functionInliningDeprecated = JlmOptCommandLineOptions::OptimizationId::iln;
  auto invariantValueRedirection = JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection;
  auto nodePushOut = JlmOptCommandLineOptions::OptimizationId::NodePushOut;
  auto nodePushOutDeprecated = JlmOptCommandLineOptions::OptimizationId::psh;
  auto nodePullIn = JlmOptCommandLineOptions::OptimizationId::NodePullIn;
  auto nodePullInDeprecated = JlmOptCommandLineOptions::OptimizationId::pll;
  auto nodeReduction = JlmOptCommandLineOptions::OptimizationId::NodeReduction;
  auto nodeReductionDeprecated = JlmOptCommandLineOptions::OptimizationId::red;
  auto thetaGammaInversion = JlmOptCommandLineOptions::OptimizationId::ThetaGammaInversion;
  auto thetaGammaInversionDeprecated = JlmOptCommandLineOptions::OptimizationId::ivt;
  auto loopUnrolling = JlmOptCommandLineOptions::OptimizationId::LoopUnrolling;
  auto loopUnrollingDeprecated = JlmOptCommandLineOptions::OptimizationId::url;

  cl::list<JlmOptCommandLineOptions::OptimizationId> optimizationIds(
    cl::values(
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
        commonNodeEliminationDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(commonNodeEliminationDeprecated),
        "[Deprecated] Use --CommonNodeElimination flag instead"),
      ::clEnumValN(
        deadNodeElimination,
        JlmOptCommandLineOptions::ToCommandLineArgument(deadNodeElimination),
        "Dead Node Elimination"),
      ::clEnumValN(
        deadNodeEliminationDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(deadNodeEliminationDeprecated),
        "[Deprecated] Use --DeadNodeElimination flag instead"),
      ::clEnumValN(
        functionInlining,
        JlmOptCommandLineOptions::ToCommandLineArgument(functionInlining),
        "Function Inlining"),
      ::clEnumValN(
        functionInliningDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(functionInliningDeprecated),
        "[Deprecated] Use --FunctionInlining flag instead"),
      ::clEnumValN(
        invariantValueRedirection,
        JlmOptCommandLineOptions::ToCommandLineArgument(invariantValueRedirection),
        "Invariant Value Redirection"),
      ::clEnumValN(
        nodePushOut,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodePushOut),
        "Node Push Out"),
      ::clEnumValN(
        nodePushOutDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodePushOutDeprecated),
        "[Deprecated] Use --NodePushOut flag instead"),
      ::clEnumValN(
        nodePullIn,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodePullIn),
        "Node Pull In"),
      ::clEnumValN(
        nodePullInDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodePullInDeprecated),
        "[Deprecated] Use --NodePullIn flag instead"),
      ::clEnumValN(
        nodeReduction,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodeReduction),
        "Node Reduction"),
      ::clEnumValN(
        nodeReductionDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(nodeReductionDeprecated),
        "[Deprecated] Use --NodeReduction flag instead"),
      ::clEnumValN(
        thetaGammaInversion,
        JlmOptCommandLineOptions::ToCommandLineArgument(thetaGammaInversion),
        "Theta-Gamma Inversion"),
      ::clEnumValN(
        thetaGammaInversionDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(thetaGammaInversionDeprecated),
        "[Deprecated] Use --ThetaGammaInversion flag instead"),
      ::clEnumValN(
        loopUnrolling,
        JlmOptCommandLineOptions::ToCommandLineArgument(loopUnrolling),
        "Loop Unrolling"),
      ::clEnumValN(
        loopUnrollingDeprecated,
        JlmOptCommandLineOptions::ToCommandLineArgument(loopUnrollingDeprecated),
        "[Deprecated] Use --LoopUnrolling flag instead")),
    cl::desc("Perform optimization"));

  cl::ParseCommandLineOptions(argc, argv);

  statisticsCollectorSettings.SetFilePath(statisticFile);

  util::HashSet<util::Statistics::Id> demandedStatistics({printStatistics.begin(), printStatistics.end()});
  statisticsCollectorSettings.SetDemandedStatistics(std::move(demandedStatistics));

  CommandLineOptions_ = JlmOptCommandLineOptions::Create(
    inputFile,
    outputFile,
    outputFormat,
    std::move(statisticsCollectorSettings),
    std::move(optimizationIds));

  return *CommandLineOptions_;
}

const JlmOptCommandLineOptions &
JlmOptCommandLineParser::Parse(int argc, char ** argv)
{
  static JlmOptCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

JlmHlsCommandLineParser::~JlmHlsCommandLineParser() noexcept
= default;

const JlmHlsCommandLineOptions &
JlmHlsCommandLineParser::ParseCommandLineArguments(int argc, char ** argv)
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

  cl::opt<std::string> inputFile(
    cl::Positional,
    cl::desc("<input>"));

  cl::opt<std::string> outputFolder(
    "o",
    cl::desc("Write output to <folder>"),
    cl::value_desc("folder"));

  cl::opt<std::string> hlsFunction(
    "hls-function",
    cl::Prefix,
    cl::desc("Function that should be accelerated"),
    cl::value_desc("hls-function"));

  cl::opt<bool> extractHlsFunction(
    "extract",
    cl::Prefix,
    cl::desc("Extracts function specified by hls-function"));

  cl::opt<bool> useCirct(
    "circt",
    cl::Prefix,
    cl::desc("Use CIRCT to generate FIRRTL"));

  cl::opt<JlmHlsCommandLineOptions::OutputFormat> format(
    cl::values(
      ::clEnumValN(
        JlmHlsCommandLineOptions::OutputFormat::Firrtl,
        "fir",
        "Output FIRRTL [default]"),
      ::clEnumValN(
        JlmHlsCommandLineOptions::OutputFormat::Dot,
        "dot",
        "Output DOT graph")),
    cl::desc("Select output format"));

  cl::ParseCommandLineOptions(argc, argv);

  if (outputFolder.empty())
    throw jlm::util::error("jlm-hls no output directory provided, i.e, -o.\n");

  if (extractHlsFunction && hlsFunction.empty())
    throw jlm::util::error("jlm-hls: --hls-function is not specified.\n         which is required for --extract\n");

  CommandLineOptions_.InputFile_ = inputFile;
  CommandLineOptions_.HlsFunction_ = std::move(hlsFunction);
  CommandLineOptions_.OutputFolder_ = outputFolder;
  CommandLineOptions_.ExtractHlsFunction_ = extractHlsFunction;
  CommandLineOptions_.UseCirct_ = useCirct;
  CommandLineOptions_.OutputFormat_ = format;

  return CommandLineOptions_;
}

const JlmHlsCommandLineOptions &
JlmHlsCommandLineParser::Parse(int argc, char ** argv)
{
  static JlmHlsCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

JhlsCommandLineParser::~JhlsCommandLineParser() noexcept
= default;

const JhlsCommandLineOptions &
JhlsCommandLineParser::ParseCommandLineArguments(int argc, char **argv)
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

  cl::opt<bool> generateFirrtl(
    "firrtl",
    cl::ValueDisallowed,
    cl::desc("Generate firrtl"));

  cl::opt<bool> useCirct(
    "circt",
    cl::Prefix,
    cl::desc("Use CIRCT to generate FIRRTL"));

  cl::ParseCommandLineOptions(argc, argv);

  /* Process parsed options */

  static std::unordered_map<std::string, JhlsCommandLineOptions::OptimizationLevel> Olvlmap(
    {
      {"0", JhlsCommandLineOptions::OptimizationLevel::O0},
      {"1", JhlsCommandLineOptions::OptimizationLevel::O1},
      {"2", JhlsCommandLineOptions::OptimizationLevel::O2},
      {"3", JhlsCommandLineOptions::OptimizationLevel::O3}
    });

  static std::unordered_map<std::string, JhlsCommandLineOptions::LanguageStandard> stdmap(
    {
      {"Gnu89", JhlsCommandLineOptions::LanguageStandard::Gnu89},
      {"Gnu99", JhlsCommandLineOptions::LanguageStandard::Gnu99},
      {"C89", JhlsCommandLineOptions::LanguageStandard::C89},
      {"C99", JhlsCommandLineOptions::LanguageStandard::C99},
      {"C11", JhlsCommandLineOptions::LanguageStandard::C11},
      {"C++98", JhlsCommandLineOptions::LanguageStandard::Cpp98},
      {"C++03", JhlsCommandLineOptions::LanguageStandard::Cpp03},
      {"C++11", JhlsCommandLineOptions::LanguageStandard::Cpp11},
      {"C++14", JhlsCommandLineOptions::LanguageStandard::Cpp14}
    });

  if (!optimizationLevel.empty()) {
    auto it = Olvlmap.find(optimizationLevel);
    if (it == Olvlmap.end()) {
      std::cerr << "Unknown optimization level.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.OptimizationLevel_ = it->second;
  }

  if (!languageStandard.empty()) {
    auto it = stdmap.find(languageStandard);
    if (it == stdmap.end()) {
      std::cerr << "Unknown language standard.\n";
      exit(EXIT_FAILURE);
    }
    CommandLineOptions_.LanguageStandard_ = it->second;
  }

  if (inputFiles.empty()) {
    std::cerr << "jlc: no input files.\n";
    exit(EXIT_FAILURE);
  }

  if (inputFiles.size() > 1 && noLinking && !outputFile.empty()) {
    std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
    exit(EXIT_FAILURE);
  }

  if (!hlsFunction.empty()) {
    CommandLineOptions_.Hls_ = true;
    CommandLineOptions_.HlsFunctionRegex_ = hlsFunction.front();
  }

  if (hlsFunction.size() > 1) {
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
  CommandLineOptions_.UseCirct_ = useCirct;

  for (auto & inputFile : inputFiles) {
    if (IsObjectFile(inputFile)) {
      /* FIXME: print a warning like clang if noLinking is true */
      CommandLineOptions_.Compilations_.push_back({
                                                    inputFile,
                                                    util::filepath(""),
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
                                                  mF.empty() ? CreateDependencyFileFromFile(inputFile) : util::filepath(mF),
                                                  CreateObjectFileFromFile(inputFile),
                                                  mT.empty() ? CreateObjectFileFromFile(inputFile).name() : mT,
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
      CommandLineOptions_.OutputFile_ = util::filepath(outputFile);
    }
  }

  return CommandLineOptions_;
}

bool
JhlsCommandLineParser::IsObjectFile(const util::filepath & file)
{
  return file.suffix() == "o";
}

util::filepath
JhlsCommandLineParser::CreateObjectFileFromFile(const util::filepath & f)
{
  return {f.path() + f.base() + ".o"};
}

util::filepath
JhlsCommandLineParser::CreateDependencyFileFromFile(const util::filepath & f)
{
  return {f.path() + f.base() + ".d"};
}

const JhlsCommandLineOptions &
JhlsCommandLineParser::Parse(int argc, char ** argv)
{
  static JhlsCommandLineParser parser;
  return parser.ParseCommandLineArguments(argc, argv);
}

}