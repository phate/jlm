/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

namespace jlm
{

JlcCommandGraphGenerator::~JlcCommandGraphGenerator() noexcept
= default;

filepath
JlcCommandGraphGenerator::CreateJlmOptCommandOutputFile(const filepath & inputFile)
{
  return strfmt("/tmp/tmp-", inputFile.base(), "-jlm-opt-out.ll");
}

filepath
JlcCommandGraphGenerator::CreateParserCommandOutputFile(const filepath & inputFile)
{
  return strfmt("/tmp/tmp-", inputFile.base(), "-clang-out.ll");
}

ClangCommand::LanguageStandard
JlcCommandGraphGenerator::ConvertLanguageStandard(const JlcCommandLineOptions::LanguageStandard & languageStandard)
{
  static std::unordered_map<JlcCommandLineOptions::LanguageStandard, ClangCommand::LanguageStandard>
    map({
          {JlcCommandLineOptions::LanguageStandard::None,  ClangCommand::LanguageStandard::Unspecified},
          {JlcCommandLineOptions::LanguageStandard::Gnu89, ClangCommand::LanguageStandard::Gnu89},
          {JlcCommandLineOptions::LanguageStandard::Gnu99, ClangCommand::LanguageStandard::Gnu99},
          {JlcCommandLineOptions::LanguageStandard::C89,   ClangCommand::LanguageStandard::C89},
          {JlcCommandLineOptions::LanguageStandard::C99,   ClangCommand::LanguageStandard::C99},
          {JlcCommandLineOptions::LanguageStandard::C11,   ClangCommand::LanguageStandard::C11},
          {JlcCommandLineOptions::LanguageStandard::Cpp98, ClangCommand::LanguageStandard::Cpp98},
          {JlcCommandLineOptions::LanguageStandard::Cpp03, ClangCommand::LanguageStandard::Cpp03},
          {JlcCommandLineOptions::LanguageStandard::Cpp11, ClangCommand::LanguageStandard::Cpp11},
          {JlcCommandLineOptions::LanguageStandard::Cpp14, ClangCommand::LanguageStandard::Cpp14}
        });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

LlcCommand::OptimizationLevel
JlcCommandGraphGenerator::ConvertOptimizationLevel(const JlcCommandLineOptions::OptimizationLevel & optimizationLevel)
{
    static std::unordered_map<JlcCommandLineOptions::OptimizationLevel, LlcCommand::OptimizationLevel>
    map({
          {JlcCommandLineOptions::OptimizationLevel::O0, LlcCommand::OptimizationLevel::O0},
          {JlcCommandLineOptions::OptimizationLevel::O1, LlcCommand::OptimizationLevel::O1},
          {JlcCommandLineOptions::OptimizationLevel::O2, LlcCommand::OptimizationLevel::O2},
          {JlcCommandLineOptions::OptimizationLevel::O3, LlcCommand::OptimizationLevel::O3}
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

CommandGraph::Node &
JlcCommandGraphGenerator::CreateParserCommand(
  CommandGraph & commandGraph,
  const JlcCommandLineOptions::Compilation & compilation,
  const JlcCommandLineOptions & commandLineOptions)
{
  return ClangCommand::CreateParsingCommand(
    commandGraph,
    compilation.InputFile(),
    CreateParserCommandOutputFile(compilation.InputFile()),
    compilation.DependencyFile(),
    commandLineOptions.IncludePaths_,
    commandLineOptions.MacroDefinitions_,
    commandLineOptions.Warnings_,
    commandLineOptions.Flags_,
    commandLineOptions.Verbose_,
    commandLineOptions.Rdynamic_,
    commandLineOptions.Suppress_,
    commandLineOptions.UsePthreads_,
    commandLineOptions.Md_,
    compilation.Mt(),
    ConvertLanguageStandard(commandLineOptions.LanguageStandard_),
    {});
}

std::unique_ptr<CommandGraph>
JlcCommandGraphGenerator::GenerateCommandGraph(const JlcCommandLineOptions & commandLineOptions)
{
  auto commandGraph = CommandGraph::Create();

  std::vector<CommandGraph::Node *> leafNodes;
  for (auto & compilation: commandLineOptions.Compilations_)
  {
    auto lastNode = &commandGraph->GetEntryNode();

    if (compilation.RequiresParsing())
    {
      auto & parserCommandNode = CreateParserCommand(
        *commandGraph,
        compilation,
        commandLineOptions);

      lastNode->AddEdge(parserCommandNode);
      lastNode = &parserCommandNode;
    }

    if (compilation.RequiresOptimization())
    {
      /*
       * If a default optimization level has been specified (-O) and no specific jlm options
       * have been specified (-J) then use a default set of optimizations.
       */
      std::vector<JlmOptCommand::Optimization> optimizations;
      if (commandLineOptions.JlmOptOptimizations_.empty()
          && commandLineOptions.OptimizationLevel_ == JlcCommandLineOptions::OptimizationLevel::O3)
      {
        /*
         * Only -O3 sets default optimizations
         */
        optimizations = {
          JlmOptCommand::Optimization::FunctionInlining,
          JlmOptCommand::Optimization::InvariantValueRedirection,
          JlmOptCommand::Optimization::NodeReduction,
          JlmOptCommand::Optimization::DeadNodeElimination,
          JlmOptCommand::Optimization::ThetaGammaInversion,
          JlmOptCommand::Optimization::InvariantValueRedirection,
          JlmOptCommand::Optimization::DeadNodeElimination,
          JlmOptCommand::Optimization::NodePushOut,
          JlmOptCommand::Optimization::InvariantValueRedirection,
          JlmOptCommand::Optimization::DeadNodeElimination,
          JlmOptCommand::Optimization::NodeReduction,
          JlmOptCommand::Optimization::CommonNodeElimination,
          JlmOptCommand::Optimization::DeadNodeElimination,
          JlmOptCommand::Optimization::NodePullIn,
          JlmOptCommand::Optimization::InvariantValueRedirection,
          JlmOptCommand::Optimization::DeadNodeElimination,
          JlmOptCommand::Optimization::LoopUnrolling,
          JlmOptCommand::Optimization::InvariantValueRedirection
        };
      }

      auto & jlmOptCommandNode = JlmOptCommand::Create(
        *commandGraph,
        CreateParserCommandOutputFile(compilation.InputFile()),
        CreateJlmOptCommandOutputFile(compilation.InputFile()),
        optimizations);
      lastNode->AddEdge(jlmOptCommandNode);
      lastNode = &jlmOptCommandNode;
    }

    if (compilation.RequiresAssembly())
    {
      auto &llvmLlcCommandNode = LlcCommand::Create(
        *commandGraph,
        CreateJlmOptCommandOutputFile(compilation.InputFile()),
        compilation.OutputFile(),
        ConvertOptimizationLevel(commandLineOptions.OptimizationLevel_),
        LlcCommand::RelocationModel::Static);
      lastNode->AddEdge(llvmLlcCommandNode);
      lastNode = &llvmLlcCommandNode;
    }

    leafNodes.push_back(lastNode);
  }

  std::vector<filepath> linkerInputFiles;
  for (auto & compilation: commandLineOptions.Compilations_)
  {
    if (compilation.RequiresLinking())
      linkerInputFiles.push_back(compilation.OutputFile());
  }

  if (!linkerInputFiles.empty())
  {
    auto &linkerCommandNode = ClangCommand::CreateLinkerCommand(
      *commandGraph,
      linkerInputFiles,
      commandLineOptions.OutputFile_,
      commandLineOptions.LibraryPaths_,
      commandLineOptions.Libraries_,
      commandLineOptions.UsePthreads_);

    for (const auto &leafNode: leafNodes)
      leafNode->AddEdge(linkerCommandNode);

    leafNodes.clear();
    leafNodes.push_back(&linkerCommandNode);
  }

  for (auto & leafNode: leafNodes)
    leafNode->AddEdge(commandGraph->GetExitNode());

  if (commandLineOptions.OnlyPrintCommands_)
    commandGraph = PrintCommandsCommand::Create(std::move(commandGraph));

  return commandGraph;
}

}