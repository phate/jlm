/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

#include <unistd.h>

namespace jlm::tooling
{

JlcCommandGraphGenerator::~JlcCommandGraphGenerator() noexcept = default;

util::FilePath
JlcCommandGraphGenerator::CreateJlmOptCommandOutputFile(const util::FilePath & inputFile)
{
  return util::FilePath::CreateUniqueFileName(
      util::FilePath::TempDirectoryPath(),
      inputFile.base() + "-",
      "-jlm-opt.ll");
}

util::FilePath
JlcCommandGraphGenerator::CreateParserCommandOutputFile(const util::FilePath & inputFile)
{
  return util::FilePath::CreateUniqueFileName(
      util::FilePath::TempDirectoryPath(),
      inputFile.base() + "-",
      "-clang.ll");
}

ClangCommand::LanguageStandard
JlcCommandGraphGenerator::ConvertLanguageStandard(
    const JlcCommandLineOptions::LanguageStandard & languageStandard)
{
  static std::unordered_map<JlcCommandLineOptions::LanguageStandard, ClangCommand::LanguageStandard>
      map({ { JlcCommandLineOptions::LanguageStandard::None,
              ClangCommand::LanguageStandard::Unspecified },
            { JlcCommandLineOptions::LanguageStandard::Gnu89,
              ClangCommand::LanguageStandard::Gnu89 },
            { JlcCommandLineOptions::LanguageStandard::Gnu99,
              ClangCommand::LanguageStandard::Gnu99 },
            { JlcCommandLineOptions::LanguageStandard::C89, ClangCommand::LanguageStandard::C89 },
            { JlcCommandLineOptions::LanguageStandard::C99, ClangCommand::LanguageStandard::C99 },
            { JlcCommandLineOptions::LanguageStandard::C11, ClangCommand::LanguageStandard::C11 },
            { JlcCommandLineOptions::LanguageStandard::Cpp98,
              ClangCommand::LanguageStandard::Cpp98 },
            { JlcCommandLineOptions::LanguageStandard::Cpp03,
              ClangCommand::LanguageStandard::Cpp03 },
            { JlcCommandLineOptions::LanguageStandard::Cpp11,
              ClangCommand::LanguageStandard::Cpp11 },
            { JlcCommandLineOptions::LanguageStandard::Cpp14,
              ClangCommand::LanguageStandard::Cpp14 } });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

LlcCommand::OptimizationLevel
JlcCommandGraphGenerator::ConvertOptimizationLevel(
    const JlcCommandLineOptions::OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<JlcCommandLineOptions::OptimizationLevel, LlcCommand::OptimizationLevel>
      map({ { JlcCommandLineOptions::OptimizationLevel::O0, LlcCommand::OptimizationLevel::O0 },
            { JlcCommandLineOptions::OptimizationLevel::O1, LlcCommand::OptimizationLevel::O1 },
            { JlcCommandLineOptions::OptimizationLevel::O2, LlcCommand::OptimizationLevel::O2 },
            { JlcCommandLineOptions::OptimizationLevel::O3, LlcCommand::OptimizationLevel::O3 } });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

CommandGraph::Node &
JlcCommandGraphGenerator::CreateParserCommand(
    CommandGraph & commandGraph,
    const util::FilePath & outputFile,
    const JlcCommandLineOptions::Compilation & compilation,
    const JlcCommandLineOptions & commandLineOptions)
{
  return ClangCommand::CreateParsingCommand(
      commandGraph,
      compilation.InputFile(),
      outputFile,
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
  for (auto & compilation : commandLineOptions.Compilations_)
  {
    auto lastNode = &commandGraph->GetEntryNode();

    if (compilation.RequiresParsing())
    {
      auto & parserCommandNode = CreateParserCommand(
          *commandGraph,
          CreateParserCommandOutputFile(compilation.InputFile()),
          compilation,
          commandLineOptions);

      lastNode->AddEdge(parserCommandNode);
      lastNode = &parserCommandNode;
    }

    if (compilation.RequiresOptimization())
    {
      auto clangCommand = util::assertedCast<ClangCommand>(&lastNode->GetCommand());

      util::StatisticsCollectorSettings statisticsCollectorSettings(
          commandLineOptions.JlmOptPassStatistics_,
          util::FilePath::TempDirectoryPath(),
          compilation.InputFile().base());

      JlmOptCommandLineOptions jlmOptCommandLineOptions(
          clangCommand->OutputFile(),
          JlmOptCommandLineOptions::InputFormat::Llvm,
          CreateJlmOptCommandOutputFile(compilation.InputFile()),
          JlmOptCommandLineOptions::OutputFormat::Llvm,
          std::move(statisticsCollectorSettings),
          jlm::llvm::RvsdgTreePrinter::Configuration({}),
          commandLineOptions.JlmOptOptimizations_,
          false);

      auto & jlmOptCommandNode =
          JlmOptCommand::Create(*commandGraph, "jlm-opt", std::move(jlmOptCommandLineOptions));
      lastNode->AddEdge(jlmOptCommandNode);
      lastNode = &jlmOptCommandNode;
    }

    if (compilation.RequiresAssembly())
    {
      auto jlmOptCommand = util::assertedCast<JlmOptCommand>(&lastNode->GetCommand());
      auto & llvmLlcCommandNode = LlcCommand::Create(
          *commandGraph,
          jlmOptCommand->GetCommandLineOptions().GetOutputFile(),
          compilation.OutputFile(),
          ConvertOptimizationLevel(commandLineOptions.OptimizationLevel_),
          LlcCommand::RelocationModel::Static);
      lastNode->AddEdge(llvmLlcCommandNode);
      lastNode = &llvmLlcCommandNode;
    }

    leafNodes.push_back(lastNode);
  }

  std::vector<util::FilePath> linkerInputFiles;
  for (auto & compilation : commandLineOptions.Compilations_)
  {
    if (compilation.RequiresLinking())
      linkerInputFiles.push_back(compilation.OutputFile());
  }

  if (!linkerInputFiles.empty())
  {
    auto & linkerCommandNode = ClangCommand::CreateLinkerCommand(
        *commandGraph,
        linkerInputFiles,
        commandLineOptions.OutputFile_,
        commandLineOptions.LibraryPaths_,
        commandLineOptions.Libraries_,
        commandLineOptions.UsePthreads_);

    for (const auto & leafNode : leafNodes)
      leafNode->AddEdge(linkerCommandNode);

    leafNodes.clear();
    leafNodes.push_back(&linkerCommandNode);
  }

  for (auto & leafNode : leafNodes)
    leafNode->AddEdge(commandGraph->GetExitNode());

  if (commandLineOptions.OnlyPrintCommands_)
    commandGraph = PrintCommandsCommand::Create(std::move(commandGraph));

  return commandGraph;
}

JhlsCommandGraphGenerator::~JhlsCommandGraphGenerator() noexcept = default;

util::FilePath
JhlsCommandGraphGenerator::CreateParserCommandOutputFile(
    const util::FilePath & tmpDirectory,
    const util::FilePath & inputFile)
{
  return util::FilePath::CreateUniqueFileName(tmpDirectory, inputFile.base() + "-", "-clang.ll");
}

util::FilePath
JhlsCommandGraphGenerator::CreateJlmOptCommandOutputFile(
    const util::FilePath & tmpDirectory,
    const util::FilePath & inputFile)
{
  return util::FilePath::CreateUniqueFileName(tmpDirectory, inputFile.base() + "-", "-jlm-opt.ll");
}

ClangCommand::LanguageStandard
JhlsCommandGraphGenerator::ConvertLanguageStandard(
    const JhlsCommandLineOptions::LanguageStandard & languageStandard)
{
  static std::unordered_map<
      JhlsCommandLineOptions::LanguageStandard,
      ClangCommand::LanguageStandard>
      map({ { JhlsCommandLineOptions::LanguageStandard::None,
              ClangCommand::LanguageStandard::Unspecified },
            { JhlsCommandLineOptions::LanguageStandard::Gnu89,
              ClangCommand::LanguageStandard::Gnu89 },
            { JhlsCommandLineOptions::LanguageStandard::Gnu99,
              ClangCommand::LanguageStandard::Gnu99 },
            { JhlsCommandLineOptions::LanguageStandard::C89, ClangCommand::LanguageStandard::C89 },
            { JhlsCommandLineOptions::LanguageStandard::C99, ClangCommand::LanguageStandard::C99 },
            { JhlsCommandLineOptions::LanguageStandard::C11, ClangCommand::LanguageStandard::C11 },
            { JhlsCommandLineOptions::LanguageStandard::Cpp98,
              ClangCommand::LanguageStandard::Cpp98 },
            { JhlsCommandLineOptions::LanguageStandard::Cpp03,
              ClangCommand::LanguageStandard::Cpp03 },
            { JhlsCommandLineOptions::LanguageStandard::Cpp11,
              ClangCommand::LanguageStandard::Cpp11 },
            { JhlsCommandLineOptions::LanguageStandard::Cpp14,
              ClangCommand::LanguageStandard::Cpp14 } });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

LlcCommand::OptimizationLevel
JhlsCommandGraphGenerator::ConvertOptimizationLevel(
    const JhlsCommandLineOptions::OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<
      JhlsCommandLineOptions::OptimizationLevel,
      LlcCommand::OptimizationLevel>
      map({ { JhlsCommandLineOptions::OptimizationLevel::O0, LlcCommand::OptimizationLevel::O0 },
            { JhlsCommandLineOptions::OptimizationLevel::O1, LlcCommand::OptimizationLevel::O1 },
            { JhlsCommandLineOptions::OptimizationLevel::O2, LlcCommand::OptimizationLevel::O2 },
            { JhlsCommandLineOptions::OptimizationLevel::O3, LlcCommand::OptimizationLevel::O3 } });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::unique_ptr<CommandGraph>
JhlsCommandGraphGenerator::GenerateCommandGraph(const JhlsCommandLineOptions & commandLineOptions)
{
  std::unique_ptr<CommandGraph> commandGraph(new CommandGraph());

  std::vector<CommandGraph::Node *> leaves;
  std::vector<CommandGraph::Node *> llir;
  std::vector<util::FilePath> llir_files;

  // Create directory in /tmp for storing temporary files
  std::string tmp_identifier = "jhls-";
  for (const auto & compilation : commandLineOptions.Compilations_)
  {
    tmp_identifier += compilation.InputFile().name() + "-";
    if (tmp_identifier.length() > 30)
      break;
  }

  const auto tmp_folder =
      util::FilePath::CreateUniqueFileName(util::FilePath::TempDirectoryPath(), tmp_identifier, "");
  auto & mkdir = MkdirCommand::Create(*commandGraph, tmp_folder);
  commandGraph->GetEntryNode().AddEdge(mkdir);

  for (const auto & compilation : commandLineOptions.Compilations_)
  {
    CommandGraph::Node * last = &mkdir;

    if (compilation.RequiresParsing())
    {
      auto & parserNode = ClangCommand::CreateParsingCommand(
          *commandGraph,
          compilation.InputFile(),
          CreateParserCommandOutputFile(tmp_folder, compilation.InputFile()),
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
          { ClangCommand::ClangArgument::DisableO0OptNone });

      last->AddEdge(parserNode);
      last = &parserNode;

      // HLS links all files to a single IR
      // Need to know when the IR has been generated for all input files
      llir.push_back(&parserNode);
      llir_files.push_back(dynamic_cast<ClangCommand *>(&parserNode.GetCommand())->OutputFile());
    }

    leaves.push_back(last);
  }

  // link all llir into one so inlining can be done across files for HLS
  util::FilePath ll_merged(tmp_folder.to_str() + "merged.ll");
  auto & ll_link = LlvmLinkCommand::Create(*commandGraph, llir_files, ll_merged, true, true);
  // Add edges between each c.parse and the ll_link
  for (const auto & ll : llir)
  {
    ll->AddEdge(ll_link);
  }

  // need to already run m2r here
  util::FilePath ll_m2r1(tmp_folder.to_str() + "merged.m2r.ll");
  auto & m2r1 = LlvmOptCommand::Create(
      *commandGraph,
      ll_merged,
      ll_m2r1,
      true,
      { LlvmOptCommand::Optimization::Mem2Reg });
  ll_link.AddEdge(m2r1);
  auto & extract = JlmHlsExtractCommand::Create(
      *commandGraph,
      dynamic_cast<LlvmOptCommand *>(&m2r1.GetCommand())->OutputFile(),
      commandLineOptions.HlsFunctionRegex_,
      commandLineOptions.OutputFile_);
  m2r1.AddEdge(extract);
  util::FilePath ll_m2r2(tmp_folder.to_str() + "function.hls.ll");
  auto & m2r2 = LlvmOptCommand::Create(
      *commandGraph,
      dynamic_cast<JlmHlsExtractCommand *>(&extract.GetCommand())->HlsFunctionFile(),
      ll_m2r2,
      true,
      { LlvmOptCommand::Optimization::Mem2Reg });
  extract.AddEdge(m2r2);
  // hls
  auto & hls = JlmHlsCommand::Create(
      *commandGraph,
      dynamic_cast<LlvmOptCommand *>(&m2r2.GetCommand())->OutputFile(),
      commandLineOptions.OutputFile_,
      commandLineOptions.JlmHls_);
  m2r2.AddEdge(hls);

  auto linkerInputFiles = util::FilePath(commandLineOptions.OutputFile_.to_str() + ".re*.ll");
  auto mergedFile = util::FilePath(commandLineOptions.OutputFile_.to_str() + ".merged.ll");
  auto & llvmLink =
      LlvmLinkCommand::Create(*commandGraph, { linkerInputFiles }, mergedFile, true, false);
  hls.AddEdge(llvmLink);

  auto & compileMerged = LlcCommand::Create(
      *commandGraph,
      mergedFile,
      util::FilePath(commandLineOptions.OutputFile_.to_str() + ".o"),
      LlcCommand::OptimizationLevel::O3,
      LlcCommand::RelocationModel::Pic);
  llvmLink.AddEdge(compileMerged);

  for (const auto & leave : leaves)
    leave->AddEdge(commandGraph->GetExitNode());

  if (commandLineOptions.OnlyPrintCommands_)
    commandGraph = PrintCommandsCommand::Create(std::move(commandGraph));

  return commandGraph;
}

}
