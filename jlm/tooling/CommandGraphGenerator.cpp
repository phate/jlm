/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

#include <unistd.h>

namespace jlm::tooling
{

JlcCommandGraphGenerator::~JlcCommandGraphGenerator() noexcept = default;

util::filepath
JlcCommandGraphGenerator::CreateJlmOptCommandOutputFile(const util::filepath & inputFile)
{
  return util::filepath::CreateUniqueFile(
      std::filesystem::temp_directory_path().string(),
      inputFile.base() + "-",
      "-jlm-opt.ll");
}

util::filepath
JlcCommandGraphGenerator::CreateParserCommandOutputFile(const util::filepath & inputFile)
{
  return util::filepath::CreateUniqueFile(
      std::filesystem::temp_directory_path().string(),
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
    const util::filepath & outputFile,
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
      auto clangCommand = util::AssertedCast<ClangCommand>(&lastNode->GetCommand());
      auto statisticsFilePath = util::StatisticsCollectorSettings::CreateUniqueStatisticsFile(
          util::filepath(std::filesystem::temp_directory_path()),
          compilation.InputFile());
      util::StatisticsCollectorSettings statisticsCollectorSettings(
          statisticsFilePath,
          commandLineOptions.JlmOptPassStatistics_);

      JlmOptCommandLineOptions jlmOptCommandLineOptions(
          clangCommand->OutputFile(),
          CreateJlmOptCommandOutputFile(compilation.InputFile()),
          JlmOptCommandLineOptions::OutputFormat::Llvm,
          statisticsCollectorSettings,
          commandLineOptions.JlmOptOptimizations_);

      auto & jlmOptCommandNode =
          JlmOptCommand::Create(*commandGraph, "jlm-opt", std::move(jlmOptCommandLineOptions));
      lastNode->AddEdge(jlmOptCommandNode);
      lastNode = &jlmOptCommandNode;
    }

    if (compilation.RequiresAssembly())
    {
      auto jlmOptCommand = util::AssertedCast<JlmOptCommand>(&lastNode->GetCommand());
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

  std::vector<util::filepath> linkerInputFiles;
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

util::filepath
JhlsCommandGraphGenerator::CreateParserCommandOutputFile(
    const util::filepath & tmpDirectory,
    const util::filepath & inputFile)
{
  return util::filepath::CreateUniqueFile(tmpDirectory, inputFile.base() + "-", "-clang.ll");
}

util::filepath
JhlsCommandGraphGenerator::CreateJlmOptCommandOutputFile(
    const util::filepath & tmpDirectory,
    const util::filepath & inputFile)
{
  return util::filepath::CreateUniqueFile(tmpDirectory, inputFile.base() + "-", "-jlm-opt.ll");
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
  std::vector<util::filepath> llir_files;

  // Create directory in /tmp for storing temporary files
  std::string tmp_identifier;
  for (const auto & compilation : commandLineOptions.Compilations_)
  {
    tmp_identifier += compilation.InputFile().name() + "_";
    if (tmp_identifier.length() > 30)
      break;
  }
  srandom((unsigned)time(nullptr) * getpid());
  tmp_identifier += std::to_string(random());
  util::filepath tmp_folder(
      std::filesystem::temp_directory_path().string() + "/" + tmp_identifier + "/");
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
          CreateParserCommandOutputFile(tmp_folder, compilation.InputFile()).to_str(),
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
  util::filepath ll_merged(tmp_folder.to_str() + "merged.ll");
  auto & ll_link = LlvmLinkCommand::Create(*commandGraph, llir_files, ll_merged, true, true);
  // Add edges between each c.parse and the ll_link
  for (const auto & ll : llir)
  {
    ll->AddEdge(ll_link);
  }

  // need to already run m2r here
  util::filepath ll_m2r1(tmp_folder.to_str() + "merged.m2r.ll");
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
      tmp_folder.to_str());
  m2r1.AddEdge(extract);
  util::filepath ll_m2r2(tmp_folder.to_str() + "function.m2r.ll");
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
      tmp_folder.to_str(),
      commandLineOptions.UseCirct_);
  m2r2.AddEdge(hls);

  if (!commandLineOptions.GenerateFirrtl_)
  {
    util::filepath verilogfile(tmp_folder.to_str() + "jlm_hls.v");
    auto & firrtl = FirtoolCommand::Create(
        *commandGraph,
        dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->FirrtlFile(),
        verilogfile);
    hls.AddEdge(firrtl);
    util::filepath assemblyFile(tmp_folder.to_str() + "hls.o");
    auto inputFile = dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->LlvmFile();
    auto & asmnode = LlcCommand::Create(
        *commandGraph,
        commandLineOptions.Hls_ ? inputFile
                                : CreateJlmOptCommandOutputFile(tmp_folder, inputFile).to_str(),
        assemblyFile,
        ConvertOptimizationLevel(commandLineOptions.OptimizationLevel_),
        commandLineOptions.Hls_ ? LlcCommand::RelocationModel::Pic
                                : LlcCommand::RelocationModel::Static);
    hls.AddEdge(asmnode);

    std::vector<util::filepath> lnkifiles;
    for (const auto & compilation : commandLineOptions.Compilations_)
    {
      if (compilation.RequiresLinking() && !compilation.RequiresParsing())
        lnkifiles.push_back(compilation.OutputFile());
    }
    lnkifiles.push_back(assemblyFile);
    auto & verilatorCommandNode = VerilatorCommand::Create(
        *commandGraph,
        verilogfile,
        lnkifiles,
        dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->HarnessFile(),
        commandLineOptions.OutputFile_,
        tmp_folder,
        commandLineOptions.LibraryPaths_,
        commandLineOptions.Libraries_);
    firrtl.AddEdge(verilatorCommandNode);
    verilatorCommandNode.AddEdge(commandGraph->GetExitNode());
  }

  std::vector<util::filepath> lnkifiles;
  for (const auto & c : commandLineOptions.Compilations_)
  {
    if (c.RequiresLinking())
      lnkifiles.push_back(c.OutputFile());
  }

  for (const auto & leave : leaves)
    leave->AddEdge(commandGraph->GetExitNode());

  if (commandLineOptions.OnlyPrintCommands_)
    commandGraph = PrintCommandsCommand::Create(std::move(commandGraph));

  return commandGraph;
}

}
