/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/tooling/Command.hpp>
#include <jlm/llvm/tooling/CommandGraph.hpp>
#include <jlm/llvm/tooling/CommandGraphGenerator.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

#include <unistd.h>

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
      std::vector<JlmOptCommand::Optimization> optimizations;
      if (!commandLineOptions.JlmOptOptimizations_.empty()) {
        static std::unordered_map<std::string, JlmOptCommand::Optimization>map(
        {
          {"AASteensgaardAgnostic", JlmOptCommand::Optimization::AASteensgaardAgnostic},
          {"AASteensgaardRegionAware", JlmOptCommand::Optimization::AASteensgaardRegionAware},
          {"cne", JlmOptCommand::Optimization::CommonNodeElimination},
          {"dne", JlmOptCommand::Optimization::DeadNodeElimination},
          {"iln", JlmOptCommand::Optimization::FunctionInlining},
          {"InvariantValueRedirection", JlmOptCommand::Optimization::InvariantValueRedirection},
          {"psh", JlmOptCommand::Optimization::NodePushOut},
          {"pll", JlmOptCommand::Optimization::NodePullIn},
          {"red", JlmOptCommand::Optimization::NodeReduction},
          {"ivt", JlmOptCommand::Optimization::ThetaGammaInversion},
          {"url", JlmOptCommand::Optimization::LoopUnrolling},
        });
        for (const auto & jlmOpt : commandLineOptions.JlmOptOptimizations_)
	{
          JLM_ASSERT(map.find(jlmOpt) != map.end());
          optimizations.push_back(map[jlmOpt]);
	}
      /*
       * If a default optimization level has been specified (-O) and no specific jlm options
       * have been specified (-J) then use a default set of optimizations.
       */
      } else if (commandLineOptions.JlmOptOptimizations_.empty()
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

JhlsCommandGraphGenerator::~JhlsCommandGraphGenerator() noexcept
= default;

filepath
JhlsCommandGraphGenerator::CreateParserCommandOutputFile(const filepath & inputFile)
{
  return {"tmp-" + inputFile.base() + "-clang-out.ll"};
}

filepath
JhlsCommandGraphGenerator::CreateJlmOptCommandOutputFile(const filepath & inputFile)
{
  return {"tmp-" + inputFile.base() + "-jlm-opt-out.ll"};
}

ClangCommand::LanguageStandard
JhlsCommandGraphGenerator::ConvertLanguageStandard(const JhlsCommandLineOptions::LanguageStandard & languageStandard)
{
  static std::unordered_map<JhlsCommandLineOptions::LanguageStandard, ClangCommand::LanguageStandard>map(
    {
      {JhlsCommandLineOptions::LanguageStandard::None,  ClangCommand::LanguageStandard::Unspecified},
      {JhlsCommandLineOptions::LanguageStandard::Gnu89, ClangCommand::LanguageStandard::Gnu89},
      {JhlsCommandLineOptions::LanguageStandard::Gnu99, ClangCommand::LanguageStandard::Gnu99},
      {JhlsCommandLineOptions::LanguageStandard::C89,   ClangCommand::LanguageStandard::C89},
      {JhlsCommandLineOptions::LanguageStandard::C99,   ClangCommand::LanguageStandard::C99},
      {JhlsCommandLineOptions::LanguageStandard::C11,   ClangCommand::LanguageStandard::C11},
      {JhlsCommandLineOptions::LanguageStandard::Cpp98, ClangCommand::LanguageStandard::Cpp98},
      {JhlsCommandLineOptions::LanguageStandard::Cpp03, ClangCommand::LanguageStandard::Cpp03},
      {JhlsCommandLineOptions::LanguageStandard::Cpp11, ClangCommand::LanguageStandard::Cpp11},
      {JhlsCommandLineOptions::LanguageStandard::Cpp14, ClangCommand::LanguageStandard::Cpp14}
    });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

LlcCommand::OptimizationLevel
JhlsCommandGraphGenerator::ConvertOptimizationLevel(const JhlsCommandLineOptions::OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<JhlsCommandLineOptions::OptimizationLevel, LlcCommand::OptimizationLevel> map(
    {
      {JhlsCommandLineOptions::OptimizationLevel::O0, LlcCommand::OptimizationLevel::O0},
      {JhlsCommandLineOptions::OptimizationLevel::O1, LlcCommand::OptimizationLevel::O1},
      {JhlsCommandLineOptions::OptimizationLevel::O2, LlcCommand::OptimizationLevel::O2},
      {JhlsCommandLineOptions::OptimizationLevel::O3, LlcCommand::OptimizationLevel::O3}
    });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::unique_ptr<CommandGraph>
JhlsCommandGraphGenerator::GenerateCommandGraph(const JhlsCommandLineOptions & commandLineOptions)
{
  std::unique_ptr<CommandGraph> commandGraph(new CommandGraph());

  std::vector<CommandGraph::Node*> leaves;
  std::vector<CommandGraph::Node*> llir;
  std::vector<jlm::filepath> llir_files;

  // Create directory in /tmp for storing temporary files
  std::string tmp_identifier;
  for (const auto & compilation : commandLineOptions.Compilations_) {
    tmp_identifier += compilation.InputFile().name() + "_";
    if (tmp_identifier.length() > 30)
      break;
  }
  srandom((unsigned) time(nullptr) * getpid());
  tmp_identifier += std::to_string(random());
  jlm::filepath tmp_folder("/tmp/" + tmp_identifier + "/");
  auto & mkdir = MkdirCommand::Create(*commandGraph, tmp_folder);
  commandGraph->GetEntryNode().AddEdge(mkdir);

  for (const auto & compilation : commandLineOptions.Compilations_) {
    CommandGraph::Node * last = &mkdir;

    if (compilation.RequiresParsing()) {
      auto & parserNode = ClangCommand::CreateParsingCommand(
        *commandGraph,
        compilation.InputFile(),
        tmp_folder.to_str() + CreateParserCommandOutputFile(compilation.InputFile()).to_str(),
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
        {ClangCommand::ClangArgument::DisableO0OptNone});

      last->AddEdge(parserNode);
      last = &parserNode;

      // HLS links all files to a single IR
      // Need to know when the IR has been generated for all input files
      llir.push_back(&parserNode);
      llir_files.push_back(
        dynamic_cast<ClangCommand*>(&parserNode.GetCommand())->OutputFile());
    }

    leaves.push_back(last);
  }

  // link all llir into one so inlining can be done across files for HLS
  jlm::filepath ll_merged(tmp_folder.to_str()+"merged.ll");
  auto & ll_link = LlvmLinkCommand::Create(
    *commandGraph,
    llir_files,
    ll_merged,
    true,
    true);
  // Add edges between each c.parse and the ll_link
  for (const auto & ll : llir) {
    ll->AddEdge(ll_link);
  }

  // need to already run m2r here
  jlm::filepath  ll_m2r1(tmp_folder.to_str()+"merged.m2r.ll");
  auto & m2r1 = LlvmOptCommand::Create(
    *commandGraph,
    ll_merged,
    ll_m2r1,
    true,
    {LlvmOptCommand::Optimization::Mem2Reg});
  ll_link.AddEdge(m2r1);
  auto & extract = JlmHlsExtractCommand::Create(
    *commandGraph,
    dynamic_cast<LlvmOptCommand *>(&m2r1.GetCommand())->OutputFile(),
    commandLineOptions.HlsFunctionRegex_,
    tmp_folder.to_str());
  m2r1.AddEdge(extract);
  jlm::filepath  ll_m2r2(tmp_folder.to_str()+"function.m2r.ll");
  auto & m2r2 = LlvmOptCommand::Create(
    *commandGraph,
    dynamic_cast<JlmHlsExtractCommand *>(&extract.GetCommand())->HlsFunctionFile(),
    ll_m2r2,
    true,
    {LlvmOptCommand::Optimization::Mem2Reg});
  extract.AddEdge(m2r2);
  // hls
  auto & hls = JlmHlsCommand::Create(
    *commandGraph,
    dynamic_cast<LlvmOptCommand *>(&m2r2.GetCommand())->OutputFile(),
    tmp_folder.to_str(),
    commandLineOptions.UseCirct_);
  m2r2.AddEdge(hls);

  if (!commandLineOptions.GenerateFirrtl_) {
    jlm::filepath verilogfile(tmp_folder.to_str()+"jlm_hls.v");
    auto & firrtl = FirtoolCommand::Create(
      *commandGraph,
      dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->FirrtlFile(),
      verilogfile);
    hls.AddEdge(firrtl);
    filepath assemblyFile(tmp_folder.to_str() + "hls.o");
    auto inputFile = dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->LlvmFile();
    auto & asmnode = LlcCommand::Create(
      *commandGraph,
      commandLineOptions.Hls_
      ? inputFile
      : tmp_folder.to_str() + CreateJlmOptCommandOutputFile(inputFile).to_str(),
      assemblyFile,
      ConvertOptimizationLevel(commandLineOptions.OptimizationLevel_),
      commandLineOptions.Hls_
      ? LlcCommand::RelocationModel::Pic
      : LlcCommand::RelocationModel::Static);
    hls.AddEdge(asmnode);

    std::vector<jlm::filepath> lnkifiles;
    for (const auto & compilation : commandLineOptions.Compilations_) {
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

  std::vector<jlm::filepath> lnkifiles;
  for (const auto & c : commandLineOptions.Compilations_) {
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
