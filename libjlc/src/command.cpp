/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/command.hpp>
#include <jlm/util/strfmt.hpp>

#include <deque>
#include <functional>
#include <iostream>
#include <memory>

namespace jlm {

/* command generation */

static LlcCommand::OptimizationLevel
ToLlcCommandOptimizationLevel(const JlcCommandLineOptions::OptimizationLevel & optimizationLevel)
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

static ClangCommand::LanguageStandard
ToPrscmdLanguageStandard(const JlcCommandLineOptions::LanguageStandard & languageStandard)
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

static std::string
create_optcmd_ofile(const std::string & ifile)
{
  return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

static std::string
create_prscmd_ofile(const std::string & ifile)
{
  return strfmt("tmp-", ifile, "-clang-out.ll");
}

std::unique_ptr<CommandGraph>
generate_commands(const JlcCommandLineOptions & commandLineOptions)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	for (const auto & c : commandLineOptions.Compilations_) {
		auto last = &pgraph->GetEntryNode();

		if (c.RequiresParsing()) {
			auto & prsnode = ClangCommand::CreateParsingCommand(
        *pgraph,
        c.InputFile(),
        "/tmp/" + create_prscmd_ofile(c.InputFile().base()),
        c.DependencyFile(),
        commandLineOptions.IncludePaths_,
        commandLineOptions.MacroDefinitions_,
        commandLineOptions.Warnings_,
        commandLineOptions.Flags_,
        commandLineOptions.Verbose_,
        commandLineOptions.Rdynamic_,
        commandLineOptions.Suppress_,
        commandLineOptions.UsePthreads_,
        commandLineOptions.Md_,
        c.Mt(),
        ToPrscmdLanguageStandard(commandLineOptions.LanguageStandard_),
        {});

      last->AddEdge(prsnode);
			last = &prsnode;
		}

    if (c.RequiresOptimization()) {
      /*
       * If a default optimization level has been specified (-O) and no specific jlm options
       * have been specified (-J) then use a default set of optimizations.
       */
      std::vector<JlmOptCommand::Optimization> optimizations;
      if (commandLineOptions.JlmOptOptimizations_.empty()
      && commandLineOptions.OptimizationLevel_ == JlcCommandLineOptions::OptimizationLevel::O3) {
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

      auto & optnode = JlmOptCommand::Create(
        *pgraph,
        "/tmp/" + create_prscmd_ofile(c.InputFile().base()),
        "/tmp/" + create_optcmd_ofile(c.InputFile().base()),
        optimizations);
      last->AddEdge(optnode);
      last = &optnode;
    }

		if (c.RequiresAssembly()) {
			auto & asmnode = LlcCommand::Create(
        *pgraph,
        "/tmp/" + create_optcmd_ofile(c.InputFile().base()),
        c.OutputFile(),
        ToLlcCommandOptimizationLevel(commandLineOptions.OptimizationLevel_),
        LlcCommand::RelocationModel::Static);
      last->AddEdge(asmnode);
			last = &asmnode;
		}

		leaves.push_back(last);
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : commandLineOptions.Compilations_) {
		if (c.RequiresLinking())
			lnkifiles.push_back(c.OutputFile());
	}

  if (!lnkifiles.empty()) {
    auto & linkerCommandNode = ClangCommand::CreateLinkerCommand(
      *pgraph,
      lnkifiles,
      commandLineOptions.OutputFile_,
      commandLineOptions.LibraryPaths_,
      commandLineOptions.Libraries_,
      commandLineOptions.UsePthreads_);

    for (const auto & leave : leaves)
      leave->AddEdge(linkerCommandNode);

    leaves.clear();
    leaves.push_back(&linkerCommandNode);
  }

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (commandLineOptions.OnlyPrintCommands_)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

	return pgraph;
}

}
