/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/command.hpp>
#include <jlc/llvmpaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <deque>
#include <functional>
#include <iostream>
#include <memory>

namespace jlm {

/* command generation */

static LlcCommand::OptimizationLevel
ToLlcCommandOptimizationLevel(optlvl optimizationLevel)
{
  static std::unordered_map<optlvl, LlcCommand::OptimizationLevel>
    map({
          {optlvl::O0, LlcCommand::OptimizationLevel::O0},
          {optlvl::O1, LlcCommand::OptimizationLevel::O1},
          {optlvl::O2, LlcCommand::OptimizationLevel::O2},
          {optlvl::O3, LlcCommand::OptimizationLevel::O3}
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

static ClangCommand::LanguageStandard
ToPrscmdLanguageStandard(standard languageStandard)
{
  static std::unordered_map<standard, ClangCommand::LanguageStandard>
    map({
          {standard::none,  ClangCommand::LanguageStandard::Unspecified},
          {standard::gnu89, ClangCommand::LanguageStandard::Gnu89},
          {standard::gnu99, ClangCommand::LanguageStandard::Gnu99},
          {standard::c89,   ClangCommand::LanguageStandard::C89},
          {standard::c99,   ClangCommand::LanguageStandard::C99},
          {standard::c11,   ClangCommand::LanguageStandard::C11},
          {standard::cpp98, ClangCommand::LanguageStandard::Cpp98},
          {standard::cpp03, ClangCommand::LanguageStandard::Cpp03},
          {standard::cpp11, ClangCommand::LanguageStandard::Cpp11},
          {standard::cpp14, ClangCommand::LanguageStandard::Cpp14}
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
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	for (const auto & c : opts.compilations) {
		auto last = &pgraph->GetEntryNode();

		if (c.parse()) {
			auto & prsnode = ClangCommand::CreateParsingCommand(
				*pgraph,
				c.ifile(),
        "/tmp/" + create_prscmd_ofile(c.ifile().base()),
				c.DependencyFile(),
				opts.includepaths,
				opts.macros,
				opts.warnings,
				opts.flags,
				opts.verbose,
				opts.rdynamic,
				opts.suppress,
				opts.pthread,
				opts.MD,
				c.Mt(),
        ToPrscmdLanguageStandard(opts.std),
        {});

      last->AddEdge(prsnode);
			last = &prsnode;
		}

    if (c.optimize()) {
      /*
       * If a default optimization level has been specified (-O) and no specific jlm options
       * have been specified (-J) then use a default set of optimizations.
       */
      std::vector<JlmOptCommand::Optimization> optimizations;
      if (opts.jlmopts.empty() && opts.Olvl == optlvl::O3) {
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
        "/tmp/" + create_prscmd_ofile(c.ifile().base()),
        "/tmp/" + create_optcmd_ofile(c.ifile().base()),
        optimizations);
      last->AddEdge(optnode);
      last = &optnode;
    }

		if (c.assemble()) {
			auto & asmnode = LlcCommand::Create(
        *pgraph,
        "/tmp/" + create_optcmd_ofile(c.ifile().base()),
        c.ofile(),
        ToLlcCommandOptimizationLevel(opts.Olvl),
        LlcCommand::RelocationModel::Static);
      last->AddEdge(asmnode);
			last = &asmnode;
		}

		leaves.push_back(last);
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

  if (!lnkifiles.empty()) {
    auto & linkerCommandNode = ClangCommand::CreateLinkerCommand(
      *pgraph,
      lnkifiles,
      opts.lnkofile,
      opts.libpaths,
      opts.libs,
      opts.pthread);

    for (const auto & leave : leaves)
      leave->AddEdge(linkerCommandNode);

    leaves.clear();
    leaves.push_back(&linkerCommandNode);
  }

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (opts.only_print_commands)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

	return pgraph;
}

}
