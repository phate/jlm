/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jhls/cmdline.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/util/strfmt.hpp>

#include <unistd.h>

#include <iostream>
#include <unordered_map>

namespace jlm {

/* command generation */

static LlcCommand::OptimizationLevel
ToLlcCommandOptimizationLevel(const JhlsCommandLineOptions::OptimizationLevel & optimizationLevel)
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

static ClangCommand::LanguageStandard
ToPrscmdLanguageStandard(const JhlsCommandLineOptions::LanguageStandard & languageStandard)
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
generate_commands(const jlm::JhlsCommandLineOptions & opts)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	std::vector<CommandGraph::Node*> llir;
	std::vector<jlm::filepath> llir_files;

	// Create directory in /tmp for storing temporary files
	std::string tmp_identifier;
	for (const auto & c : opts.Compilations_) {
		tmp_identifier += c.InputFile().name() + "_";
		if (tmp_identifier.length() > 30)
			break;
	}
	srandom((unsigned) time(nullptr) * getpid());
	tmp_identifier += std::to_string(random());
	jlm::filepath tmp_folder("/tmp/" + tmp_identifier + "/");
	auto & mkdir = MkdirCommand::Create(*pgraph, tmp_folder);
  pgraph->GetEntryNode().AddEdge(mkdir);

	for (const auto & c : opts.Compilations_) {
		CommandGraph::Node * last = &mkdir;

		if (c.RequiresParsing()) {
			auto & prsnode = ClangCommand::CreateParsingCommand(
				*pgraph,
        c.InputFile(),
        tmp_folder.to_str() + create_prscmd_ofile(c.InputFile().base()),
				c.DependencyFile(),
				opts.IncludePaths_,
				opts.MacroDefinitions_,
				opts.Warnings_,
				opts.Flags_,
				opts.Verbose_,
				opts.Rdynamic_,
				opts.Suppress_,
				opts.UsePthreads_,
				opts.Md_,
				c.Mt(),
        ToPrscmdLanguageStandard(opts.LanguageStandard_),
        {ClangCommand::ClangArgument::DisableO0OptNone});

      last->AddEdge(prsnode);
			last = &prsnode;

			// HLS links all files to a single IR
			// Need to know when the IR has been generated for all input files
			llir.push_back(&prsnode);
			llir_files.push_back(
				dynamic_cast<ClangCommand*>(&prsnode.GetCommand())->OutputFile());
		}

		leaves.push_back(last);
	}

	// link all llir into one so inlining can be done across files for HLS
	jlm::filepath ll_merged(tmp_folder.to_str()+"merged.ll");
	auto & ll_link = LlvmLinkCommand::Create(
    *pgraph,
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
    *pgraph,
    ll_merged,
    ll_m2r1,
    true,
    {LlvmOptCommand::Optimization::Mem2Reg});
  ll_link.AddEdge(m2r1);
	auto & extract = JlmHlsExtractCommand::Create(
				*pgraph,
        dynamic_cast<LlvmOptCommand *>(&m2r1.GetCommand())->OutputFile(),
				opts.HlsFunctionRegex_,
				tmp_folder.to_str());
  m2r1.AddEdge(extract);
	jlm::filepath  ll_m2r2(tmp_folder.to_str()+"function.m2r.ll");
	auto & m2r2 = LlvmOptCommand::Create(
				*pgraph,
        dynamic_cast<JlmHlsExtractCommand *>(&extract.GetCommand())->HlsFunctionFile(),
				ll_m2r2,
        true,
        {LlvmOptCommand::Optimization::Mem2Reg});
  extract.AddEdge(m2r2);
	// hls
	auto & hls = JlmHlsCommand::Create(
				*pgraph,
        dynamic_cast<LlvmOptCommand *>(&m2r2.GetCommand())->OutputFile(),
				tmp_folder.to_str(),
				opts.UseCirct_);
  m2r2.AddEdge(hls);

	if (!opts.GenerateFirrtl_) {
    jlm::filepath verilogfile(tmp_folder.to_str()+"jlm_hls.v");
    auto & firrtl = FirtoolCommand::Create(
      *pgraph,
      dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->FirrtlFile(),
      verilogfile);
    hls.AddEdge(firrtl);
    jlm::filepath asmofile(tmp_folder.to_str()+"hls.o");
    auto inputFile = dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->LlvmFile();
    auto & asmnode = LlcCommand::Create(
      *pgraph,
      opts.Hls_
      ? inputFile
      : tmp_folder.to_str() + create_optcmd_ofile(inputFile.base()),
      asmofile,
      ToLlcCommandOptimizationLevel(opts.OptimizationLevel_),
      opts.Hls_
      ? LlcCommand::RelocationModel::Pic
      : LlcCommand::RelocationModel::Static);
    hls.AddEdge(asmnode);

		std::vector<jlm::filepath> lnkifiles;
		for (const auto & c : opts.Compilations_) {
			if (c.RequiresLinking() && !c.RequiresParsing())
				lnkifiles.push_back(c.OutputFile());
		}
		lnkifiles.push_back(asmofile);
		auto & verinode = VerilatorCommand::Create(
				*pgraph,
				verilogfile,
				lnkifiles,
        dynamic_cast<JlmHlsCommand *>(&hls.GetCommand())->HarnessFile(),
				opts.OutputFile_,
				tmp_folder,
				opts.LibraryPaths_,
				opts.Libraries_);
    firrtl.AddEdge(verinode);
    verinode.AddEdge(pgraph->GetExitNode());
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.Compilations_) {
		if (c.RequiresLinking())
			lnkifiles.push_back(c.OutputFile());
	}

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (opts.OnlyPrintCommands_)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

	return pgraph;
}

} // jlm

int
main(int argc, char ** argv)
{
	jlm::JhlsCommandLineOptions options;
	parse_cmdline(argc, argv, options);

	auto pgraph = generate_commands(options);
  pgraph->Run();

	return 0;
}
