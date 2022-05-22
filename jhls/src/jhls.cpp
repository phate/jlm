/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jhls/cmdline.hpp>
#include <jhls/command.hpp>
#include <jlm/util/strfmt.hpp>

#include <iostream>
#include <unordered_map>

namespace jlm {

/* command generation */

static cgencmd::OptimizationLevel
ToLlcCommandOptimizationLevel(optlvl optimizationLevel)
{
  static std::unordered_map<optlvl, cgencmd::OptimizationLevel>
    map({
          {optlvl::O0, cgencmd::OptimizationLevel::O0},
          {optlvl::O1, cgencmd::OptimizationLevel::O1},
          {optlvl::O2, cgencmd::OptimizationLevel::O2},
          {optlvl::O3, cgencmd::OptimizationLevel::O3}
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

static std::string
create_optcmd_ofile(const std::string & ifile)
{
  return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

std::unique_ptr<CommandGraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	std::vector<CommandGraph::Node*> llir;
	std::vector<jlm::filepath> llir_files;

	// Create directory in /tmp for storing temporary files
	std::string tmp_identifier;
	for (const auto & c : opts.compilations) {
		tmp_identifier += c.ifile().name() + "_";
		if (tmp_identifier.length() > 30)
			break;
	}
	srandom((unsigned) time(nullptr) * getpid());
	tmp_identifier += std::to_string(random());
	jlm::filepath tmp_folder("/tmp/" + tmp_identifier + "/");
	auto mkdir = mkdircmd::create(pgraph.get(), tmp_folder);
  pgraph->GetEntryNode().AddEdge(*mkdir);

	for (const auto & c : opts.compilations) {
		CommandGraph::Node * last = mkdir;

		if (c.parse()) {
			auto prsnode = prscmd::create(
				pgraph.get(),
				c.ifile(),
				c.DependencyFile(),
				tmp_folder,
				opts.includepaths,
				opts.macros,
				opts.warnings,
				opts.flags,
				opts.verbose,
				opts.rdynamic,
				opts.suppress,
				opts.pthread,
				opts.MD,
				true,
				c.Mt(),
				opts.std);

      last->AddEdge(*prsnode);
			last = prsnode;

			// HLS links all files to a single IR
			// Need to know when the IR has been generated for all input files
			llir.push_back(prsnode);
			llir_files.push_back(
				dynamic_cast<prscmd*>(&prsnode->GetCommand())->ofile());
		}

		leaves.push_back(last);
	}

	// link all llir into one so inlining can be done across files for HLS
	jlm::filepath ll_merged(tmp_folder.to_str()+"merged.ll");
	auto ll_link = lllnkcmd::create(pgraph.get(), llir_files, ll_merged);
	// Add edges between each c.parse and the ll_link
	for (const auto & ll : llir) {
    ll->AddEdge(*ll_link);
	}

	// need to already run m2r here
	jlm::filepath  ll_m2r1(tmp_folder.to_str()+"merged.m2r.ll");
	auto m2r1 = m2rcmd::create(pgraph.get(), ll_merged, ll_m2r1);
  ll_link->AddEdge(*m2r1);
	auto extract = extractcmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r1->GetCommand())->ofile(),
				opts.hls_function_regex,
				tmp_folder.to_str());
  m2r1->AddEdge(*extract);
	jlm::filepath  ll_m2r2(tmp_folder.to_str()+"function.m2r.ll");
	auto m2r2 = m2rcmd::create(
				pgraph.get(),
				dynamic_cast<extractcmd*>(&extract->GetCommand())->functionfile(),
				ll_m2r2);
  extract->AddEdge(*m2r2);
	// hls
	auto hls = hlscmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r2->GetCommand())->ofile(),
				tmp_folder.to_str(),
				opts.circt);
  m2r2->AddEdge(*hls);

	if (!opts.generate_firrtl) {
    jlm::filepath verilogfile(tmp_folder.to_str()+"jlm_hls.v");
    auto firrtl = firrtlcmd::create(
      pgraph.get(),
      dynamic_cast<hlscmd*>(&hls->GetCommand())->firfile(),
      verilogfile);
    hls->AddEdge(*firrtl);
    jlm::filepath asmofile(tmp_folder.to_str()+"hls.o");
    auto inputFile = dynamic_cast<hlscmd*>(&hls->GetCommand())->llfile();
    auto asmnode = cgencmd::create(
      pgraph.get(),
      opts.hls
      ? inputFile
      : tmp_folder.to_str() + create_optcmd_ofile(inputFile.base()),
      asmofile,
      ToLlcCommandOptimizationLevel(opts.Olvl),
      opts.hls
      ? cgencmd::RelocationModel::Pic
      : cgencmd::RelocationModel::Static);
    hls->AddEdge(*asmnode);

		std::vector<jlm::filepath> lnkifiles;
		for (const auto & c : opts.compilations) {
			if (c.link() && !c.parse())
				lnkifiles.push_back(c.ofile());
		}
		lnkifiles.push_back(asmofile);
		auto verinode = verilatorcmd::create(
				pgraph.get(),
				verilogfile,
				lnkifiles,
				dynamic_cast<hlscmd*>(&hls->GetCommand())->harnessfile(),
				opts.lnkofile,
				tmp_folder,
				opts.libpaths,
				opts.libs);
    firrtl->AddEdge(*verinode);
    verinode->AddEdge(pgraph->GetExitNode());
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (opts.only_print_commands)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

	return pgraph;
}

} // jlm

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	parse_cmdline(argc, argv, options);

	auto pgraph = generate_commands(options);
  pgraph->Run();

	return 0;
}
