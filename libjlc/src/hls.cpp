/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/cmdline.hpp>
#include <jlc/command.hpp>

#include <iostream>

namespace jlm {

/* command generation */

std::unique_ptr<passgraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<passgraph> pgraph(new passgraph());

	std::vector<passgraph_node*> leaves;
	std::vector<passgraph_node*> llir;
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
	pgraph->entry()->add_edge(mkdir);

	for (const auto & c : opts.compilations) {
		passgraph_node* last = mkdir;

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

			last->add_edge(prsnode);
			last = prsnode;

			// HLS links all files to a single IR
			// Need to know when the IR has been generated for all input files
			llir.push_back(prsnode);
			llir_files.push_back(
				dynamic_cast<prscmd*>(&prsnode->cmd())->ofile());
		}

		leaves.push_back(last);
	}

	// link all llir into one so inlining can be done across files for HLS
	jlm::filepath ll_merged(tmp_folder.to_str()+"merged.ll");
	auto ll_link = lllnkcmd::create(pgraph.get(), llir_files, ll_merged);
	// Add edges between each c.parse and the ll_link
	for (const auto & ll : llir) {
		ll->add_edge(ll_link);
	}

	// need to already run m2r here
	jlm::filepath  ll_m2r1(tmp_folder.to_str()+"merged.m2r.ll");
	auto m2r1 = m2rcmd::create(pgraph.get(), ll_merged, ll_m2r1);
	ll_link->add_edge(m2r1);
	auto extract = extractcmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r1->cmd())->ofile(),
				opts.hls_function_regex,
				tmp_folder.to_str());
	m2r1->add_edge(extract);
	jlm::filepath  ll_m2r2(tmp_folder.to_str()+"function.m2r.ll");
	auto m2r2 = m2rcmd::create(
				pgraph.get(),
				dynamic_cast<extractcmd*>(&extract->cmd())->functionfile(),
				ll_m2r2);
	extract->add_edge(m2r2);
	// hls
	auto hls = hlscmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r2->cmd())->ofile(),
				tmp_folder.to_str(),
				opts.circt);
	m2r2->add_edge(hls);

	if (!opts.generate_firrtl) {
		jlm::filepath verilogfile(tmp_folder.to_str()+"jlm_hls.v");
		auto firrtl = firrtlcmd::create(
				pgraph.get(),
				dynamic_cast<hlscmd*>(&hls->cmd())->firfile(),
				verilogfile);
		hls->add_edge(firrtl);
		jlm::filepath asmofile(tmp_folder.to_str()+"hls.o");
		auto asmnode = cgencmd::create(
				pgraph.get(),
				dynamic_cast<hlscmd*>(&hls->cmd())->llfile(),
				asmofile,
				tmp_folder,
				opts.hls,
				opts.Olvl);
		hls->add_edge(asmnode);

		std::vector<jlm::filepath> lnkifiles;
		for (const auto & c : opts.compilations) {
			if (c.link() && !c.parse())
				lnkifiles.push_back(c.ofile());
		}
		lnkifiles.push_back(asmofile);
		// TODO: remove old verilator folder first
		auto verinode = verilatorcmd::create(
				pgraph.get(),
				verilogfile,
				lnkifiles,
				dynamic_cast<hlscmd*>(&hls->cmd())->harnessfile(),
				opts.lnkofile,
				tmp_folder,
				opts.libpaths,
				opts.libs);
		firrtl->add_edge(verinode);
		verinode->add_edge(pgraph->exit());
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

	for (const auto & leave : leaves)
		leave->add_edge(pgraph->exit());

	if (opts.only_print_commands) {
		std::unique_ptr<passgraph> pg(new passgraph());
		auto printnode = printcmd::create(pg.get(), std::move(pgraph));
		pg->entry()->add_edge(printnode);
		printnode->add_edge(pg->exit());
		pgraph = std::move(pg);
	}

	return pgraph;
}

} // jlm

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	parse_cmdline(argc, argv, options);

	auto pgraph = generate_commands(options);
	pgraph->run();

	return 0;
}
