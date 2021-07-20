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
				false,
				c.Mt(),
				opts.std);

			last->add_edge(prsnode);
			last = prsnode;
		}

		if (c.optimize()) {
			auto optnode = optcmd::create(
				pgraph.get(),
				c.ifile(),
				tmp_folder,
				opts.jlmopts,
				opts.Olvl);
			last->add_edge(optnode);
			last = optnode;
		}

		if (c.assemble()) {
			auto asmnode = cgencmd::create(
				pgraph.get(),
				c.ifile(),
				c.ofile(),
				tmp_folder,
				false,
				opts.Olvl);
			last->add_edge(asmnode);
			last = asmnode;
		}

		leaves.push_back(last);
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

	if (!lnkifiles.empty()) {
		auto lnknode = lnkcmd::create(pgraph.get(), lnkifiles, opts.lnkofile,
			opts.libpaths, opts.libs, opts.pthread);
		for (const auto & leave : leaves)
			leave->add_edge(lnknode);

		leaves.clear();
		leaves.push_back(lnknode);
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
