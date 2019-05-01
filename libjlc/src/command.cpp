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

std::unique_ptr<passgraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<passgraph> pgraph(new passgraph());

	std::vector<passgraph_node*> leaves;
	for (const auto & c : opts.compilations) {
		passgraph_node * last = pgraph->entry();

		if (c.parse()) {
			auto prsnode = prscmd::create(pgraph.get(), c.ifile(), opts.includepaths, opts.macros,
				opts.warnings, opts.std);
			last->add_edge(prsnode);
			last = prsnode;
		}

		if (c.optimize()) {
			auto optnode = optcmd::create(pgraph.get(), c.ifile());
			last->add_edge(optnode);
			last = optnode;
		}

		if (c.assemble()) {
			auto asmnode = cgencmd::create(pgraph.get(), c.ifile(), c.ofile(), opts.Olvl);
			last->add_edge(asmnode);
			last = asmnode;
		}

		leaves.push_back(last);
	}

	std::vector<jlm::file> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

	if (!lnkifiles.empty()) {
		auto lnknode = lnkcmd::create(pgraph.get(), lnkifiles, opts.lnkofile,
			opts.libpaths, opts.libs);
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

/* parser command */

static std::string
create_prscmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-clang-out.ll");
}

prscmd::~prscmd()
{}

std::string
prscmd::to_str() const
{
	auto f = ifile_.base();

	std::string Ipaths;
	for (const auto & Ipath : Ipaths_)
		Ipaths += "-I" + Ipath + " ";

	std::string Dmacros;
	for (const auto & Dmacro : Dmacros_)
		Dmacros += "-D" + Dmacro + " ";

	std::string Wwarnings;
	for (const auto & Wwarning : Wwarnings_)
		Wwarnings += "-W" + Wwarning + " ";

	return strfmt(
	  "clang "
	, Wwarnings, " "
	, std_ != standard::none ? "-std="+jlm::to_str(std_)+" " : ""
	, Dmacros, " "
	, Ipaths, " "
	, "-S -emit-llvm "
	, "-o /tmp/", create_prscmd_ofile(f), " "
	, ifile_.to_str()
	);
}

void
prscmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* optimization command */

static std::string
create_optcmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

optcmd::~optcmd()
{}

std::string
optcmd::to_str() const
{
	auto f = ifile_.base();

	return strfmt(
	  "jlm-opt "
	, "--llvm "
	, "/tmp/", create_prscmd_ofile(f), " > /tmp/", create_optcmd_ofile(f)
	);
}

void
optcmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* code generator command */

cgencmd::~cgencmd()
{}

std::string
cgencmd::to_str() const
{
	return strfmt(
	  "llc "
	, "-", jlm::to_str(ol_), " "
	, "-filetype=obj "
	, "-o ", ofile_.to_str()
	, " /tmp/", create_optcmd_ofile(ifile_.base())
	);
}

void
cgencmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* linker command */

lnkcmd::~lnkcmd()
{}

std::string
lnkcmd::to_str() const
{
	std::string ifiles;
	for (const auto & ifile : ifiles_)
		ifiles += ifile.to_str() + " ";

	std::string Lpaths;
	for (const auto & Lpath : Lpaths_)
		Lpaths += "-L" + Lpath + " ";

	std::string libs;
	for (const auto & lib : libs_)
		libs += "-l" + lib + " ";

	return strfmt(
	  "clang "
	, "-O0 "
	, ifiles
	, "-o ", ofile_.to_str(), " "
	, Lpaths
	, libs
	);
}

void
lnkcmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* print command */

printcmd::~printcmd()
{}

std::string
printcmd::to_str() const
{
	return "PRINTCMD";
}

void
printcmd::run() const
{
	for (const auto & node : topsort(pgraph_.get())) {
		if (node != pgraph_->entry() && node != pgraph_->exit())
			std::cout << node->cmd().to_str() << "\n";
	}
}

}
