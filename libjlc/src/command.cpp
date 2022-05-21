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
#include <algorithm>

namespace jlm {

/* command generation */

std::unique_ptr<CommandGraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	for (const auto & c : opts.compilations) {
		auto last = &pgraph->GetEntryNode();

		if (c.parse()) {
			auto prsnode = prscmd::create(
				pgraph.get(),
				c.ifile(),
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
				opts.std);

      last->AddEdge(*prsnode);
			last = prsnode;
		}

		if (c.optimize()) {
			auto optnode = optcmd::create(pgraph.get(), c.ifile(), opts.jlmopts, opts.Olvl);
      last->AddEdge(*optnode);
			last = optnode;
		}

		if (c.assemble()) {
			auto asmnode = cgencmd::create(pgraph.get(), c.ifile(), c.ofile(), opts.Olvl);
      last->AddEdge(*asmnode);
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
      leave->AddEdge(*lnknode);

		leaves.clear();
		leaves.push_back(lnknode);
	}

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (opts.only_print_commands)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

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
prscmd::replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

std::string
prscmd::ToString() const
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

	std::string flags;
	for (const auto & flag : flags_)
		flags += "-f" + flag + " ";

	std::string arguments;
	if (verbose_)
	  arguments += "-v ";

	if (rdynamic_)
	  arguments += "-rdynamic ";

	if (suppress_)
	  arguments += "-w ";

	if (pthread_)
	  arguments += "-pthread ";

	if (MD_) {
		arguments += "-MD ";
		arguments += "-MF " + dependencyFile_.to_str() + " ";
		arguments += "-MT " + mT_ + " ";
	}

	return strfmt(
	  clangpath.to_str() + " "
	, arguments, " "
	, Wwarnings, " "
	, flags, " "
	, std_ != standard::none ? "-std="+jlm::to_str(std_)+" " : ""
	, replace_all(Dmacros, std::string("\""), std::string("\\\"")), " "
	, Ipaths, " "
	, "-S -emit-llvm "
	, "-o /tmp/", create_prscmd_ofile(f), " "
	, ifile_.to_str()
	);
}

void
prscmd::Run() const
{
	if (system(ToString().c_str()))
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
optcmd::ToString() const
{
	auto f = ifile_.base();

	std::string jlmopts;
	for (const auto & jlmopt : jlmopts_)
		jlmopts += "--" + jlmopt + " ";

	/*
		If a default optimization level has been specified (-O) and no specific jlm-options 
		have been specified (-J) then use a default set of optimizations.
	 */
	if (jlmopts.empty()) {
		/*
			Only -O3 sets default optimizations
		*/
		if (ol_ == optlvl::O3) {
			jlmopts  = "--iln --InvariantValueRedirection --red --dne --ivt --InvariantValueRedirection ";
      jlmopts += "--dne --psh --InvariantValueRedirection --dne ";
			jlmopts += "--red --cne --dne --pll --InvariantValueRedirection --dne --url --InvariantValueRedirection ";
		}
	}

	return strfmt(
	  "jlm-opt "
	, "--llvm "
	, jlmopts
	, "/tmp/", create_prscmd_ofile(f), " > /tmp/", create_optcmd_ofile(f)
	);
}

void
optcmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

/* code generator command */

cgencmd::~cgencmd()
{}

std::string
cgencmd::ToString() const
{
	return strfmt(
	  llcpath.to_str() + " "
	, "-", jlm::to_str(ol_), " "
	, "-filetype=obj "
	, "-o ", ofile_.to_str()
	, " /tmp/", create_optcmd_ofile(ifile_.base())
	);
}

void
cgencmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

/* linker command */

lnkcmd::~lnkcmd()
{}

std::string
lnkcmd::ToString() const
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

	std::string arguments;
	if (pthread_)
	  arguments += "-pthread ";

	return strfmt(
	  clangpath.to_str() + " "
	, "-no-pie -O0 "
        , arguments
	, ifiles
	, "-o ", ofile_.to_str(), " "
	, Lpaths
	, libs
	);
}

void
lnkcmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

}
