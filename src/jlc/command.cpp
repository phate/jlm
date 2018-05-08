/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlc/command.hpp>
#include <jlm/util/strfmt.hpp>

#include <iostream>
#include <memory>

namespace jlm {

/* command generation */

std::vector<std::unique_ptr<command>>
generate_commands(const jlm::cmdline_options & opts)
{
	std::vector<std::unique_ptr<command>> cmds;

	for (const auto & ifile : opts.ifiles) {
		cmds.push_back(std::make_unique<prscmd>(ifile, opts.includepaths, opts.macros,
			opts.warnings, opts.std));
		cmds.push_back(std::make_unique<optcmd>(ifile));
		cmds.push_back(std::make_unique<cgencmd>(ifile, opts.Olvl));
	}

	if (!opts.no_linking) {
		cmds.push_back(std::make_unique<lnkcmd>(opts.ifiles, opts.ofile, opts.libpaths, opts.libs));
	}

	if (opts.only_print_commands) {
		std::unique_ptr<command> cmd(new printcmd(std::move(cmds)));
		cmds.push_back(std::move(cmd));
	}

	return cmds;
}

/* parser command */

static std::string
create_prscmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-clang-out.ll");
}

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
prscmd::execute() const
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
optcmd::execute() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* code generator command */

static std::string
create_cgencmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-llc-out.o");
}

std::string
cgencmd::to_str() const
{
	auto f = ifile_.base();

	return strfmt(
	  "llc "
	, "-", jlm::to_str(ol_), " "
	, "-filetype=obj "
	, "-o /tmp/", create_cgencmd_ofile(f), " /tmp/", create_optcmd_ofile(f)
	);
}

void
cgencmd::execute() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* linker command */

std::string
lnkcmd::to_str() const
{
	std::string ifiles;
	for (const auto & ifile : ifiles_)
		ifiles += "/tmp/" + create_cgencmd_ofile(ifile.base()) + " ";

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
lnkcmd::execute() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* print command */

std::string
printcmd::to_str() const
{
	std::string str;
	for (const auto & cmd : cmds_)
		str += cmd->to_str() + "\n";

	return str;
}

void
printcmd::execute() const
{
	std::cout << to_str();
}

}
