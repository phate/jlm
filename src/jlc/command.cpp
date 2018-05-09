/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlc/command.hpp>
#include <jlm/util/strfmt.hpp>

#include <iostream>
#include <memory>

namespace jlm {

static jlm::file
create_cgencmd_ofile(const jlm::file & ifile)
{
	return jlm::file(strfmt("/tmp/", ifile.base(), "-llc-out.o"));
}

/* command generation */

std::vector<std::unique_ptr<command>>
generate_commands(const jlm::cmdline_options & opts)
{
	std::vector<std::unique_ptr<command>> cmds;

	for (const auto & ifile : opts.ifiles) {
		if (opts.enable_parser) {
			cmds.push_back(std::make_unique<prscmd>(ifile, opts.includepaths, opts.macros,
				opts.warnings, opts.std));
		}

		if (opts.enable_optimizer)
			cmds.push_back(std::make_unique<optcmd>(ifile));

		if (opts.enable_assembler) {
			auto cgenofile = !opts.enable_linker ? opts.ofile : create_cgencmd_ofile(ifile);
			cmds.push_back(std::make_unique<cgencmd>(ifile, cgenofile, opts.Olvl));
		}
	}

	if (opts.enable_linker) {
		std::vector<jlm::file> ifiles;
		for (const auto & ifile : opts.ifiles)
			ifiles.push_back(opts.enable_assembler ? create_cgencmd_ofile(ifile) : ifile);

		cmds.push_back(std::make_unique<lnkcmd>(ifiles, opts.ofile, opts.libpaths, opts.libs));
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
