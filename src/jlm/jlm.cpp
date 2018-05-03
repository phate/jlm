/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/cmdline.hpp>
#include <jlm/util/strfmt.hpp>

#include <iostream>
#include <vector>

static std::string
file(const std::string & filepath)
{
	auto pos = filepath.find_last_of("/");
	if (pos == std::string::npos)
		return filepath;

	return filepath.substr(pos+1, filepath.size()-pos);
}

static std::string
create_frontend_ofilename(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-clang-out.ll");
}

static std::string
create_opt_ofilename(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

static std::string
create_cgen_ofilename(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-llc-out.o");
}

static std::string
create_frontend_command(
	const std::string & ifilepath,
	const std::vector<std::string> & includepaths)
{
	auto f = file(ifilepath);

	std::string incpaths;
	for (const auto & incpath : includepaths)
		incpaths += "-I" + incpath + " ";

	return strfmt(
	  "clang "
	, incpaths, " "
	, "-S -emit-llvm "
	, "-o /tmp/", create_frontend_ofilename(f), " "
	, ifilepath
	);
}

static std::string
create_opt_command(const std::string & ifilepath)
{
	auto f = file(ifilepath);

	return strfmt(
	  "jlm-opt "
	, "--llvm "
	, "/tmp/", create_frontend_ofilename(f), " > /tmp/", create_opt_ofilename(f)
	);
}

static std::string
create_cgen_command(const std::string & ifilepath)
{
	auto f = file(ifilepath);

	return strfmt(
	  "llc "
	, "-filetype=obj "
	, "-o /tmp/", create_cgen_ofilename(f), " /tmp/", create_opt_ofilename(f)
	);
}

static std::string
create_link_command(const jlm::cmdline_options & options)
{
	std::string ifiles;
	for (const auto & ifilepath : options.ifilepaths)
		ifiles += "/tmp/" + create_cgen_ofilename(file(ifilepath)) + " ";

	std::string libpaths;
	for (const auto & libpath : options.libpaths)
		libpaths += "-L" + libpath + " ";

	return strfmt(
	  "clang "
	, "-O0 "
	, ifiles
	, "-o ", options.ofilepath, " "
	, libpaths
	);
}

static std::vector<std::string>
create_commands(const jlm::cmdline_options & options)
{
	std::vector<std::string> commands;
	for (const auto ifilepath : options.ifilepaths) {
		commands.push_back(create_frontend_command(ifilepath, options.includepaths));
		commands.push_back(create_opt_command(ifilepath));
		commands.push_back(create_cgen_command(ifilepath));
	}

	commands.push_back(create_link_command(options));

	return commands;
}

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	jlm::parse_cmdline(argc, argv, options);

	auto commands = create_commands(options);

	if (options.only_print_commands) {
		for (const auto & cmd : commands)
			std::cout << cmd << "\n";
		return 0;
	}

	for (const auto & cmd : commands) {
		if (system(cmd.c_str()))
			exit(EXIT_FAILURE);
	}

	return 0;
}
