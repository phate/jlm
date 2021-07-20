/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_CMDLINE_HPP
#define JLM_JLC_CMDLINE_HPP

#include <jlm/util/file.hpp>

#include <string>
#include <vector>

namespace jlm {

enum class optlvl {O0, O1, O2, O3};

std::string
to_str(const optlvl & ol);

enum class standard {none, gnu89, gnu99, c89, c90, c99, c11, cpp98, cpp03, cpp11, cpp14};

std::string
to_str(const standard & std);

class compilation {
public:
	compilation(
		const jlm::filepath & ifile,
		const jlm::filepath & dependencyFile,
		const jlm::filepath & ofile,
		const std::string & mT,
		bool parse,
		bool optimize,
		bool assemble,
		bool link)
	: link_(link)
	, parse_(parse)
	, optimize_(optimize)
	, assemble_(assemble)
	, ifile_(ifile)
	, ofile_(ofile)
	, dependencyFile_(dependencyFile)
	, mT_(mT)
	{}

	const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}

	const jlm::filepath &
	DependencyFile() const noexcept
	{
		return dependencyFile_;
	}

	const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	const std::string &
	Mt() const noexcept
	{
		return mT_;
	}

	void
	set_ofile(const jlm::filepath & ofile)
	{
		ofile_ = ofile;
	}

	bool
	parse() const noexcept
	{
		return parse_;
	}

	bool
	optimize() const noexcept
	{
		return optimize_;
	}

	bool
	assemble() const noexcept
	{
		return assemble_;
	}

	bool
	link() const noexcept
	{
		return link_;
	}

private:
	bool link_;
	bool parse_;
	bool optimize_;
	bool assemble_;
	jlm::filepath ifile_;
	jlm::filepath ofile_;
	jlm::filepath dependencyFile_;
	const std::string mT_;
};

class cmdline_options {
public:
	cmdline_options()
	: only_print_commands(false)
	, generate_debug_information(false)
	, verbose(false)
	, rdynamic(false)
	, suppress(false)
	, pthread(false)
	, MD(false)
	, Olvl(optlvl::O0)
	, std(standard::none)
	, lnkofile("a.out")
	{}

	bool only_print_commands;
	bool generate_debug_information;
	bool verbose;
	bool rdynamic;
	bool suppress;
	bool pthread;
	bool generate_firrtl;
	bool circt;
	bool hls = false;

	bool MD;

	optlvl Olvl;
	standard std;
	jlm::filepath lnkofile;
	std::vector<std::string> libs;
	std::vector<std::string> macros;
	std::vector<std::string> libpaths;
	std::vector<std::string> warnings;
	std::vector<std::string> includepaths;
	std::vector<std::string> flags;
	std::vector<std::string> jlmopts;

	std::vector<compilation> compilations;
	std::string hls_function_regex;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

#endif
