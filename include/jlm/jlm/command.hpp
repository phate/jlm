/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM_COMMAND_HPP
#define JLM_JLM_COMMAND_HPP

#include <jlm/jlm/cmdline.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

class command {
public:

	virtual std::string
	to_str() const = 0;

	virtual void
	execute() const = 0;
};

std::vector<std::unique_ptr<command>>
generate_commands(const jlm::cmdline_options & options);

/* parser command */

class prscmd final : public command {
public:
	prscmd(
		const std::string & ifile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros)
	: ifile_(ifile)
	, Ipaths_(Ipaths)
	, Dmacros_(Dmacros)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	std::string ifile_;
	std::vector<std::string> Ipaths_;
	std::vector<std::string> Dmacros_;
};

/* optimization command */

class optcmd final : public command {
public:
	optcmd(const std::string & ifile)
	: ifile_(ifile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	std::string ifile_;
};

/* code generator command */

class cgencmd final : public command {
public:
	cgencmd(
		const std::string & ifile,
		const optlvl & ol)
	: ol_(ol)
	, ifile_(ifile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	optlvl ol_;
	std::string ifile_;
};

/* linker command */

class lnkcmd final : public command {
public:
	lnkcmd(
		const std::vector<std::string> & ifiles,
		const std::string & ofile,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs)
	: ofile_(ofile)
	, libs_(libs)
	, ifiles_(ifiles)
	, Lpaths_(Lpaths)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	std::string ofile_;
	std::vector<std::string> libs_;
	std::vector<std::string> ifiles_;
	std::vector<std::string> Lpaths_;
};

/* print command */

class printcmd final : public command {
public:
	printcmd(
		std::vector<std::unique_ptr<command>> cmds)
	: cmds_(std::move(cmds))
	{}

	printcmd(const printcmd&) = delete;

	printcmd(printcmd&&) = delete;

	printcmd &
	operator=(const printcmd&) = delete;

	printcmd &
	operator=(printcmd&&)	= delete;

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	std::vector<std::unique_ptr<command>> cmds_;
};

}

#endif
