/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlm/jlc/cmdline.hpp>

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
		const jlm::file & ifile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const standard & std)
	: std_(std)
	, ifile_(ifile)
	, Ipaths_(Ipaths)
	, Dmacros_(Dmacros)
	, Wwarnings_(Wwarnings)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	standard std_;
	jlm::file ifile_;
	std::vector<std::string> Ipaths_;
	std::vector<std::string> Dmacros_;
	std::vector<std::string> Wwarnings_;
};

/* optimization command */

class optcmd final : public command {
public:
	optcmd(const jlm::file & ifile)
	: ifile_(ifile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

private:
	jlm::file ifile_;
};

/* code generator command */

class cgencmd final : public command {
public:
	cgencmd(
		const jlm::file & ifile,
		const jlm::file & ofile,
		const optlvl & ol)
	: ol_(ol)
	, ifile_(ifile)
	, ofile_(ofile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	execute() const override;

	inline const jlm::file &
	ofile() const noexcept
	{
		return ofile_;
	}

private:
	optlvl ol_;
	jlm::file ifile_;
	jlm::file ofile_;
};

/* linker command */

class lnkcmd final : public command {
public:
	lnkcmd(
		const std::vector<jlm::file> & ifiles,
		const jlm::file & ofile,
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

	inline const jlm::file &
	ofile() const noexcept
	{
		return ofile_;
	}

private:
	jlm::file ofile_;
	std::vector<std::string> libs_;
	std::vector<jlm::file> ifiles_;
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
