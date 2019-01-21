/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlm/jlc/cmdline.hpp>
#include <jlm/jlm/driver/passgraph.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

std::unique_ptr<passgraph>
generate_commands(const jlm::cmdline_options & options);

/* parser command */

class prscmd final : public command {
public:
	virtual
	~prscmd();

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
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::file & ifile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const standard & std)
	{
		std::unique_ptr<prscmd> cmd(new prscmd(ifile, Ipaths, Dmacros, Wwarnings, std));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

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
	virtual
	~optcmd();

	optcmd(const jlm::file & ifile)
	: ifile_(ifile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::file & ifile)
	{
		return passgraph_node::create(pgraph, std::make_unique<optcmd>(ifile));
	}

private:
	jlm::file ifile_;
};

/* code generator command */

class cgencmd final : public command {
public:
	virtual
	~cgencmd();

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
	run() const override;

	inline const jlm::file &
	ofile() const noexcept
	{
		return ofile_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::file & ifile,
		const jlm::file & ofile,
		const optlvl & ol)
	{
		return passgraph_node::create(pgraph, std::make_unique<cgencmd>(ifile, ofile, ol));
	}

private:
	optlvl ol_;
	jlm::file ifile_;
	jlm::file ofile_;
};

/* linker command */

class lnkcmd final : public command {
public:
	virtual
	~lnkcmd();

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
	run() const override;

	inline const jlm::file &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::file> &
	ifiles() const noexcept
	{
		return ifiles_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const std::vector<jlm::file> & ifiles,
		const jlm::file & ofile,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs)
	{
		std::unique_ptr<lnkcmd> cmd(new lnkcmd(ifiles, ofile, Lpaths, libs));
		return passgraph_node::create(pgraph, std::move(cmd));
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
	virtual
	~printcmd();

	printcmd(
		std::unique_ptr<passgraph> pgraph)
	: pgraph_(std::move(pgraph))
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
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		std::unique_ptr<passgraph> pg)
	{
		return passgraph_node::create(pgraph, std::make_unique<printcmd>(std::move(pg)));
	}

private:
	std::unique_ptr<passgraph> pgraph_;
};

}

#endif
