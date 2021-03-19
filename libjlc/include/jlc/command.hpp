/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlc/cmdline.hpp>
#include <jlm/driver/passgraph.hpp>

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
		const jlm::filepath & ifile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const std::vector<std::string> & flags,
		bool verbose,
		bool rdynamic,
		bool suppress,
		const standard & std)
	: std_(std)
	, ifile_(ifile)
	, Ipaths_(Ipaths)
	, Dmacros_(Dmacros)
	, Wwarnings_(Wwarnings)
	, flags_(flags)
	, verbose_(verbose)
	, rdynamic_(rdynamic)
	, suppress_(suppress)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const std::vector<std::string> & flags,
		bool verbose,
		bool rdynamic,
		bool suppress,
		const standard & std)
	{
		std::unique_ptr<prscmd> cmd(new prscmd(ifile, Ipaths, Dmacros, Wwarnings, flags, verbose, rdynamic, suppress, std));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	standard std_;
	jlm::filepath ifile_;
	std::vector<std::string> Ipaths_;
	std::vector<std::string> Dmacros_;
	std::vector<std::string> Wwarnings_;
	std::vector<std::string> flags_;
	bool verbose_;
	bool rdynamic_;
	bool suppress_;
};

/* optimization command */

class optcmd final : public command {
public:
	virtual
	~optcmd();

	optcmd(
		const jlm::filepath & ifile,
		const std::vector<std::string> & jlmopts,
		const optlvl & ol)
	: ifile_(ifile)
	, jlmopts_(jlmopts)
	, ol_(ol)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const std::vector<std::string> & jlmopts,
		const optlvl & ol)
	{
		return passgraph_node::create(pgraph, std::make_unique<optcmd>(ifile, jlmopts, ol));
	}

private:
	jlm::filepath ifile_;
	std::vector<std::string> jlmopts_;
	optlvl ol_;
};

/* code generator command */

class cgencmd final : public command {
public:
	virtual
	~cgencmd();

	cgencmd(
		const jlm::filepath & ifile,
		const jlm::filepath & ofile,
		const optlvl & ol)
	: ol_(ol)
	, ifile_(ifile)
	, ofile_(ofile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & ofile,
		const optlvl & ol)
	{
		return passgraph_node::create(pgraph, std::make_unique<cgencmd>(ifile, ofile, ol));
	}

private:
	optlvl ol_;
	jlm::filepath ifile_;
	jlm::filepath ofile_;
};

/* linker command */

class lnkcmd final : public command {
public:
	virtual
	~lnkcmd();

	lnkcmd(
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile,
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

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::filepath> &
	ifiles() const noexcept
	{
		return ifiles_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs)
	{
		std::unique_ptr<lnkcmd> cmd(new lnkcmd(ifiles, ofile, Lpaths, libs));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	std::vector<std::string> libs_;
	std::vector<jlm::filepath> ifiles_;
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
