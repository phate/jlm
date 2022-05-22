/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlc/cmdline.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

std::unique_ptr<CommandGraph>
generate_commands(const jlm::cmdline_options & options);

/* parser command */

class prscmd final : public Command {
public:
	virtual
	~prscmd();

	prscmd(
		const jlm::filepath & ifile,
		const jlm::filepath & dependencyFile,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const std::vector<std::string> & flags,
		bool verbose,
		bool rdynamic,
		bool suppress,
		bool pthread,
		bool MD,
		const std::string & mT,
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
	, pthread_(pthread)
	, MD_(MD)
	, mT_(mT)
	, dependencyFile_(dependencyFile)
	{}

	virtual std::string
	ToString() const override;

	virtual void
	Run() const override;

	static CommandGraph::Node *
	create(
    CommandGraph * pgraph,
    const jlm::filepath & ifile,
    const jlm::filepath & dependencyFile,
    const std::vector<std::string> & Ipaths,
    const std::vector<std::string> & Dmacros,
    const std::vector<std::string> & Wwarnings,
    const std::vector<std::string> & flags,
    bool verbose,
    bool rdynamic,
    bool suppress,
    bool pthread,
    bool MD,
    const std::string & mT,
    const standard & std)
	{
		std::unique_ptr<prscmd> cmd(new prscmd(
			ifile,
			dependencyFile,
			Ipaths,
			Dmacros,
			Wwarnings,
			flags,
			verbose,
			rdynamic,
			suppress,
			pthread,
			MD,
			mT,
			std));

		return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
	}

private:
	static std::string
	replace_all(std::string str, const std::string& from, const std::string& to);

	standard std_;
	jlm::filepath ifile_;
	std::vector<std::string> Ipaths_;
	std::vector<std::string> Dmacros_;
	std::vector<std::string> Wwarnings_;
	std::vector<std::string> flags_;
	bool verbose_;
	bool rdynamic_;
	bool suppress_;
	bool pthread_;
	bool MD_;
	std::string mT_;
	jlm::filepath dependencyFile_;
};

/* optimization command */

class optcmd final : public Command {
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
	ToString() const override;

	virtual void
	Run() const override;

	static CommandGraph::Node *
	create(
    CommandGraph * pgraph,
    const jlm::filepath & ifile,
    const std::vector<std::string> & jlmopts,
    const optlvl & ol)
	{
		return &CommandGraph::Node::Create(*pgraph, std::make_unique<optcmd>(ifile, jlmopts, ol));
	}

private:
	jlm::filepath ifile_;
	std::vector<std::string> jlmopts_;
	optlvl ol_;
};

}

#endif
