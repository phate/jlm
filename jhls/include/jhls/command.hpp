/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JHLS_COMMAND_HPP
#define JLM_JHLS_COMMAND_HPP

#include <jhls/cmdline.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>

#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

namespace jlm {

class verilatorcmd final : public Command {
public:
	virtual
	~verilatorcmd(){}

	verilatorcmd(
			const jlm::filepath & vfile,
			const std::vector<jlm::filepath> & lfiles,
			const jlm::filepath & hfile,
			const jlm::filepath & ofile,
            const jlm::filepath & tmpfolder,
			const std::vector<std::string> & Lpaths,
			const std::vector<std::string> & libs)
	: ofile_(ofile)
	, vfile_(vfile)
	, hfile_(hfile)
	, tmpfolder_(tmpfolder)
	, libs_(libs)
	, lfiles_(lfiles)
	, Lpaths_(Lpaths)
	{}

	virtual std::string
	ToString() const override;

	virtual void
	Run() const override;

	inline const jlm::filepath &
	vfile() const noexcept
	{
		return vfile_;
	}

	inline const jlm::filepath &
	hfile() const noexcept
	{
		return hfile_;
	}

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::filepath> &
	lfiles() const noexcept
	{
		return lfiles_;
	}

	static CommandGraph::Node *
	create(
    CommandGraph * pgraph,
    const jlm::filepath & vfile,
    const std::vector<jlm::filepath> & lfiles,
    const jlm::filepath & hfile,
    const jlm::filepath & ofile,
    const jlm::filepath & tmpfolder,
    const std::vector<std::string> & Lpaths,
    const std::vector<std::string> & libs)
	{
		std::unique_ptr<verilatorcmd> cmd(new verilatorcmd(vfile, lfiles, hfile, ofile, tmpfolder, Lpaths, libs));
		return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	jlm::filepath vfile_;
	jlm::filepath hfile_;
	jlm::filepath tmpfolder_;
	std::vector<std::string> libs_;
	std::vector<jlm::filepath> lfiles_;
	std::vector<std::string> Lpaths_;
};

class firrtlcmd final : public Command {
public:
	virtual
	~firrtlcmd(){}

	firrtlcmd(
		const jlm::filepath & ifile,
		const jlm::filepath & ofile)
	: ofile_(ofile)
	, ifile_(ifile)
	{}

	virtual std::string
	ToString() const override;

	virtual void
	Run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}

	static CommandGraph::Node *
	create(
    CommandGraph * pgraph,
    const jlm::filepath & ifile,
    const jlm::filepath & ofile)
	{
		std::unique_ptr<firrtlcmd> cmd(new firrtlcmd(ifile, ofile));
		return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	jlm::filepath ifile_;
};

std::basic_string<char>
gcd();

}

#endif
