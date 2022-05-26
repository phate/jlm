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

class hlscmd final : public Command {
public:
	virtual
	~hlscmd(){}

	hlscmd(
		const jlm::filepath & ifile,
		const std::string & outfolder,
	        const bool &circt)
	: ifile_(ifile)
	, outfolder_(outfolder)
	, circt_(circt)
	{}

	virtual std::string
	ToString() const override;

	virtual void
	Run() const override;

	inline const jlm::filepath
	firfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls.fir");
	}

	inline const jlm::filepath
	llfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls_rest.ll");
	}

	inline const jlm::filepath
	harnessfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls_harness.cpp");
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
    const std::string & outfolder,
    const bool &circt)
	{
		std::unique_ptr<hlscmd> cmd(new hlscmd(ifile, outfolder, circt));
		return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
	}

private:
	jlm::filepath ifile_;
	std::string outfolder_;
	bool circt_;
};

class extractcmd final : public Command {
public:
	virtual
	~extractcmd(){}

    extractcmd(
		const jlm::filepath & ifile,
		const std::string & function,
		const std::string & outfolder)
	: ifile_(ifile)
	, function_(function)
	, outfolder_(outfolder)
	{}

	virtual std::string
	ToString() const override;

	virtual void
	Run() const override;

	inline const jlm::filepath
	functionfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls_function.ll");
	}

	inline const jlm::filepath
	llfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls_rest.ll");
	}

	inline const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}
	inline const std::string &
	function() const noexcept
	{
		return function_;
	}

	static CommandGraph::Node *
	create(
    CommandGraph * pgraph,
    const jlm::filepath & ifile,
    const std::string & function,
    const std::string & outfolder)
	{
		std::unique_ptr<extractcmd> cmd(new extractcmd(ifile, function, outfolder));
		return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
	}

private:
	jlm::filepath ifile_;
	std::string function_;
	std::string outfolder_;
};

}

#endif
